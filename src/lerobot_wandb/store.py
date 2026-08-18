# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin W&B wrappers for uploading and transactionally downloading versioned Artifacts.

Every operation returns a :class:`MaterializedArtifact` carrying both the requested and resolved
(immutable) reference — see the terminology in the repository README — never
only one or the other. Importing this module does not require W&B; the optional SDK is checked only
when an upload or download is actually attempted.
"""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE
from packaging.version import Version

from .refs import ArtifactRef, parse_artifact_ref

if TYPE_CHECKING:
    import wandb

MODEL_ARTIFACT_TYPE = "model"


def _require_package(package: str, *, install_hint: str) -> None:
    """Raise an actionable error when ``package`` is not importable."""
    try:
        importlib.import_module(package)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{package} is required but not installed. Install it with: {install_hint}"
        ) from e


class ArtifactTypeMismatchError(ValueError):
    """A fetched artifact's declared type doesn't match what the caller expected."""


class RegistryLinkRefusedError(ValueError):
    """A Registry link was requested for a version that cannot be rolled out on its own."""


class PromotionNotVisibleError(RuntimeError):
    """An alias was applied, but the reference still resolves to a different version."""


class DownloadDestinationNotEmptyError(ValueError):
    """A download destination contains content that must not be overwritten."""


@dataclass(frozen=True, slots=True)
class MaterializedArtifact:
    """The result of uploading or downloading a W&B Artifact.

    ``requested_ref`` is what the caller asked for (a collection path for an upload, or the
    possibly-mutable-alias reference string for a download). ``resolved_ref`` is always the
    immutable ``entity/project/name:vN`` W&B actually resolved to. ``local_path`` is the directory
    that now holds (or, for an upload, already held) the artifact's contents on disk — the source
    directory for an upload, the download destination for a download, and ``None`` when the
    artifact was only *referenced* and never fetched (see :func:`declare_input`).
    ``registry_collection`` is the unified-Registry collection name the artifact was linked into
    (see :func:`link_to_registry`), or ``None`` when the caller didn't request a registry link.
    """

    requested_ref: str
    resolved_ref: str
    local_path: Path | None
    version: str
    digest: str
    metadata: dict[str, Any]
    registry_collection: str | None = None


def _wandb_sdk() -> Any:
    """Return a supported W&B SDK, raising only when an SDK operation is requested."""
    _require_package("wandb", install_hint="pip install lerobot-wandb")

    import wandb

    if Version(version("wandb")) < Version("0.24.1"):
        raise RuntimeError(
            "lerobot-wandb requires wandb>=0.24.1 because wandb 0.24.0 can silently fail to upload "
            "run data. Upgrade the training extra before using artifact transfers."
        )
    return wandb


def link_to_registry(
    run: wandb.sdk.wandb_run.Run,
    artifact: wandb.Artifact,
    *,
    collection: str,
    aliases: Sequence[str] = (),
) -> str:
    """Link an already-logged ``artifact`` into the ``collection`` of W&B's unified Registry.

    Uses ``Run.link_artifact(artifact, target_path="wandb-registry-model/<collection>")`` — the
    unified Registry — rather than ``Run.link_model()``, which hardcodes W&B's legacy, sunsetting
    Model Registry (``project="model-registry"``). The standalone package uses only the unified
    Registry path.

    Returns the ``target_path`` the artifact was linked into.
    """
    target_path = f"wandb-registry-model/{collection}"
    run.link_artifact(artifact, target_path=target_path, aliases=list(aliases) if aliases else None)
    return target_path


def _registry_refusal_from_manifest(artifact: wandb.Artifact) -> str | None:
    """Why ``artifact`` must not be linked into the Registry, judged without downloading it.

    Mirrors ``validate_model_directory`` plus ``registry_link_refusal``, which together decide
    whether a *local* directory is a loadable, self-contained policy. Both check only file
    existence, so ``manifest.entries`` — keyed by in-artifact path — answers the same question for
    a remote version at the cost of no bytes.

    Both halves matter. ``WandBLogger.log_policy`` uploads a full-weight periodic checkpoint as
    ``model.safetensors`` alone, with no ``config.json``: weights present, still unloadable, since
    ``PreTrainedConfig.from_pretrained`` needs the config.
    """
    entries = artifact.manifest.entries
    if CONFIG_NAME not in entries:
        return (
            f"the artifact has no {CONFIG_NAME}, so it cannot be loaded as a LeRobot policy "
            "(a periodic training checkpoint is uploaded as weights alone)"
        )

    # Deferred: `inspect` pulls datasets/pandas/pyarrow (the `dataset` extra), and importing this
    # module must stay possible on a base install — see the module docstring.
    from .inspect import registry_link_refusal

    return registry_link_refusal(
        is_self_contained=SAFETENSORS_SINGLE_FILE in entries,
        base_model_name_or_path=(artifact.metadata or {}).get("base_model_name_or_path"),
    )


def promote_model(
    ref: str | ArtifactRef,
    *,
    alias: str,
    registry_collection: str | None = None,
) -> MaterializedArtifact:
    """Give an already-logged model version ``alias``, and optionally link *that* version.

    Promotion has to act on the exact immutable version a rollout evaluated. Re-uploading the
    downloaded policy would produce a different version carrying no edge to the rollout that
    justified it, so nothing here writes bytes: the version is fetched by reference, aliased in
    place, and linked as-is.

    Runless by design, unlike every other operation in this module. ``Api().artifact`` reads,
    ``save()`` on a committed artifact takes the update path rather than logging anything, and
    ``Artifact.link`` reaches the Registry without a ``Run``. A run here would exist only to
    exist — no inputs, no outputs, no metrics.

    Deployability is judged from the artifact's file manifest by
    :func:`_registry_refusal_from_manifest`, not from a download and not from its stored
    ``is_self_contained``: the manifest carries the same file-existence signal the local checks
    use, costs no bytes, and is not mutable after the fact.

    Raises:
        ArtifactTypeMismatchError: ``ref`` is not a ``model`` artifact.
        RegistryLinkRefusedError: a Registry link was requested for a version that cannot be
            rolled out on its own. The alias is not applied either — the whole command is refused,
            because a half-done promotion is worse than none.
        PromotionNotVisibleError: the alias was applied but the reference still resolves
            elsewhere. Any requested Registry link was already made by then — two calls, no
            transaction, so the choice is which partial state, not whether one exists.
    """
    parsed = ref if isinstance(ref, ArtifactRef) else parse_artifact_ref(ref)
    wandb = _wandb_sdk()

    artifact = wandb.Api().artifact(str(parsed))
    if artifact.type != MODEL_ARTIFACT_TYPE:
        raise ArtifactTypeMismatchError(
            f"Expected an artifact of type {MODEL_ARTIFACT_TYPE!r} but {parsed} is of type {artifact.type!r}."
        )

    if registry_collection is not None:
        refusal = _registry_refusal_from_manifest(artifact)
        if refusal is not None:
            raise RegistryLinkRefusedError(
                f"Refusing to link {artifact.qualified_name} into Registry collection "
                f"{registry_collection!r}: {refusal}."
            )

        # Linked first: no transaction covers both writes, and a failed link this way changes
        # nothing, where aliasing first would leave `production` on a version that never reached
        # the Registry.
        artifact.link(f"wandb-registry-model/{registry_collection}", aliases=[alias])

    if alias not in artifact.aliases:
        artifact.aliases = [*artifact.aliases, alias]
        artifact.save()

    # Check the postcondition, not the mechanism: that W&B takes an alias off the version holding
    # it is server-enforced and unprovable from a test, but "the ref now resolves to what we
    # promoted" is one read.
    resolved = wandb.Api().artifact(f"{parsed.entity}/{parsed.project}/{parsed.name}:{alias}")
    if resolved.version != artifact.version:
        raise PromotionNotVisibleError(
            f"Applied alias {alias!r} to {artifact.qualified_name}, but "
            f"{parsed.entity}/{parsed.project}/{parsed.name}:{alias} still resolves to "
            f"{resolved.version}. Do not treat {alias!r} as pointing at {artifact.version}; any "
            f"Registry link requested here was already made."
        )

    return MaterializedArtifact(
        requested_ref=str(parsed),
        resolved_ref=artifact.qualified_name,
        local_path=None,
        version=artifact.version,
        digest=artifact.digest,
        metadata=dict(artifact.metadata or {}),
        registry_collection=registry_collection,
    )


def upload_directory(
    run: wandb.sdk.wandb_run.Run,
    directory: Path | str,
    *,
    name: str,
    artifact_type: str,
    aliases: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    registry_collection: str | None = None,
) -> MaterializedArtifact:
    """Upload ``directory`` as a new version of the ``name`` Artifact collection.

    Waits for W&B to confirm the upload is fully committed before returning, so a caller that
    gets a result back knows the artifact genuinely exists in W&B, not just that local files were
    queued for upload. When ``registry_collection`` is given, the committed version is additionally
    linked into that unified-Registry collection via :func:`link_to_registry`, after the commit
    (``.wait()``) — never before — so only a durably-logged version is ever linked.
    """
    directory = Path(directory)
    requested_ref = f"{run.entity}/{run.project}/{name}"
    wandb = _wandb_sdk()

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=dict(metadata) if metadata else None)
    artifact.add_dir(str(directory))

    logged = run.log_artifact(artifact, aliases=list(aliases) if aliases else None)
    logged.wait()

    if registry_collection is not None:
        link_to_registry(run, logged, collection=registry_collection, aliases=aliases)

    return MaterializedArtifact(
        requested_ref=requested_ref,
        resolved_ref=logged.qualified_name,
        local_path=directory,
        version=logged.version,
        digest=logged.digest,
        metadata=dict(logged.metadata or {}),
        registry_collection=registry_collection,
    )


def _use_artifact(run: wandb.sdk.wandb_run.Run, ref: ArtifactRef, expected_type: str) -> wandb.Artifact:
    """Draw the lineage edge from ``run`` to ``ref`` and check the type before any bytes move."""
    _wandb_sdk()
    artifact = run.use_artifact(str(ref))
    if artifact.type != expected_type:
        raise ArtifactTypeMismatchError(
            f"Expected an artifact of type {expected_type!r} but {ref} is of type {artifact.type!r}."
        )
    return artifact


def declare_input(
    run: wandb.sdk.wandb_run.Run,
    ref: str | ArtifactRef,
    *,
    expected_type: str,
) -> MaterializedArtifact:
    """Declare ``ref`` as an input of ``run`` for lineage, without downloading it.

    ``use_artifact`` both draws the lineage edge and resolves a possibly-mutable alias to an
    immutable version, so this costs one metadata call and no bytes — the point when a caller needs
    to record *which* model produced something it is not going to load. The returned
    :class:`MaterializedArtifact` has ``local_path=None`` precisely because nothing was fetched.

    Raises:
        ArtifactTypeMismatchError: the artifact's declared type isn't ``expected_type``.
    """
    parsed = ref if isinstance(ref, ArtifactRef) else parse_artifact_ref(ref)
    artifact = _use_artifact(run, parsed, expected_type)
    return MaterializedArtifact(
        requested_ref=str(parsed),
        resolved_ref=artifact.qualified_name,
        local_path=None,
        version=artifact.version,
        digest=artifact.digest,
        metadata=dict(artifact.metadata or {}),
    )


def download_artifact(
    run: wandb.sdk.wandb_run.Run,
    ref: str | ArtifactRef,
    *,
    expected_type: str,
    download_root: Path | str,
    validator: Callable[[Path], object] | None = None,
) -> MaterializedArtifact:
    """Declare ``ref`` as a run input and transactionally materialize it at ``download_root``.

    The artifact is downloaded into a temporary sibling directory. If supplied, ``validator`` runs
    against that staged directory. Only a complete, valid result is atomically promoted to the final
    destination, so interrupted downloads and validation failures never poison the caller-visible
    path. A pre-existing empty directory is accepted; any file, symlink, or nonempty directory is
    rejected and never modified.

    Raises:
        DownloadDestinationNotEmptyError: ``download_root`` is not absent or an empty directory.
        ArtifactTypeMismatchError: the fetched artifact's declared type isn't ``expected_type``.
            Raised before any download happens.
        Exception: Any download or validation failure, after cleaning up the private staging path.
    """
    parsed = ref if isinstance(ref, ArtifactRef) else parse_artifact_ref(ref)
    download_root = Path(download_root)

    destination_was_empty = False
    if download_root.exists() or download_root.is_symlink():
        if download_root.is_symlink() or not download_root.is_dir() or any(download_root.iterdir()):
            raise DownloadDestinationNotEmptyError(
                f"{download_root} already contains or points to local content. Point at an empty "
                "directory or a path that does not exist."
            )
        destination_was_empty = True

    artifact = _use_artifact(run, parsed, expected_type)

    download_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{download_root.name}.download-", dir=download_root.parent))
    removed_empty_destination = False

    try:
        staged_path = Path(artifact.download(root=str(staging_root)))
        if validator is not None:
            validator(staged_path)

        if destination_was_empty:
            # rmdir is deliberately race-safe: it fails rather than removing newly-created content.
            download_root.rmdir()
            removed_empty_destination = True

        os.replace(staged_path, download_root)
    except Exception:
        if removed_empty_destination and not download_root.exists():
            download_root.mkdir()
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    return MaterializedArtifact(
        requested_ref=str(parsed),
        resolved_ref=artifact.qualified_name,
        local_path=download_root,
        version=artifact.version,
        digest=artifact.digest,
        metadata=dict(artifact.metadata or {}),
    )
