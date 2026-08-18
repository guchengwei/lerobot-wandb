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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")

import wandb

from lerobot_wandb.refs import parse_artifact_ref
from lerobot_wandb.store import (
    ArtifactTypeMismatchError,
    DownloadDestinationNotEmptyError,
    MaterializedArtifact,
    PromotionNotVisibleError,
    RegistryLinkRefusedError,
    declare_input,
    download_artifact,
    link_to_registry,
    promote_model,
    upload_directory,
)


class _FakeArtifact:
    """Small stand-in for ``wandb.Artifact`` and ``Run.use_artifact`` results."""

    def __init__(self, name=None, type=None, metadata=None, **_kwargs):  # noqa: A002
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.added_dirs = []
        self.entity = "my-team"
        self.project = "my-project"
        self.version = "v7"
        self.digest = "abc123digest"
        self._download_root = None

    def add_dir(self, local_path, **_kwargs):
        self.added_dirs.append(local_path)

    def wait(self, timeout=None):
        if ":" not in self.name:
            self.name = f"{self.name}:{self.version}"
        return self

    @property
    def qualified_name(self):
        return f"{self.entity}/{self.project}/{self.name}"

    def download(self, root=None, **_kwargs):
        self._download_root = root
        return root


# ---------------------------------------------------------------------------
# upload_directory
# ---------------------------------------------------------------------------


def test_upload_directory_logs_artifact_with_expected_shape(tmp_path, monkeypatch):
    created = {}

    def _fake_artifact_ctor(name, type, metadata=None, **kwargs):  # noqa: A002
        artifact = _FakeArtifact(name=name, type=type, metadata=metadata)
        created["artifact"] = artifact
        return artifact

    monkeypatch.setattr(wandb, "Artifact", _fake_artifact_ctor)

    run = MagicMock()
    run.entity = "my-team"
    run.project = "my-project"
    run.log_artifact.side_effect = lambda artifact, aliases=None: artifact

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    result = upload_directory(
        run,
        dataset_dir,
        name="pick-cube",
        artifact_type="dataset",
        aliases=["latest", "raw"],
        metadata={"fps": 30},
    )

    artifact = created["artifact"]
    assert artifact.type == "dataset"
    assert artifact.metadata == {"fps": 30}
    assert artifact.added_dirs == [str(dataset_dir)]

    run.log_artifact.assert_called_once()
    call_args = run.log_artifact.call_args
    assert call_args.args[0] is artifact
    assert call_args.kwargs["aliases"] == ["latest", "raw"]

    assert isinstance(result, MaterializedArtifact)
    assert result.requested_ref == "my-team/my-project/pick-cube"
    assert result.resolved_ref == "my-team/my-project/pick-cube:v7"
    assert result.local_path == dataset_dir
    assert result.version == "v7"
    assert result.digest == "abc123digest"
    assert result.metadata == {"fps": 30}


def test_upload_directory_waits_for_commit(tmp_path, monkeypatch):
    waited = []

    class _WaitTrackingArtifact(_FakeArtifact):
        def wait(self, timeout=None):
            waited.append(True)
            return super().wait(timeout=timeout)

    monkeypatch.setattr(
        wandb,
        "Artifact",
        lambda name, type, metadata=None: _WaitTrackingArtifact(name=name, type=type, metadata=metadata),
    )

    run = MagicMock()
    run.entity = "e"
    run.project = "p"
    run.log_artifact.side_effect = lambda artifact, aliases=None: artifact

    upload_directory(run, tmp_path, name="n", artifact_type="dataset")
    assert waited == [True]


# ---------------------------------------------------------------------------
# link_to_registry
# ---------------------------------------------------------------------------


def test_link_to_registry_targets_unified_registry_collection():
    run = MagicMock()
    artifact = MagicMock()

    target_path = link_to_registry(run, artifact, collection="pick-cube-policy", aliases=["candidate"])

    assert target_path == "wandb-registry-model/pick-cube-policy"
    run.link_artifact.assert_called_once_with(
        artifact, target_path="wandb-registry-model/pick-cube-policy", aliases=["candidate"]
    )


def test_link_to_registry_without_aliases_passes_none():
    run = MagicMock()
    artifact = MagicMock()

    link_to_registry(run, artifact, collection="pick-cube-policy")

    run.link_artifact.assert_called_once_with(
        artifact, target_path="wandb-registry-model/pick-cube-policy", aliases=None
    )


def test_upload_directory_without_registry_collection_never_links(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wandb,
        "Artifact",
        lambda name, type, metadata=None: _FakeArtifact(name=name, type=type, metadata=metadata),
    )
    run = MagicMock()
    run.entity = "e"
    run.project = "p"
    run.log_artifact.side_effect = lambda artifact, aliases=None: artifact

    result = upload_directory(run, tmp_path, name="n", artifact_type="model")

    run.link_artifact.assert_not_called()
    assert result.registry_collection is None


def test_upload_directory_with_registry_collection_links_after_wait(tmp_path, monkeypatch):
    call_order = []

    class _TrackingArtifact(_FakeArtifact):
        def wait(self, timeout=None):
            call_order.append("wait")
            return super().wait(timeout=timeout)

    monkeypatch.setattr(
        wandb,
        "Artifact",
        lambda name, type, metadata=None: _TrackingArtifact(name=name, type=type, metadata=metadata),
    )
    run = MagicMock()
    run.entity = "e"
    run.project = "p"
    run.log_artifact.side_effect = lambda artifact, aliases=None: artifact
    run.link_artifact.side_effect = lambda *a, **kw: call_order.append("link")

    result = upload_directory(
        run,
        tmp_path,
        name="n",
        artifact_type="model",
        aliases=["candidate"],
        registry_collection="pick-cube-policy",
    )

    assert call_order == ["wait", "link"]
    run.link_artifact.assert_called_once()
    call_kwargs = run.link_artifact.call_args.kwargs
    assert call_kwargs["target_path"] == "wandb-registry-model/pick-cube-policy"
    assert call_kwargs["aliases"] == ["candidate"]
    assert result.registry_collection == "pick-cube-policy"


# ---------------------------------------------------------------------------
# download_artifact
# ---------------------------------------------------------------------------


def _run_with(artifact):
    run = MagicMock()
    run.use_artifact.return_value = artifact
    return run


def test_download_artifact_declares_input_and_atomically_materializes(tmp_path):
    fake = _FakeArtifact(name="pick-cube:v2", type="dataset")
    fake.version = "v2"
    run = _run_with(fake)
    destination = tmp_path / "materialized"

    result = download_artifact(
        run,
        "my-team/my-project/pick-cube:latest",
        expected_type="dataset",
        download_root=destination,
    )

    run.use_artifact.assert_called_once_with("my-team/my-project/pick-cube:latest")
    assert Path(fake._download_root).parent == tmp_path
    assert Path(fake._download_root).name.startswith(".materialized.download-")
    assert not Path(fake._download_root).exists()
    assert destination.is_dir()
    assert isinstance(result, MaterializedArtifact)
    assert result.requested_ref == "my-team/my-project/pick-cube:latest"
    assert result.resolved_ref == "my-team/my-project/pick-cube:v2"
    assert result.version == "v2"
    assert result.local_path == destination


def test_download_artifact_accepts_parsed_ref(tmp_path):
    fake = _FakeArtifact(name="pick-cube:v2", type="dataset")
    fake.version = "v2"
    run = _run_with(fake)
    destination = tmp_path / "materialized"

    ref = parse_artifact_ref("my-team/my-project/pick-cube:v2")
    result = download_artifact(run, ref, expected_type="dataset", download_root=destination)

    run.use_artifact.assert_called_once_with("my-team/my-project/pick-cube:v2")
    assert result.requested_ref == "my-team/my-project/pick-cube:v2"


def test_download_artifact_rejects_type_mismatch_without_downloading(tmp_path):
    fake = _FakeArtifact(name="candidate-model:v0", type="model")
    run = _run_with(fake)

    with pytest.raises(ArtifactTypeMismatchError):
        download_artifact(
            run,
            "my-team/my-project/candidate-model:v0",
            expected_type="dataset",
            download_root=tmp_path / "materialized",
        )

    assert fake._download_root is None


def test_download_artifact_rejects_nonempty_destination_without_touching_it(tmp_path):
    destination = tmp_path / "materialized"
    destination.mkdir()
    sentinel = destination / "unrelated.txt"
    sentinel.write_text("keep me")
    run = MagicMock()

    with pytest.raises(DownloadDestinationNotEmptyError):
        download_artifact(
            run,
            "my-team/my-project/pick-cube:v0",
            expected_type="dataset",
            download_root=destination,
        )

    run.use_artifact.assert_not_called()
    assert sentinel.read_text() == "keep me"


def test_download_artifact_rejects_existing_file_destination(tmp_path):
    destination = tmp_path / "materialized"
    destination.write_text("keep me")
    run = MagicMock()

    with pytest.raises(DownloadDestinationNotEmptyError):
        download_artifact(
            run,
            "my-team/my-project/pick-cube:v0",
            expected_type="dataset",
            download_root=destination,
        )

    run.use_artifact.assert_not_called()
    assert destination.read_text() == "keep me"


def test_download_artifact_accepts_empty_existing_destination(tmp_path):
    destination = tmp_path / "materialized"
    destination.mkdir()
    fake = _FakeArtifact(name="pick-cube:v0", type="dataset")
    fake.version = "v0"

    result = download_artifact(
        _run_with(fake),
        "my-team/my-project/pick-cube:v0",
        expected_type="dataset",
        download_root=destination,
    )

    assert destination.is_dir()
    assert result.local_path == destination


def test_download_artifact_validates_staging_before_promotion(tmp_path):
    destination = tmp_path / "materialized"
    fake = _FakeArtifact(name="pick-cube:v0", type="dataset")
    fake.version = "v0"
    validated = []

    def _validate(path: Path):
        validated.append(path)
        assert path != destination
        (path / "validated.txt").write_text("ok")

    result = download_artifact(
        _run_with(fake),
        "my-team/my-project/pick-cube:v0",
        expected_type="dataset",
        download_root=destination,
        validator=_validate,
    )

    assert len(validated) == 1
    assert result.local_path == destination
    assert (destination / "validated.txt").read_text() == "ok"


def test_download_artifact_cleans_staging_after_download_failure(tmp_path):
    destination = tmp_path / "materialized"

    class _FailingArtifact(_FakeArtifact):
        def download(self, root=None, **_kwargs):
            self._download_root = root
            Path(root, "partial.txt").write_text("partial")
            raise RuntimeError("network failed")

    fake = _FailingArtifact(name="pick-cube:v0", type="dataset")

    with pytest.raises(RuntimeError, match="network failed"):
        download_artifact(
            _run_with(fake),
            "my-team/my-project/pick-cube:v0",
            expected_type="dataset",
            download_root=destination,
        )

    assert not destination.exists()
    assert not Path(fake._download_root).exists()


def test_download_artifact_cleans_staging_after_validation_failure(tmp_path):
    destination = tmp_path / "materialized"
    fake = _FakeArtifact(name="pick-cube:v0", type="dataset")

    def _reject(_path: Path):
        raise ValueError("invalid dataset")

    with pytest.raises(ValueError, match="invalid dataset"):
        download_artifact(
            _run_with(fake),
            "my-team/my-project/pick-cube:v0",
            expected_type="dataset",
            download_root=destination,
            validator=_reject,
        )

    assert not destination.exists()
    assert not Path(fake._download_root).exists()


# ---------------------------------------------------------------------------
# declare_input
# ---------------------------------------------------------------------------


def test_declare_input_resolves_the_ref_without_downloading_anything():
    """Lineage-only: the edge is drawn and the alias resolved, but no bytes are fetched, so there
    is no local path to report.
    """
    # W&B resolves the mutable alias, so `use_artifact` hands back the immutable version.
    artifact = _FakeArtifact(name="pick-cube-policy:v7", type="model", metadata={"policy": "act"})
    run = MagicMock()
    run.use_artifact.return_value = artifact

    result = declare_input(run, "my-team/my-project/pick-cube-policy:latest", expected_type="model")

    run.use_artifact.assert_called_once_with("my-team/my-project/pick-cube-policy:latest")
    assert artifact._download_root is None
    assert result.local_path is None
    assert result.requested_ref == "my-team/my-project/pick-cube-policy:latest"
    assert result.resolved_ref == "my-team/my-project/pick-cube-policy:v7"
    assert result.version == "v7"
    assert result.metadata == {"policy": "act"}


def test_declare_input_rejects_the_wrong_artifact_type():
    run = MagicMock()
    run.use_artifact.return_value = _FakeArtifact(name="pick-cube:latest", type="dataset")

    with pytest.raises(ArtifactTypeMismatchError, match="type 'dataset'"):
        declare_input(run, "my-team/my-project/pick-cube:latest", expected_type="model")


def test_declare_input_rejects_a_malformed_ref_before_calling_wandb():
    run = MagicMock()

    with pytest.raises(ValueError):
        declare_input(run, "not-a-ref", expected_type="model")

    run.use_artifact.assert_not_called()


# ---------------------------------------------------------------------------
# promote_model
#
# These tests prove which SDK calls `promote_model` makes and which it does not. They cannot
# prove what W&B's servers do with those calls: that assigning an alias to v3 takes it off v2 is
# enforced server-side, and a fake says yes regardless. `test_wandb_sdk_surface.py` pins that the
# methods called here still exist on the real package; the alias actually moving was verified by
# hand against a live project, and is recorded in the PR that added this command.
# ---------------------------------------------------------------------------


class _FakePromotableArtifact(_FakeArtifact):
    """A committed artifact as ``Api().artifact()`` returns it: aliases, a manifest, no bytes."""

    def __init__(self, *, entries=(CONFIG_NAME, SAFETENSORS_SINGLE_FILE), aliases=("v3",), **kwargs):
        super().__init__(**kwargs)
        self.aliases = list(aliases)
        self.manifest = SimpleNamespace(entries=dict.fromkeys(entries, object()))
        self.saved = 0
        self.links = []

    def save(self):
        self.saved += 1

    def link(self, target_path, aliases=None):
        self.links.append((target_path, aliases))


def _fake_api(artifact, monkeypatch, *, alias_resolves_to=None):
    """`Api().artifact()` is called twice: once for the requested ref, once to re-resolve the alias.

    `alias_resolves_to` stands in for a server that did *not* move the alias.
    """
    calls = []

    def _artifact(name):
        calls.append(name)
        # Call 1 is the requested ref; call 2 is the alias read-back.
        if len(calls) > 1 and alias_resolves_to is not None:
            return alias_resolves_to
        return artifact

    api = MagicMock()
    api.artifact.side_effect = _artifact
    monkeypatch.setattr(wandb, "Api", lambda *a, **k: api)
    return api


def test_promote_model_aliases_the_existing_version_and_uploads_nothing(monkeypatch):
    artifact = _FakePromotableArtifact(name="pick-cube-policy:v3", type="model")
    api = _fake_api(artifact, monkeypatch)

    result = promote_model(
        "my-team/my-project/pick-cube-policy:v3",
        alias="production",
        registry_collection="pick-cube-policy",
    )

    assert [c.args[0] for c in api.artifact.call_args_list] == [
        "my-team/my-project/pick-cube-policy:v3",
        # The alias is re-resolved to prove the promotion is visible, not assumed.
        "my-team/my-project/pick-cube-policy:production",
    ]
    assert artifact.aliases == ["v3", "production"]
    assert artifact.saved == 1
    assert artifact.links == [("wandb-registry-model/pick-cube-policy", ["production"])]
    # Nothing was uploaded: the version promoted is the version asked for, same digest.
    assert artifact.added_dirs == []
    assert result.resolved_ref == "my-team/my-project/pick-cube-policy:v3"
    assert result.digest == "abc123digest"
    assert result.local_path is None
    assert result.registry_collection == "pick-cube-policy"


def test_promote_model_without_a_collection_touches_only_the_project_alias(monkeypatch):
    artifact = _FakePromotableArtifact(name="pick-cube-policy:v3", type="model")
    _fake_api(artifact, monkeypatch)

    result = promote_model("my-team/my-project/pick-cube-policy:v3", alias="production")

    assert artifact.aliases == ["v3", "production"]
    assert artifact.links == []
    assert result.registry_collection is None


def test_promote_model_is_idempotent_for_an_alias_already_on_the_version(monkeypatch):
    artifact = _FakePromotableArtifact(name="pick-cube-policy:v3", type="model", aliases=("production",))
    _fake_api(artifact, monkeypatch)

    promote_model("my-team/my-project/pick-cube-policy:v3", alias="production")

    assert artifact.aliases == ["production"]
    assert artifact.saved == 0


def test_promote_model_rejects_a_non_model_artifact(monkeypatch):
    artifact = _FakePromotableArtifact(name="pick-cube:v3", type="dataset")
    _fake_api(artifact, monkeypatch)

    with pytest.raises(ArtifactTypeMismatchError, match="type 'dataset'"):
        promote_model("my-team/my-project/pick-cube:v3", alias="production")

    assert artifact.saved == 0


def test_promote_model_refuses_to_register_a_version_that_cannot_be_rolled_out(monkeypatch):
    """The manifest is the whole check: an adapter-only version has no `model.safetensors`.

    Refused before the alias is applied — a version left aliased `production` but unlinked is a
    worse outcome than a command that did nothing.
    """
    artifact = _FakePromotableArtifact(
        name="pick-cube-policy:v3",
        type="model",
        entries=(CONFIG_NAME, "adapter_config.json", "adapter_model.safetensors"),
        metadata={"base_model_name_or_path": "lerobot/pi0"},
    )
    _fake_api(artifact, monkeypatch)

    with pytest.raises(RegistryLinkRefusedError, match="lerobot/pi0"):
        promote_model(
            "my-team/my-project/pick-cube-policy:v3",
            alias="production",
            registry_collection="pick-cube-policy",
        )

    assert artifact.saved == 0
    assert artifact.links == []
    assert artifact.aliases == ["v3"]


def test_promote_model_allows_an_adapter_only_alias_without_a_registry_link(monkeypatch):
    """Matches `model upload`, which uploads an adapter-only checkpoint and refuses only the link."""
    artifact = _FakePromotableArtifact(
        name="pick-cube-policy:v3",
        type="model",
        entries=(CONFIG_NAME, "adapter_config.json", "adapter_model.safetensors"),
    )
    _fake_api(artifact, monkeypatch)

    promote_model("my-team/my-project/pick-cube-policy:v3", alias="candidate")

    assert artifact.aliases == ["v3", "candidate"]


def test_promote_model_rejects_a_malformed_ref_before_calling_wandb(monkeypatch):
    api = MagicMock()
    monkeypatch.setattr(wandb, "Api", lambda *a, **k: api)

    with pytest.raises(ValueError):
        promote_model("not-a-ref", alias="production")

    api.artifact.assert_not_called()


def test_promote_model_fails_when_the_alias_still_resolves_elsewhere(monkeypatch):
    """The one server behaviour no mock can vouch for, turned into a checked postcondition.

    If W&B ever stops moving an alias off the version that held it, two versions would claim
    `production` and the CLI would report success. This is what makes that loud instead.
    """
    artifact = _FakePromotableArtifact(name="pick-cube-policy:v3", type="model")
    stale = _FakePromotableArtifact(name="pick-cube-policy:v2", type="model")
    stale.version = "v2"
    _fake_api(artifact, monkeypatch, alias_resolves_to=stale)

    with pytest.raises(PromotionNotVisibleError, match="still resolves to v2"):
        promote_model(
            "my-team/my-project/pick-cube-policy:v3",
            alias="production",
            registry_collection="pick-cube-policy",
        )

    # The link ran first (see `promote_model` on ordering), so it has already happened when the
    # alias turns out not to be visible. That residual state is why the error names it.
    assert artifact.links == [("wandb-registry-model/pick-cube-policy", ["production"])]


def test_promote_model_refuses_a_weights_only_periodic_checkpoint(monkeypatch):
    """Weights present is not the same as loadable.

    `WandBLogger.log_policy` uploads a full-weight periodic checkpoint as `model.safetensors`
    alone. It passes an "is it self-contained" check that looks only at weights, and still cannot
    be loaded as a policy, because `PreTrainedConfig.from_pretrained` needs `config.json`.
    """
    artifact = _FakePromotableArtifact(
        name="pick-cube-policy:v3", type="model", entries=(SAFETENSORS_SINGLE_FILE,)
    )
    _fake_api(artifact, monkeypatch)

    with pytest.raises(RegistryLinkRefusedError, match=CONFIG_NAME):
        promote_model(
            "my-team/my-project/pick-cube-policy:v3",
            alias="production",
            registry_collection="pick-cube-policy",
        )

    assert artifact.saved == 0
    assert artifact.links == []


def test_promote_model_links_before_moving_the_project_alias(monkeypatch):
    """A failed link must leave nothing changed, not a `production` alias with no Registry entry."""
    order = []

    class _FailingLinkArtifact(_FakePromotableArtifact):
        def link(self, target_path, aliases=None):
            order.append("link")
            raise RuntimeError("permission denied on wandb-registry-model/pick-cube-policy")

        def save(self):
            order.append("save")
            super().save()

    artifact = _FailingLinkArtifact(name="pick-cube-policy:v3", type="model")
    _fake_api(artifact, monkeypatch)

    with pytest.raises(RuntimeError, match="permission denied"):
        promote_model(
            "my-team/my-project/pick-cube-policy:v3",
            alias="production",
            registry_collection="pick-cube-policy",
        )

    assert order == ["link"]
    assert artifact.aliases == ["v3"]
