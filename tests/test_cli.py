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

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from typing import Any

from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

pytest.importorskip("lerobot", reason="lerobot is required (install a supported lerobot release)")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pyav_utils import get_codec

from lerobot_wandb import cli
from lerobot_wandb.dataset_transfer import TransferDataset
from lerobot_wandb.inspect import DatasetDirectoryError, DatasetDirectoryMetadata, ModelDirectoryError
from lerobot_wandb.store import ArtifactTypeMismatchError, MaterializedArtifact

require_h264 = pytest.mark.skipif(
    get_codec("h264") is None, reason="'h264' encoder not in local FFmpeg build"
)

_ACTION_FEATURE = {"dtype": "float32", "shape": (6,), "names": None}


def _write_minimal_dataset(root: Path) -> None:
    """A tiny, genuinely valid local LeRobot dataset (`root` must not already exist).

    Built with the real dataset writer rather than hand-assembled JSON so it stays valid as
    `inspect.validate_dataset_directory`'s requirements evolve.
    """
    dataset = LeRobotDataset.create(
        repo_id="tests/wandb-artifacts-cli",
        fps=30,
        features={"action": _ACTION_FEATURE},
        root=root,
        robot_type="so101",
        use_videos=False,
        video_backend="pyav",
        metadata_buffer_size=1,
    )
    dataset.add_frame({"action": np.zeros(6, dtype=np.float32), "task": "task-0"})
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


def _write_minimal_model(root: Path) -> None:
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / CONFIG_NAME).write_text(json.dumps({"type": "act"}))
    (root / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")


def _fake_run():
    run = MagicMock()
    run.entity = "my-team"
    run.project = "my-project"
    return run


def _materialized_upload_result():
    return MaterializedArtifact(
        requested_ref="my-team/my-project/pick-cube",
        resolved_ref="my-team/my-project/pick-cube:v0",
        local_path=Path("/tmp/does-not-matter"),
        version="v0",
        digest="digest",
        metadata={},
    )


def test_dataset_upload_validates_before_touching_wandb(tmp_path, monkeypatch):
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())
    upload_calls = []
    monkeypatch.setattr(
        cli,
        "upload_directory",
        lambda *a, **kw: upload_calls.append((a, kw)) or _materialized_upload_result(),
    )

    empty_root = tmp_path / "not-a-dataset"
    empty_root.mkdir()

    with pytest.raises(DatasetDirectoryError):
        cli.main(["dataset", "upload", "--root", str(empty_root), "--project", "p", "--name", "n"])

    assert init_calls == []
    assert upload_calls == []


def test_dataset_upload_happy_path(tmp_path, monkeypatch, capsys):
    dataset_root = tmp_path / "dataset"
    _write_minimal_dataset(dataset_root)

    run = _fake_run()
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or run)
    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None):
        upload_calls.append(
            {
                "run": passed_run,
                "directory": Path(directory),
                "name": name,
                "artifact_type": artifact_type,
                "aliases": list(aliases),
                "metadata": metadata,
            }
        )
        return _materialized_upload_result()

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)

    cli.main(
        [
            "dataset",
            "upload",
            "--root",
            str(dataset_root),
            "--project",
            "my-project",
            "--entity",
            "my-team",
            "--name",
            "pick-cube",
            "--alias",
            "raw",
            "--alias",
            "clean",
        ]
    )

    assert init_calls[0]["project"] == "my-project"
    assert init_calls[0]["entity"] == "my-team"
    assert init_calls[0]["mode"] == "online"
    run.finish.assert_called_once()

    assert len(upload_calls) == 1
    call = upload_calls[0]
    assert call["directory"] == dataset_root
    assert call["name"] == "pick-cube"
    assert call["artifact_type"] == "dataset"
    assert call["aliases"] == ["raw", "clean"]
    assert call["metadata"]["schema_version"] == "v3.0"

    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube:v0" in out
    assert "raw" in out and "clean" in out


def test_dataset_upload_finishes_run_even_on_upload_failure(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    _write_minimal_dataset(dataset_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    def _boom(*a, **kw):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(cli, "upload_directory", _boom)

    with pytest.raises(RuntimeError):
        cli.main(["dataset", "upload", "--root", str(dataset_root), "--project", "p", "--name", "n"])

    run.finish.assert_called_once()


def test_transfer_commands_do_not_expose_offline_or_disabled_modes(tmp_path):
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "dataset",
                "upload",
                "--root",
                str(tmp_path),
                "--project",
                "p",
                "--name",
                "n",
                "--mode",
                "offline",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "dataset",
                "download",
                "--ref",
                "e/p/n:v0",
                "--root",
                str(tmp_path / "dataset"),
                "--mode",
                "disabled",
            ]
        )


def test_dataset_download_rejects_malformed_ref_before_touching_wandb(tmp_path, monkeypatch):
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())

    with pytest.raises(ValueError):
        cli.main(["dataset", "download", "--ref", "not-a-valid-ref", "--root", str(tmp_path)])

    assert init_calls == []


def test_dataset_download_happy_path(tmp_path, monkeypatch, capsys):
    run = _fake_run()
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or run)

    dest = tmp_path / "materialized"
    validator_calls = []

    def _fake_download(passed_run, ref, *, expected_type, download_root, validator=None):
        _write_minimal_dataset(Path(download_root))
        validator_calls.append(validator)
        validator(Path(download_root))
        return MaterializedArtifact(
            requested_ref=str(ref),
            resolved_ref="my-team/my-project/pick-cube:v3",
            local_path=Path(download_root),
            version="v3",
            digest="digest",
            metadata={},
        )

    monkeypatch.setattr(cli, "download_artifact", _fake_download)

    cli.main(["dataset", "download", "--ref", "my-team/my-project/pick-cube:latest", "--root", str(dest)])

    assert init_calls[0]["entity"] == "my-team"
    assert init_calls[0]["project"] == "my-project"
    assert init_calls[0]["mode"] == "online"
    assert validator_calls == [cli.validate_transfer_dataset]
    run.finish.assert_called_once()

    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube:v3" in out
    assert str(dest) in out


def test_dataset_download_allows_logging_the_run_in_a_different_project(tmp_path, monkeypatch):
    """A read-only source project must not also be the mandatory lineage-run destination."""
    run = _fake_run()
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or run)

    dest = tmp_path / "materialized"
    download_calls = []

    def _fake_download(passed_run, ref, *, expected_type, download_root, validator=None):
        download_calls.append(str(ref))
        _write_minimal_dataset(Path(download_root))
        validator(Path(download_root))
        return MaterializedArtifact(
            requested_ref=str(ref),
            resolved_ref="source-team/source-project/pick-cube:v3",
            local_path=Path(download_root),
            version="v3",
            digest="digest",
            metadata={},
        )

    monkeypatch.setattr(cli, "download_artifact", _fake_download)

    cli.main(
        [
            "dataset",
            "download",
            "--ref",
            "source-team/source-project/pick-cube:latest",
            "--root",
            str(dest),
            "--entity",
            "my-own-team",
            "--project",
            "my-own-project",
        ]
    )

    assert init_calls[0]["entity"] == "my-own-team"
    assert init_calls[0]["project"] == "my-own-project"
    assert init_calls[0]["mode"] == "online"
    assert download_calls == ["source-team/source-project/pick-cube:latest"]


def test_dataset_download_rejects_result_missing_required_files(tmp_path, monkeypatch):
    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"

    def _fake_download_incomplete(passed_run, ref, *, expected_type, download_root, validator=None):
        Path(download_root).mkdir(parents=True, exist_ok=True)
        validator(Path(download_root))
        raise AssertionError("validator should have rejected the incomplete directory")

    monkeypatch.setattr(cli, "download_artifact", _fake_download_incomplete)

    with pytest.raises(DatasetDirectoryError):
        cli.main(["dataset", "download", "--ref", "my-team/my-project/pick-cube:latest", "--root", str(dest)])

    run.finish.assert_called_once()


# ---------------------------------------------------------------------------
# model upload / download
# ---------------------------------------------------------------------------


def _materialized_model_upload_result(registry_collection=None):
    return MaterializedArtifact(
        requested_ref="my-team/my-project/pick-cube-policy",
        resolved_ref="my-team/my-project/pick-cube-policy:v0",
        local_path=Path("/tmp/does-not-matter"),
        version="v0",
        digest="digest",
        metadata={},
        registry_collection=registry_collection,
    )


def test_model_upload_validates_before_touching_wandb(tmp_path, monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("wandb.init should not be called before validation")

    monkeypatch.setattr(cli.wandb, "init", _boom)

    not_a_model = tmp_path / "not-a-model"
    not_a_model.mkdir()

    with pytest.raises(ModelDirectoryError):
        cli.main(["model", "upload", "--root", str(not_a_model), "--project", "p", "--name", "n"])


def test_model_upload_happy_path_without_registry(tmp_path, monkeypatch, capsys):
    model_root = tmp_path / "model"
    _write_minimal_model(model_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None, **kwargs):
        upload_calls.append(
            {"artifact_type": artifact_type, "registry_collection": kwargs.get("registry_collection")}
        )
        return _materialized_model_upload_result()

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)

    cli.main(["model", "upload", "--root", str(model_root), "--project", "p", "--entity", "e", "--name", "n"])

    run.finish.assert_called_once()
    assert upload_calls[0]["artifact_type"] == "model"
    assert upload_calls[0]["registry_collection"] is None

    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube-policy:v0" in out
    assert "Linked into registry collection" not in out


def test_model_upload_happy_path_with_registry(tmp_path, monkeypatch, capsys):
    model_root = tmp_path / "model"
    _write_minimal_model(model_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None, **kwargs):
        upload_calls.append(kwargs.get("registry_collection"))
        return _materialized_model_upload_result(registry_collection="pick-cube-policy")

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)

    cli.main(
        [
            "model",
            "upload",
            "--root",
            str(model_root),
            "--project",
            "p",
            "--name",
            "n",
            "--registry-collection",
            "pick-cube-policy",
        ]
    )

    assert upload_calls == ["pick-cube-policy"]
    out = capsys.readouterr().out
    assert "Linked into registry collection: pick-cube-policy" in out


def test_model_upload_finishes_run_even_on_upload_failure(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    _write_minimal_model(model_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    def _boom(*a, **kw):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(cli, "upload_directory", _boom)

    with pytest.raises(RuntimeError):
        cli.main(["model", "upload", "--root", str(model_root), "--project", "p", "--name", "n"])

    run.finish.assert_called_once()


def test_model_download_rejects_malformed_ref_before_touching_wandb(tmp_path, monkeypatch):
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())

    with pytest.raises(ValueError):
        cli.main(["model", "download", "--ref", "not-a-valid-ref", "--root", str(tmp_path)])

    assert init_calls == []


def test_model_download_happy_path(tmp_path, monkeypatch, capsys):
    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"
    download_calls = []
    validator_calls = []

    def _fake_download(passed_run, ref, *, expected_type, download_root, validator=None):
        download_calls.append(expected_type)
        _write_minimal_model(Path(download_root))
        validator_calls.append(validator)
        validator(Path(download_root))
        return MaterializedArtifact(
            requested_ref=str(ref),
            resolved_ref="my-team/my-project/pick-cube-policy:v3",
            local_path=Path(download_root),
            version="v3",
            digest="digest",
            metadata={},
        )

    monkeypatch.setattr(cli, "download_artifact", _fake_download)

    cli.main(
        ["model", "download", "--ref", "my-team/my-project/pick-cube-policy:latest", "--root", str(dest)]
    )

    assert download_calls == ["model"]
    assert validator_calls == [cli.validate_model_directory]
    run.finish.assert_called_once()
    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube-policy:v3" in out
    assert str(dest) in out


def test_model_download_rejects_result_missing_required_files(tmp_path, monkeypatch):
    """The store must validate the staged download before promoting it to ``--root``.

    Mirrors ``test_dataset_download_rejects_result_missing_required_files``: the validator is
    invoked by ``download_artifact`` itself, while the download is still staged, not by the CLI
    after the fact — so a rejecting validator must stop promotion before it ever happens.
    """
    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"

    def _fake_download_incomplete(passed_run, ref, *, expected_type, download_root, validator=None):
        Path(download_root).mkdir(parents=True, exist_ok=True)
        validator(Path(download_root))
        raise AssertionError("validator should have rejected the incomplete directory")

    monkeypatch.setattr(cli, "download_artifact", _fake_download_incomplete)

    with pytest.raises(ModelDirectoryError):
        cli.main(
            ["model", "download", "--ref", "my-team/my-project/pick-cube-policy:latest", "--root", str(dest)]
        )

    run.finish.assert_called_once()


@pytest.mark.parametrize("destination_exists", [False, True])
def test_model_download_leaves_destination_untouched_when_staged_model_is_invalid(
    tmp_path, monkeypatch, destination_exists
):
    """Regression test for staging an invalid model: it must never be promoted to ``--root``.

    Exercises the real ``download_artifact`` (not a mock of it) end to end through
    ``cmd_model_download``, so it proves the CLI actually wires ``validate_model_directory`` in as
    the store's ``validator`` — a mocked ``download_artifact`` would hide a missing wire-up.
    """
    run = _fake_run()

    class _InvalidStagedArtifact:
        type = "model"
        version = "v0"
        digest = "digest"
        metadata = {}
        entity = "my-team"
        project = "my-project"
        name = "pick-cube-policy:v0"

        @property
        def qualified_name(self):
            return f"{self.entity}/{self.project}/{self.name}"

        def download(self, root=None, **_kwargs):
            # Staged content has config.json but no weights: fails validate_model_directory.
            root_path = Path(root)
            root_path.mkdir(parents=True, exist_ok=True)
            (root_path / CONFIG_NAME).write_text("{}")
            return root

    run.use_artifact = lambda ref: _InvalidStagedArtifact()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)

    dest = tmp_path / "materialized"
    if destination_exists:
        dest.mkdir()

    with pytest.raises(ModelDirectoryError):
        cli.main(
            ["model", "download", "--ref", "my-team/my-project/pick-cube-policy:latest", "--root", str(dest)]
        )

    if destination_exists:
        assert dest.is_dir()
        assert list(dest.iterdir()) == []
    else:
        assert not dest.exists()
    run.finish.assert_called_once()


def _write_rollout_dataset(root: Path, *, episodes: int, with_video: bool) -> None:
    """A genuinely valid rollout dataset (a rollout is a LeRobotDataset; see ADR 0004)."""
    camera_key = "observation.images.cam"
    features: dict[str, dict[str, Any]] = {"action": _ACTION_FEATURE}
    if with_video:
        features[camera_key] = {
            "dtype": "video",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id="tests/rollout_wandb-cli",
        fps=10,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=with_video,
        video_backend="pyav",
        metadata_buffer_size=1,
    )
    for _ in range(episodes):
        for _ in range(4):
            frame = {"action": np.zeros(6, dtype=np.float32), "task": "pick the cube"}
            if with_video:
                frame[camera_key] = np.zeros((32, 32, 3), dtype=np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


def _model_input_result():
    return MaterializedArtifact(
        requested_ref="my-team/my-project/pick-cube-policy:latest",
        resolved_ref="my-team/my-project/pick-cube-policy:v3",
        local_path=None,
        version="v3",
        digest="digest",
        metadata={},
    )


def _rollout_upload_result():
    return MaterializedArtifact(
        requested_ref="my-team/my-project/pick-cube-rollout",
        resolved_ref="my-team/my-project/pick-cube-rollout:v0",
        local_path=Path("/tmp/does-not-matter"),
        version="v0",
        digest="digest",
        metadata={},
    )


def _rollout_argv(root: Path, **overrides) -> list[str]:
    args = {
        "--root": str(root),
        "--project": "my-project",
        "--entity": "my-team",
        "--name": "pick-cube-rollout",
        "--model-ref": "my-team/my-project/pick-cube-policy:latest",
        "--episodes-succeeded": "2",
    }
    args.update(overrides)
    return ["rollout", "upload", *[value for pair in args.items() for value in pair]]


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"--episodes-succeeded": "99"}, ValueError),
        ({"--episodes-succeeded": "-1"}, ValueError),
        ({"--model-ref": "not-a-ref"}, ValueError),
    ],
    ids=["successes_above_episode_count", "negative_successes", "malformed_model_ref"],
)
def test_rollout_upload_rejects_bad_input_before_creating_a_run(
    tmp_path, monkeypatch, overrides, expected_error
):
    """Every local check runs before `wandb.init`, so a typo never leaves an empty run behind."""
    rollout_root = tmp_path / "rollout"
    _write_rollout_dataset(rollout_root, episodes=3, with_video=False)

    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())
    monkeypatch.setattr(cli, "upload_directory", lambda *a, **kw: _rollout_upload_result())

    with pytest.raises(expected_error):
        cli.main(_rollout_argv(rollout_root, **overrides))

    assert init_calls == []


def test_rollout_upload_rejects_a_directory_that_is_not_a_dataset(tmp_path, monkeypatch):
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())

    empty_root = tmp_path / "not-a-dataset"
    empty_root.mkdir()

    with pytest.raises(DatasetDirectoryError):
        cli.main(_rollout_argv(empty_root))

    assert init_calls == []


def test_rollout_upload_rejects_v21_transfer_layout_before_creating_a_run(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollout"
    rollout_root.mkdir()
    transfer = TransferDataset(
        root=rollout_root,
        layout="v2.1",
        metadata=DatasetDirectoryMetadata(
            schema_version="v2.1",
            robot_type="so101",
            fps=30,
            total_episodes=1,
            total_frames=1,
            total_tasks=1,
            camera_keys=(),
            video_keys=(),
            git_commit=None,
        ),
        info={},
    )
    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: transfer)
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())

    with pytest.raises(DatasetDirectoryError, match="v2.1"):
        cli.main(_rollout_argv(rollout_root))

    assert init_calls == []


@require_h264
def test_rollout_upload_happy_path(tmp_path, monkeypatch, capsys):
    rollout_root = tmp_path / "rollout"
    _write_rollout_dataset(rollout_root, episodes=3, with_video=True)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    declared = []
    monkeypatch.setattr(
        cli,
        "declare_input",
        lambda passed_run, ref, *, expected_type: (
            declared.append((str(ref), expected_type)) or _model_input_result()
        ),
    )
    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None, **kwargs):
        upload_calls.append({"artifact_type": artifact_type, "metadata": metadata, "name": name})
        return _rollout_upload_result()

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)
    video_calls = []

    def _fake_video(path, **kw):
        # The moment the CLI hands the path to wandb: the derived preview must exist right then.
        assert Path(path).is_file()
        video_calls.append((path, kw))
        return f"video:{path}"

    monkeypatch.setattr(cli.wandb, "Video", _fake_video)

    cli.main(_rollout_argv(rollout_root))

    # The model is a run input for lineage, and is never downloaded.
    assert declared == [("my-team/my-project/pick-cube-policy:latest", "model")]

    # Its own artifact type, distinct from a training dataset.
    assert upload_calls[0]["artifact_type"] == "rollout"

    # Dataset facts and rollout facts both travel with the artifact.
    metadata = upload_calls[0]["metadata"]
    assert metadata["schema_version"] == "v3.0"
    assert metadata["episodes"] == 3
    assert metadata["successes"] == 2
    assert metadata["success_rate"] == pytest.approx(2 / 3)
    assert metadata["frames"] == 12
    assert metadata["duration_s"] == pytest.approx(1.2)
    assert metadata["model_artifact_requested_ref"] == "my-team/my-project/pick-cube-policy:latest"
    assert metadata["model_artifact_resolved_ref"] == "my-team/my-project/pick-cube-policy:v3"

    # ...and the rollout facts are visible in the run UI, built from the same summary object.
    run.summary.update.assert_called_once()
    logged_summary = run.summary.update.call_args[0][0]
    assert logged_summary["success_rate"] == pytest.approx(2 / 3)
    assert logged_summary.items() <= metadata.items()

    # Exactly one derived preview reaches the run, and nothing else: the original (AV1) video stays
    # in the artifact root, the preview lives outside it, and W&B receives an explicit MP4 format.
    assert len(video_calls) == 1
    preview_path_arg, video_kwargs = video_calls[0]
    assert video_kwargs == {"format": "mp4"}
    assert rollout_root not in Path(preview_path_arg).parents  # outside the artifact root
    assert sorted(rollout_root.rglob("*.mp4")) == [
        rollout_root / "videos/observation.images.cam/chunk-000/file-000.mp4"
    ]  # original AV1 still the only file in the artifact root
    assert preview_path_arg != str(sorted(rollout_root.rglob("*.mp4"))[0])  # not the source
    assert run.log.call_count == 1

    # The recorded path locates the file inside the artifact, where the metadata will be read from
    # — never this machine's copy, which won't exist wherever the artifact is materialized next.
    assert metadata["representative_video_path"] == "videos/observation.images.cam/chunk-000/file-000.mp4"
    assert str(rollout_root) not in metadata["representative_video_path"]

    run.finish.assert_called_once()

    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube-rollout:v0" in out
    assert "my-team/my-project/pick-cube-policy:v3" in out
    assert "success rate: 66.7%" in out
    assert "episode(s) 0, 1, 2" in out


def test_rollout_upload_without_video_logs_no_run_media(tmp_path, monkeypatch, capsys):
    rollout_root = tmp_path / "rollout"
    _write_rollout_dataset(rollout_root, episodes=2, with_video=False)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(cli, "declare_input", lambda *a, **kw: _model_input_result())
    monkeypatch.setattr(cli, "upload_directory", lambda *a, **kw: _rollout_upload_result())
    preview_prep = MagicMock()
    monkeypatch.setattr(cli, "prepare_rollout_preview", preview_prep)

    cli.main(_rollout_argv(rollout_root))

    preview_prep.assert_not_called()
    run.log.assert_not_called()
    run.finish.assert_called_once()
    assert "nothing logged as run media" in capsys.readouterr().out


def test_rollout_upload_preview_failure_aborts_before_wandb_init(tmp_path, monkeypatch):
    """A preview that cannot be prepared (e.g. no h264 encoder) must not create an empty run."""
    rollout_root = tmp_path / "rollout"
    _write_rollout_dataset(rollout_root, episodes=2, with_video=True)

    def _boom(*a, **kw):
        raise RuntimeError("no h264 encoder")

    monkeypatch.setattr(cli, "prepare_rollout_preview", _boom)
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())

    with pytest.raises(RuntimeError, match="no h264 encoder"):
        cli.main(_rollout_argv(rollout_root))

    assert init_calls == []


def test_rollout_upload_rejects_a_preview_temp_dir_inside_the_rollout_root(tmp_path, monkeypatch):
    """A preview inside the artifact root would be uploaded with it; the CLI refuses before any
    run exists, whatever the temp-dir mechanism returns.
    """
    rollout_root = tmp_path / "rollout"
    _write_rollout_dataset(rollout_root, episodes=2, with_video=True)

    real_temporary_directory = tempfile.TemporaryDirectory
    monkeypatch.setattr(
        cli.tempfile,
        "TemporaryDirectory",
        lambda *a, **kw: real_temporary_directory(dir=rollout_root),
    )
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs) or _fake_run())

    with pytest.raises(ValueError, match="outside the rollout root"):
        cli.main(_rollout_argv(rollout_root))

    assert init_calls == []


@require_h264
def test_rollout_upload_preview_ignores_a_tmpdir_inside_the_rollout_root(tmp_path, monkeypatch):
    """TMPDIR must not be able to place the preview inside the artifact: the temp dir is pinned
    beside the rollout root instead of wherever the environment points.
    """
    rollout_root = tmp_path / "rollout"
    _write_rollout_dataset(rollout_root, episodes=2, with_video=True)

    monkeypatch.setenv("TMPDIR", str(rollout_root))
    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(cli, "declare_input", lambda *a, **kw: _model_input_result())
    monkeypatch.setattr(cli, "upload_directory", lambda *a, **kw: _rollout_upload_result())
    video_calls = []
    monkeypatch.setattr(cli.wandb, "Video", lambda path, **kw: video_calls.append(path) or f"video:{path}")

    cli.main(_rollout_argv(rollout_root))

    assert len(video_calls) == 1
    preview = Path(video_calls[0])
    assert rollout_root not in preview.parents
    assert preview.parent.parent == rollout_root.parent  # pinned beside the root, not under it


@require_h264
def test_rollout_upload_preview_still_exists_when_run_finishes(tmp_path, monkeypatch):
    """The temp preview dir outlives run.finish() — wandb reads the file at finish time."""
    rollout_root = tmp_path / "rollout"
    _write_rollout_dataset(rollout_root, episodes=2, with_video=True)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(cli, "declare_input", lambda *a, **kw: _model_input_result())
    monkeypatch.setattr(cli, "upload_directory", lambda *a, **kw: _rollout_upload_result())
    video_calls = []
    monkeypatch.setattr(
        cli.wandb, "Video", lambda path, **kw: video_calls.append((path, kw)) or f"video:{path}"
    )

    def _finish_checks_preview_alive():
        # The exact moment wandb reads the file: run.log already happened (video_calls populated),
        # so a premature temp-dir cleanup would surface right here as a missing file.
        preview_path = Path(video_calls[0][0])
        assert preview_path.is_file()

    run.finish.side_effect = _finish_checks_preview_alive

    cli.main(_rollout_argv(rollout_root))

    run.finish.assert_called_once()


def test_rollout_upload_finishes_the_run_when_the_model_ref_is_not_a_model(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollout"
    _write_rollout_dataset(rollout_root, episodes=2, with_video=False)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    upload_calls = []
    monkeypatch.setattr(
        cli, "upload_directory", lambda *a, **kw: upload_calls.append(kw) or _rollout_upload_result()
    )

    def _wrong_type(*a, **kw):
        raise ArtifactTypeMismatchError("not a model")

    monkeypatch.setattr(cli, "declare_input", _wrong_type)

    with pytest.raises(ArtifactTypeMismatchError):
        cli.main(_rollout_argv(rollout_root))

    assert upload_calls == []  # nothing is uploaded without a lineage edge
    run.finish.assert_called_once()


def _write_adapter_only_model(root: Path, *, base_model: str = "lerobot/pi0_base") -> None:
    """A PEFT checkpoint with no full weights: loadable only if the base model is available."""
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / CONFIG_NAME).write_text(json.dumps({"type": "pi0"}))
    (root / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base_model}))
    (root / "adapter_model.safetensors").write_bytes(b"adapter")


def test_model_upload_refuses_to_register_an_adapter_only_checkpoint(tmp_path, monkeypatch, capsys):
    """The Artifact still uploads, but it is not linked into the Registry: a Registry collection is
    where a team looks for something deployable, and this cannot be rolled out on its own.
    """
    model_root = tmp_path / "model"
    _write_adapter_only_model(model_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None, **kwargs):
        upload_calls.append({"registry_collection": kwargs.get("registry_collection"), "metadata": metadata})
        return _materialized_model_upload_result()

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)

    cli.main(
        [
            "model",
            "upload",
            "--root",
            str(model_root),
            "--project",
            "p",
            "--name",
            "n",
            "--registry-collection",
            "pick-cube-policy",
        ]
    )

    # Uploaded, but never linked.
    assert len(upload_calls) == 1
    assert upload_calls[0]["registry_collection"] is None

    # The refusal travels with the artifact, not only in a log line the operator may not see.
    metadata = upload_calls[0]["metadata"]
    assert metadata["is_self_contained"] is False
    assert "lerobot/pi0_base" in metadata["registry_link_refused_reason"]

    out = capsys.readouterr().out
    assert "NOT linked into registry collection pick-cube-policy" in out
    assert "Linked into registry collection:" not in out


def test_model_upload_still_registers_a_full_weights_checkpoint(tmp_path, monkeypatch, capsys):
    """The refusal is specific to adapter-only checkpoints; ordinary ones are unaffected."""
    model_root = tmp_path / "model"
    _write_minimal_model(model_root)

    run = _fake_run()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    upload_calls = []

    def _fake_upload(passed_run, directory, *, name, artifact_type, aliases=(), metadata=None, **kwargs):
        upload_calls.append({"registry_collection": kwargs.get("registry_collection"), "metadata": metadata})
        return _materialized_model_upload_result(registry_collection="pick-cube-policy")

    monkeypatch.setattr(cli, "upload_directory", _fake_upload)

    cli.main(
        [
            "model",
            "upload",
            "--root",
            str(model_root),
            "--project",
            "p",
            "--name",
            "n",
            "--registry-collection",
            "pick-cube-policy",
        ]
    )

    assert upload_calls[0]["registry_collection"] == "pick-cube-policy"
    assert "registry_link_refused_reason" not in upload_calls[0]["metadata"]
    assert "Linked into registry collection: pick-cube-policy" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# model promote
# ---------------------------------------------------------------------------


def test_model_promote_rejects_malformed_ref_before_touching_wandb(monkeypatch):
    promote_calls = []
    monkeypatch.setattr(cli, "promote_model", lambda *a, **k: promote_calls.append((a, k)))

    with pytest.raises(ValueError):
        cli.main(["model", "promote", "--ref", "not-a-valid-ref", "--alias", "production"])

    assert promote_calls == []


def test_model_promote_passes_the_parsed_ref_through_and_never_starts_a_run(monkeypatch, capsys):
    """`promote` is the one command with no `wandb.init`: it aliases a version that already exists.

    Proves the CLI wiring and that no run is created. What the server does with the alias is
    outside what any mocked test can see — see the note in `test_store.py`.
    """
    init_calls = []
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: init_calls.append(kwargs))

    captured = {}

    def _fake_promote(ref, *, alias, registry_collection=None):
        captured["ref"] = str(ref)
        captured["alias"] = alias
        captured["registry_collection"] = registry_collection
        return MaterializedArtifact(
            requested_ref=str(ref),
            resolved_ref="my-team/my-project/pick-cube-policy:v3",
            local_path=None,
            version="v3",
            digest="abc123digest",
            metadata={},
            registry_collection=registry_collection,
        )

    monkeypatch.setattr(cli, "promote_model", _fake_promote)

    cli.main(
        [
            "model",
            "promote",
            "--ref",
            "my-team/my-project/pick-cube-policy:v3",
            "--alias",
            "production",
            "--registry-collection",
            "pick-cube-policy",
        ]
    )

    assert init_calls == []
    assert captured == {
        "ref": "my-team/my-project/pick-cube-policy:v3",
        "alias": "production",
        "registry_collection": "pick-cube-policy",
    }
    out = capsys.readouterr().out
    assert "my-team/my-project/pick-cube-policy:v3" in out
    assert "production" in out
    assert "abc123digest" in out
    assert "pick-cube-policy" in out


def test_help_omits_removed_workspace_command():
    parser = cli.build_parser()

    assert "workspace" not in parser.format_help().lower()

    with pytest.raises(SystemExit):
        parser.parse_args(["workspace", "create", "--entity", "e", "--project", "p"])


def test_upload_commands_import_without_wandb_workspaces():
    """Existing transfer commands keep working without the removed SDK."""
    code = "import sys\nsys.modules['wandb_workspaces'] = None\nfrom lerobot_wandb import cli  # noqa: F401\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr


def test_main_reports_missing_lerobot_cleanly(monkeypatch, capsys):
    """A LeRobot-dependent command without LeRobot installed must exit nonzero with
    the actionable compatibility message — never an import traceback."""
    monkeypatch.setattr("lerobot_wandb.compatibility.get_installed_lerobot_version", lambda: None)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["dataset", "upload", "--root", ".", "--project", "p", "--name", "n"])
    assert excinfo.value.code == 1
    assert "not installed" in capsys.readouterr().out
