# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("wandb", reason="wandb is required")

from lerobot_wandb import cli
from lerobot_wandb.dataset_preview import PreparedDatasetPreview, PreparedPreviewBatch
from lerobot_wandb.dataset_transfer import DatasetPreviewSource, TransferDataset
from lerobot_wandb.inspect import DatasetDirectoryMetadata


def _transfer_dataset(
    root: Path,
    *,
    video_keys: tuple[str, ...] = ("observation.images.wrist",),
) -> TransferDataset:
    return TransferDataset(
        root=root,
        layout="v2.1",
        metadata=DatasetDirectoryMetadata(
            schema_version="v2.1",
            robot_type="so101",
            fps=30,
            total_episodes=11,
            total_frames=22,
            total_tasks=1,
            camera_keys=video_keys,
            video_keys=video_keys,
            git_commit=None,
        ),
        info={},
    )


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        root=root,
        entity="my-team",
        project="my-project",
        name="pick-cube-v21",
        aliases=["raw"],
        preview_episodes=[10],
        preview_all=False,
        preview_max_episodes=50,
        no_preview=False,
    )


def test_dataset_preview_all_is_mutually_exclusive_with_explicit_episodes(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "dataset",
                "upload",
                "--root",
                "dataset",
                "--project",
                "project",
                "--name",
                "name",
                "--preview-all",
                "--preview-episode",
                "1",
            ]
        )

    assert "not allowed with argument" in capsys.readouterr().err


def test_dataset_preview_all_has_a_positive_configurable_default_limit():
    parser = cli.build_parser()
    base = ["dataset", "upload", "--root", "dataset", "--project", "project", "--name", "name"]

    args = parser.parse_args([*base, "--preview-all"])

    assert args.preview_all is True
    assert args.preview_max_episodes == 50
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--preview-all", "--preview-max-episodes", "0"])


def test_default_representative_media_key_is_schema_neutral():
    source = DatasetPreviewSource(
        episode=7,
        video_key="observation.images.front/left",
        relative_path=Path("video.mp4"),
        is_representative=True,
    )

    assert cli._dataset_media_key(source, 0) == (
        "dataset_video/representative/observation.images.front%2Fleft"
    )


def test_dataset_upload_logs_playable_preview_and_keeps_it_through_finish(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    source = root / "videos/chunk-000/observation.images.wrist/episode_000010.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")

    dataset = _transfer_dataset(root)
    selected = DatasetPreviewSource(
        episode=10,
        video_key="observation.images.wrist",
        relative_path=source.relative_to(root),
        start_timestamp_s=1.0,
        end_timestamp_s=2.5,
    )
    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: dataset)
    monkeypatch.setattr(
        cli,
        "select_dataset_preview_sources",
        lambda _dataset, **_kwargs: [selected],
    )

    state: dict[str, object] = {"prepared": False, "preview": None}

    def _prepare_batch(passed_root: Path, passed_sources, destination_dir: Path) -> PreparedPreviewBatch:
        assert passed_root == root.resolve()
        assert list(passed_sources) == [selected]
        assert destination_dir.is_dir()
        destination = destination_dir / "preview-000000.mp4"
        destination.write_bytes(b"h264-preview")
        state["prepared"] = True
        state["preview"] = destination
        return PreparedPreviewBatch(
            previews=(
                PreparedDatasetPreview(
                    source=selected,
                    path=destination,
                    bytes=destination.stat().st_size,
                    used_source=False,
                ),
            ),
            total_bytes=destination.stat().st_size,
            canonical_bytes=100,
            budget_bytes=20,
        )

    monkeypatch.setattr(cli, "prepare_dataset_previews", _prepare_batch)

    run = MagicMock()
    run.entity = "my-team"
    run.project = "my-project"
    init_calls = []

    def _init(**kwargs):
        assert state["prepared"] is True
        init_calls.append(kwargs)
        return run

    monkeypatch.setattr(cli.wandb, "init", _init)
    video_calls = []

    def _video(path, **kwargs):
        video_calls.append((path, kwargs))
        return f"video:{path}"

    monkeypatch.setattr(cli.wandb, "Video", _video)

    upload_calls = []

    def _upload(passed_run, directory, **kwargs):
        upload_calls.append((passed_run, Path(directory), kwargs))
        return SimpleNamespace(resolved_ref="my-team/my-project/pick-cube-v21:v0")

    monkeypatch.setattr(cli, "upload_directory", _upload)

    def _finish():
        preview = state["preview"]
        assert isinstance(preview, Path)
        assert preview.is_file()

    run.finish.side_effect = _finish

    cli.cmd_dataset_upload(_args(root))

    assert init_calls == [
        {"entity": "my-team", "project": "my-project", "job_type": "dataset_upload", "mode": "online"}
    ]
    assert len(upload_calls) == 1
    assert upload_calls[0][1] == root
    assert upload_calls[0][2]["metadata"]["schema_version"] == "v2.1"
    run.log.assert_called_once()
    media = run.log.call_args.args[0]
    assert list(media) == ["dataset_video/episode_000010/observation.images.wrist"]
    assert str(state["preview"]) in media[next(iter(media))]
    assert video_calls == [(str(state["preview"]), {"format": "mp4"})]
    summary = run.summary.update.call_args.args[0]
    assert summary["dataset_schema_version"] == "v2.1"
    assert summary["dataset_preview_representative_episode_index"] is None
    assert summary["dataset_preview_episode_indices"] == [10]
    assert summary["dataset_artifact_resolved_ref"] == "my-team/my-project/pick-cube-v21:v0"
    run.finish.assert_called_once()
    assert not Path(state["preview"]).exists()


def test_dataset_upload_media_keys_preserve_exact_camera_identity(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    video_keys = ("observation.images.front", "observation_images_front")
    sources = []
    for video_key in video_keys:
        source = root / f"videos/{video_key}/episode_000010.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"source")
        sources.append(
            DatasetPreviewSource(
                episode=10,
                video_key=video_key,
                relative_path=source.relative_to(root),
            )
        )

    dataset = _transfer_dataset(root, video_keys=video_keys)
    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: dataset)
    monkeypatch.setattr(
        cli,
        "select_dataset_preview_sources",
        lambda _dataset, **_kwargs: sources,
    )

    def _prepare_batch(_root: Path, passed_sources, destination_dir: Path) -> PreparedPreviewBatch:
        previews = []
        for index, source in enumerate(passed_sources):
            destination = destination_dir / f"preview-{index:06d}.mp4"
            destination.write_bytes(b"h264-preview")
            previews.append(
                PreparedDatasetPreview(
                    source=source,
                    path=destination,
                    bytes=destination.stat().st_size,
                    used_source=False,
                )
            )
        return PreparedPreviewBatch(
            previews=tuple(previews),
            total_bytes=sum(item.bytes for item in previews),
            canonical_bytes=100,
            budget_bytes=100,
        )

    monkeypatch.setattr(cli, "prepare_dataset_previews", _prepare_batch)
    run = MagicMock()
    monkeypatch.setattr(cli.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(cli.wandb, "Video", lambda path, **kwargs: f"video:{path}")
    monkeypatch.setattr(
        cli,
        "upload_directory",
        lambda *args, **kwargs: SimpleNamespace(resolved_ref="my-team/my-project/pick-cube-v21:v0"),
    )

    cli.cmd_dataset_upload(_args(root))

    media = run.log.call_args.args[0]
    assert len(media) == 2
    assert len(set(media)) == 2
    encoded_cameras = [key.rsplit("/", 1)[1] for key in media]
    assert encoded_cameras == ["observation.images.front", "observation_images_front"]


def test_dataset_preview_failure_happens_before_wandb_init(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    source = root / "videos/chunk-000/observation.images.wrist/episode_000010.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    dataset = _transfer_dataset(root)
    selected = DatasetPreviewSource(10, "observation.images.wrist", source.relative_to(root))

    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: dataset)
    monkeypatch.setattr(
        cli,
        "select_dataset_preview_sources",
        lambda _dataset, **_kwargs: [selected],
    )

    def _fail_preview(*_args, **_kwargs):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(cli, "prepare_dataset_previews", _fail_preview)
    init = MagicMock()
    monkeypatch.setattr(cli.wandb, "init", init)

    with pytest.raises(RuntimeError, match="encoder unavailable"):
        cli.cmd_dataset_upload(_args(root))

    init.assert_not_called()


def test_dataset_no_preview_uploads_canonical_root_without_generating_media(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    dataset = _transfer_dataset(root)
    args = _args(root)
    args.no_preview = True
    select = MagicMock()
    prepare = MagicMock()
    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: dataset)
    monkeypatch.setattr(cli, "select_dataset_preview_sources", select)
    run = MagicMock()
    monkeypatch.setattr(cli.wandb, "init", lambda **_kwargs: run)
    uploaded_roots = []

    def _upload(_run, directory, **_kwargs):
        uploaded_roots.append(Path(directory))
        return SimpleNamespace(resolved_ref="my-team/my-project/pick-cube-v21:v0")

    monkeypatch.setattr(cli, "upload_directory", _upload)

    cli.cmd_dataset_upload(args)

    assert uploaded_roots == [root]
    select.assert_not_called()
    prepare.assert_not_called()
    run.log.assert_not_called()
    run.finish.assert_called_once()
