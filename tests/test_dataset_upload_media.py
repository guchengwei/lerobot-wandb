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
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("wandb", reason="wandb is required")

from lerobot_wandb import cli
from lerobot_wandb.dataset_preview import (
    PreparedDatasetPreview,
    PreparedPreviewBatch,
    PreviewBudgetExceededError,
)
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
        force_preview_budget=False,
    )


def _prepared_batch(
    source: DatasetPreviewSource,
    destination_dir: Path,
    *,
    total_bytes: int = 5,
    budget_bytes: int = 10,
) -> PreparedPreviewBatch:
    destination = destination_dir / "preview-000000.mp4"
    destination.write_bytes(b"preview")
    preview = PreparedDatasetPreview(
        source=source,
        path=destination,
        bytes=total_bytes,
        used_source=False,
    )
    return PreparedPreviewBatch(
        previews=(preview,),
        total_bytes=total_bytes,
        canonical_bytes=100,
        budget_bytes=budget_bytes,
    )


class _FakeStdin(io.StringIO):
    def __init__(self, response: str = "", *, tty: bool) -> None:
        super().__init__(response)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _patch_dataset_upload(
    monkeypatch,
    dataset: TransferDataset,
    source: DatasetPreviewSource,
    batch_factory,
):
    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: dataset)
    monkeypatch.setattr(
        cli,
        "select_dataset_preview_sources",
        lambda _dataset, **_kwargs: [source],
    )
    prepare_calls = []

    def _prepare(_root, sources, destination_dir, **kwargs):
        prepare_calls.append((list(sources), kwargs))
        return batch_factory(destination_dir)

    monkeypatch.setattr(cli, "prepare_dataset_previews", _prepare)
    run = MagicMock()
    run.entity = "my-team"
    run.project = "my-project"
    monkeypatch.setattr(cli.wandb, "init", lambda **_kwargs: run)
    monkeypatch.setattr(cli.wandb, "Video", lambda path, **_kwargs: f"video:{path}")
    monkeypatch.setattr(
        cli,
        "upload_directory",
        lambda *args, **kwargs: SimpleNamespace(resolved_ref="my-team/my-project/name:v0"),
    )
    return run, prepare_calls


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


def test_force_preview_budget_is_dataset_upload_only():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "dataset",
            "upload",
            "--root",
            "dataset",
            "--project",
            "project",
            "--name",
            "name",
            "--force-preview-budget",
        ]
    )

    assert args.force_preview_budget is True
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "model",
                "upload",
                "--root",
                "model",
                "--project",
                "project",
                "--name",
                "name",
                "--force-preview-budget",
            ]
        )


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

    def _prepare_batch(
        passed_root: Path,
        passed_sources,
        destination_dir: Path,
        *,
        progress_callback,
    ) -> PreparedPreviewBatch:
        assert passed_root == root.resolve()
        assert list(passed_sources) == [selected]
        assert destination_dir.is_dir()
        assert callable(progress_callback)
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

    def _prepare_batch(
        _root: Path,
        passed_sources,
        destination_dir: Path,
        *,
        progress_callback,
    ) -> PreparedPreviewBatch:
        assert callable(progress_callback)
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


def test_under_budget_preparation_proceeds_without_prompt(tmp_path, monkeypatch, capsys):
    root = tmp_path / "dataset"
    root.mkdir()
    source_path = root / "episode.mp4"
    source_path.write_bytes(b"source")
    source = DatasetPreviewSource(10, "camera.front", source_path.relative_to(root))
    dataset = _transfer_dataset(root)
    run, prepare_calls = _patch_dataset_upload(
        monkeypatch,
        dataset,
        source,
        lambda destination: _prepared_batch(source, destination),
    )
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin("n\n", tty=True))

    cli.cmd_dataset_upload(_args(root))

    assert run.finish.call_count == 1
    assert len(prepare_calls) == 1
    assert callable(prepare_calls[0][1]["progress_callback"])
    assert "Upload anyway?" not in capsys.readouterr().err


@pytest.mark.parametrize("response", ["y\n", "yes\n"])
def test_over_budget_interactive_affirmative_uses_prepared_batch(
    tmp_path,
    monkeypatch,
    response,
    capsys,
):
    root = tmp_path / "dataset"
    root.mkdir()
    source_path = root / "episode.mp4"
    source_path.write_bytes(b"source")
    source = DatasetPreviewSource(10, "camera.front", source_path.relative_to(root))
    dataset = _transfer_dataset(root)
    prepared_path = {}

    def _batch_factory(destination):
        batch = _prepared_batch(
            source,
            destination,
            total_bytes=15,
            budget_bytes=10,
        )
        prepared_path["path"] = batch.previews[0].path
        return batch

    run, prepare_calls = _patch_dataset_upload(
        monkeypatch,
        dataset,
        source,
        _batch_factory,
    )

    def _finish():
        assert prepared_path["path"].is_file()

    run.finish.side_effect = _finish
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(response, tty=True))

    cli.cmd_dataset_upload(_args(root))

    assert run.finish.call_count == 1
    assert len(prepare_calls) == 1
    assert not prepared_path["path"].exists()
    error = capsys.readouterr().err
    assert error.index("Prepared 1 preview") < error.index("Starting W&B upload...")


@pytest.mark.parametrize(
    "response",
    ["\n", "n\n", "", "maybe\n"],
    ids=["default", "no", "eof", "invalid"],
)
def test_over_budget_interactive_nonaffirmative_rejects_before_init(tmp_path, monkeypatch, response):
    root = tmp_path / "dataset"
    root.mkdir()
    source_path = root / "episode.mp4"
    source_path.write_bytes(b"source")
    source = DatasetPreviewSource(10, "camera.front", source_path.relative_to(root))
    dataset = _transfer_dataset(root)
    _patch_dataset_upload(
        monkeypatch,
        dataset,
        source,
        lambda destination: _prepared_batch(
            source,
            destination,
            total_bytes=15,
            budget_bytes=10,
        ),
    )
    init = MagicMock()
    monkeypatch.setattr(cli.wandb, "init", init)
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(response, tty=True))

    with pytest.raises(PreviewBudgetExceededError):
        cli.cmd_dataset_upload(_args(root))

    init.assert_not_called()


def test_over_budget_non_tty_rejects_with_actionable_message(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    source_path = root / "episode.mp4"
    source_path.write_bytes(b"source")
    source = DatasetPreviewSource(10, "camera.front", source_path.relative_to(root))
    dataset = _transfer_dataset(root)
    _patch_dataset_upload(
        monkeypatch,
        dataset,
        source,
        lambda destination: _prepared_batch(
            source,
            destination,
            total_bytes=15,
            budget_bytes=10,
        ),
    )
    init = MagicMock()
    monkeypatch.setattr(cli.wandb, "init", init)
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin("y\n", tty=False))

    with pytest.raises(PreviewBudgetExceededError) as error:
        cli.cmd_dataset_upload(_args(root))

    message = str(error.value)
    assert "Select fewer episodes" in message
    assert "--preview-episode" in message
    assert "--no-preview" in message
    assert "--force-preview-budget" in message
    init.assert_not_called()


def test_force_preview_budget_skips_prompt_without_reencoding(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    source_path = root / "episode.mp4"
    source_path.write_bytes(b"source")
    source = DatasetPreviewSource(10, "camera.front", source_path.relative_to(root))
    dataset = _transfer_dataset(root)
    run, prepare_calls = _patch_dataset_upload(
        monkeypatch,
        dataset,
        source,
        lambda destination: _prepared_batch(
            source,
            destination,
            total_bytes=15,
            budget_bytes=10,
        ),
    )
    args = _args(root)
    args.force_preview_budget = True
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin("n\n", tty=True))

    cli.cmd_dataset_upload(args)

    assert run.finish.call_count == 1
    assert len(prepare_calls) == 1


def test_force_preview_budget_does_not_bypass_preview_episode_limit(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    dataset = _transfer_dataset(root)
    monkeypatch.setattr(cli, "inspect_transfer_dataset", lambda _root: dataset)
    selection = MagicMock(side_effect=ValueError("exceeds --preview-max-episodes"))
    monkeypatch.setattr(cli, "select_dataset_preview_sources", selection)
    prepare = MagicMock()
    init = MagicMock()
    monkeypatch.setattr(cli, "prepare_dataset_previews", prepare)
    monkeypatch.setattr(cli.wandb, "init", init)
    args = _args(root)
    args.preview_all = True
    args.force_preview_budget = True

    with pytest.raises(ValueError, match="preview-max-episodes"):
        cli.cmd_dataset_upload(args)

    prepare.assert_not_called()
    init.assert_not_called()


def test_progress_renderer_identifies_sources_and_bounds_redirected_output():
    source = DatasetPreviewSource(
        episode=12,
        video_key="observation.images.front",
        relative_path=Path("video"),
    )
    stream = io.StringIO()
    renderer = cli._PreviewProgressRenderer(stream)

    def event(phase, completed=None, total_work=None, unit=None):
        return SimpleNamespace(
            index=1,
            total=1,
            source=source,
            phase=phase,
            completed=completed,
            total_work=total_work,
            unit=unit,
        )

    renderer(event("start"))
    for completed in range(101):
        renderer(event("progress", completed=completed, total_work=100, unit="frames"))
    renderer(event("complete", completed=100, total_work=100, unit="frames"))

    lines = stream.getvalue().splitlines()
    assert len(lines) <= 15
    assert all("episode 12" in line and "observation.images.front" in line for line in lines)
    assert any("60%" in line or "70%" in line for line in lines)
    assert any("100% done" in line for line in lines)

    unknown_stream = io.StringIO()
    unknown_renderer = cli._PreviewProgressRenderer(unknown_stream)
    unknown_renderer(event("start"))
    for completed in range(1001):
        unknown_renderer(event("progress", completed=completed, unit="frames"))
    unknown_renderer(event("complete", completed=1000, unit="frames"))
    unknown_lines = unknown_stream.getvalue().splitlines()
    assert len(unknown_lines) <= 15
    assert any("frames" in line for line in unknown_lines)
    assert not any("%" in line for line in unknown_lines)
