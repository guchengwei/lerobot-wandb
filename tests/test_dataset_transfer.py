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

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("lerobot", reason="lerobot is required (install a supported lerobot release)")

from lerobot.datasets.pyav_utils import get_codec

from lerobot_wandb import dataset_transfer
from lerobot_wandb.compatibility import LeRobotCompatibilityError
from lerobot_wandb.dataset_transfer import (
    DatasetPreviewSource,
    TransferDataset,
    inspect_transfer_dataset,
    prepare_dataset_preview,
    select_dataset_preview_sources,
)
from lerobot_wandb.inspect import DatasetDirectoryError, DatasetDirectoryMetadata

require_h264 = pytest.mark.skipif(
    get_codec("h264") is None, reason="'h264' encoder not in local FFmpeg build"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_v21_dataset(
    root: Path,
    *,
    cameras: tuple[str, ...] = ("observation.images.wrist",),
    video_path: str | None = None,
) -> None:
    chunks_size = 1000
    video_path_template: str = (
        video_path or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    features = {
        "action": {"dtype": "float32", "shape": [1], "names": ["motor"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera in cameras:
        features[camera] = {"dtype": "video", "shape": [3, 8, 8], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": "so101",
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": chunks_size,
        "total_videos": 2 * len(cameras),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": video_path_template,
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick"], "length": 2},
            {"episode_index": 1, "tasks": ["pick"], "length": 2},
        ],
    )
    _write_jsonl(
        root / "meta/episodes_stats.jsonl",
        [
            {"episode_index": 0, "stats": {}},
            {"episode_index": 1, "stats": {}},
        ],
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "pick"}])

    for episode in range(2):
        data_path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "action": pa.array([[0.0], [1.0]], type=pa.list_(pa.float32(), 1)),
                "timestamp": pa.array([0.0, 1 / 30], type=pa.float32()),
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "episode_index": pa.array([episode, episode], type=pa.int64()),
                "index": pa.array([episode * 2, episode * 2 + 1], type=pa.int64()),
                "task_index": pa.array([0, 0], type=pa.int64()),
            }
        )
        pq.write_table(table, data_path)
        for camera in cameras:
            video = root / video_path_template.format(
                episode_chunk=episode // chunks_size,
                episode_index=episode,
                video_key=camera,
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"test-video-bytes")


def test_transfer_validation_reports_compatibility_before_reading_dataset(tmp_path, monkeypatch):
    calls: list[str] = []

    def _reject_incompatible_install() -> None:
        calls.append("compatibility")
        raise LeRobotCompatibilityError("LeRobot is unavailable")

    monkeypatch.setattr(dataset_transfer, "check_lerobot_compatible", _reject_incompatible_install)

    with pytest.raises(LeRobotCompatibilityError, match="unavailable"):
        inspect_transfer_dataset(tmp_path / "missing")

    assert calls == ["compatibility"]


def test_v21_transfer_accepts_episode_per_file_layout(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root)

    dataset = inspect_transfer_dataset(root)

    assert dataset.layout == "v2.1"
    assert dataset.metadata.schema_version == "v2.1"
    assert dataset.metadata.total_episodes == 2
    assert dataset.metadata.video_keys == ("observation.images.wrist",)
    preview = select_dataset_preview_sources(dataset)[0]
    assert preview.episode == 0
    assert preview.relative_path == Path("videos/chunk-000/observation.images.wrist/episode_000000.mp4")


def test_v21_default_preview_is_one_deterministic_video(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root, cameras=("observation.images.front", "observation.images.wrist"))
    dataset = inspect_transfer_dataset(root)

    previews = select_dataset_preview_sources(dataset)

    assert [(preview.episode, preview.video_key, preview.is_representative) for preview in previews] == [
        (0, "observation.images.front", True),
        (0, "observation.images.wrist", True),
    ]
    assert previews[0].relative_path == Path("videos/chunk-000/observation.images.front/episode_000000.mp4")


def test_v21_preview_episode_selects_every_camera(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root, cameras=("observation.images.front", "observation.images.wrist"))
    dataset = inspect_transfer_dataset(root)

    previews = select_dataset_preview_sources(dataset, episodes=[1])

    assert [preview.episode for preview in previews] == [1, 1]
    assert [preview.video_key for preview in previews] == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    assert all("episode_000001.mp4" in str(preview.relative_path) for preview in previews)


def test_v21_repeatable_preview_episode_selects_multiple_episodes(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root, cameras=("observation.images.front", "observation.images.wrist"))
    dataset = inspect_transfer_dataset(root)

    previews = select_dataset_preview_sources(dataset, episodes=[0, 1])

    assert [(preview.episode, preview.video_key) for preview in previews] == [
        (0, "observation.images.front"),
        (0, "observation.images.wrist"),
        (1, "observation.images.front"),
        (1, "observation.images.wrist"),
    ]


def test_v21_preview_all_selects_every_episode_and_camera(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root, cameras=("observation.images.front", "observation.images.wrist"))
    dataset = inspect_transfer_dataset(root)

    previews = select_dataset_preview_sources(dataset, preview_all=True)
    assert len(previews) == dataset.metadata.total_episodes * len(dataset.metadata.video_keys)

    assert [(preview.episode, preview.video_key) for preview in previews] == [
        (0, "observation.images.front"),
        (0, "observation.images.wrist"),
        (1, "observation.images.front"),
        (1, "observation.images.wrist"),
    ]


def test_preview_all_selects_more_than_former_episode_limit(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root)
    dataset = inspect_transfer_dataset(root)
    dataset = replace(dataset, metadata=replace(dataset.metadata, total_episodes=60))

    sources = select_dataset_preview_sources(dataset, preview_all=True)

    assert len(sources) == dataset.metadata.total_episodes * len(dataset.metadata.video_keys)


def test_preview_all_rejects_explicit_episode_selectors(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root)
    dataset = inspect_transfer_dataset(root)

    with pytest.raises(DatasetDirectoryError, match="mutually exclusive"):
        select_dataset_preview_sources(dataset, episodes=[0], preview_all=True)


def test_v21_missing_video_is_rejected(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root)
    (root / "videos/chunk-000/observation.images.wrist/episode_000001.mp4").unlink()

    with pytest.raises(DatasetDirectoryError, match="missing v2.1 video for episode 1"):
        inspect_transfer_dataset(root)


def test_v21_preview_rejects_video_template_without_episode_per_file_proof(tmp_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root, video_path="videos/{video_key}/chunk-{episode_chunk:03d}.mp4")

    with pytest.raises(DatasetDirectoryError, match="episode-per-file"):
        inspect_transfer_dataset(root)


def test_exact_episode_validates_range_before_empty_preview_short_circuit(tmp_path):
    metadata = DatasetDirectoryMetadata(
        schema_version="v3.0",
        robot_type="so101",
        fps=30,
        total_episodes=0,
        total_frames=0,
        total_tasks=0,
        camera_keys=("observation.images.wrist",),
        video_keys=("observation.images.wrist",),
        git_commit=None,
    )
    dataset = TransferDataset(root=tmp_path, layout="v3", metadata=metadata, info={})

    with pytest.raises(DatasetDirectoryError, match="outside the dataset range"):
        select_dataset_preview_sources(dataset, episodes=[0])


def test_exact_episode_without_video_returns_no_preview(tmp_path):
    metadata = DatasetDirectoryMetadata(
        schema_version="v3.0",
        robot_type="so101",
        fps=30,
        total_episodes=1,
        total_frames=1,
        total_tasks=1,
        camera_keys=(),
        video_keys=(),
        git_commit=None,
    )
    dataset = TransferDataset(root=tmp_path, layout="v3", metadata=metadata, info={})

    assert select_dataset_preview_sources(dataset, episodes=[0]) == []


@pytest.mark.parametrize(
    "video_path",
    ["videos/{}/episode.mp4", "videos/{video_key.missing}/episode.mp4"],
    ids=["positional-field", "attribute-field"],
)
def test_v21_malformed_video_template_is_wrapped_as_dataset_error(tmp_path, video_path):
    root = tmp_path / "v21"
    _write_v21_dataset(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_path"] = video_path
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(DatasetDirectoryError, match="cannot resolve v2.1 video"):
        inspect_transfer_dataset(root)


@pytest.mark.parametrize(
    ("episode", "expected_path", "expected_start", "expected_end"),
    [
        (0, "videos/observation.images.wrist/chunk-000/file-000.mp4", 0.0, 1.0),
        (1, "videos/observation.images.wrist/chunk-000/file-000.mp4", 1.0, 2.5),
        (2, "videos/observation.images.wrist/chunk-000/file-000.mp4", 2.5, 4.0),
    ],
    ids=["first", "middle", "last"],
)
def test_v3_exact_episode_resolves_shared_chunk_boundaries(
    tmp_path, monkeypatch, episode, expected_path, expected_start, expected_end
):
    metadata = DatasetDirectoryMetadata(
        schema_version="v3.0",
        robot_type="so101",
        fps=30,
        total_episodes=3,
        total_frames=120,
        total_tasks=1,
        camera_keys=("observation.images.wrist",),
        video_keys=("observation.images.wrist",),
        git_commit=None,
    )
    dataset = TransferDataset(
        root=tmp_path,
        layout="v3",
        metadata=metadata,
        info={"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"},
    )
    monkeypatch.setattr(
        dataset_transfer._lerobot,
        "load_episodes",
        lambda _root: [
            {
                "episode_index": index,
                "videos/observation.images.wrist/chunk_index": 0,
                "videos/observation.images.wrist/file_index": 0,
                "videos/observation.images.wrist/from_timestamp": start,
                "videos/observation.images.wrist/to_timestamp": end,
            }
            for index, (start, end) in enumerate(((0.0, 1.0), (1.0, 2.5), (2.5, 4.0)))
        ],
    )

    assert select_dataset_preview_sources(dataset, episodes=[episode]) == [
        DatasetPreviewSource(
            episode=episode,
            video_key="observation.images.wrist",
            relative_path=Path(expected_path),
            start_timestamp_s=expected_start,
            end_timestamp_s=expected_end,
        )
    ]


def test_v3_preview_all_selects_every_episode_and_camera(tmp_path, monkeypatch):
    video_keys = ("observation.images.front", "observation.images.wrist")
    metadata = DatasetDirectoryMetadata(
        schema_version="v3.0",
        robot_type="so101",
        fps=30,
        total_episodes=2,
        total_frames=60,
        total_tasks=1,
        camera_keys=video_keys,
        video_keys=video_keys,
        git_commit=None,
    )
    dataset = TransferDataset(
        root=tmp_path,
        layout="v3",
        metadata=metadata,
        info={"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"},
    )
    monkeypatch.setattr(
        dataset_transfer._lerobot,
        "load_episodes",
        lambda _root: [
            {
                "episode_index": episode,
                **{
                    field: value
                    for video_key in video_keys
                    for field, value in (
                        (f"videos/{video_key}/chunk_index", 0),
                        (f"videos/{video_key}/file_index", 0),
                        (f"videos/{video_key}/from_timestamp", float(episode)),
                        (f"videos/{video_key}/to_timestamp", float(episode + 1)),
                    )
                },
            }
            for episode in range(2)
        ],
    )

    previews = select_dataset_preview_sources(dataset, preview_all=True)

    assert [
        (preview.episode, preview.video_key, preview.start_timestamp_s, preview.end_timestamp_s)
        for preview in previews
    ] == [
        (0, "observation.images.front", 0.0, 1.0),
        (0, "observation.images.wrist", 0.0, 1.0),
        (1, "observation.images.front", 1.0, 2.0),
        (1, "observation.images.wrist", 1.0, 2.0),
    ]


def test_prepare_dataset_preview_trims_exact_episode_to_h264(tmp_path, monkeypatch):
    source = tmp_path / "shared-chunk.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "preview.mp4"
    calls = []

    def _prepare(input_path, output_path, **kwargs):
        calls.append((input_path, output_path, kwargs))
        output_path.write_bytes(b"episode-only-h264")
        return output_path

    monkeypatch.setattr(dataset_transfer, "_prepare_dataset_preview", _prepare)

    result = prepare_dataset_preview(
        source,
        destination,
        start_timestamp_s=1.0,
        end_timestamp_s=2.5,
    )

    assert result == destination
    assert calls == [
        (
            source,
            destination,
            {
                "start_timestamp_s": 1.0,
                "end_timestamp_s": 2.5,
                "exact_source": None,
                "profile": dataset_transfer.DEFAULT_PREVIEW_PROFILE,
            },
        )
    ]


@require_h264
@pytest.mark.parametrize(
    ("start", "end", "expected_channel"),
    [(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 2)],
    ids=["first", "middle", "last"],
)
def test_prepare_dataset_preview_contains_only_selected_episode_frames(
    tmp_path, start, end, expected_channel
):
    source = tmp_path / "shared-chunk.mp4"
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    with av.open(str(source), mode="w") as container:
        stream = container.add_stream("h264", rate=2)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for frame_index in range(6):
            episode = frame_index // 2
            pixels = np.full((16, 16, 3), colors[episode], dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = frame_index
            frame.time_base = Fraction(1, 2)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    destination = tmp_path / f"episode-{expected_channel}.mp4"
    prepare_dataset_preview(
        source,
        destination,
        start_timestamp_s=start,
        end_timestamp_s=end,
    )

    with av.open(str(destination)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(frames) == 2
    assert all(frame.mean(axis=(0, 1)).argmax() == expected_channel for frame in frames)


def test_v3_default_preview_uses_declared_representative_path(tmp_path, monkeypatch):
    metadata = DatasetDirectoryMetadata(
        schema_version="v3.0",
        robot_type="so101",
        fps=30,
        total_episodes=20,
        total_frames=100,
        total_tasks=1,
        camera_keys=("observation.images.wrist",),
        video_keys=("observation.images.wrist",),
        git_commit=None,
    )
    dataset = TransferDataset(
        root=tmp_path,
        layout="v3",
        metadata=metadata,
        info={"video_path": "recordings/{video_key}/{chunk_index:03d}-{file_index:03d}.mkv"},
    )
    monkeypatch.setattr(
        dataset_transfer._lerobot,
        "load_episodes",
        lambda _root: [
            {
                "episode_index": episode,
                "videos/observation.images.wrist/chunk_index": 2,
                "videos/observation.images.wrist/file_index": 3,
                "videos/observation.images.wrist/from_timestamp": float(episode),
                "videos/observation.images.wrist/to_timestamp": float(episode + 1),
            }
            for episode in range(20)
        ],
    )

    assert select_dataset_preview_sources(dataset) == [
        DatasetPreviewSource(
            episode=0,
            video_key="observation.images.wrist",
            relative_path=Path("recordings/observation.images.wrist/002-003.mkv"),
            start_timestamp_s=0.0,
            end_timestamp_s=1.0,
            is_representative=True,
        )
    ]


def test_future_dataset_schema_is_rejected_instead_of_using_v3_reader(tmp_path):
    root = tmp_path / "future"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v4.0"}), encoding="utf-8")

    with pytest.raises(DatasetDirectoryError, match="unsupported dataset schema"):
        inspect_transfer_dataset(root)
