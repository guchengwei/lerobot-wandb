# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from lerobot_wandb.dataset_preview import (
    DatasetPreviewSource,
    PreviewEncodingError,
    prepare_dataset_preview,
    prepare_dataset_previews,
    probe_video,
)


def _write_video(
    path: Path,
    *,
    width: int = 800,
    height: int = 400,
    fps: int = 30,
    frames: int = 60,
    encoder_options: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps, options=encoder_options)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            pixels = np.full((height, width, 3), index % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_prepare_profile_scales_caps_fps_and_slices(tmp_path):
    source = tmp_path / "shared.mp4"
    destination = tmp_path / "preview.mp4"
    _write_video(source)

    result = prepare_dataset_preview(
        source,
        destination,
        start_timestamp_s=0.5,
        end_timestamp_s=1.5,
    )

    assert result == destination
    probe = probe_video(destination)
    assert probe.codec in {"h264", "avc1", "libx264"}
    assert probe.pixel_format == "yuv420p"
    assert probe.width == 640
    assert probe.height == 320
    assert probe.fps <= 15
    with av.open(str(destination)) as container:
        assert len(list(container.decode(video=0))) == 15


def test_prepare_profile_uses_a_two_second_gop(tmp_path):
    source = tmp_path / "source.mp4"
    destination = tmp_path / "preview.mp4"
    _write_video(source, width=64, height=32, fps=30, frames=150)

    prepare_dataset_preview(source, destination, exact_source=False)

    with av.open(str(destination)) as container:
        keyframe_indexes = [index for index, frame in enumerate(container.decode(video=0)) if frame.key_frame]

    assert keyframe_indexes == [0, 30, 60]


def test_fast_path_requires_exact_episode_proof_not_source_crf_or_gop(tmp_path):
    source = tmp_path / "episode.mp4"
    destination = tmp_path / "preview.mp4"
    # Deliberately use a non-profile CRF and one-frame GOP. The source fast path cannot reliably
    # recover those encoder settings; its proof is the exact episode plus observable media fields.
    _write_video(
        source,
        width=640,
        height=400,
        fps=15,
        frames=15,
        encoder_options={"crf": "18", "g": "1", "keyint_min": "1", "sc_threshold": "0"},
    )

    result = prepare_dataset_preview(source, destination, exact_source=True)
    assert result == source
    assert not destination.exists()

    # Timestamp-free v3 callers still have to opt into an exact-source proof. A shared chunk must
    # not be mislabeled as one episode merely because its bytes happen to be browser-compatible.
    result = prepare_dataset_preview(source, destination, exact_source=False)
    assert result == destination
    assert destination.is_file()


def test_prepare_batch_returns_over_budget_derivatives_for_explicit_policy(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    source = root / "episode.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "previews"
    preview_source = DatasetPreviewSource(episode=0, video_key="camera", relative_path=Path("episode.mp4"))

    def _fake_prepare(_source, output, **_kwargs):
        output.write_bytes(b"x" * 5)
        return output

    monkeypatch.setattr("lerobot_wandb.dataset_preview.prepare_dataset_preview", _fake_prepare)
    # Canonical directory is ten bytes, so its 20% budget is two bytes. The measured five-byte
    # derivative is returned for the caller to approve without re-encoding or deleting it.
    (root / "meta.bin").write_bytes(b"1234")

    batch = prepare_dataset_previews(root, [preview_source], destination)

    assert batch.total_bytes == 5
    assert batch.budget_bytes == 2
    assert batch.over_budget
    assert (destination / "preview-000000.mp4").read_bytes() == b"x" * 5


def test_prepare_rejects_timestamp_range_without_frames(tmp_path):
    source = tmp_path / "episode.mp4"
    destination = tmp_path / "preview.mp4"
    _write_video(source, fps=15, frames=15)

    with pytest.raises(PreviewEncodingError, match="contains no decodable frames"):
        prepare_dataset_preview(source, destination, start_timestamp_s=2.0, end_timestamp_s=3.0)


def test_prepare_batch_reports_known_timestamp_progress(tmp_path):
    root = tmp_path / "dataset"
    source = root / "shared.mp4"
    _write_video(source, fps=15, frames=30)
    preview_source = DatasetPreviewSource(
        episode=4,
        video_key="observation.images.front",
        relative_path=Path("shared.mp4"),
        start_timestamp_s=0.5,
        end_timestamp_s=1.5,
    )
    events = []

    batch = prepare_dataset_previews(
        root, [preview_source], tmp_path / "previews", progress_callback=events.append
    )

    assert batch.previews[0].used_source is False
    assert [event.phase for event in events][0] == "start"
    assert [event.phase for event in events][-1] == "complete"
    progress = [event for event in events if event.phase == "progress"]
    assert progress
    assert all(event.index == 1 and event.total == 1 for event in events)
    assert all(event.source is preview_source for event in events)
    assert all(event.unit == "seconds" for event in progress)
    assert all(event.total_work == 1.0 for event in progress)
    completed = [float(event.completed) for event in progress]
    assert completed == sorted(completed)
    assert all(0.0 <= value <= 1.0 for value in completed)
    assert events[-1].completed == events[-1].total_work == 1.0


def test_prepare_batch_reports_known_whole_file_progress(tmp_path):
    root = tmp_path / "dataset"
    source = root / "episode.mp4"
    _write_video(source, fps=15, frames=30)
    preview_source = DatasetPreviewSource(episode=4, video_key="camera", relative_path=Path("episode.mp4"))
    events = []

    prepare_dataset_previews(root, [preview_source], tmp_path / "previews", progress_callback=events.append)

    progress = [event for event in events if event.phase == "progress"]
    assert progress
    assert all(event.unit == "seconds" for event in progress)
    assert all(event.total_work is not None and event.total_work > 0 for event in progress)
    completed = [float(event.completed) for event in progress]
    assert completed == sorted(completed)
    assert all(
        0.0 <= value <= float(event.total_work) for value, event in zip(completed, progress, strict=True)
    )
    assert events[-1].completed == events[-1].total_work


def test_prepare_batch_reports_frame_activity_without_duration(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    source = root / "episode.mp4"
    _write_video(source, fps=15, frames=15)
    preview_source = DatasetPreviewSource(episode=4, video_key="camera", relative_path=Path("episode.mp4"))
    events = []
    monkeypatch.setattr("lerobot_wandb.dataset_preview._container_duration_s", lambda *_args: None)

    prepare_dataset_previews(root, [preview_source], tmp_path / "previews", progress_callback=events.append)

    progress = [event for event in events if event.phase == "progress"]
    assert progress
    assert all(event.unit == "frames" and event.total_work is None for event in progress)
    assert [event.completed for event in progress] == sorted(event.completed for event in progress)
    assert events[-1].phase == "complete"
    assert events[-1].unit == "frames"
    assert events[-1].total_work is None
    assert events[-1].completed == progress[-1].completed


def test_fast_path_reports_start_and_completion(tmp_path):
    root = tmp_path / "dataset"
    source = root / "episode.mp4"
    _write_video(source, width=640, height=400, fps=15, frames=15)
    preview_source = DatasetPreviewSource(episode=4, video_key="camera", relative_path=Path("episode.mp4"))
    events = []

    batch = prepare_dataset_previews(
        root, [preview_source], tmp_path / "previews", progress_callback=events.append
    )

    assert batch.previews[0].used_source
    assert [event.phase for event in events] == ["start", "complete"]
    assert all(event.index == 1 and event.total == 1 for event in events)
    assert all(event.source is preview_source for event in events)


def test_prepare_batch_cleans_generated_derivatives_after_encoding_failure(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "one.mp4").write_bytes(b"one")
    (root / "two.mp4").write_bytes(b"two")
    destination = tmp_path / "previews"
    sources = [
        DatasetPreviewSource(episode=0, video_key="camera", relative_path=Path("one.mp4")),
        DatasetPreviewSource(episode=1, video_key="camera", relative_path=Path("two.mp4")),
    ]

    def _fake_prepare(_source, output, **_kwargs):
        output.write_bytes(b"partial derivative")
        if output.name == "preview-000001.mp4":
            raise PreviewEncodingError("synthetic encoder failure")
        return output

    monkeypatch.setattr("lerobot_wandb.dataset_preview.prepare_dataset_preview", _fake_prepare)

    with pytest.raises(PreviewEncodingError, match="synthetic encoder failure"):
        prepare_dataset_previews(root, sources, destination)

    assert not (destination / "preview-000000.mp4").exists()
    assert not (destination / "preview-000001.mp4").exists()
