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
    PreviewBudgetExceededError,
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


def test_prepare_batch_enforces_aggregate_budget_before_publication(tmp_path, monkeypatch):
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
    # derivative must fail instead of silently reducing quality or dropping the camera.
    (root / "meta.bin").write_bytes(b"1234")

    with pytest.raises(PreviewBudgetExceededError, match="5 bytes.*2 bytes") as error:
        prepare_dataset_previews(root, [preview_source], destination)

    assert error.value.measured_bytes == 5
    assert error.value.budget_bytes == 2
    assert not (destination / "preview-000000.mp4").exists()


def test_prepare_rejects_timestamp_range_without_frames(tmp_path):
    source = tmp_path / "episode.mp4"
    destination = tmp_path / "preview.mp4"
    _write_video(source, fps=15, frames=15)

    with pytest.raises(PreviewEncodingError, match="contains no decodable frames"):
        prepare_dataset_preview(source, destination, start_timestamp_s=2.0, end_timestamp_s=3.0)
