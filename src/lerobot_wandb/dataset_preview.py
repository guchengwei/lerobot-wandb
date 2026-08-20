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
"""Bounded, schema-neutral dataset review media.

The dataset transfer layer resolves a logical episode to a :class:`DatasetPreviewSource`.  This
module owns everything after that boundary: the one Workspace key contract, the deterministic
derivative-encoding profile, and the aggregate byte guard.  The profile's CRF and GOP are applied
only when this module encodes a derivative; source fast-path eligibility uses observable container,
codec, pixel-format, geometry, and frame-rate fields and does not pretend to recover them. It
intentionally uses PyAV
directly rather than LeRobot's video helpers.  The companion can therefore retain its
``lerobot>=0.6.1,<0.6.2`` compatibility promise while keeping canonical dataset files untouched.

Preview preparation is a local operation.  Callers must create the destination directory outside
the canonical dataset root and keep it alive through ``run.finish()`` when the resulting paths are
logged as W&B Run Media.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

PREVIEW_MAX_WIDTH = 640
PREVIEW_MAX_FPS = 15
PREVIEW_CRF = 32
PREVIEW_GOP_SECONDS = 2
PREVIEW_MAX_BYTES = 250 * 1024 * 1024
PREVIEW_CANONICAL_RATIO = 0.20


@dataclass(frozen=True, slots=True)
class DatasetPreviewSource:
    """One exact episode/camera request resolved against a dataset directory.

    ``relative_path`` is always relative to the canonical dataset root.  v2.1's episode-per-file
    source has no timestamp bounds; v3's shared source carries the exact half-open range from its
    episode metadata.  The latter is deliberately never assumed to be a complete source file,
    even when the source happens to contain only one episode.
    """

    episode: int | None
    video_key: str
    relative_path: Path
    start_timestamp_s: float | None = None
    end_timestamp_s: float | None = None
    is_representative: bool = False

    @property
    def is_exact_source_file(self) -> bool:
        """Whether the source file itself is exactly the requested episode.

        v2.1 proves this through its episode-per-file contract.  A v3 source has timestamp bounds
        and is consequently sliced/re-encoded unless a future resolver explicitly introduces a
        stronger proof type.
        """

        return self.start_timestamp_s is None and self.end_timestamp_s is None

    @property
    def has_timestamp_range(self) -> bool:
        return self.start_timestamp_s is not None or self.end_timestamp_s is not None


@dataclass(frozen=True, slots=True)
class PreviewProgressEvent:
    """Structured progress for preparing one dataset preview.

    ``index`` is one-based within the selected batch.  A known duration or timestamp range uses
    seconds for ``completed`` and ``total_work``; when duration metadata is unavailable, the
    encoder reports frame activity with an unknown total instead of inventing a percentage.
    """

    index: int
    total: int
    source: DatasetPreviewSource
    phase: str
    completed: int | float | None
    total_work: int | float | None
    unit: str | None


@dataclass(frozen=True, slots=True)
class PreviewProfile:
    """Deterministic encoding profile for derivatives created for browser review media."""

    max_width: int = PREVIEW_MAX_WIDTH
    max_fps: int = PREVIEW_MAX_FPS
    crf: int = PREVIEW_CRF
    gop_seconds: int = PREVIEW_GOP_SECONDS
    codec: str = "h264"
    pixel_format: str = "yuv420p"
    container: str = "mp4"


DEFAULT_PREVIEW_PROFILE = PreviewProfile()


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """The source properties needed to decide whether the fast path is safe."""

    codec: str
    pixel_format: str
    width: int
    height: int
    fps: float
    duration_s: float | None
    container: str

    def fits(self, profile: PreviewProfile = DEFAULT_PREVIEW_PROFILE) -> bool:
        """Return whether observable source fields fit the browser fast-path constraints.

        Container metadata does not reliably expose the source encoder's CRF or GOP settings, so
        this predicate intentionally checks only the observable fields above. The aggregate byte
        budget bounds the complete Run Media batch independently of those unobservable settings.
        """

        return (
            self.codec in {profile.codec, "avc1", "libx264"}
            and self.pixel_format == profile.pixel_format
            and self.container == profile.container
            and self.width > 0
            and self.height > 0
            and self.width <= profile.max_width
            and self.width % 2 == 0
            and self.height % 2 == 0
            and self.fps > 0
            and self.fps <= profile.max_fps + 1e-6
        )


@dataclass(frozen=True, slots=True)
class PreparedDatasetPreview:
    """One media path and its measured contribution to the aggregate preview budget."""

    source: DatasetPreviewSource
    path: Path
    bytes: int
    used_source: bool


@dataclass(frozen=True, slots=True)
class PreparedPreviewBatch:
    """All prepared media, measured before W&B initialization."""

    previews: tuple[PreparedDatasetPreview, ...]
    total_bytes: int
    canonical_bytes: int
    budget_bytes: int

    @property
    def over_budget(self) -> bool:
        """Whether the measured previews exceed the configured aggregate byte budget."""

        return self.total_bytes > self.budget_bytes

    @property
    def episode_indices(self) -> tuple[int, ...]:
        """Return the selected episode indexes in stable order.

        A batch contains one item per camera, so callers that need run-visible provenance should
        not infer the selection by counting media items.  Sorting also keeps summary metadata
        deterministic when a caller supplied explicit selectors in a different order.
        """

        return tuple(
            sorted(
                {preview.source.episode for preview in self.previews if preview.source.episode is not None}
            )
        )

    @property
    def representative_episode(self) -> int | None:
        """The selected representative episode, if the batch contains one."""

        representatives = {
            preview.source.episode
            for preview in self.previews
            if preview.source.episode is not None and preview.source.is_representative
        }
        return next(iter(representatives)) if len(representatives) == 1 else None


class PreviewBudgetExceededError(ValueError):
    """The caller rejected prepared Run Media that exceeds the aggregate byte budget."""

    def __init__(self, *, measured_bytes: int, budget_bytes: int, preview_count: int) -> None:
        self.measured_bytes = measured_bytes
        self.budget_bytes = budget_bytes
        self.preview_count = preview_count
        measured = _format_bytes(measured_bytes)
        budget = _format_bytes(budget_bytes)
        super().__init__(
            f"Prepared dataset review media totals {measured} across {preview_count} item(s), "
            f"exceeding the configured preview budget of {budget}. Select fewer episodes with "
            "--preview-episode, omit --preview-all, use --no-preview, or rerun with "
            "--force-preview-budget to approve the measured overage."
        )


class PreviewEncodingError(RuntimeError):
    """A source could not be converted into the deterministic preview profile."""


class _PreviewProgress:
    """Adapt encoder work into bounded, monotonic structured progress events.

    The encoder emits an event for each transcoded frame; output throttling belongs to the CLI
    renderer rather than this library-level adapter.
    """

    __slots__ = ("callback", "completed", "index", "source", "total", "total_work", "unit")

    def __init__(
        self,
        callback: Callable[[PreviewProgressEvent], None] | None,
        *,
        index: int,
        total: int,
        source: DatasetPreviewSource,
    ) -> None:
        self.callback = callback
        self.index = index
        self.total = total
        self.source = source
        self.completed: int | float | None = None
        self.total_work: int | float | None = None
        self.unit: str | None = None

    def _emit(self, phase: str) -> None:
        if self.callback is None:
            return
        self.callback(
            PreviewProgressEvent(
                index=self.index,
                total=self.total,
                source=self.source,
                phase=phase,
                completed=self.completed,
                total_work=self.total_work,
                unit=self.unit,
            )
        )

    def start(self) -> None:
        self._emit("start")

    def configure(self, *, total_work: float | None, unit: str) -> None:
        self.total_work = total_work
        self.unit = unit
        self.completed = 0 if unit == "frames" else 0.0

    def progress(self, completed: int | float) -> None:
        if self.unit == "seconds" and self.total_work is not None:
            bounded = min(max(float(completed), 0.0), float(self.total_work))
            previous = float(self.completed) if self.completed is not None else 0.0
            self.completed = max(previous, bounded)
        elif self.unit == "frames":
            previous = int(self.completed) if self.completed is not None else 0
            self.completed = max(previous, int(completed))
        else:
            self.completed = completed
        self._emit("progress")

    def complete(self) -> None:
        if self.total_work is not None:
            self.completed = self.total_work
        self._emit("complete")



def canonical_directory_bytes(root: Path | str) -> int:
    """Measure canonical dataset bytes without following directory symlinks."""

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def preview_budget_bytes(canonical_bytes: int) -> int:
    """Return ``min(250 MiB, 20% of canonical directory bytes)``."""

    if canonical_bytes < 0:
        raise ValueError(f"Canonical byte count cannot be negative, got {canonical_bytes}.")
    return min(PREVIEW_MAX_BYTES, math.floor(canonical_bytes * PREVIEW_CANONICAL_RATIO))


def probe_video(path: Path | str) -> VideoProbe:
    """Read source stream properties without changing the source file."""

    source = Path(path)
    try:
        with av.open(str(source), mode="r") as container:
            if not container.streams.video:
                raise PreviewEncodingError(f"Preview source has no video stream: {source}")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate is not None else 0.0
            codec = (stream.codec_context.name or "").lower()
            pixel_format = (stream.pix_fmt or stream.codec_context.pix_fmt or "").lower()
            container_name = (container.format.name or "").lower()
            duration_s = None
            if stream.duration is not None and stream.time_base is not None:
                duration_s = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration_s = float(container.duration / av.time_base)
            return VideoProbe(
                codec=codec,
                pixel_format=pixel_format,
                width=int(stream.width),
                height=int(stream.height),
                fps=fps,
                duration_s=duration_s,
                container="mp4" if "mp4" in container_name else container_name,
            )
    except PreviewEncodingError:
        raise
    except Exception as error:
        raise PreviewEncodingError(f"Could not inspect preview source {source}: {error}") from error


def _container_duration_s(
    stream: av.video.stream.VideoStream, container: av.container.InputContainer
) -> float | None:
    """Return a finite positive duration from stream or container metadata, if available."""

    duration_s: float | None = None
    if stream.duration is not None and stream.time_base is not None:
        duration_s = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration_s = float(container.duration / av.time_base)
    if duration_s is None or not math.isfinite(duration_s) or duration_s <= 0:
        return None
    return duration_s


def is_browser_compatible(path: Path | str, *, profile: PreviewProfile = DEFAULT_PREVIEW_PROFILE) -> bool:
    """Return whether ``path`` matches observable source-side fast-path constraints.

    This does not claim that a source was encoded with the derivative profile's CRF or GOP.
    """

    try:
        return probe_video(path).fits(profile)
    except PreviewEncodingError:
        return False


def prepare_dataset_preview(
    source: Path | str,
    destination: Path | str,
    *,
    start_timestamp_s: float | None = None,
    end_timestamp_s: float | None = None,
    exact_source: bool | None = None,
    profile: PreviewProfile = DEFAULT_PREVIEW_PROFILE,
    _progress_reporter: _PreviewProgress | None = None,
) -> Path:
    """Prepare one exact episode preview and return its media path.

    The source fast path is intentionally narrow: no timestamp range, a source proven to be one
    episode, and all observable browser-compatibility checks passing. CRF and GOP are not source
    predicates because they cannot be reliably recovered; they are applied only by the derivative
    encoder, while ``prepare_dataset_previews`` measures the aggregate byte budget. Callers that
    have no stronger proof should leave ``exact_source`` unset; the presence of a timestamp range
    already disables the fast path.
    """

    source = Path(source)
    destination = Path(destination)
    if (start_timestamp_s is None) != (end_timestamp_s is None):
        raise ValueError(
            "Dataset preview timestamp bounds must be supplied together: "
            f"start={start_timestamp_s}, end={end_timestamp_s}."
        )
    if start_timestamp_s is not None:
        if end_timestamp_s is None:
            raise ValueError("Dataset preview end timestamp is required when start is supplied.")
        if not (math.isfinite(start_timestamp_s) and math.isfinite(end_timestamp_s)):
            raise ValueError(
                "Dataset preview timestamp bounds must be finite: "
                f"start={start_timestamp_s}, end={end_timestamp_s}."
            )
        if start_timestamp_s < 0:
            raise ValueError(
                f"Dataset preview start timestamp must be non-negative, got {start_timestamp_s}."
            )
        if end_timestamp_s <= start_timestamp_s:
            raise ValueError(
                f"Dataset preview end timestamp ({end_timestamp_s}) must be greater than "
                f"start ({start_timestamp_s})."
            )
    if exact_source is None:
        # A generic caller cannot prove that a timestamp-free source contains exactly one
        # episode. Dataset selection supplies the v2.1 proof explicitly; v3 supplies bounds.
        exact_source = False
    if exact_source and start_timestamp_s is None and is_browser_compatible(source, profile=profile):
        return source

    return _encode_preview(
        source,
        destination,
        start_timestamp_s=start_timestamp_s,
        end_timestamp_s=end_timestamp_s,
        profile=profile,
        progress_reporter=_progress_reporter,
    )


def prepare_dataset_previews(
    root: Path | str,
    sources: Iterable[DatasetPreviewSource],
    destination_dir: Path | str,
    *,
    profile: PreviewProfile = DEFAULT_PREVIEW_PROFILE,
    progress_callback: Callable[[PreviewProgressEvent], None] | None = None,
) -> PreparedPreviewBatch:
    """Prepare and measure one selected preview batch before ``wandb.init``."""

    root = Path(root).resolve()
    destination_dir = Path(destination_dir).resolve()
    if root == destination_dir or root in destination_dir.parents:
        raise ValueError(
            f"Preview destination {destination_dir} must be outside the canonical dataset root {root}."
        )
    destination_dir.mkdir(parents=True, exist_ok=True)

    selected_sources = tuple(sources)
    canonical_bytes = canonical_directory_bytes(root)
    budget_bytes = preview_budget_bytes(canonical_bytes)
    prepared: list[PreparedDatasetPreview] = []
    generated_paths: list[Path] = []
    try:
        total = len(selected_sources)
        for index, source in enumerate(selected_sources):
            source_path = (root / source.relative_path).resolve()
            try:
                source_path.relative_to(root)
            except ValueError as error:
                raise PreviewEncodingError(
                    f"Preview source {source.relative_path} resolves outside dataset root {root}."
                ) from error
            if not source_path.is_file():
                raise PreviewEncodingError(f"Preview source does not exist: {source_path}")
            destination = destination_dir / f"preview-{index:06d}.mp4"
            generated_paths.append(destination)
            progress_reporter = _PreviewProgress(
                progress_callback,
                index=index + 1,
                total=total,
                source=source,
            )
            progress_reporter.start()
            path = prepare_dataset_preview(
                source_path,
                destination,
                start_timestamp_s=source.start_timestamp_s,
                end_timestamp_s=source.end_timestamp_s,
                exact_source=source.is_exact_source_file,
                profile=profile,
                _progress_reporter=progress_reporter if progress_callback is not None else None,
            )
            size = path.stat().st_size
            prepared.append(
                PreparedDatasetPreview(
                    source=source,
                    path=path,
                    bytes=size,
                    used_source=path.resolve() == source_path,
                )
            )
            progress_reporter.complete()
        total_bytes = sum(item.bytes for item in prepared)
        return PreparedPreviewBatch(
            previews=tuple(prepared),
            total_bytes=total_bytes,
            canonical_bytes=canonical_bytes,
            budget_bytes=budget_bytes,
        )
    except Exception:
        # Encoded derivatives are disposable.  A fast-path source is canonical and must never be
        # removed while cleaning up a failed batch.
        canonical_paths = {item.path.resolve() for item in prepared if item.used_source}
        for path in generated_paths:
            if path.resolve() not in canonical_paths:
                path.unlink(missing_ok=True)
        raise


def _encode_preview(
    source: Path,
    destination: Path,
    *,
    start_timestamp_s: float | None,
    end_timestamp_s: float | None,
    profile: PreviewProfile,
    progress_reporter: _PreviewProgress | None = None,
) -> Path:
    """Decode, select, scale, and encode one source with PyAV."""

    if destination.resolve() == source.resolve():
        raise PreviewEncodingError("Preview destination must differ from its source path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with av.open(str(source), mode="r") as input_container:
            if not input_container.streams.video:
                raise PreviewEncodingError(f"Preview source has no video stream: {source}")
            input_stream = input_container.streams.video[0]
            input_rate = input_stream.average_rate or input_stream.base_rate
            input_fps = float(input_rate) if input_rate is not None else float(profile.max_fps)
            if not math.isfinite(input_fps) or input_fps <= 0:
                input_fps = float(profile.max_fps)
            target_fps = min(input_fps, float(profile.max_fps))
            duration_s = _container_duration_s(input_stream, input_container)
            if progress_reporter is not None:
                if start_timestamp_s is not None and end_timestamp_s is not None:
                    progress_reporter.configure(
                        total_work=end_timestamp_s - start_timestamp_s,
                        unit="seconds",
                    )
                elif duration_s is not None:
                    progress_reporter.configure(total_work=duration_s, unit="seconds")
                else:
                    progress_reporter.configure(total_work=None, unit="frames")
            target_rate = Fraction(target_fps).limit_denominator(1000)
            output_width, output_height = _preview_dimensions(
                int(input_stream.width), int(input_stream.height), profile.max_width
            )
            gop_frames = max(1, round(float(target_rate) * profile.gop_seconds))

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".mp4", prefix=f".{destination.name}.", dir=destination.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)

            with av.open(
                str(temporary_path),
                mode="w",
                format=profile.container,
                options={"movflags": "faststart"},
            ) as output_container:
                output_stream = output_container.add_stream(
                    "libx264",
                    rate=target_rate,
                    options={
                        "crf": str(profile.crf),
                        "g": str(gop_frames),
                        "keyint_min": str(gop_frames),
                        # A fixed two-second GOP is part of the preview profile.  Without this,
                        # x264 may insert scene-cut keyframes well before the requested interval.
                        "sc_threshold": "0",
                    },
                )
                output_stream.width = output_width
                output_stream.height = output_height
                output_stream.pix_fmt = profile.pixel_format

                next_output_time = start_timestamp_s if start_timestamp_s is not None else 0.0
                frame_index = 0
                for decoded_index, frame in enumerate(input_container.decode(input_stream)):
                    frame_time = frame.time
                    if frame_time is None or not math.isfinite(frame_time):
                        frame_time = decoded_index / input_fps
                    if start_timestamp_s is not None and frame_time < start_timestamp_s:
                        continue
                    if end_timestamp_s is not None and frame_time >= end_timestamp_s:
                        break
                    if frame_time + 1e-9 < next_output_time:
                        continue

                    converted = frame.reformat(
                        width=output_width,
                        height=output_height,
                        format=profile.pixel_format,
                    )
                    converted.pts = frame_index
                    converted.time_base = Fraction(1, 1) / target_rate
                    for packet in output_stream.encode(converted):
                        output_container.mux(packet)
                    frame_index += 1
                    if progress_reporter is not None:
                        progress_time = (
                            frame_time - start_timestamp_s if start_timestamp_s is not None else frame_time
                        )
                        if progress_reporter.unit == "frames":
                            progress_reporter.progress(frame_index)
                        else:
                            progress_reporter.progress(progress_time)
                    next_output_time += 1.0 / float(target_rate)

                if frame_index == 0:
                    raise PreviewEncodingError(
                        f"The requested preview range contains no decodable frames in {source}: "
                        f"start={start_timestamp_s}, end={end_timestamp_s}."
                    )
                for packet in output_stream.encode():
                    output_container.mux(packet)

        if temporary_path is None:
            raise PreviewEncodingError("Preview encoder did not create a temporary output.")
        os.replace(temporary_path, destination)
        temporary_path = None
        if not destination.is_file():
            raise PreviewEncodingError(f"Preview encoder produced no output: {destination}")
        return destination
    except PreviewEncodingError:
        raise
    except Exception as error:
        raise PreviewEncodingError(f"Could not encode preview {source} -> {destination}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _preview_dimensions(width: int, height: int, max_width: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise PreviewEncodingError(f"Video dimensions must be positive, got {width}x{height}.")
    scaled_width = min(width, max_width)
    scaled_width = max(2, scaled_width - (scaled_width % 2))
    scaled_height = max(2, round(height * scaled_width / width))
    scaled_height -= scaled_height % 2
    scaled_height = max(2, scaled_height)
    return scaled_width, scaled_height


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MiB ({value:,} bytes)"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB ({value:,} bytes)"
    return f"{value:,} bytes"
