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
"""Version-aware dataset transfer validation and review-video selection.

Current LeRobot intentionally refuses to read v2.1 datasets and asks callers to migrate them to
v3.0. That is the right behavior for the current training reader, but it is too strict for a
transport tool: ``lerobot-wandb`` must be able to preserve a canonical v2.1 directory byte-for-byte
for consumers such as GR00T that still require the legacy episode-per-file layout.

This module therefore keeps two contracts separate:

- v3.0 uses the existing current-LeRobot validator unchanged;
- v2.1 is validated locally against the final v2.1 on-disk contract, without invoking the current
  reader's backward-compatibility gate.

The same version-aware validator is used on upload and download, so a v2.1 artifact still round-trips
through W&B even though the installed current LeRobot reader would not train from it directly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from packaging.version import InvalidVersion, Version

from . import lerobot_adapter as _lerobot
from .compatibility import check_lerobot_compatible
from .dataset_preview import (
    DEFAULT_PREVIEW_PROFILE,
    DatasetPreviewSource,
    PreviewProfile,
    is_browser_compatible,
    prepare_dataset_preview as _prepare_dataset_preview,
)
from .inspect import DatasetDirectoryError, DatasetDirectoryMetadata, inspect_dataset_directory

V21_INFO_PATH = Path("meta/info.json")
V21_EPISODES_PATH = Path("meta/episodes.jsonl")
V21_EPISODES_STATS_PATH = Path("meta/episodes_stats.jsonl")
V21_TASKS_PATH = Path("meta/tasks.jsonl")

DatasetLayout = Literal["v2.1", "v3"]
DEFAULT_PREVIEW_MAX_EPISODES = 50


@dataclass(frozen=True, slots=True)
class TransferDataset:
    """A validated dataset together with the layout facts transfer/review code needs."""

    root: Path
    layout: DatasetLayout
    metadata: DatasetDirectoryMetadata
    info: dict[str, Any]


def inspect_transfer_dataset(root: Path | str) -> TransferDataset:
    """Validate a v3.0 or canonical v2.1 dataset for byte-preserving transfer."""
    # Preserve the companion CLI's compatibility boundary before touching the on-disk format. This
    # keeps a missing LeRobot install actionable even when the caller points at a malformed path.
    check_lerobot_compatible()
    root = Path(root)
    info = _load_info_json(root)
    version = _parse_schema_version(root, info)

    if version.major == 3:
        metadata = inspect_dataset_directory(root)
        return TransferDataset(root=root, layout="v3", metadata=metadata, info=info)

    if version == Version("2.1"):
        metadata = _validate_v21_dataset(root, info)
        return TransferDataset(root=root, layout="v2.1", metadata=metadata, info=info)

    raise DatasetDirectoryError(
        f"{root}/{V21_INFO_PATH} declares unsupported dataset schema {version}. "
        "lerobot-wandb transfer supports only v2.1 and the installed current v3.x format."
    )


def validate_transfer_dataset(root: Path | str) -> DatasetDirectoryMetadata:
    """Validate ``root`` for artifact transfer and return its extracted metadata."""
    return inspect_transfer_dataset(root).metadata


def select_dataset_preview_sources(
    dataset: TransferDataset,
    *,
    episodes: Sequence[int] = (),
    preview_all: bool = False,
    max_episodes: int = DEFAULT_PREVIEW_MAX_EPISODES,
) -> list[DatasetPreviewSource]:
    """Select bounded review media without changing the canonical artifact.

    With no explicit episode request, exactly one deterministic representative source is selected.
    Explicit episodes and ``preview_all`` select every camera for each selected episode. v2.1
    sources are already episode-per-file; v3 sources carry the exact timestamp range to trim from
    their shared video file. ``preview_all`` is refused when the dataset exceeds ``max_episodes``.
    """
    if preview_all and episodes:
        raise DatasetDirectoryError(
            "--preview-all and --preview-episode are mutually exclusive; choose one review mode."
        )
    if preview_all and max_episodes <= 0:
        raise DatasetDirectoryError(f"preview maximum must be greater than zero, got {max_episodes}.")
    if preview_all and dataset.metadata.total_episodes > max_episodes:
        raise DatasetDirectoryError(
            f"--preview-all selected {dataset.metadata.total_episodes} episodes, exceeding the configured "
            f"maximum of {max_episodes}. Raise --preview-max-episodes explicitly to allow this upload."
        )

    is_representative = not preview_all and not episodes
    selected = list(range(dataset.metadata.total_episodes)) if preview_all else list(dict.fromkeys(episodes))
    for episode in selected:
        if isinstance(episode, bool) or not isinstance(episode, Integral):
            raise DatasetDirectoryError(f"preview episode must be an integer, got {episode!r}.")
        if episode < 0 or episode >= dataset.metadata.total_episodes:
            raise DatasetDirectoryError(
                f"preview episode {episode} is outside the dataset range "
                f"0..{dataset.metadata.total_episodes - 1}."
            )
    selected = [int(episode) for episode in selected]

    if dataset.metadata.total_episodes == 0 or not dataset.metadata.video_keys:
        return []

    # The default representative is deliberately one exact episode and *every* camera.  Keeping
    # this selection schema-neutral is what lets a Workspace key configured for one dataset
    # schema work for the other. Episode zero is the stable representative index; it is also
    # surfaced in run summary metadata so users never have to infer it from a key.
    selected = selected or [0]
    video_keys = tuple(sorted(dataset.metadata.video_keys))
    if dataset.layout == "v2.1":
        return [
            DatasetPreviewSource(
                episode=episode,
                video_key=video_key,
                relative_path=_v21_video_path(dataset.root, dataset.info, episode, video_key),
                is_representative=is_representative,
            )
            for episode in selected
            for video_key in video_keys
        ]

    video_path = dataset.info.get("video_path")
    if not isinstance(video_path, str):
        raise DatasetDirectoryError(
            f"{dataset.root}/{V21_INFO_PATH} cannot resolve episode previews: no video_path template."
        )
    rows: dict[int, dict[str, Any]] = {}
    try:
        episode_rows = _lerobot.load_episodes(dataset.root)
        for row_number, row in enumerate(episode_rows):
            raw_episode = row.get("episode_index")
            if isinstance(raw_episode, bool) or not isinstance(raw_episode, Integral) or raw_episode < 0:
                raise ValueError(f"row {row_number} has invalid episode_index={raw_episode!r}")
            episode_index = int(raw_episode)
            if episode_index in rows:
                raise ValueError(f"duplicate episode_index={episode_index}")
            rows[episode_index] = row
    except DatasetDirectoryError:
        raise
    except Exception as error:
        raise DatasetDirectoryError(
            f"{dataset.root} episode metadata could not be read for preview selection: {error}"
        ) from error
    resolved: list[DatasetPreviewSource] = []
    for episode in selected:
        row = rows.get(episode)
        if row is None:
            raise DatasetDirectoryError(
                f"{dataset.root} episode metadata has no row for requested preview episode {episode}."
            )
        for video_key in video_keys:
            try:
                raw_chunk_index = row[f"videos/{video_key}/chunk_index"]
                raw_file_index = row[f"videos/{video_key}/file_index"]
                if (
                    isinstance(raw_chunk_index, bool)
                    or not isinstance(raw_chunk_index, Integral)
                    or raw_chunk_index < 0
                ):
                    raise ValueError(f"invalid chunk_index={raw_chunk_index!r}")
                if (
                    isinstance(raw_file_index, bool)
                    or not isinstance(raw_file_index, Integral)
                    or raw_file_index < 0
                ):
                    raise ValueError(f"invalid file_index={raw_file_index!r}")
                chunk_index = int(raw_chunk_index)
                file_index = int(raw_file_index)
                raw_start_timestamp = row[f"videos/{video_key}/from_timestamp"]
                raw_end_timestamp = row[f"videos/{video_key}/to_timestamp"]
                if isinstance(raw_start_timestamp, bool) or isinstance(raw_end_timestamp, bool):
                    raise ValueError("timestamps must be numeric, not boolean")
                start_timestamp_s = float(raw_start_timestamp)
                end_timestamp_s = float(raw_end_timestamp)
                if not (
                    math.isfinite(start_timestamp_s)
                    and math.isfinite(end_timestamp_s)
                    and start_timestamp_s >= 0
                    and end_timestamp_s > start_timestamp_s
                ):
                    raise ValueError(f"invalid timestamp range {start_timestamp_s}..{end_timestamp_s}")
                relative_path = _safe_template_path(
                    dataset.root,
                    video_path,
                    f"video {video_key!r}",
                    video_key=video_key,
                    chunk_index=chunk_index,
                    file_index=file_index,
                    episode_index=episode,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetDirectoryError(
                    f"{dataset.root} cannot resolve v3 preview episode {episode}, camera {video_key!r}: "
                    f"{error}"
                ) from error
            resolved.append(
                DatasetPreviewSource(
                    episode=episode,
                    video_key=video_key,
                    relative_path=relative_path,
                    start_timestamp_s=start_timestamp_s,
                    end_timestamp_s=end_timestamp_s,
                    is_representative=is_representative,
                )
            )
    return resolved


def prepare_dataset_preview(
    source: Path,
    destination: Path,
    *,
    start_timestamp_s: float | None = None,
    end_timestamp_s: float | None = None,
    exact_source: bool | None = None,
    profile: PreviewProfile = DEFAULT_PREVIEW_PROFILE,
) -> Path:
    """Create browser-playable review media through the companion-local preview boundary."""
    return _prepare_dataset_preview(
        source,
        destination,
        start_timestamp_s=start_timestamp_s,
        end_timestamp_s=end_timestamp_s,
        exact_source=exact_source,
        profile=profile,
    )


def _is_browser_h264(path: Path) -> bool:
    return is_browser_compatible(path, profile=DEFAULT_PREVIEW_PROFILE)


def _load_info_json(root: Path) -> dict[str, Any]:
    path = root / V21_INFO_PATH
    if not root.is_dir():
        raise DatasetDirectoryError(f"{root} is not a directory.")
    try:
        with path.open(encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetDirectoryError(f"{path} could not be read as JSON: {error}") from error
    if not isinstance(info, dict):
        raise DatasetDirectoryError(f"{path} must contain a JSON object.")
    return info


def _parse_schema_version(root: Path, info: dict[str, Any]) -> Version:
    value = info.get("codebase_version")
    if not isinstance(value, str):
        raise DatasetDirectoryError(f"{root}/{V21_INFO_PATH} has no string codebase_version.")
    try:
        return Version(value)
    except InvalidVersion as error:
        raise DatasetDirectoryError(
            f"{root}/{V21_INFO_PATH} has invalid codebase_version {value!r}."
        ) from error


def _validate_v21_dataset(root: Path, info: dict[str, Any]) -> DatasetDirectoryMetadata:
    fps = _positive_int(root, info, "fps")
    total_episodes = _nonnegative_int(root, info, "total_episodes")
    total_frames = _nonnegative_int(root, info, "total_frames")
    total_tasks = _nonnegative_int(root, info, "total_tasks")
    chunks_size = _positive_int(root, info, "chunks_size")

    features = info.get("features")
    if not isinstance(features, dict) or not all(
        isinstance(key, str) and isinstance(value, dict) for key, value in features.items()
    ):
        raise DatasetDirectoryError(f"{root}/{V21_INFO_PATH} has an invalid features object.")

    data_path = info.get("data_path")
    video_path = info.get("video_path")
    if not isinstance(data_path, str) or not data_path:
        raise DatasetDirectoryError(f"{root}/{V21_INFO_PATH} has no usable data_path template.")

    camera_keys = tuple(
        sorted(key for key, feature in features.items() if feature.get("dtype") in {"image", "video"})
    )
    video_keys = tuple(sorted(key for key, feature in features.items() if feature.get("dtype") == "video"))
    if video_keys and (not isinstance(video_path, str) or not video_path):
        raise DatasetDirectoryError(f"{root}/{V21_INFO_PATH} declares video features without video_path.")

    episodes = _load_jsonl(root / V21_EPISODES_PATH, required=total_episodes > 0)
    if len(episodes) != total_episodes:
        raise DatasetDirectoryError(
            f"{root}/{V21_EPISODES_PATH} contains {len(episodes)} rows; expected {total_episodes}."
        )
    episode_indices = [_row_index(root, row, "episode_index", V21_EPISODES_PATH) for row in episodes]
    if episode_indices != list(range(total_episodes)):
        raise DatasetDirectoryError(
            f"{root}/{V21_EPISODES_PATH} must contain episodes ordered from 0 to {total_episodes - 1}."
        )

    stats_rows = _load_jsonl(root / V21_EPISODES_STATS_PATH, required=total_episodes > 0)
    if len(stats_rows) != total_episodes:
        raise DatasetDirectoryError(
            f"{root}/{V21_EPISODES_STATS_PATH} contains {len(stats_rows)} rows; expected {total_episodes}."
        )
    stats_indices = [_row_index(root, row, "episode_index", V21_EPISODES_STATS_PATH) for row in stats_rows]
    if stats_indices != list(range(total_episodes)):
        raise DatasetDirectoryError(
            f"{root}/{V21_EPISODES_STATS_PATH} must contain one row per episode in order."
        )

    tasks = _load_jsonl(root / V21_TASKS_PATH, required=total_tasks > 0)
    if len(tasks) != total_tasks:
        raise DatasetDirectoryError(
            f"{root}/{V21_TASKS_PATH} contains {len(tasks)} rows; expected {total_tasks}."
        )
    task_indices = [_row_index(root, row, "task_index", V21_TASKS_PATH) for row in tasks]
    if task_indices != list(range(total_tasks)):
        raise DatasetDirectoryError(f"{root}/{V21_TASKS_PATH} must contain tasks ordered by task_index.")

    expected_parquet_columns = {key for key, feature in features.items() if feature.get("dtype") != "video"}
    video_paths_by_key: dict[str, dict[Path, int]] = {video_key: {} for video_key in video_keys}
    counted_frames = 0
    for episode, row in enumerate(episodes):
        length = _row_positive_int(root, row, "length", episode)
        data_rel = _safe_template_path(
            root,
            data_path,
            "data",
            episode_chunk=episode // chunks_size,
            episode_index=episode,
        )
        data_file = root / data_rel
        if not data_file.is_file():
            raise DatasetDirectoryError(f"{root} is missing v2.1 data file: {data_rel}.")
        try:
            parquet = pq.ParquetFile(data_file)
        except Exception as error:
            raise DatasetDirectoryError(f"{data_file} is not readable Parquet: {error}") from error
        if parquet.metadata.num_rows != length:
            raise DatasetDirectoryError(
                f"{data_file} contains {parquet.metadata.num_rows} rows; episode {episode} declares {length}."
            )
        missing_columns = sorted(expected_parquet_columns - set(parquet.schema_arrow.names))
        if missing_columns:
            raise DatasetDirectoryError(
                f"{data_file} is missing declared non-video feature column(s): {', '.join(missing_columns)}."
            )
        counted_frames += length

        for video_key in video_keys:
            relative = _v21_video_path(root, info, episode, video_key)
            previous_episode = video_paths_by_key[video_key].get(relative)
            if previous_episode is not None:
                raise DatasetDirectoryError(
                    f"{root}/{V21_INFO_PATH} cannot prove v2.1 episode-per-file videos for "
                    f"{video_key!r}: episodes {previous_episode} and {episode} resolve to {relative}."
                )
            video_paths_by_key[video_key][relative] = episode
            if not (root / relative).is_file():
                raise DatasetDirectoryError(
                    f"{root} is missing v2.1 video for episode {episode}, {video_key!r}: {relative}."
                )

    if counted_frames != total_frames:
        raise DatasetDirectoryError(
            f"{root} declares total_frames={total_frames}, but episode metadata sums to {counted_frames}."
        )

    robot_type = info.get("robot_type")
    if robot_type is not None and not isinstance(robot_type, str):
        raise DatasetDirectoryError(f"{root}/{V21_INFO_PATH} robot_type must be a string or null.")

    return DatasetDirectoryMetadata(
        schema_version=str(info["codebase_version"]),
        robot_type=robot_type,
        fps=fps,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=total_tasks,
        camera_keys=camera_keys,
        video_keys=video_keys,
        git_commit=_lerobot.lerobot_git_commit(),
    )


def _v21_video_path(root: Path, info: dict[str, Any], episode: int, video_key: str) -> Path:
    chunks_size = _positive_int(root, info, "chunks_size")
    template = info.get("video_path")
    if not isinstance(template, str):
        raise DatasetDirectoryError(f"{root}/{V21_INFO_PATH} has no video_path template.")
    return _safe_template_path(
        root,
        template,
        f"video {video_key!r}",
        episode_chunk=episode // chunks_size,
        episode_index=episode,
        video_key=video_key,
    )


def _safe_template_path(root: Path, template: str, payload: str, **values: Any) -> Path:
    try:
        path = Path(template.format(**values))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise DatasetDirectoryError(
            f"{root}/{V21_INFO_PATH} cannot resolve v2.1 {payload} path: {error}"
        ) from error
    if path.is_absolute() or ".." in path.parts:
        raise DatasetDirectoryError(
            f"{root}/{V21_INFO_PATH} resolves {payload} outside the dataset root: {path}."
        )
    try:
        (root / path).resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise DatasetDirectoryError(
            f"{root}/{V21_INFO_PATH} resolves {payload} outside the dataset root: {path}."
        ) from error
    return path


def _load_jsonl(path: Path, *, required: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        if not required:
            return []
        raise DatasetDirectoryError(f"{path} is required for this v2.1 dataset.")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise DatasetDirectoryError(f"{path}:{line_number} must contain a JSON object.")
                rows.append(row)
    except json.JSONDecodeError as error:
        raise DatasetDirectoryError(f"{path} contains invalid JSONL: {error}") from error
    except OSError as error:
        raise DatasetDirectoryError(f"{path} could not be read: {error}") from error
    return rows


def _row_index(root: Path, row: dict[str, Any], key: str, path: Path) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetDirectoryError(f"{root}/{path} has invalid {key}={value!r}.")
    return value


def _row_positive_int(root: Path, row: dict[str, Any], key: str, episode: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetDirectoryError(
            f"{root}/{V21_EPISODES_PATH} episode {episode} has invalid {key}={value!r}."
        )
    return value


def _nonnegative_int(root: Path, info: dict[str, Any], key: str) -> int:
    value = info.get(key)
    if isinstance(value, bool):
        value = None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise DatasetDirectoryError(f"{root}/{V21_INFO_PATH} has invalid {key}={info.get(key)!r}.")
    return value


def _positive_int(root: Path, info: dict[str, Any], key: str) -> int:
    value = _nonnegative_int(root, info, key)
    if value <= 0:
        raise DatasetDirectoryError(f"{root}/{V21_INFO_PATH} has invalid {key}={value!r}; expected > 0.")
    return value
