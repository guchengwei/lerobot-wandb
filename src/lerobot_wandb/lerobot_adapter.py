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
"""The single boundary for ``lerobot`` imports in ``lerobot_wandb``.

No other companion module imports LeRobot. This adapter owns:

- installed LeRobot presence/version validation (delegated to ``compatibility``);
- lazy, compatibility-gated imports of the LeRobot dataset metadata/readers and video
  re-encoding helpers the integration uses;
- locating the LeRobot checkout commit for metadata stamping.

Importing this module never touches LeRobot: symbols are resolved lazily on first
access, and LeRobot-dependent accessors raise
:class:`~lerobot_wandb.compatibility.LeRobotCompatibilityError` (never an import
traceback) when LeRobot is absent or unsupported.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from .compatibility import LeRobotCompatibilityError, check_lerobot_compatible, get_allow_unsupported

if TYPE_CHECKING:
    # Type-checker-only aliases for the lazy symbols below; ruff sees no usage because
    # resolution goes through __getattr__ and annotations are strings (PEP 563).
    from lerobot.configs.video import RGBEncoderConfig  # noqa: F401
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION  # noqa: F401
    from lerobot.datasets.utils import (  # noqa: F401
        DATA_DIR,
        DEFAULT_TASKS_PATH,
        EPISODES_DIR,
        INFO_PATH,
        STATS_PATH,
        DatasetInfo,
    )
    from lerobot.utils.constants import DEFAULT_FEATURES  # noqa: F401

# symbol -> (lerobot module, attribute name)
_LAZY_SYMBOLS = {
    "CODEBASE_VERSION": ("lerobot.datasets.dataset_metadata", "CODEBASE_VERSION"),
    "DEFAULT_FEATURES": ("lerobot.utils.constants", "DEFAULT_FEATURES"),
    "DATA_DIR": ("lerobot.datasets.utils", "DATA_DIR"),
    "DEFAULT_TASKS_PATH": ("lerobot.datasets.utils", "DEFAULT_TASKS_PATH"),
    "EPISODES_DIR": ("lerobot.datasets.utils", "EPISODES_DIR"),
    "INFO_PATH": ("lerobot.datasets.utils", "INFO_PATH"),
    "STATS_PATH": ("lerobot.datasets.utils", "STATS_PATH"),
    "DatasetInfo": ("lerobot.datasets.utils", "DatasetInfo"),
    "RGBEncoderConfig": ("lerobot.configs.video", "RGBEncoderConfig"),
}

_IO_UTILS = "lerobot.datasets.io_utils"
_FEATURE_UTILS = "lerobot.datasets.feature_utils"
_VIDEO_UTILS = "lerobot.datasets.video_utils"

_loaded: dict[str, ModuleType] = {}


def _module(module_name: str) -> ModuleType:
    """Import a LeRobot submodule lazily, gated on installed-version compatibility."""
    check_lerobot_compatible(allow_unsupported=get_allow_unsupported())
    if module_name not in _loaded:
        try:
            _loaded[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise LeRobotCompatibilityError(
                f"The installed LeRobot distribution is missing module {module_name!r}, which "
                "this lerobot-wandb command needs. Install a complete lerobot installation."
            ) from e
    return _loaded[module_name]


def __getattr__(name: str) -> Any:
    """Resolve LeRobot constants/types lazily so importing this module stays LeRobot-free."""
    entry = _LAZY_SYMBOLS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = entry
    return getattr(_module(module_name), attribute)


# --- Dataset metadata/readers ------------------------------------------------


def load_info(root: Any) -> Any:
    """See ``lerobot.datasets.io_utils.load_info``."""
    return _module(_IO_UTILS).load_info(root)


def load_episodes(root: Any) -> Any:
    """See ``lerobot.datasets.io_utils.load_episodes``."""
    return _module(_IO_UTILS).load_episodes(root)


def load_stats(root: Any) -> Any:
    """See ``lerobot.datasets.io_utils.load_stats``."""
    return _module(_IO_UTILS).load_stats(root)


def load_tasks(root: Any) -> Any:
    """See ``lerobot.datasets.io_utils.load_tasks``."""
    return _module(_IO_UTILS).load_tasks(root)


def load_nested_dataset(root: Any, *, features: Any) -> Any:
    """See ``lerobot.datasets.io_utils.load_nested_dataset``."""
    return _module(_IO_UTILS).load_nested_dataset(root, features=features)


def get_hf_features_from_features(features: Any) -> Any:
    """See ``lerobot.datasets.feature_utils.get_hf_features_from_features``."""
    return _module(_FEATURE_UTILS).get_hf_features_from_features(features)


def check_version_compatibility(repo_id: Any, codebase_version: Any, expected: Any) -> None:
    """See ``lerobot.datasets.utils.check_version_compatibility``."""
    _module("lerobot.datasets.utils").check_version_compatibility(repo_id, codebase_version, expected)


def codebase_version() -> str:
    """The dataset schema version constant of the installed LeRobot."""
    return _module("lerobot.datasets.dataset_metadata").CODEBASE_VERSION


# --- Video re-encoding ------------------------------------------------------


def reencode_video(
    source: Any,
    destination: Any,
    *,
    video_encoder: Any,
    overwrite: bool,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> None:
    """See ``lerobot.datasets.video_utils.reencode_video``."""
    _module(_VIDEO_UTILS).reencode_video(
        source,
        destination,
        video_encoder=video_encoder,
        overwrite=overwrite,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )


# --- Checkout provenance ----------------------------------------------------


def lerobot_git_commit() -> str | None:
    """Best-effort commit of the installed LeRobot checkout; ``None`` when it has none.

    Locates the installed ``lerobot`` package (a wheel install has no checkout, so this
    resolves to ``None``; a fork development install resolves to its git repo and
    returns the current HEAD).
    """
    try:
        spec = find_spec("lerobot")
    except (ImportError, ModuleNotFoundError):
        return None
    if spec is None or spec.origin is None:
        return None
    package_dir = Path(spec.origin).resolve().parent
    return _git_commit_of(package_dir, "src/lerobot")


def _git_commit_of(package_dir: Path, src_layout: str) -> str | None:
    """Return the git HEAD commit whose checkout contains ``src_layout``, or ``None``.

    Resolved to an absolute path (rather than passing the bare "git") so bandit's B607
    partial-executable-path check doesn't flag a PATH-search invocation: this always
    runs a fixed literal argv with no shell and no user input, but bandit can't see that.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        root_result = subprocess.run(
            [git, "rev-parse", "--show-toplevel"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if root_result.returncode != 0:
        return None

    repo_root = Path(root_result.stdout.strip()).resolve()
    if (repo_root / src_layout).resolve() != package_dir:
        return None
    try:
        commit_result = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if commit_result.returncode != 0:
        return None
    return commit_result.stdout.strip() or None
