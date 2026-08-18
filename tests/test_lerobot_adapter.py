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

"""The adapter is the sole boundary for `lerobot` imports in lerobot_wandb.

It must be importable — and its non-LeRobot surface usable — without LeRobot
installed, and it must turn missing/unsupported LeRobot into actionable
compatibility errors instead of import tracebacks.
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lerobot_wandb import compatibility as compatibility_module, lerobot_adapter as adapter
from lerobot_wandb.compatibility import (
    LeRobotCompatibilityError,
    set_allow_unsupported,
)


def test_importing_the_adapter_does_not_import_lerobot():
    preamble = textwrap.dedent(
        """
        import builtins
        blocked = ("lerobot",)
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == package or name.startswith(package + ".") for package in blocked):
                raise ModuleNotFoundError(name + " deliberately unavailable")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            preamble
            + textwrap.dedent(
                """
                import lerobot_wandb
                import lerobot_wandb.lerobot_adapter
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_lerobot_turns_into_actionable_error(monkeypatch):
    monkeypatch.setattr(compatibility_module, "get_installed_lerobot_version", lambda: None)
    with pytest.raises(LeRobotCompatibilityError) as excinfo:
        adapter.load_info(".")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "not installed" in message
    assert "pip install" in message


def test_unsupported_lerobot_turns_into_actionable_error(monkeypatch):
    monkeypatch.setattr(compatibility_module, "get_installed_lerobot_version", lambda: "0.5.0")
    with pytest.raises(LeRobotCompatibilityError) as excinfo:
        adapter.load_info(".")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "Unsupported LeRobot version 0.5.0" in message
    assert "--allow-unsupported-lerobot" in message


def test_unsupported_lerobot_override_allows_access(monkeypatch):
    pytest.importorskip("lerobot")
    monkeypatch.setattr(compatibility_module, "get_installed_lerobot_version", lambda: "0.5.0")
    try:
        set_allow_unsupported(True)
        assert adapter.DATA_DIR == "data"
    finally:
        set_allow_unsupported(False)


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("DATA_DIR", "data"),
        ("EPISODES_DIR", "meta/episodes"),
        ("INFO_PATH", "meta/info.json"),
        ("STATS_PATH", "meta/stats.json"),
        ("DEFAULT_TASKS_PATH", "meta/tasks.parquet"),
    ],
)
def test_schema_constants_match_lerobot(attribute, expected):
    pytest.importorskip("lerobot")
    assert getattr(adapter, attribute) == expected


def test_loader_accessors_reach_the_real_lerobot_io():
    pytest.importorskip("lerobot")
    assert adapter.codebase_version().startswith("v")
    with pytest.raises(FileNotFoundError):
        adapter.load_info(Path("/definitely/not/a/lerobot/dataset"))


def test_lerobot_git_commit_is_commit_hash_or_none():
    commit = adapter.lerobot_git_commit()
    if commit is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", commit)
