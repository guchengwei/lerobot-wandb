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

"""LeRobot presence/version validation for the companion distribution.

The base ``lerobot-wandb`` distribution deliberately has no hard ``lerobot``
dependency, so commands that need LeRobot structures validate at runtime
instead: absent LeRobot, and unsupported installed versions, must fail with a
concise actionable message — never a traceback and never silent continuation.
"""

from importlib.metadata import PackageNotFoundError

import pytest

from lerobot_wandb import compatibility as compatibility_module
from lerobot_wandb.compatibility import (
    SUPPORTED_LEROBOT_RANGE,
    LeRobotCompatibilityError,
    check_lerobot_compatible,
    get_allow_unsupported,
    get_installed_lerobot_version,
    set_allow_unsupported,
)


def test_supported_version_passes():
    assert check_lerobot_compatible(installed_version="0.6.1") == "0.6.1"


def test_missing_lerobot_raises_install_guidance(monkeypatch):
    monkeypatch.setattr(compatibility_module, "get_installed_lerobot_version", lambda: None)
    with pytest.raises(LeRobotCompatibilityError) as excinfo:
        check_lerobot_compatible()
    message = str(excinfo.value)
    assert "not installed" in message
    assert "pip install" in message


def test_unsupported_version_raises_with_supported_range():
    with pytest.raises(LeRobotCompatibilityError) as excinfo:
        check_lerobot_compatible(installed_version="0.5.0")
    message = str(excinfo.value)
    assert "Unsupported LeRobot version 0.5.0" in message
    assert SUPPORTED_LEROBOT_RANGE in message
    assert "--allow-unsupported-lerobot" in message


def test_next_release_is_unsupported_until_validated():
    with pytest.raises(LeRobotCompatibilityError):
        check_lerobot_compatible(installed_version="0.6.2")


def test_unsupported_version_allowed_with_override():
    assert check_lerobot_compatible(installed_version="0.5.0", allow_unsupported=True) == "0.5.0"


def test_missing_lerobot_is_always_fatal_even_with_override(monkeypatch):
    monkeypatch.setattr(compatibility_module, "get_installed_lerobot_version", lambda: None)
    with pytest.raises(LeRobotCompatibilityError):
        check_lerobot_compatible(allow_unsupported=True)


def test_detection_reads_installed_metadata(monkeypatch):
    def fake_version(name):
        assert name == "lerobot"
        return "0.6.1"

    monkeypatch.setattr(compatibility_module, "version", fake_version)
    assert get_installed_lerobot_version() == "0.6.1"
    assert check_lerobot_compatible() == "0.6.1"


def test_detection_returns_none_when_lerobot_absent(monkeypatch):
    def fake_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("lerobot_wandb.compatibility.version", fake_version)
    assert get_installed_lerobot_version() is None
    with pytest.raises(LeRobotCompatibilityError):
        check_lerobot_compatible()


def test_global_override_flag_roundtrip():
    try:
        assert get_allow_unsupported() is False
        set_allow_unsupported(True)
        assert get_allow_unsupported() is True
        assert check_lerobot_compatible(installed_version="0.5.0") == "0.5.0"
    finally:
        set_allow_unsupported(False)
    assert get_allow_unsupported() is False
    with pytest.raises(LeRobotCompatibilityError):
        check_lerobot_compatible(installed_version="0.5.0")


def test_global_override_does_not_affect_missing_lerobot(monkeypatch):
    monkeypatch.setattr(compatibility_module, "get_installed_lerobot_version", lambda: None)
    try:
        set_allow_unsupported(True)
        with pytest.raises(LeRobotCompatibilityError):
            check_lerobot_compatible()
    finally:
        set_allow_unsupported(False)
