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
"""Presence and version validation for the installed LeRobot distribution.

The base ``lerobot-wandb`` distribution deliberately does not hard-depend on ``lerobot``
(see the comment in the repository ``pyproject.toml``): it is a companion
installed into an environment that may already contain LeRobot. LeRobot-dependent
commands therefore validate at runtime instead of at install time, failing fast with a
concise actionable message when LeRobot is absent or outside the supported range.

Version checks are a fast guard, not a substitute for structural dataset validation:
a custom fork can carry the same version number yet differ in schema, so the existing
directory/schema checks remain authoritative.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet

# The LeRobot release range this companion release is tested against. Must stay in sync
# with the ``lerobot`` extra in the repository ``pyproject.toml``.
SUPPORTED_LEROBOT_RANGE = ">=0.6.1,<0.6.2"

_SPECIFIERS = SpecifierSet(SUPPORTED_LEROBOT_RANGE)

# Opt-in escape hatch for experimental use of a custom fork whose version falls outside
# the supported range. Never enables running without LeRobot at all.
_allow_unsupported = False


class LeRobotCompatibilityError(RuntimeError):
    """LeRobot is absent or outside the range this companion release supports."""


def get_installed_lerobot_version() -> str | None:
    """Return the installed ``lerobot`` distribution version, or ``None`` if absent."""
    try:
        return version("lerobot")
    except PackageNotFoundError:
        return None


def check_lerobot_compatible(
    *,
    installed_version: str | None = None,
    allow_unsupported: bool | None = None,
) -> str:
    """Validate the installed LeRobot against :data:`SUPPORTED_LEROBOT_RANGE`.

    Args:
        installed_version: Version to judge; ``None`` detects it from installed metadata.
        allow_unsupported: Skip the range check for an installed (but unsupported)
            version. ``None`` (the default) consults the global override set by
            :func:`set_allow_unsupported`; pass an explicit bool to force a choice
            regardless of that state. Absent LeRobot is always fatal — the override
            never substitutes for an install.

    Returns:
        The judged version string.

    Raises:
        LeRobotCompatibilityError: LeRobot is absent, or its version is unsupported and
            the (parameter or global) override is not enabled.
    """
    if installed_version is None:
        installed_version = get_installed_lerobot_version()

    if installed_version is None:
        raise LeRobotCompatibilityError(
            "LeRobot is required by this lerobot-wandb command but is not installed.\n"
            "Install a compatible release with: pip install 'lerobot-wandb[lerobot]' "
            "(or install the lerobot package your environment already uses)."
        )

    if allow_unsupported is None:
        allow_unsupported = get_allow_unsupported()

    if installed_version not in _SPECIFIERS and not allow_unsupported:
        raise LeRobotCompatibilityError(
            f"Unsupported LeRobot version {installed_version}.\n"
            f"This lerobot-wandb release supports {SUPPORTED_LEROBOT_RANGE}.\n"
            "Install a compatible lerobot-wandb release or use the documented experimental "
            "override: --allow-unsupported-lerobot"
        )

    return installed_version


def set_allow_unsupported(allow: bool) -> None:
    """Enable or disable the global unsupported-version override.

    Set once at command startup from the ``--allow-unsupported-lerobot`` flag; read by
    the adapter when it lazily touches LeRobot. Opt-in only: the default is always
    ``False``.
    """
    global _allow_unsupported
    _allow_unsupported = allow


def get_allow_unsupported() -> bool:
    """Whether the unsupported-version override is currently enabled."""
    return _allow_unsupported
