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

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

_CHECK_WHEEL = importlib.util.spec_from_file_location(
    "check_wheel", Path(__file__).parents[1] / "scripts" / "check_wheel.py"
)
assert _CHECK_WHEEL is not None and _CHECK_WHEEL.loader is not None
check_wheel = importlib.util.module_from_spec(_CHECK_WHEEL)
_CHECK_WHEEL.loader.exec_module(check_wheel)


def _write_sdist(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            data = b""
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))


def test_sdist_rejects_a_root_level_lerobot_namespace(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    _write_sdist(
        archive,
        [
            "lerobot_wandb-0.1.0/src/lerobot_wandb/__init__.py",
            "lerobot/bad.py",
        ],
    )

    with pytest.raises(SystemExit, match="forbidden LeRobot files"):
        check_wheel.check_sdist(archive)
