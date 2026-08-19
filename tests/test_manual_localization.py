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
"""English/Japanese manual parity checks for executable examples and assets."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def _fenced_command_blocks(path: Path) -> list[str]:
    text = path.read_text()
    return [block.strip() for block in re.findall(r"```(?:bash|shell)\n(.*?)```", text, re.DOTALL)]


def test_manuals_keep_the_same_executable_command_blocks():
    english = _fenced_command_blocks(REPO_ROOT / "MANUAL.md")
    japanese = _fenced_command_blocks(REPO_ROOT / "MANUAL.ja.md")
    assert english
    assert japanese == english


def test_manuals_keep_language_specific_assets_and_cross_links():
    english = (REPO_ROOT / "MANUAL.md").read_text()
    japanese = (REPO_ROOT / "MANUAL.ja.md").read_text()
    assert "assets/wandb-workflow-overview-en.jpg" in english
    assert "assets/wandb-workflow-overview-ja.jpg" in japanese
    assert "MANUAL.ja.md" in english
    assert "MANUAL.md" in japanese
    assert "日本語" in japanese
