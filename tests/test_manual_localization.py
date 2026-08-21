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
"""English/Japanese README parity checks for executable examples and assets."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def _fenced_command_blocks(path: Path) -> list[str]:
    text = path.read_text()
    return [block.strip() for block in re.findall(r"```(?:bash|shell)\n(.*?)```", text, re.DOTALL)]


def test_readmes_keep_the_same_executable_command_blocks():
    english = _fenced_command_blocks(REPO_ROOT / "README.md")
    japanese = _fenced_command_blocks(REPO_ROOT / "README.ja.md")
    assert english
    assert japanese == english


def test_readmes_keep_language_specific_assets_and_cross_links():
    english = (REPO_ROOT / "README.md").read_text()
    japanese = (REPO_ROOT / "README.ja.md").read_text()
    assert "assets/wandb-workflow-overview-en.jpg" in english
    assert "assets/wandb-workflow-overview-ja.jpg" in japanese
    assert "README.ja.md" in english
    assert "README.md" in japanese
    assert "日本語" in english
    assert "English" in japanese


def test_japanese_training_boundary_matches_the_upstream_companion_scope():
    japanese = " ".join((REPO_ROOT / "README.ja.md").read_text().split())
    assert "### 4. LeRobot で学習する" in japanese
    assert "`lerobot-wandb` は `lerobot-train` をラップせず" in japanese
    assert "学習完了時にモデルを自動公開することもありません" in japanese
    assert "ロボットの制御ループには入りません" in japanese
