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
"""Public documentation contract for the standalone companion manual."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
DOCS = (REPO_ROOT / "README.md", REPO_ROOT / "MANUAL.md", REPO_ROOT / "MANUAL.ja.md")
STALE_FORK_MARKERS = (
    "packages/lerobot-wandb",
    "--dataset.artifact_ref",
    "--wandb.model_artifact_name",
    "--wandb.registered_model_name",
    "uv sync",
    "guchengwei/lerobot.git",
)


def test_manual_files_and_workflow_assets_exist():
    assert (REPO_ROOT / "MANUAL.md").is_file()
    assert (REPO_ROOT / "MANUAL.ja.md").is_file()
    for image in ("assets/wandb-workflow-overview-en.jpg", "assets/wandb-workflow-overview-ja.jpg"):
        assert (REPO_ROOT / image).is_file(), image


def test_readme_navigates_to_both_manuals():
    readme = (REPO_ROOT / "README.md").read_text()
    assert re.search(r"\[Manual\]\(\./MANUAL\.md\)", readme)
    assert re.search(r"\[Manual \(日本語\)\]\(\./MANUAL\.ja\.md\)", readme)
    assert "PyPI" in readme and "future" in readme.lower()


def test_manual_images_use_existing_relative_assets_and_alt_text():
    for path in (REPO_ROOT / "MANUAL.md", REPO_ROOT / "MANUAL.ja.md"):
        text = path.read_text()
        links = re.findall(r"!\[([^]]+)\]\(([^)]+)\)", text)
        assert links, path
        for alt, target in links:
            assert alt.strip(), path
            assert not target.startswith(("http://", "https://")), target
            assert (path.parent / target).is_file(), target


def test_public_docs_do_not_recommend_fork_only_paths_or_flags():
    for path in DOCS:
        text = path.read_text()
        for marker in STALE_FORK_MARKERS:
            assert marker not in text, f"{marker!r} remains in {path.name}"


def test_manual_documents_the_portable_command_route():
    english = (REPO_ROOT / "MANUAL.md").read_text()
    required_markers = (
        "companion distribution, not a native LeRobot plugin",
        ">=0.6.1,<0.6.2",
        'pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"',
        "lerobot-wandb dataset download",
        "--root ./datasets/pick-cube",
        "lerobot-train",
        "--dataset.root=./datasets/pick-cube",
        "lerobot-wandb model upload",
        "lerobot-wandb model promote",
    )
    for marker in required_markers:
        assert marker in english, marker
