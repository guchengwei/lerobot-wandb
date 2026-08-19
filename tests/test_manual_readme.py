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
"""Public documentation contract for the LeRobot companion manual."""

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
    for asset in ("assets/wandb-workflow-overview-en.svg", "assets/wandb-workflow-overview-ja.svg"):
        assert (REPO_ROOT / asset).is_file(), asset
    assert not (REPO_ROOT / "assets/wandb-workflow-overview-en.jpg").exists()
    assert not (REPO_ROOT / "assets/wandb-workflow-overview-ja.jpg").exists()


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
        "W&B companion integration",
        "generic plugin contract for this integration",
        "not presented as a native plugin",
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


def test_user_docs_frame_lerobot_wandb_as_a_companion_alongside_upstream_lerobot():
    english = (REPO_ROOT / "README.md").read_text() + (REPO_ROOT / "MANUAL.md").read_text()
    japanese = (REPO_ROOT / "MANUAL.ja.md").read_text()
    assert "LeRobot W&B companion integration" in english
    assert "alongside an existing upstream LeRobot" in english
    assert "generic plugin contract for this integration" in english
    assert "upstream LeRobot と同じ environment" in japanese
    assert "generic plugin contract" in japanese
    assert "lerobot-record" in english
    assert "lerobot-record" in japanese
    for text in (english, japanese):
        assert "standalone" not in text.lower()


def test_readme_train_example_uses_the_upstream_dataset_root_and_checkpoint_layout():
    readme = (REPO_ROOT / "README.md").read_text()
    for marker in (
        "--dataset.repo_id=local/pick-cube",
        "--dataset.root=./datasets/pick-cube",
        "--root ./outputs/train/act_pick_cube/checkpoints/last/pretrained_model",
    ):
        assert marker in readme, marker


def test_manual_model_validation_describes_structural_checks_only():
    english = " ".join((REPO_ROOT / "MANUAL.md").read_text().split())
    japanese = " ".join((REPO_ROOT / "MANUAL.ja.md").read_text().split())
    for marker in (
        "structural validation",
        "expected configuration and weight files",
        "does not load or execute the weights",
        "model-specific validation before rollout",
    ):
        assert marker in english, marker
    for marker in (
        "構造検証",
        "config file と weight file",
        "weight を load/execute しません",
        "rollout 前に model-specific validation",
    ):
        assert marker in japanese, marker
    for path, false_markers in (
        (
            "MANUAL.md",
            ("loadable policy directory", "checks that the checkpoint can be loaded"),
        ),
        (
            "MANUAL.ja.md",
            ("load 可能な local policy directory", "checkpoint を load できることを確認"),
        ),
    ):
        text = " ".join((REPO_ROOT / path).read_text().split())
        for marker in false_markers:
            assert marker not in text, f"{marker!r} remains in {path}"


def test_workflow_assets_show_only_the_portable_boundary():
    required_markers = (
        "W&amp;B dataset Artifact",
        "dataset download/materialize",
        "local dataset tree",
        "upstream lerobot-train --dataset.root",
        "local trained model",
        "model upload/promote",
        "No automatic training lifecycle",
    )
    forbidden_markers = ("Auto-Upload", "W&amp;B SDK", "automatic training result", "all data saved")
    for name in ("wandb-workflow-overview-en.svg", "wandb-workflow-overview-ja.svg"):
        text = (REPO_ROOT / "assets" / name).read_text()
        for marker in required_markers:
            assert marker in text, f"{marker!r} missing from {name}"
        for marker in forbidden_markers:
            assert marker not in text, f"{marker!r} remains in {name}"
