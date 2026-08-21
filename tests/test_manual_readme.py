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
"""Public documentation contract for the LeRobot companion READMEs."""

import re
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
PRIMARY_DOCS = (REPO_ROOT / "README.md", REPO_ROOT / "README.ja.md")
LEGACY_DOCS = (REPO_ROOT / "MANUAL.md", REPO_ROOT / "MANUAL.ja.md")
DOCS = (*PRIMARY_DOCS, *LEGACY_DOCS)
STALE_FORK_MARKERS = (
    "packages/lerobot-wandb",
    "--dataset.artifact_ref",
    "--wandb.model_artifact_name",
    "--wandb.registered_model_name",
    "uv sync",
    "guchengwei/lerobot.git",
)


def test_readme_files_and_workflow_assets_exist():
    for path in (*PRIMARY_DOCS, *LEGACY_DOCS):
        assert path.is_file(), path
    for asset in (
        "assets/wandb-workflow-overview-en.jpg",
        "assets/wandb-workflow-overview-ja.jpg",
    ):
        assert (REPO_ROOT / asset).is_file(), asset


def test_readmes_cross_link_and_legacy_manuals_redirect():
    english = (REPO_ROOT / "README.md").read_text()
    japanese = (REPO_ROOT / "README.ja.md").read_text()
    manual = (REPO_ROOT / "MANUAL.md").read_text()
    manual_ja = (REPO_ROOT / "MANUAL.ja.md").read_text()

    assert re.search(r"\[日本語\]\(\./README\.ja\.md\)", english)
    assert re.search(r"\[English\]\(\./README\.md\)", japanese)
    assert re.search(r"\[README\]\(\./README\.md\)", manual)
    assert re.search(r"\[README\]\(\./README\.ja\.md\)", manual_ja)
    assert "PyPI" in english and "not been published" in english


def test_readme_images_use_existing_relative_assets_and_alt_text():
    for path in PRIMARY_DOCS:
        text = path.read_text()
        links = re.findall(r"!\[([^]]+)\]\(([^)]+)\)", text)
        assert links, path
        for alt, target in links:
            assert alt.strip(), path
            assert not target.startswith(("http://", "https://")), target
            assert (path.parent / target).is_file(), target


def test_docs_and_manifest_do_not_reference_companion_svg_assets():
    for path in (*DOCS, REPO_ROOT / "MANIFEST.in"):
        text = path.read_text()
        assert ".svg" not in text, f"SVG reference remains in {path.name}"


def test_public_docs_do_not_recommend_fork_only_paths_or_flags():
    for path in DOCS:
        text = path.read_text()
        for marker in STALE_FORK_MARKERS:
            assert marker not in text, f"{marker!r} remains in {path.name}"


def test_english_readme_documents_the_companion_command_route():
    english = (REPO_ROOT / "README.md").read_text()
    required_markers = (
        "companion CLI for an existing LeRobot installation",
        "not a native LeRobot plugin",
        ">=0.6.1,<0.6.2",
        "uv venv --python 3.12",
        'uv pip install "lerobot[core_scripts,training,feetech]==0.6.1"',
        'pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"',
        "lerobot-wandb dataset download",
        '--root "$TRAIN_DATASET_ROOT"',
        "lerobot-train",
        '--dataset.root="$TRAIN_DATASET_ROOT"',
        "lerobot-wandb model upload",
        "lerobot-wandb model promote",
    )
    for marker in required_markers:
        assert marker in english, marker


def test_readmes_document_safe_companion_uninstall():
    cases = (
        (
            REPO_ROOT / "README.md",
            "Uninstall",
            (
                "LeRobot remains installed and is not modified",
                "does not delete local datasets",
                "does not delete remote W&B Artifacts",
                "does not remove W&B authentication or configuration",
            ),
        ),
        (
            REPO_ROOT / "README.ja.md",
            "アンインストール",
            (
                "LeRobot 自体は削除も変更もされません",
                "ローカルのデータセット",
                "W&B 上の Artifact、Run、Registry object も削除しません",
                "W&B の認証情報や設定もそのまま残ります",
            ),
        ),
    )
    uninstall_prefixes = (
        ("uv", "pip", "uninstall"),
        ("pip", "uninstall"),
        ("python", "-m", "pip", "uninstall"),
    )

    for path, heading, required_markers in cases:
        text = path.read_text()
        match = re.search(
            rf"^## {re.escape(heading)}\s*$\n(?P<section>.*?)(?=^## |\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match, f"missing {heading!r} section in {path.name}"
        section = match.group("section")

        assert "uv pip uninstall lerobot-wandb" in section
        for marker in required_markers:
            assert marker in section, (path.name, marker)

        bash_blocks = re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL)
        assert bash_blocks, f"missing bash uninstall example in {path.name}"
        for block in bash_blocks:
            for line in block.splitlines():
                tokens = shlex.split(line)
                for prefix in uninstall_prefixes:
                    if tuple(tokens[: len(prefix)]) != prefix:
                        continue
                    targets = [token for token in tokens[len(prefix) :] if not token.startswith("-")]
                    assert "lerobot" not in targets, (
                        f"{path.name} must not recommend uninstalling LeRobot: {line!r}"
                    )


def test_user_docs_frame_lerobot_wandb_as_a_companion_alongside_upstream_lerobot():
    english = (REPO_ROOT / "README.md").read_text()
    japanese = (REPO_ROOT / "README.ja.md").read_text()
    assert "companion CLI for an existing LeRobot installation" in english
    assert "separate distribution, not a native LeRobot plugin" in english
    assert "既存の LeRobot 環境に追加して使います" in japanese
    assert "LeRobot のネイティブプラグインではありません" in japanese
    assert "lerobot-record" in english
    assert "lerobot-record" in japanese
    for text in (english, japanese):
        assert "standalone" not in text.lower()
        assert "portable" not in text.lower()


def test_readme_train_example_uses_the_upstream_dataset_root_and_checkpoint_layout():
    readme = (REPO_ROOT / "README.md").read_text()
    for marker in (
        '--dataset.repo_id="local/$DATASET_NAME"',
        '--dataset.root="$TRAIN_DATASET_ROOT"',
        '--wandb.enable=true',
        '--wandb.entity="$WANDB_ENTITY"',
        '--wandb.project="$WANDB_PROJECT"',
        '--wandb.disable_artifact=true',
        'export POLICY_ROOT="$TRAIN_OUTPUT/checkpoints/last/pretrained_model"',
        '--root "$POLICY_ROOT"',
    ):
        assert marker in readme, marker


def test_readmes_use_reusable_workflow_names_instead_of_sample_artifact_names():
    for path in PRIMARY_DOCS:
        text = path.read_text()
        for variable in (
            "DATASET_NAME",
            "POLICY_NAME",
            "ROLLOUT_NAME",
            "TRAIN_DATASET_ROOT",
            "POLICY_ROOT",
            "ROLLOUT_ROOT",
        ):
            assert f"${variable}" in text, (path, variable)
        for sample_name in ("pick-cube", "pick_cube", "so101-pick-cube"):
            assert sample_name not in text, (path, sample_name)


def test_readmes_describe_structural_model_checks_only():
    english = " ".join((REPO_ROOT / "README.md").read_text().split())
    japanese = " ".join((REPO_ROOT / "README.ja.md").read_text().split())
    for marker in (
        "expected configuration and weight files",
        "does not load or execute the weights",
        "policy-specific validation separately",
    ):
        assert marker in english, marker
    for marker in (
        "必要な設定ファイルと重みファイル",
        "重みをロードしたり実行したりはしません",
        "ポリシー固有の検証は別途実施",
    ):
        assert marker in japanese, marker
    for text, false_markers in (
        (english, ("loadable policy directory", "checks that the checkpoint can be loaded")),
        (japanese, ("load 可能な local policy directory", "checkpoint を load できることを確認")),
    ):
        for marker in false_markers:
            assert marker not in text, marker


def test_materialized_data_requires_a_completed_artifact_download():
    english = " ".join((REPO_ROOT / "README.md").read_text().split())
    japanese = " ".join((REPO_ROOT / "README.ja.md").read_text().split())
    for marker in (
        "validate local directories before upload and after download",
        "Once the download finishes",
        "reads that local tree directly",
        "W&B is not needed to access the training data",
        "training process still needs W&B connectivity to sync metrics",
    ):
        assert marker in english, marker
    for marker in (
        "アップロード前とダウンロード後にローカルディレクトリを検証",
        "ダウンロード完了後",
        "ローカルディレクトリを直接読み込む",
        "学習データを読むために W&B 接続は必要ありません",
        "学習中も W&B への接続が必要です",
    ):
        assert marker in japanese, marker


def test_overview_and_companion_boundaries_are_explicit():
    english = (REPO_ROOT / "README.md").read_text()
    japanese = (REPO_ROOT / "README.ja.md").read_text()
    for marker in (
        "The diagram shows the full LeRobot and W&B workflow",
        "handles the Artifact steps around recording, training, and rollout",
        "the robot-facing commands remain standard LeRobot commands",
        "What this companion does not do",
        "does not reproduce those hooks",
        "streaming recorder or deployment controller",
    ):
        assert marker in english, marker
    for marker in (
        "LeRobot と W&B を組み合わせた全体の流れ",
        "記録・学習・ロールアウトの前後で Artifact を扱う",
        "ロボットを動かすコマンドは LeRobot",
        "このツールが行わないこと",
        "そのフックは再実装していません",
        "ストリーミングレコーダーやデプロイメントコントローラー",
    ):
        assert marker in japanese, marker
