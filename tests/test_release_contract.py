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
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
_TAG_CHECK = importlib.util.spec_from_file_location(
    "check_release_tag", REPO_ROOT / "scripts" / "check_release_tag.py"
)
assert _TAG_CHECK is not None and _TAG_CHECK.loader is not None
check_release_tag = importlib.util.module_from_spec(_TAG_CHECK)
_TAG_CHECK.loader.exec_module(check_release_tag)


def test_release_tag_matches_pyproject_version(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "example"\nversion = "0.1.0"\n')

    assert check_release_tag.validate_tag_version("v0.1.0", pyproject) == "0.1.0"


def test_release_tag_mismatch_is_rejected(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "example"\nversion = "0.1.0"\n')

    with pytest.raises(ValueError, match="does not match pyproject.toml version"):
        check_release_tag.validate_tag_version("v0.1.1", pyproject)


def test_ci_runs_a_built_wheel_with_the_lerobot_extra():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "scripts/fresh_extra_smoke.sh" in workflow
    assert "dist/*.whl" in workflow
    assert "[lerobot]" in workflow


def test_release_validates_artifacts_and_waits_for_all_gates():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()

    assert "scripts/check_release_tag.py" in workflow
    assert ".venv/bin/python -m build --outdir dist" in workflow
    assert "dist/*.whl" in workflow
    assert "dist/*.tar.gz" in workflow
    assert "needs: [build, compatibility, coexistence]" in workflow
    assert "vars.ENABLE_PYPI_PUBLISH == 'true'" in workflow
