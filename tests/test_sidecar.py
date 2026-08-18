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

import json
import subprocess
import sys
import textwrap

from lerobot_wandb.sidecar import ArtifactSidecar, read_sidecar, write_sidecar


def test_write_then_read_roundtrips(tmp_path):
    sidecar = ArtifactSidecar(
        requested_ref="my-team/my-project/pick-cube:latest",
        resolved_ref="my-team/my-project/pick-cube:v3",
        version="v3",
        digest="abc123",
    )
    write_sidecar(tmp_path, sidecar)
    assert read_sidecar(tmp_path) == sidecar


def test_read_absent_sidecar_returns_none(tmp_path):
    assert read_sidecar(tmp_path) is None


def test_read_unparsable_sidecar_returns_none(tmp_path):
    (tmp_path / ".wandb_artifact.json").write_text("{not json")
    assert read_sidecar(tmp_path) is None


def test_sidecar_json_shape_is_identity_record(tmp_path):
    write_sidecar(
        tmp_path,
        ArtifactSidecar(requested_ref="a/b/c:latest", resolved_ref="a/b/c:v1", version="v1", digest="d"),
    )
    data = json.loads((tmp_path / ".wandb_artifact.json").read_text())
    assert data == {
        "requested_ref": "a/b/c:latest",
        "resolved_ref": "a/b/c:v1",
        "version": "v1",
        "digest": "d",
    }


def test_sidecar_imports_without_wandb_or_lerobot(tmp_path):
    preamble = textwrap.dedent(
        """
        import builtins
        blocked = ("wandb", "lerobot")
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
                from lerobot_wandb.sidecar import ArtifactSidecar, read_sidecar, write_sidecar
                write_sidecar(".", ArtifactSidecar("a/b/c:latest", "a/b/c:v1", "v1", "d"))
                assert read_sidecar(".") == ArtifactSidecar("a/b/c:latest", "a/b/c:v1", "v1", "d")
                """
            ),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
