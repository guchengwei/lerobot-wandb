#!/usr/bin/env python3
"""Check the standalone distribution's wheel and sdist ownership contract."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        package_names = sorted(name for name in names if name.startswith("lerobot_wandb/"))
        forbidden = [name for name in names if name.startswith("lerobot/")]
        if forbidden:
            raise SystemExit(f"wheel installs forbidden LeRobot files: {forbidden[:5]}")
        if not package_names:
            raise SystemExit("wheel does not contain the lerobot_wandb package")

        entry_point_files = [name for name in names if name.endswith("/entry_points.txt")]
        if len(entry_point_files) != 1:
            raise SystemExit(f"expected one entry_points.txt, found {entry_point_files}")
        entry_points = archive.read(entry_point_files[0]).decode()
        console_scripts = [
            line.strip() for line in entry_points.splitlines() if line.strip() and not line.startswith("[")
        ]
        expected = ["lerobot-wandb = lerobot_wandb.cli:main"]
        if console_scripts != expected:
            raise SystemExit(f"unexpected console scripts: {console_scripts!r}")


def check_sdist(path: Path) -> None:
    with tarfile.open(path) as archive:
        names = archive.getnames()
        # Tar member names use POSIX separators even when an archive is inspected on Windows.
        # Check path components rather than a slash-delimited substring so a root-level
        # ``lerobot/...`` member cannot evade the ownership contract.
        forbidden = [name for name in names if "lerobot" in PurePosixPath(name).parts]
        if forbidden:
            raise SystemExit(f"sdist contains forbidden LeRobot files: {forbidden[:5]}")
        if not any(name.endswith("/src/lerobot_wandb/__init__.py") for name in names):
            raise SystemExit("sdist does not contain src/lerobot_wandb/__init__.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for artifact in args.artifacts:
        if artifact.suffix == ".whl":
            check_wheel(artifact)
        elif artifact.name.endswith(".tar.gz"):
            check_sdist(artifact)
        else:
            raise SystemExit(f"unsupported build artifact: {artifact}")
    print("package ownership and entry-point checks passed")


if __name__ == "__main__":
    main()
