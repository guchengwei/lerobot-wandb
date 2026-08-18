#!/usr/bin/env python3
"""Validate that a release tag names the version in ``pyproject.toml``."""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path


def validate_tag_version(tag: str, pyproject: Path = Path("pyproject.toml")) -> str:
    """Return the project version when ``tag`` is the matching ``v<version>`` tag."""

    tag_name = tag.removeprefix("refs/tags/")
    if not tag_name.startswith("v"):
        raise ValueError(f"release tag {tag!r} must use the v<version> form")

    with pyproject.open("rb") as stream:
        document = tomllib.load(stream)
    try:
        project_version = document["project"]["version"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{pyproject} does not define project.version") from error
    if not isinstance(project_version, str) or not project_version:
        raise ValueError(f"{pyproject} project.version must be a non-empty string")

    tag_version = tag_name.removeprefix("v")
    if tag_version != project_version:
        raise ValueError(
            f"release tag {tag_name!r} does not match pyproject.toml version {project_version!r}"
        )
    return project_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tag",
        nargs="?",
        default=os.environ.get("GITHUB_REF_NAME"),
        help="release tag (defaults to GITHUB_REF_NAME)",
    )
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()
    if args.tag is None:
        parser.error("a release tag is required (or set GITHUB_REF_NAME)")

    try:
        version = validate_tag_version(args.tag, args.pyproject)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"release tag {args.tag!r} matches project version {version}")


if __name__ == "__main__":
    main()
