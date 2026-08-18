#!/usr/bin/env bash
# Verify a built wheel's fresh-environment ``[lerobot]`` installation contract.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PATH_TO_WHEEL" >&2
  exit 2
fi

WHEEL="$1"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPECTED_LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"

if [[ ! -f "$WHEEL" ]]; then
  echo "wheel does not exist: $WHEEL" >&2
  exit 2
fi

"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' \
  || { echo "fresh extra smoke requires Python 3.12+ (set PYTHON_BIN)" >&2; exit 2; }

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> Installing the built wheel with [lerobot] in a fresh environment"
"$PYTHON_BIN" -m venv "$WORKDIR/env"
PIP="$WORKDIR/env/bin/pip"
PYTHON="$WORKDIR/env/bin/python"

"$PIP" install --disable-pip-version-check -q "${WHEEL}[lerobot]"
"$PYTHON" -m pip check

(
  cd "$WORKDIR"
  env -u PYTHONPATH EXPECTED_LEROBOT_VERSION="$EXPECTED_LEROBOT_VERSION" "$PYTHON" - <<'PY'
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path

expected_lerobot = os.environ["EXPECTED_LEROBOT_VERSION"]
for distribution_name, module_name, expected_version in (
    ("lerobot", "lerobot", expected_lerobot),
    ("lerobot-wandb", "lerobot_wandb", None),
):
    distribution = metadata.distribution(distribution_name)
    version = distribution.version
    if expected_version is not None and version != expected_version:
        raise SystemExit(
            f"{distribution_name} resolved to {version}, expected {expected_version}"
        )
    location = Path(distribution.locate_file(""))
    if not location.is_dir():
        raise SystemExit(f"{distribution_name} location does not exist: {location}")
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    if location.resolve() not in module_path.parents:
        raise SystemExit(f"{module_name} imported from outside {location}: {module_path}")
    print(f"    {distribution_name} {version}: {location}")
PY
)

env -u PYTHONPATH "$WORKDIR/env/bin/lerobot-train" --help >/dev/null
env -u PYTHONPATH "$WORKDIR/env/bin/lerobot-wandb" --help >/dev/null
echo "    LeRobot ${EXPECTED_LEROBOT_VERSION}, both imports/locations, both CLIs, pip check: OK"
