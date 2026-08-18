#!/usr/bin/env bash
# Standalone coexistence and uninstall smoke for the lerobot-wandb wheel.
#
# Requires Python 3.12+, pip, and network access to the supported upstream LeRobot
# release. No W&B credentials are needed: this script uses --help and local tests.
# Set PYTHON_BIN explicitly when the host's python3 is older than 3.12.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' \
  || { echo "coexistence smoke requires Python 3.12+ (set PYTHON_BIN)" >&2; exit 2; }

echo "==> Creating isolated build environment"
"$PYTHON_BIN" -m venv "$WORKDIR/build-env"
BUILD_PYTHON="$WORKDIR/build-env/bin/python"
"$BUILD_PYTHON" -m pip install --disable-pip-version-check -q build==1.3.0

echo "==> Building the companion wheel and sdist"
(
  cd "$REPO_ROOT"
  env -u PYTHONPATH "$BUILD_PYTHON" -m build --wheel --sdist --outdir "$WORKDIR/dist" >/dev/null
)
WHEEL="$(find "$WORKDIR/dist" -maxdepth 1 -name 'lerobot_wandb-*.whl' -print -quit)"
test -n "$WHEEL"

echo "==> Installing upstream LeRobot ${LEROBOT_VERSION} first"
"$PYTHON_BIN" -m venv "$WORKDIR/env"
PIP="$WORKDIR/env/bin/pip"
PYTHON="$WORKDIR/env/bin/python"
export PATH="$WORKDIR/env/bin:$PATH"

"$PIP" install --disable-pip-version-check -q "lerobot==${LEROBOT_VERSION}"
BEFORE_VERSION="$("$PYTHON" -c 'import importlib.metadata as m; print(m.version("lerobot"))')"
BEFORE_LOCATION="$("$PYTHON" -c 'import importlib.metadata as m; print(m.distribution("lerobot")._path)')"

snapshot_files() {
  "$PYTHON" - "$1" <<'PY'
import importlib.metadata as metadata
import sys

distribution = metadata.distribution(sys.argv[1])
for path in sorted(distribution.files or (), key=str):
    print(path)
PY
}

snapshot_commands() {
  "$PYTHON" - <<'PY'
import importlib.metadata as metadata
import shutil

distribution = metadata.distribution("lerobot")
for entry_point in sorted(distribution.entry_points, key=lambda item: item.name):
    if entry_point.group == "console_scripts":
        print(f"{entry_point.name}\t{shutil.which(entry_point.name) or ''}")
PY
}

snapshot_files lerobot >"$WORKDIR/lerobot-before.files"
snapshot_commands >"$WORKDIR/lerobot-before.commands"

echo "==> Installing companion into the existing environment"
"$PIP" install --disable-pip-version-check -q "${WHEEL}[test]"
AFTER_VERSION="$("$PYTHON" -c 'import importlib.metadata as m; print(m.version("lerobot"))')"
AFTER_LOCATION="$("$PYTHON" -c 'import importlib.metadata as m; print(m.distribution("lerobot")._path)')"
test "$BEFORE_VERSION" = "$AFTER_VERSION"
test "$BEFORE_LOCATION" = "$AFTER_LOCATION"
snapshot_files lerobot >"$WORKDIR/lerobot-after-install.files"
snapshot_commands >"$WORKDIR/lerobot-after-install.commands"
cmp -s "$WORKDIR/lerobot-before.files" "$WORKDIR/lerobot-after-install.files"
cmp -s "$WORKDIR/lerobot-before.commands" "$WORKDIR/lerobot-after-install.commands"
echo "    LeRobot unchanged by companion install: $AFTER_VERSION"

"$PYTHON" -m pip check
"$WORKDIR/env/bin/lerobot-train" --help >/dev/null
"$WORKDIR/env/bin/lerobot-wandb" --help >/dev/null
(
  cd "$WORKDIR"
  env -u PYTHONPATH "$PYTHON" -m pytest -q "$REPO_ROOT/tests" >/dev/null
)
echo "    pip check, upstream CLI, companion CLI, package tests: OK"

echo "==> Checking wheel ownership and uninstall isolation"
snapshot_files lerobot-wandb >"$WORKDIR/wandb-before.files"

"$PIP" uninstall --disable-pip-version-check -q -y lerobot-wandb
test ! -e "$WORKDIR/env/bin/lerobot-wandb"
AFTER_COMPANION_UNINSTALL_VERSION="$("$PYTHON" -c 'import importlib.metadata as m; print(m.version("lerobot"))')"
AFTER_COMPANION_UNINSTALL_LOCATION="$("$PYTHON" -c 'import importlib.metadata as m; print(m.distribution("lerobot")._path)')"
test "$BEFORE_VERSION" = "$AFTER_COMPANION_UNINSTALL_VERSION"
test "$BEFORE_LOCATION" = "$AFTER_COMPANION_UNINSTALL_LOCATION"
snapshot_files lerobot >"$WORKDIR/lerobot-after-companion-uninstall.files"
snapshot_commands >"$WORKDIR/lerobot-after-companion-uninstall.commands"
cmp -s "$WORKDIR/lerobot-before.files" "$WORKDIR/lerobot-after-companion-uninstall.files"
cmp -s "$WORKDIR/lerobot-before.commands" "$WORKDIR/lerobot-after-companion-uninstall.commands"
echo "    LeRobot remains after companion uninstall: $AFTER_COMPANION_UNINSTALL_VERSION"

"$PIP" install --disable-pip-version-check -q --no-deps "$WHEEL"
"$PIP" uninstall --disable-pip-version-check -q -y lerobot

"$PYTHON" - "$WORKDIR/lerobot-before.files" "$WORKDIR/wandb-before.files" <<'PY'
from pathlib import Path
import site
import sys

site_packages = Path(site.getsitepackages()[0])

def paths(snapshot):
    return [(site_packages / line.strip()).resolve() for line in Path(snapshot).read_text().splitlines()]

lerobot_remaining = [str(path) for path in paths(sys.argv[1]) if path.exists()]
if lerobot_remaining:
    raise SystemExit("LeRobot files survived LeRobot uninstall: " + ", ".join(lerobot_remaining[:5]))

companion_missing = [str(path) for path in paths(sys.argv[2]) if not path.exists()]
if companion_missing:
    raise SystemExit("companion files deleted by LeRobot uninstall: " + ", ".join(companion_missing[:5]))
print("    LeRobot files removed and companion files remain")
PY

if [[ -e "$WORKDIR/env/bin/lerobot-train" ]]; then
  echo "LeRobot CLI survived its uninstall" >&2
  exit 1
fi

mkdir -p "$WORKDIR/fake-dataset"
set +e
OUTPUT="$("$WORKDIR/env/bin/lerobot-wandb" dataset upload \
  --root "$WORKDIR/fake-dataset" --project smoke --name smoke 2>&1)"
RC=$?
set -e
test "$RC" -ne 0
echo "$OUTPUT" | grep -q "not installed"
echo "    missing-LeRobot command has an actionable error (rc=$RC)"

echo "==> All standalone coexistence checks passed"
