"""Package version, read from installed metadata so it cannot drift from the wheel."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lerobot-wandb")
except PackageNotFoundError:
    __version__ = "0.1.0"
