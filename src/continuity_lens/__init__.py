"""Continuity Lens: a masked-future video continuity research probe."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("continuity-lens")
except PackageNotFoundError:  # pragma: no cover - editable source checkout
    __version__ = "0.1.0"
