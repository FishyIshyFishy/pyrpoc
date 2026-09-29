"""Icons shipped in ``pyrpoc/assets/``, looked up by name."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QIcon

# ``pyrpoc/assets``, two levels up from ``pyrpoc/panels/components``.
ASSETS = Path(__file__).resolve().parents[2] / "assets"


def asset_icon(name: str) -> QIcon:
    """The icon ``assets/<name>.svg`` (or ``.png``). A missing file is a
    packaging bug, so it raises rather than falling back to text."""
    for suffix in (".svg", ".png"):
        path = ASSETS / f"{name}{suffix}"
        if path.is_file():
            return QIcon(str(path))
    raise FileNotFoundError(f"no icon named {name!r} in {ASSETS}")
