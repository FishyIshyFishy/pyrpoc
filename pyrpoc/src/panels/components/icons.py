"""Icons shipped in ``pyrpoc/assets/``, looked up by name."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QIcon

# ``pyrpoc/assets``, beside ``pyrpoc/src``.
ASSETS = Path(__file__).resolve().parents[3] / "assets"


def asset_icon(name: str) -> QIcon | None:
    """The icon ``assets/<name>.svg`` (or ``.png``), or None if there is none.

    None rather than an empty icon, so a caller can fall back to text: a
    button with a blank face is worse than one with a word on it.
    """
    for suffix in (".svg", ".png"):
        path = ASSETS / f"{name}{suffix}"
        if path.is_file():
            return QIcon(str(path))
    return None
