"""Entry point: build the controller and show the main window."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_qt_fontdir() -> None:
    """On Windows, point Qt at the system fonts so text renders (matches old GUI)."""
    if os.name != "nt" or os.environ.get("QT_QPA_FONTDIR"):
        return
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if fonts.is_dir():
        os.environ["QT_QPA_FONTDIR"] = str(fonts)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    configure_qt_fontdir()
    app = QApplication(sys.argv)
    try:
        from pyrpoc_next.gui.styles.theme import apply_dark_theme

        apply_dark_theme(app)
    except Exception:
        pass

    from pyrpoc_next.core import Controller
    from pyrpoc_next.gui import MainWindow

    window = MainWindow(Controller())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
