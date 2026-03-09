from __future__ import annotations

import os
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

from pyrpoc.gui.styles.theme_manager import ThemeController
from pyrpoc.services.app_controller import AppController


def _configure_qt_fontdir() -> None:
    if os.name != "nt":
        return
    if os.environ.get("QT_QPA_FONTDIR"):
        return

    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for candidate in (windir / "Fonts", Path(r"C:\Windows\Fonts")):
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return


def main() -> int:
    _configure_qt_fontdir()
    app = QApplication(sys.argv)
    theme_controller = ThemeController(app)
    theme_controller.apply_saved_or_default()

    controller = AppController(theme_controller=theme_controller)
    controller.main_window.resize(1400, 850)
    controller.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
