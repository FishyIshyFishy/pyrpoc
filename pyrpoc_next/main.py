"""Entry point: build the controller and show the main window."""

from __future__ import annotations

import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication

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
