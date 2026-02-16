from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from pyrpoc.services.app_controller import AppController


def main() -> int:
    app = QApplication(sys.argv)
    controller = AppController()
    controller.main_window.resize(1400, 850)
    controller.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
