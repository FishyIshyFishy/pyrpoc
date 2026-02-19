from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from pyrpoc.gui.styles.theme_manager import ThemeController
from pyrpoc.services.app_controller import AppController


def main() -> int:
    app = QApplication(sys.argv)
    theme_controller = ThemeController(app)
    theme_controller.apply_saved_or_default()

    controller = AppController(theme_controller=theme_controller)
    app.aboutToQuit.connect(controller.session_coordinator.save_now)
    controller.main_window.resize(1400, 850)
    controller.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
