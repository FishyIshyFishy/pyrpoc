from __future__ import annotations

import os
from pathlib import Path
import sys

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QWidget

from pyrpoc.shell.app import Application
from pyrpoc.shell.display_bridge import DisplayBridge
from pyrpoc.shell.theme.manager import ThemeController
from pyrpoc.shell.window import MainWindow


def configure_qt_fontdir() -> None:
    if os.name != "nt":
        return
    if os.environ.get("QT_QPA_FONTDIR"):
        return

    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for candidate in (windir / "Fonts", Path(r"C:\Windows\Fonts")):
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return


def fit_to_available_screen(
    window: QWidget,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Size and position a window so it lands inside a visible screen area.

    Qt's default placement can drop a window outside the desktop on multi-monitor
    high-DPI setups: Qt reports a secondary screen's origin in native pixels while
    its size stays logical, which leaves a hole in the logical coordinate space
    that default placement can land in. Anchoring to a real availableGeometry()
    keeps the window reachable. Call once before show() to pick the spot, and
    again afterwards to re-clamp now that the frame margins are known.
    """
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        if width is not None and height is not None:
            window.resize(width, height)
        return

    avail = screen.availableGeometry()
    if avail.isEmpty():
        if width is not None and height is not None:
            window.resize(width, height)
        return

    frame_margin = window.frameGeometry().size() - window.size()
    target_width = window.width() if width is None else width
    target_height = window.height() if height is None else height
    window.resize(
        min(target_width, avail.width() - frame_margin.width()),
        min(target_height, avail.height() - frame_margin.height()),
    )

    frame = window.frameGeometry()
    frame.moveCenter(avail.center())
    frame.moveLeft(max(avail.left(), min(frame.left(), avail.right() - frame.width() + 1)))
    frame.moveTop(max(avail.top(), min(frame.top(), avail.bottom() - frame.height() + 1)))
    window.move(frame.topLeft())


def build(theme_controller: ThemeController) -> tuple[Application, MainWindow]:
    """Build the application and its window. Shared by main() and the tests."""
    app = Application()
    DisplayBridge(app, app)          # temporary; deleted in phase 8
    window = MainWindow(app, theme_controller)
    return app, window


def main() -> int:
    configure_qt_fontdir()
    qt_app = QApplication(sys.argv)
    theme_controller = ThemeController(qt_app)
    theme_controller.apply_saved_or_default()

    app, window = build(theme_controller)

    # A fresh workbench needs a card and a galvo, or nothing can run.
    if not app.devices:
        app.add_device("daq")
        app.add_device("galvo")

    fit_to_available_screen(window, 1400, 850)
    window.show()
    fit_to_available_screen(window)
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
