from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from types import TracebackType

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from pyrpoc.src.app.gui.theme.manager import ThemeController
from pyrpoc.src.app.gui.window import MainWindow
from pyrpoc.src.app.model.application import Application
from pyrpoc.src.app.workspace.autosave import Autosave
from pyrpoc.src.app.workspace.file import WorkspaceFile, default_workspace_path

log = logging.getLogger("pyrpoc")


def report_uncaught(
    kind: type[BaseException], error: BaseException, tb: TracebackType | None
) -> None:
    """The one place an unexpected error lands. PyQt aborts the process on an
    exception escaping a slot unless this hook is set, so code elsewhere lets
    errors raise instead of guarding each slot."""
    text = "".join(traceback.format_exception(kind, error, tb))
    log.error("uncaught exception\n%s", text)
    if QApplication.instance() is not None:
        QMessageBox.critical(None, "Unexpected Error", f"{error}\n\n{text}")


def configure_qt_fontdir() -> None:
    if os.name != "nt" or os.environ.get("QT_QPA_FONTDIR"):
        return
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for candidate in (windir / "Fonts", Path(r"C:\Windows\Fonts")):
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return


def fit_to_available_screen(
    window: QWidget, width: int | None = None, height: int | None = None
) -> None:
    """Size and place a window inside a visible screen area.

    On multi-monitor high-DPI setups Qt reports a secondary screen's origin in
    native pixels but its size in logical ones, leaving a hole default
    placement can land in. Call before show() to pick the spot, and again
    after to re-clamp once the frame margins are known.
    """
    screen = window.screen() or QGuiApplication.primaryScreen()
    avail = screen.availableGeometry() if screen is not None else None
    if avail is None or avail.isEmpty():
        if width is not None and height is not None:
            window.resize(width, height)
        return

    frame_margin = window.frameGeometry().size() - window.size()
    window.resize(
        min(width or window.width(), avail.width() - frame_margin.width()),
        min(height or window.height(), avail.height() - frame_margin.height()),
    )
    frame = window.frameGeometry()
    frame.moveCenter(avail.center())
    frame.moveLeft(max(avail.left(), min(frame.left(), avail.right() - frame.width() + 1)))
    frame.moveTop(max(avail.top(), min(frame.top(), avail.bottom() - frame.height() + 1)))
    window.move(frame.topLeft())


def build(
    theme_controller: ThemeController, workspace_path: Path
) -> tuple[Application, MainWindow, Autosave]:
    """Build the application, its window and its autosave."""
    app = Application()
    window = MainWindow(app, theme_controller)
    autosave = Autosave(app, window, WorkspaceFile(workspace_path), parent=app)
    window.bind_workspace(autosave.save_now)
    return app, window, autosave


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    sys.excepthook = report_uncaught
    configure_qt_fontdir()
    qt_app = QApplication(sys.argv)
    theme_controller = ThemeController(qt_app)
    theme_controller.apply_saved_or_default()

    _app, window, autosave = build(theme_controller, default_workspace_path())

    fit_to_available_screen(window, 1400, 850)
    window.show()
    autosave.restore()
    fit_to_available_screen(window)
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
