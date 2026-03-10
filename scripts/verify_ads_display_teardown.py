from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QWidget
import PyQt6Ads as qtads


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
    # Headless check suitable for local and CI verification.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _configure_qt_fontdir()
    app = QApplication([])
    root = QWidget()
    manager = qtads.CDockManager(root)
    dock = qtads.CDockWidget("teardown-check")
    display_widget = QWidget()
    dock.setWidget(display_widget)
    manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)

    if hasattr(manager, "removeDockWidget"):
        manager.removeDockWidget(dock)
    detached = dock.takeWidget()
    if detached is not None:
        detached.setParent(None)
        detached.deleteLater()
    dock.deleteLater()
    root.deleteLater()
    for _ in range(3):
        app.processEvents()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
