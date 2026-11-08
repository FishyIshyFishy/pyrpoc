"""Opens the multisite view on a fake loop, with a menu to switch each way of showing what the loop
is doing."""

from __future__ import annotations

import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QColor, QPalette
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QMenuBar

from ..canvas.indicators import NAMES, Indicator
from ..view import MultisiteView
from .fake_dish import FakeDish
from .fake_loop import FakeLoop

STEP_MS = 50
UNITS = 8
SEED = 7


def dark(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    for role, colour in (
        (QPalette.ColorRole.Window, QColor(24, 26, 32)),
        (QPalette.ColorRole.WindowText, QColor(230, 233, 240)),
        (QPalette.ColorRole.Base, QColor(18, 20, 26)),
        (QPalette.ColorRole.Text, QColor(230, 233, 240)),
        (QPalette.ColorRole.Button, QColor(34, 37, 45)),
        (QPalette.ColorRole.ButtonText, QColor(230, 233, 240)),
        (QPalette.ColorRole.Highlight, QColor(110, 175, 255)),
    ):
        palette.setColor(role, colour)
    app.setPalette(palette)


class DemoWindow(QMainWindow):
    def __init__(self, loop: FakeLoop):
        super().__init__()
        self.setWindowTitle("Multisite: demo loop")
        self.view = MultisiteView(self, loop)
        self.setCentralWidget(self.view)
        self.switches: dict[Indicator, QAction] = {}
        bar = QMenuBar(self)
        shown = QMenu("Indicators", bar)
        bar.addMenu(shown)
        self.setMenuBar(bar)
        for indicator, name in NAMES.items():
            action = QAction(name, self)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(self.switch)
            shown.addAction(action)
            self.switches[indicator] = action
        self.clock = QTimer(self)
        self.clock.setInterval(STEP_MS)
        self.clock.timeout.connect(loop.step)
        self.clock.start()

    def switch(self) -> None:
        on = Indicator.NONE
        for indicator, action in self.switches.items():
            if action.isChecked():
                on |= indicator
        self.view.set_indicators(on)


def main() -> None:
    app = QApplication(sys.argv)
    dark(app)
    window = DemoWindow(FakeLoop(FakeDish(SEED), UNITS))
    window.resize(1500, 950)
    window.show()
    window.view.canvas.setFocus()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
