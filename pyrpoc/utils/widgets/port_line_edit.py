"""``PortLineEdit``: one small box for a digital port and line, written ``p0.4``."""

from __future__ import annotations

import re

from PyQt6.QtCore import QRegularExpression, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QRegularExpressionValidator, QWheelEvent
from PyQt6.QtWidgets import QLineEdit, QWidget

PATTERN = r"[pP]?(\d{1,2})\.(\d{1,2})"
LARGEST = 99
# Room for "p10.31" and the frame, and no more.
WIDTH_PX = 56


class PortLineEdit(QLineEdit):
    """A port and line typed as ``p<port>.<line>``. Up/Down or the wheel step the number
    under the cursor."""

    changed = pyqtSignal()

    def __init__(self, port: int, line: int, parent: QWidget):
        super().__init__(parent)
        self.port = port
        self.line = line
        self.setValidator(QRegularExpressionValidator(QRegularExpression(PATTERN), self))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedWidth(WIDTH_PX)
        self.setToolTip("Port and line, as p<port>.<line>; Up/Down or scroll to step")
        self.show_value()
        self.editingFinished.connect(self.on_edited)

    def show_value(self) -> None:
        self.setText(f"p{self.port}.{self.line}")

    def on_edited(self) -> None:
        # Validator-checked text can still be partial ("p3"); go back to the last whole value.
        match = re.fullmatch(PATTERN, self.text())
        if match is None:
            self.show_value()
            return
        self.set_parts(int(match.group(1)), int(match.group(2)))

    def set_parts(self, port: int, line: int) -> None:
        port, line = min(max(port, 0), LARGEST), min(max(line, 0), LARGEST)
        moved = (port, line) != (self.port, self.line)
        self.port, self.line = port, line
        self.show_value()
        if moved:
            self.changed.emit()

    def step(self, by: int) -> None:
        """Step the port if the cursor is left of the dot, else the line."""
        on_port = self.cursorPosition() <= self.text().find(".")
        cursor = self.cursorPosition()
        if on_port:
            self.set_parts(self.port + by, self.line)
        else:
            self.set_parts(self.port, self.line + by)
        self.setCursorPosition(min(cursor, len(self.text())))

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is not None and a0.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self.step(1 if a0.key() == Qt.Key.Key_Up else -1)
            return
        super().keyPressEvent(a0)

    def wheelEvent(self, a0: QWheelEvent | None) -> None:
        # Only when focused, so scrolling the panel past the box does not change it.
        if a0 is None or not self.hasFocus():
            super().wheelEvent(a0)
            return
        delta = a0.angleDelta().y()
        if delta:
            self.step(1 if delta > 0 else -1)
        a0.accept()
