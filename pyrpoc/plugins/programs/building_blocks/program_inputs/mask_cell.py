"""``MaskCell``: the editor for one mask input, which shows the mask a Mask Editor drew into it."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from pyrpoc.core.mask import Mask


class MaskCell(QWidget):
    """What one mask input holds, and Clear, which leaves it empty."""

    changed = pyqtSignal()

    def __init__(self, empty_text: str, parent: QWidget):
        super().__init__(parent)
        self.empty_text = empty_text
        self.mask: Mask | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self)
        self.label.setToolTip("Draw a mask in a Mask Editor and choose this input there")
        clear_btn = QPushButton("Clear", self)
        clear_btn.setToolTip("Leave this input without a mask")
        clear_btn.clicked.connect(self.on_clear_clicked)
        row.addWidget(self.label, 1)
        row.addWidget(clear_btn)
        self.refresh()

    def value(self) -> Mask | None:
        return self.mask

    def set_value(self, mask: Mask | None) -> None:
        self.mask = mask
        self.refresh()

    def summary(self) -> str:
        return self.empty_text if self.mask is None else self.mask.describe()

    def on_clear_clicked(self) -> None:
        self.mask = None
        self.refresh()
        self.changed.emit()

    def refresh(self) -> None:
        self.label.setText(self.summary())
