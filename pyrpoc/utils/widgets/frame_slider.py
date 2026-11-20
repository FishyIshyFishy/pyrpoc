"""``FrameSlider``: a slider through numbered frames that keeps up with new ones."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class FrameSlider(QWidget):
    """Picks one of ``count`` frames, numbered beside the slider. While it is on the last frame
    it moves to each new one; anywhere else it stays put. Hidden while there is one frame."""

    # The user moved to another frame.
    moved = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.on_value_changed)
        self.position = QLabel("", self)
        row.addWidget(self.slider, 1)
        row.addWidget(self.position)
        self.setVisible(False)

    def take_count(self, count: int) -> None:
        """Hold ``count`` frames, moving to a new last frame only if it was on the last one."""
        following = self.slider.value() >= self.slider.maximum()
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(count - 1, 0))
        if following:
            self.slider.setValue(self.slider.maximum())
        self.slider.blockSignals(False)
        self.show_position()
        self.setVisible(count > 1)

    def index(self) -> int:
        """The frame chosen, counting from 0."""
        return self.slider.value()

    def reset(self) -> None:
        self.take_count(0)

    def on_value_changed(self, _value: int) -> None:
        self.show_position()
        self.moved.emit()

    def show_position(self) -> None:
        self.position.setText(f"frame {self.index() + 1}/{self.slider.maximum() + 1}")
