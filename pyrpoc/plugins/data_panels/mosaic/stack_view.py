"""The frames one at a time, on a slider."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .image_view import ColourImage
from .layout import MosaicLayout


class StackView(QWidget):
    """Scrolls through every frame of the dataset. While the slider is on the
    last frame it follows new ones as they arrive; moved back, it stays put."""

    # The user moved to another frame.
    moved = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.frames: Sequence[np.ndarray] = ()
        self.mosaic: MosaicLayout | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.image = ColourImage(self)
        root.addWidget(self.image, 1)
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(lambda _value: self.moved.emit())
        self.position = QLabel("", self)
        row.addWidget(self.slider, 1)
        row.addWidget(self.position)
        root.addLayout(row)

    def take_frames(self, frames: Sequence[np.ndarray], mosaic: MosaicLayout | None) -> None:
        """The dataset's frames so far; the slider follows a new one only if
        it was already on the last."""
        following = self.slider.value() >= self.slider.maximum()
        self.frames, self.mosaic = frames, mosaic
        self.slider.blockSignals(True)
        self.slider.setMaximum(len(frames) - 1)
        self.slider.setEnabled(len(frames) > 1)
        if following:
            self.slider.setValue(len(frames) - 1)
        self.slider.blockSignals(False)

    def current_frame(self) -> np.ndarray:
        """The ``(C, H, W)`` frame under the slider, naming it beside the slider."""
        index = self.slider.value()
        self.position.setText(self.describe(index))
        return np.asarray(self.frames[index], dtype=np.float32)

    def describe(self, index: int) -> str:
        count = f"{index + 1}/{len(self.frames)}"
        if self.mosaic is None or index >= len(self.mosaic.tiles):
            return f"frame {count}"
        tile = self.mosaic.tiles[index]
        return f"tile {count} · row {tile.row + 1}, col {tile.col + 1}"

    def clear(self) -> None:
        self.frames, self.mosaic = (), None
        self.slider.blockSignals(True)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.position.setText("")
        self.image.clear_image()
