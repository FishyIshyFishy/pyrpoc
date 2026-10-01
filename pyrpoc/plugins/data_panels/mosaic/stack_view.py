"""The frames one at a time, on a slider."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .image_view import LevelledImage
from .layout import MosaicLayout


class StackView(QWidget):
    """Scrolls through every frame of the dataset. While the slider is on the
    last frame it follows new ones as they arrive; moved back, it stays put."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.frames: Sequence[np.ndarray] = ()
        self.mosaic: MosaicLayout | None = None
        self.channel = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.image = LevelledImage(self)
        root.addWidget(self.image, 1)
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.draw)
        self.position = QLabel("", self)
        row.addWidget(self.slider, 1)
        row.addWidget(self.position)
        root.addLayout(row)

    def show_frames(
        self, frames: Sequence[np.ndarray], mosaic: MosaicLayout | None, channel: int
    ) -> None:
        following = self.slider.value() >= self.slider.maximum()
        self.frames, self.mosaic, self.channel = frames, mosaic, channel
        self.slider.blockSignals(True)
        self.slider.setMaximum(len(frames) - 1)
        self.slider.setEnabled(len(frames) > 1)
        if following:
            self.slider.setValue(len(frames) - 1)
        self.slider.blockSignals(False)
        self.draw()

    def draw(self) -> None:
        if not self.frames:
            return
        index = self.slider.value()
        self.image.show_plane(np.asarray(self.frames[index][self.channel], dtype=np.float32))
        self.position.setText(self.describe(index))

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
        self.image.clear_plane()
