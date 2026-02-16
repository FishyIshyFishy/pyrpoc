from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout

from pyrpoc.backend_utils.data import BaseData, DataImage
from .base_display import BaseDisplay
from .display_registry import display_registry


@display_registry.register("sim_image")
class SimImageDisplay(BaseDisplay):
    DISPLAY_KEY = "sim_image"
    DISPLAY_NAME = "Simulated Image Display"
    ACCEPTED_DATA_TYPES = [DataImage]
    DISPLAY_PARAMETERS = {}

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        layout = QVBoxLayout(self)
        self._image_label = QLabel("No image yet", self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(320, 240)
        self._image_label.setStyleSheet("border: 1px solid #888;")
        layout.addWidget(self._image_label)
        self._last_pixmap: QPixmap | None = None

    def configure(self, params: dict[str, Any]) -> None:
        return None

    def render(self, data: BaseData) -> None:
        if not any(isinstance(data, accepted) for accepted in self.ACCEPTED_DATA_TYPES):
            raise TypeError(f"{self.__class__.__name__} cannot render {type(data).__name__}")

        image = np.asarray(data.value, dtype=float)
        if image.ndim != 2:
            raise ValueError("SimImageDisplay expects a 2D array")

        min_val = float(np.min(image))
        max_val = float(np.max(image))
        if max_val - min_val < 1e-12:
            normalized = np.zeros_like(image, dtype=np.uint8)
        else:
            normalized = ((image - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

        h, w = normalized.shape
        bytes_per_line = w
        qimg = QImage(normalized.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg.copy())
        self._last_pixmap = pixmap
        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def clear(self) -> None:
        self._last_pixmap = None
        self._image_label.setText("No image yet")
        self._image_label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_pixmap is not None:
            scaled = self._last_pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._image_label.setPixmap(scaled)
