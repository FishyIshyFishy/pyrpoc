"""The tiles stitched into one image."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .image_view import ColourImage
from .layout import MosaicLayout
from .stitching import AxisStep, Stitcher, composite, tile_positions


def describe_axis(name: str, step: AxisStep) -> str:
    if step.registered == 0:
        return f"{name}: no neighbours yet"
    if step.nominal:
        return f"{name}: no confident pairs of {step.registered}, using fallback"
    return f"{name}: {step.confident}/{step.registered} pairs, step {step.offset} px"


class MosaicView(QWidget):
    """Stitches whatever tiles have arrived. Registrations are kept per
    dataset, so a new tile costs only its own neighbours."""

    # The user changed the fallback overlap, so the tiles must be placed again.
    fallback_changed = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.stitcher: Stitcher | None = None
        self.stitched_id: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.image = ColourImage(self)
        root.addWidget(self.image, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Fallback overlap:", self))
        self.fallback = QDoubleSpinBox(self)
        self.fallback.setRange(0.0, 90.0)
        self.fallback.setSuffix(" %")
        self.fallback.setValue(10.0)
        self.fallback.setToolTip(
            "Used along an axis where no pair of tiles could be matched, "
            "assuming the stage runs with the scan axes"
        )
        self.fallback.valueChanged.connect(lambda _value: self.fallback_changed.emit())
        row.addWidget(self.fallback)
        self.status = QLabel("", self)
        row.addWidget(self.status, 1)
        root.addLayout(row)

    def stitch(
        self, dataset_id: str, frames: Sequence[np.ndarray], layout: MosaicLayout
    ) -> np.ndarray:
        """The tiles so far blended into one ``(C, H, W)`` image."""
        if self.stitcher is None or dataset_id != self.stitched_id:
            self.stitcher = Stitcher(layout)
            self.stitched_id = dataset_id
        count = min(len(frames), len(layout.tiles))
        steps = self.stitcher.steps(frames, self.fallback.value() / 100.0)
        positions = tile_positions(layout, steps, count)
        self.status.setText(
            f"{count}/{len(layout.tiles)} tiles · "
            f"{describe_axis('x', steps.col)} · {describe_axis('y', steps.row)}"
        )
        return composite(frames, positions)

    def clear(self) -> None:
        self.stitcher, self.stitched_id = None, None
        self.status.setText("")
        self.image.clear_image()
