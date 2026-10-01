"""The tiles stitched into one image."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .image_view import LevelledImage
from .layout import MosaicLayout
from .stitching import AxisStep, Stitcher, composite, tile_positions


def describe_axis(name: str, step: AxisStep) -> str:
    if step.registered == 0:
        return f"{name}: no neighbours yet"
    if step.nominal:
        return f"{name}: no confident pairs of {step.registered}, using fallback"
    return f"{name}: {step.confident}/{step.registered} pairs, step {step.offset} px"


class MosaicView(QWidget):
    """Re-stitches whenever tiles arrive, the channel changes, or the fallback
    overlap changes. Registrations are kept per dataset and channel, so a new
    tile costs only its own neighbours."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.frames: Sequence[np.ndarray] = ()
        self.stitcher: Stitcher | None = None
        self.stitched_id: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.image = LevelledImage(self)
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
        self.fallback.valueChanged.connect(self.draw)
        row.addWidget(self.fallback)
        self.status = QLabel("", self)
        row.addWidget(self.status, 1)
        root.addLayout(row)

    def show_frames(
        self, dataset_id: str, frames: Sequence[np.ndarray], layout: MosaicLayout, channel: int
    ) -> None:
        if self.stitcher is None or (dataset_id, channel) != (
            self.stitched_id,
            self.stitcher.channel,
        ):
            self.stitcher = Stitcher(layout, channel)
            self.stitched_id = dataset_id
        self.frames = frames
        self.draw()

    def draw(self) -> None:
        stitcher = self.stitcher
        if stitcher is None or not self.frames:
            return
        count = min(len(self.frames), len(stitcher.layout.tiles))
        steps = stitcher.steps(self.frames, self.fallback.value() / 100.0)
        positions = tile_positions(stitcher.layout, steps, count)
        planes = {index: stitcher.plane(self.frames, index) for index in positions}
        self.image.show_plane(composite(planes, positions))
        self.status.setText(
            f"{count}/{len(stitcher.layout.tiles)} tiles · "
            f"{describe_axis('x', steps.col)} · {describe_axis('y', steps.row)}"
        )

    def clear(self) -> None:
        self.frames, self.stitcher, self.stitched_id = (), None, None
        self.status.setText("")
        self.image.clear_plane()
