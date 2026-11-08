"""The survey's overview with each unit's latest frame laid where it was taken, so the dish
shows how it looks now."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene

from pyrpoc.utils.widgets.colors import color_for_index

from .feed import Backdrop, UnitFrame
from .render import levels_of, pixmap_of, rgb_of

OVERVIEW_Z, FRAME_Z = 0.0, 1.0


class BackdropLayer:
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.overview: QGraphicsPixmapItem | None = None
        self.levels = np.zeros((0, 2))
        self.colours: list[tuple[int, int, int]] = []
        # Each unit's frame on the dish, and the serial of the frame it shows.
        self.frames: dict[int, tuple[QGraphicsPixmapItem, int]] = {}

    def show_backdrop(self, backdrop: Backdrop) -> None:
        if self.overview is not None:
            self.scene.removeItem(self.overview)
        # Frames are coloured with the overview's levels, so a frame matches what is round it.
        self.levels = levels_of(backdrop.planes)
        self.colours = [color_for_index(index) for index in range(len(backdrop.channels))]
        placement = backdrop.placement
        self.overview = self.placed(
            backdrop.planes, placement.left_um, placement.top_um, placement.um_per_px, OVERVIEW_Z
        )

    def show_frames(self, frames: Mapping[int, UnitFrame], present: set[int]) -> None:
        """Lay each present unit's newest frame over its last, and take away those of units
        gone."""
        for unit in list(self.frames):
            if unit not in present:
                self.scene.removeItem(self.frames.pop(unit)[0])
        for unit, frame in frames.items():
            shown = self.frames.get(unit)
            if unit not in present or (shown is not None and shown[1] == frame.serial):
                continue
            if shown is not None:
                self.scene.removeItem(shown[0])
            tag = frame.tag
            # The newest frame lies over any older one it overlaps.
            z = FRAME_Z + frame.serial * 1e-6
            item = self.placed(frame.planes, tag.left_um, tag.top_um, frame.um_per_px, z)
            self.frames[unit] = (item, frame.serial)

    def placed(
        self, planes: np.ndarray, left_um: float, top_um: float, um_per_px: float, z: float
    ) -> QGraphicsPixmapItem:
        pixmap: QPixmap = pixmap_of(rgb_of(planes, self.levels, self.colours))
        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        # Scaled once per zoom, not on every redraw of the outlines over it.
        item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        item.setPos(left_um, top_um)
        item.setScale(um_per_px)
        item.setZValue(z)
        self.scene.addItem(item)
        return item
