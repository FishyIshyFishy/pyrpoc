"""The survey's overview, with each unit's latest frame laid where it was taken, so the dish shows
how it looks now."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene

from pyrpoc.utils.widgets.colors import color_for_index

from ..feed import Backdrop, UnitFrame
from .render import levels_of, pixmap_of, rgb_of

# How long a new frame takes to fade in over the last, in seconds.
FADE_S = 0.4
OVERVIEW_Z, FRAME_Z = 0.0, 1.0


@dataclass
class FrameItem:
    item: QGraphicsPixmapItem
    serial: int
    shown_s: float
    # The frame it replaces, removed once this one is opaque.
    under: QGraphicsPixmapItem | None


class BackdropLayer:
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.backdrop: Backdrop | None = None
        self.overview: QGraphicsPixmapItem | None = None
        self.levels = np.zeros((0, 2))
        self.colours: list[tuple[int, int, int]] = []
        self.frames: dict[int, FrameItem] = {}

    def show_backdrop(self, backdrop: Backdrop) -> None:
        """Lay out the overview, unless it is already the one shown."""
        if backdrop is self.backdrop:
            return
        if self.overview is not None:
            self.scene.removeItem(self.overview)
        self.backdrop = backdrop
        # The frames are coloured with the overview's levels, so a frame matches what is round it.
        self.levels = levels_of(backdrop.planes)
        self.colours = [color_for_index(index) for index in range(len(backdrop.channels))]
        placement = backdrop.placement
        self.overview = self.placed(
            self.pixmap(backdrop.planes),
            placement.left_um,
            placement.top_um,
            placement.um_per_px,
            OVERVIEW_Z,
        )

    def show_frames(self, frames: Mapping[int, UnitFrame], present: set[int], now_s: float) -> None:
        """Lay each unit's newest frame over its last, and drop those of units gone."""
        for unit in list(self.frames):
            if unit not in present:
                self.drop(unit)
        for unit, frame in frames.items():
            shown = self.frames.get(unit)
            if unit in present and (shown is None or shown.serial != frame.serial):
                self.lay(unit, frame, shown, now_s)

    def lay(self, unit: int, frame: UnitFrame, shown: FrameItem | None, now_s: float) -> None:
        tag = frame.tag
        item = self.placed(
            self.pixmap(frame.planes),
            tag.left_um,
            tag.top_um,
            frame.um_per_px,
            FRAME_Z + frame.serial * 1e-6,
        )
        item.setOpacity(0.0)
        if shown is not None and shown.under is not None:
            self.scene.removeItem(shown.under)
        self.frames[unit] = FrameItem(
            item, frame.serial, now_s, None if shown is None else shown.item
        )

    def drop(self, unit: int) -> None:
        shown = self.frames.pop(unit)
        self.scene.removeItem(shown.item)
        if shown.under is not None:
            self.scene.removeItem(shown.under)

    def tick(self, now_s: float) -> None:
        """Carry each fading frame on."""
        for shown in self.frames.values():
            opacity = min(1.0, (now_s - shown.shown_s) / FADE_S)
            shown.item.setOpacity(opacity)
            if opacity >= 1.0 and shown.under is not None:
                self.scene.removeItem(shown.under)
                shown.under = None

    def pixmap(self, planes: np.ndarray) -> QPixmap:
        return pixmap_of(rgb_of(planes, self.levels, self.colours))

    def placed(
        self, pixmap: QPixmap, left_um: float, top_um: float, um_per_px: float, z: float
    ) -> QGraphicsPixmapItem:
        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        # Scaled once per zoom, not on every frame the reticle and markers animate.
        item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        item.setPos(left_um, top_um)
        item.setScale(um_per_px)
        item.setZValue(z)
        self.scene.addItem(item)
        return item
