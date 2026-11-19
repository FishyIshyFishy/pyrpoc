"""The mosaic on the dish: each tile where the stage put it while the mosaic is taken, then the
stitched overview once it is done."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene

from pyrpoc.plugins.program_io.outputs import MosaicOutput

# Colours a ``(C, H, W)`` image as ``(H, W, 3)`` in 0..1.
Colouring = Callable[[np.ndarray], np.ndarray]


def pixmap_of(rgb: np.ndarray) -> QPixmap:
    data = np.ascontiguousarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    height, width, _ = data.shape
    image = QImage(data.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
    # The copy owns its pixels; the bytes it was made from go when this returns.
    return QPixmap.fromImage(image.copy())


class TileLayer:
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        # Each image laid on the dish, with the planes it is coloured from.
        self.laid: list[tuple[QGraphicsPixmapItem, np.ndarray]] = []
        self.source: MosaicOutput | None = None
        self.stitched = False

    def show(self, mosaic: MosaicOutput, colour: Colouring) -> None:
        """Lay what ``mosaic`` holds that is not laid yet: its overview once stitched, which
        takes the place of the tiles, or else each new tile."""
        if mosaic is not self.source or mosaic.is_stitched != self.stitched:
            self.clear()
            self.source, self.stitched = mosaic, mosaic.is_stitched
        if self.stitched:
            if not self.laid:
                planes, placement = mosaic.overview()
                self.lay(planes, placement.left_um, placement.top_um, placement.um_per_px, colour)
            return
        for index in range(len(self.laid), min(len(mosaic), len(mosaic.tiles))):
            left, top = mosaic.tile_corner(index)
            self.lay(np.asarray(mosaic[index]), left, top, mosaic.um_per_px, colour)

    def lay(
        self, planes: np.ndarray, left_um: float, top_um: float, um_per_px: float, colour: Colouring
    ) -> None:
        item = QGraphicsPixmapItem(pixmap_of(colour(planes)))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        # Scaled once per zoom, not on every redraw of the regions over it.
        item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        item.setPos(left_um, top_um)
        item.setScale(um_per_px)
        self.scene.addItem(item)
        self.laid.append((item, planes))

    def recolour(self, colour: Colouring) -> None:
        for item, planes in self.laid:
            item.setPixmap(pixmap_of(colour(planes)))

    def bounds(self) -> QRectF:
        """What is laid, in µm; empty with nothing laid."""
        rect = QRectF()
        for item, _planes in self.laid:
            rect = rect.united(item.sceneBoundingRect())
        return rect

    def clear(self) -> None:
        for item, _planes in self.laid:
            self.scene.removeItem(item)
        self.laid = []
        self.source, self.stitched = None, False
