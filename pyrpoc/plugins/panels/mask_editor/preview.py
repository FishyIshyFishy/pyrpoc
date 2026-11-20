"""The mask being drawn, shown whole beside the image it is drawn on."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QEvent, QRectF, Qt
from PyQt6.QtGui import QImage, QMouseEvent, QPalette, QPixmap, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget


class MaskPreview(QGraphicsView):
    """A mask as it is saved, lit pixels white on black, fitted to the view at any size. It only
    shows: nothing is drawn, zoomed or dragged here."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.gscene = QGraphicsScene(self)
        self.picture = QGraphicsPixmapItem()
        self.picture.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.gscene.addItem(self.picture)
        self.setScene(self.gscene)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Always fitted, so there is nothing to scroll to.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setInteractive(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.show_backdrop()

    def show_backdrop(self) -> None:
        self.setBackgroundBrush(self.palette().color(QPalette.ColorRole.Base))

    def changeEvent(self, event: QEvent | None) -> None:
        super().changeEvent(event)
        if event is not None and event.type() == QEvent.Type.PaletteChange:
            self.show_backdrop()

    # Ignored, so a press or the wheel here never reaches the scene or scrolls the view.
    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            event.ignore()

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            event.ignore()

    def show_image(self, image: np.ndarray) -> None:
        """Show a 2-D ``uint8`` picture of the mask."""
        grey = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = grey.shape
        qimage = QImage(grey.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        self.picture.setPixmap(QPixmap.fromImage(qimage.copy()))
        self.gscene.setSceneRect(QRectF(0, 0, width, height))
        self.fit()

    def fit(self) -> None:
        self.fitInView(self.gscene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.fit()
