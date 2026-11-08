"""``Card``: one see-through panel of the overlay, docked to an edge. A new card subclasses it and
is added to the overlay's list in ``view.py``."""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from .. import look
from ..state.snapshot import Snapshot

RADIUS = 12.0


class Anchor(Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


class Card(QFrame):
    """A titled card ``span`` pixels across its edge: wide for a side, tall for the top or
    bottom. Cards sharing a side split its length by ``stretch``."""

    def __init__(self, parent: QWidget, title: str, anchor: Anchor, span: int, stretch: int):
        super().__init__(parent)
        self.anchor, self.span, self.stretch = anchor, span, stretch
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        heading = QLabel(title.upper(), self)
        heading.setFont(look.font(8.5, True))
        heading.setStyleSheet(f"color: {look.TEXT_DIM.name()}; letter-spacing: 1.5px;")
        root.addWidget(heading)
        self.content = root

    def show_snapshot(self, snapshot: Snapshot) -> None:
        raise NotImplementedError

    def focus_widget(self) -> QWidget:
        """The widget that takes the keyboard when the overlay comes up."""
        return self

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(look.GLASS_EDGE, 1.0))
        painter.setBrush(look.GLASS)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
        painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        # A click on a card's glass is the card's, never the dish's under it.
        self.focus_widget().setFocus(Qt.FocusReason.MouseFocusReason)
