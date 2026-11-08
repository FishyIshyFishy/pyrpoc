"""``Overlay``: the layer over the dish that holds the cards. It fades in and out, lays a faint veil
over the dish while up, and takes every click and key, so nothing reaches the dish by
mistake."""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRect, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QWidget

from .. import keys, look
from ..state.snapshot import Snapshot
from .card import Anchor, Card

FADE_MS = 120
MARGIN = 16


class Overlay(QWidget):
    # Esc was pressed with nothing in a card to cancel first.
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.cards: list[Card] = []
        self.fade = QGraphicsOpacityEffect(self)
        self.fade.setOpacity(0.0)
        self.setGraphicsEffect(self.fade)
        self.animation = QPropertyAnimation(self.fade, b"opacity", self)
        self.animation.setDuration(FADE_MS)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.finished.connect(self.settle)
        self.up = False
        self.hide()

    def add(self, card: Card) -> None:
        self.cards.append(card)
        self.arrange()

    def appear(self) -> None:
        self.up = True
        self.show()
        self.raise_()
        self.fade_to(1.0)
        if self.cards:
            self.cards[0].focus_widget().setFocus()

    def vanish(self) -> None:
        self.up = False
        self.fade_to(0.0)

    def fade_to(self, opacity: float) -> None:
        self.animation.stop()
        self.fade.setEnabled(True)
        self.animation.setStartValue(self.fade.opacity())
        self.animation.setEndValue(opacity)
        self.animation.start()

    def settle(self) -> None:
        if not self.up:
            self.hide()
        # The effect redraws every card off-screen on each frame, so it is on only while fading.
        self.fade.setEnabled(not self.up)

    def show_snapshot(self, snapshot: Snapshot) -> None:
        for card in self.cards:
            card.show_snapshot(snapshot)

    def covered_left(self) -> int:
        """How far in from the left the cards there reach, in pixels."""
        lefts = [c.geometry().right() for c in self.cards if c.anchor is Anchor.LEFT]
        return max(lefts) + MARGIN if lefts else 0

    def arrange(self) -> None:
        """Dock each card to its edge, sharing the edge with the others there."""
        sides = {anchor: [c for c in self.cards if c.anchor is anchor] for anchor in Anchor}
        inner = self.rect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)
        left = max((c.span for c in sides[Anchor.LEFT]), default=-MARGIN) + MARGIN
        right = max((c.span for c in sides[Anchor.RIGHT]), default=-MARGIN) + MARGIN
        self.stack(
            sides[Anchor.LEFT], QRect(inner.left(), inner.top(), left - MARGIN, inner.height())
        )
        self.stack(
            sides[Anchor.RIGHT],
            QRect(inner.right() - right + MARGIN + 1, inner.top(), right - MARGIN, inner.height()),
        )
        middle = inner.adjusted(left, 0, -right, 0)
        for anchor in (Anchor.TOP, Anchor.BOTTOM):
            for card in sides[anchor]:
                top = middle.top() if anchor is Anchor.TOP else middle.bottom() - card.span + 1
                card.setGeometry(QRect(middle.left(), top, middle.width(), card.span))

    def stack(self, cards: list[Card], area: QRect) -> None:
        if not cards:
            return
        share = (area.height() - MARGIN * (len(cards) - 1)) // len(cards)
        for index, card in enumerate(cards):
            top = area.top() + index * (share + MARGIN)
            card.setGeometry(QRect(area.left(), top, area.width(), share))

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.arrange()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), look.VEIL)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == keys.CLOSE:
            self.dismissed.emit()

    # The overlay takes every click and scroll, so none reaches the dish while it is up.
    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            event.accept()

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            event.accept()
