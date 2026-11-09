"""``Overlay``: the queue and units cards held up over the live dish while Tab has it open. It
fades in and out, lays a faint veil over the dish, and closes on a click on the dish between the
cards, so it reads as a glance rather than a pane that stays."""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, QRect, pyqtSignal
from PyQt6.QtGui import (
    QContextMenuEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QWidget

from . import look
from .cards import GlassCard, QueueCard, Side, UnitsCard

FADE_MS = 120
MARGIN = 16
# A card's widest, and its widest share of the dish, so some dish always shows between them.
CARD_PX = 460
CARD_SHARE = 0.4


class Overlay(QWidget):
    # A click on the dish between the cards.
    dismissed = pyqtSignal()

    def __init__(self, host: QWidget):
        super().__init__(host)
        self.host = host
        host.installEventFilter(self)
        self.queue = QueueCard(self)
        self.units = UnitsCard(self)
        self.fade = QGraphicsOpacityEffect(self)
        self.fade.setOpacity(0.0)
        self.setGraphicsEffect(self.fade)
        self.animation = QPropertyAnimation(self.fade, b"opacity", self)
        self.animation.setDuration(FADE_MS)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.finished.connect(self.settle)
        self.up = False
        self.hide()

    @property
    def cards(self) -> tuple[GlassCard, ...]:
        return (self.queue, self.units)

    def appear(self) -> None:
        self.up = True
        self.setGeometry(self.host.rect())
        self.show()
        self.raise_()
        self.fade_to(1.0)
        self.queue.table.setFocus()

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
        # The effect draws every card off-screen on each frame, so it is on only while fading.
        self.fade.setEnabled(not self.up)

    def uncovered(self) -> QRect:
        """The part of the dish between the cards, in this overlay's pixels."""
        left = self.queue.geometry().right() + MARGIN
        right = self.units.geometry().left() - MARGIN
        return QRect(left, 0, max(1, right - left), self.height())

    def arrange(self) -> None:
        inner = self.rect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)
        width = min(CARD_PX, int(inner.width() * CARD_SHARE))
        for card in self.cards:
            x = inner.left() if card.side is Side.LEFT else inner.right() - width + 1
            card.setGeometry(QRect(x, inner.top(), width, inner.height()))

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        if watched is self.host and event is not None and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.host.rect())
        return False

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.arrange()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), look.VEIL)

    # Clicks on a card stay in the card; one that reaches here fell on the dish between them.
    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            event.accept()
            self.dismissed.emit()

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            event.accept()

    def contextMenuEvent(self, event: QContextMenuEvent | None) -> None:
        if event is not None:
            event.accept()
