"""``SummaryCard``: the loop at a glance, a strip of figures across the top."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QFontMetricsF, QPainter, QPaintEvent
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .. import look
from ..state.snapshot import Snapshot
from ..state.summary import Stat, summary
from .card import Anchor, Card

HEIGHT = 104
LABEL = look.font(7.5, True)
VALUE = look.font(14.0, True)
DETAIL = look.font(8.0, False)
GAP = 18


class StatRow(QWidget):
    """The figures side by side, each a label, a value and a line of detail."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stats: tuple[Stat, ...] = ()

    def show_stats(self, stats: tuple[Stat, ...]) -> None:
        if stats != self.stats:
            self.stats = stats
            self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if not self.stats:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = (self.width() - GAP * (len(self.stats) - 1)) / len(self.stats)
        for index, stat in enumerate(self.stats):
            box = QRectF(index * (width + GAP), 0, width, self.height())
            self.paint_stat(painter, box, stat)
            if index:
                painter.setPen(look.GLASS_EDGE)
                x = box.left() - GAP / 2
                painter.drawLine(QPointF(x, 4), QPointF(x, self.height() - 4))

    def paint_stat(self, painter: QPainter, box: QRectF, stat: Stat) -> None:
        y = box.top()
        painter.setFont(LABEL)
        painter.setPen(look.TEXT_DIM)
        label_h = QFontMetricsF(LABEL).height()
        painter.drawText(
            QRectF(box.left(), y, box.width(), label_h),
            Qt.AlignmentFlag.AlignLeft,
            stat.label.upper(),
        )
        y += label_h + 2
        painter.setFont(VALUE)
        painter.setPen(look.TEXT)
        value_h = QFontMetricsF(VALUE).height()
        painter.drawText(
            QRectF(box.left(), y, box.width(), value_h), Qt.AlignmentFlag.AlignLeft, stat.value
        )
        y += value_h + 1
        self.paint_detail(painter, QRectF(box.left(), y, box.width(), box.bottom() - y), stat)

    def paint_detail(self, painter: QPainter, box: QRectF, stat: Stat) -> None:
        """The detail line, after the tone's mark in its colour when the figure needs notice."""
        painter.setFont(DETAIL)
        left = box.left()
        if stat.tone is not None:
            painter.setPen(look.TONE[stat.tone])
            painter.drawText(
                QRectF(left, box.top(), 14, box.height()),
                Qt.AlignmentFlag.AlignLeft,
                stat.tone.value,
            )
            left += 14
        painter.setPen(look.TEXT_DIM)
        metrics = QFontMetricsF(DETAIL)
        text = metrics.elidedText(stat.detail, Qt.TextElideMode.ElideRight, box.right() - left)
        painter.drawText(
            QRectF(left, box.top(), box.right() - left, box.height()),
            Qt.AlignmentFlag.AlignLeft,
            text,
        )


class SummaryCard(Card):
    def __init__(self, parent: QWidget):
        super().__init__(parent, "Loop", Anchor.TOP, HEIGHT, 1)
        self.row = StatRow(self)
        self.content.addWidget(self.row, 1)

    def show_snapshot(self, snapshot: Snapshot) -> None:
        self.row.show_stats(summary(snapshot))
