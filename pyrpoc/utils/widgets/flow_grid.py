"""``FlowGrid``: a layout that puts as many equal-width cells in a row as the width holds,
and shrinks to one column instead of forcing its parent wider."""

from __future__ import annotations

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem, QWidget


class FlowGrid(QLayout):
    """Cells at least ``min_cell`` wide, in as many columns as fit. A wide cell takes a whole
    row. Taller rows follow from the width, so it reports a height for any width."""

    def __init__(self, parent: QWidget, min_cell: int, spacing: int):
        super().__init__(parent)
        self.min_cell = min_cell
        self.gap = spacing
        self.items: list[QLayoutItem] = []
        self.wide: set[int] = set()

    def add(self, widget: QWidget, *, wide: bool) -> None:
        if wide:
            self.wide.add(len(self.items))
        self.addWidget(widget)

    def addItem(self, a0: QLayoutItem | None) -> None:
        if a0 is not None:
            self.items.append(a0)

    def count(self) -> int:
        return len(self.items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        if not 0 <= index < len(self.items):
            return None
        self.wide = {i if i < index else i - 1 for i in self.wide if i != index}
        return self.items.pop(index)

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, a0: int) -> int:
        margins = self.contentsMargins()
        inner = a0 - margins.left() - margins.right()
        return self.arrange(QRect(0, 0, inner, 0), apply=False) + margins.top() + margins.bottom()

    def setGeometry(self, a0: QRect) -> None:
        super().setGeometry(a0)
        margins = self.contentsMargins()
        self.arrange(a0.adjusted(margins.left(), margins.top(), -margins.right(), 0), apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        margins = self.contentsMargins()
        narrowest = max((item.minimumSize().width() for item in self.items), default=0)
        width = narrowest + margins.left() + margins.right()
        return QSize(width, self.heightForWidth(width))

    def columns(self, width: int) -> int:
        return max(1, (width + self.gap) // (self.min_cell + self.gap))

    def arrange(self, rect: QRect, *, apply: bool) -> int:
        """Place the cells in ``rect`` (when ``apply``) and return the height they take."""
        columns = self.columns(rect.width())
        cell = (rect.width() - (columns - 1) * self.gap) // columns
        y = 0
        row: list[QLayoutItem] = []
        for index, item in enumerate(self.items):
            if index in self.wide:
                y = self.place_row(row, rect, y, cell, apply)
                row = []
                y = self.place_row([item], rect, y, rect.width(), apply)
                continue
            row.append(item)
            if len(row) == columns:
                y = self.place_row(row, rect, y, cell, apply)
                row = []
        y = self.place_row(row, rect, y, cell, apply)
        return max(0, y - self.gap)

    def place_row(
        self, row: list[QLayoutItem], rect: QRect, y: int, width: int, apply: bool
    ) -> int:
        if not row:
            return y
        height = max(
            item.heightForWidth(width) if item.hasHeightForWidth() else item.sizeHint().height()
            for item in row
        )
        if apply:
            for column, item in enumerate(row):
                item.setGeometry(
                    QRect(rect.x() + column * (width + self.gap), rect.y() + y, width, height)
                )
        return y + height + self.gap
