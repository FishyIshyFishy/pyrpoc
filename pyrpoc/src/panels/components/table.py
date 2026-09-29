"""A read-only, skimmable list table.

Pulled out of the data library panel, whose table made four decisions that
have nothing to do with what a run library specifically needs: no grid, no
in-place editing, alternating rows, one row selected at a time. The next panel
that lists records rather than editing them starts from those decisions
instead of re-making them -- columns, resize modes and cell content stay the
caller's.

Not what the acquisition form's mask table is built from: that table is
edited in place, with a combo box and two spin boxes live in every row, which
is a different kind of table doing a different job.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


# PyQt types a table's headers as optional, though a table builds both in its
# constructor. These say so once, rather than every caller checking for None.
def horizontal_header(table: QTableView) -> QHeaderView:
    header = table.horizontalHeader()
    if header is None:
        raise RuntimeError("table has no horizontal header")
    return header


def vertical_header(table: QTableView) -> QHeaderView:
    header = table.verticalHeader()
    if header is None:
        raise RuntimeError("table has no vertical header")
    return header


class ListTable(QTableWidget):
    """A ``QTableWidget`` preconfigured as a read-only, single-selection list."""

    def __init__(self, columns: list[str], parent: QWidget | None = None):
        super().__init__(0, len(columns), parent)
        self.setHorizontalHeaderLabels(columns)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # Grid lines and row numbers draw a box around every cell in a table
        # whose whole job is to be skimmed. Alternating bands separate the
        # rows with no ink of their own.
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setCornerButtonEnabled(False)
        vertical_header(self).setVisible(False)
        horizontal_header(self).setHighlightSections(False)

    def set_cell(
        self, row: int, column: int, text: str, *, right: bool = False
    ) -> QTableWidgetItem:
        """Set a cell's text, creating its item if needed, and return the item."""
        item = self.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.setItem(row, column, item)
        item.setText(text)
        if right:
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item
