"""``UnitsTable``: one row per unit, its group editable in place."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QWidget

from pyrpoc.plugins.program_io.outputs import UnitState
from pyrpoc.utils.widgets.table import horizontal_header, vertical_header

from .shown import ShownUnit

HEADINGS = ("Site", "Group", "Status", "Next visit")
# What each state is called on screen.
STATE_WORDS = {
    UnitState.WAITING: "Waiting",
    UnitState.PAUSED: "Paused",
    UnitState.LOST: "Lost track",
    UnitState.DONE: "Finished",
    UnitState.REMOVED: "Removed",
}


def next_text(unit: ShownUnit, elapsed_s: float) -> str:
    """When a unit is next due, as time from now."""
    if unit.next_due_s is None:
        return ""
    wait = unit.next_due_s - elapsed_s
    return "now" if wait <= 0 else f"in {wait:.0f} s"


class UnitsTable(QTableWidget):
    """Lists the units and reports a group edited or a row chosen."""

    regrouped = pyqtSignal(str, str)  # a unit's key and its new group
    chosen = pyqtSignal(object)  # a unit's key, or None

    def __init__(self, parent: QWidget):
        super().__init__(0, len(HEADINGS), parent)
        self.setHorizontalHeaderLabels(HEADINGS)
        vertical_header(self).setVisible(False)
        header = horizontal_header(self)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.keys: list[str] = []
        self.itemChanged.connect(self.on_item_changed)
        self.itemSelectionChanged.connect(self.on_selection_changed)

    def show_units(
        self, units: tuple[ShownUnit, ...], chosen: str | None, editable: bool, elapsed_s: float
    ) -> None:
        """One row per unit, with what a loop running on it is doing, if one is."""
        self.blockSignals(True)
        self.keys = [unit.key for unit in units]
        self.setRowCount(len(units))
        for row, unit in enumerate(units):
            group = QTableWidgetItem(unit.group)
            if not editable:
                group = self.fixed(unit.group)
            self.setItem(row, 0, self.fixed(str(unit.number)))
            self.setItem(row, 1, group)
            self.setItem(row, 2, self.fixed("" if unit.state is None else STATE_WORDS[unit.state]))
            self.setItem(row, 3, self.fixed(next_text(unit, elapsed_s)))
        self.clearSelection()
        if chosen in self.keys:
            self.selectRow(self.keys.index(chosen))
        self.blockSignals(False)

    @staticmethod
    def fixed(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 1:
            self.regrouped.emit(self.keys[item.row()], item.text().strip())

    def on_selection_changed(self) -> None:
        rows = {index.row() for index in self.selectedIndexes()}
        self.chosen.emit(self.keys[min(rows)] if rows else None)
