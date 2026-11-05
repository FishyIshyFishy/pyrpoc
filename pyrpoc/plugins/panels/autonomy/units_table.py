"""``UnitsTable``: one row per unit, its group editable in place."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QWidget

from pyrpoc.plugins.program_io.inputs import Unit
from pyrpoc.utils.widgets.table import horizontal_header, vertical_header

HEADINGS = ("Unit", "Group", "Centre (µm)")


class UnitsTable(QTableWidget):
    """Lists the units and reports a group edited or a row chosen."""

    regrouped = pyqtSignal(int, str)  # a unit's id and its new group
    chosen = pyqtSignal(object)  # a unit's id, or None

    def __init__(self, parent: QWidget):
        super().__init__(0, len(HEADINGS), parent)
        self.setHorizontalHeaderLabels(HEADINGS)
        vertical_header(self).setVisible(False)
        header = horizontal_header(self)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ids: list[int] = []
        self.itemChanged.connect(self.on_item_changed)
        self.itemSelectionChanged.connect(self.on_selection_changed)

    def show_units(self, units: tuple[Unit, ...], chosen: int | None) -> None:
        self.blockSignals(True)
        self.ids = [unit.id for unit in units]
        self.setRowCount(len(units))
        for row, unit in enumerate(units):
            x, y = unit.centre
            self.setItem(row, 0, self.fixed(str(unit.id)))
            self.setItem(row, 1, QTableWidgetItem(unit.group))
            self.setItem(row, 2, self.fixed(f"{x:.1f}, {y:.1f}"))
        self.clearSelection()
        if chosen in self.ids:
            self.selectRow(self.ids.index(chosen))
        self.blockSignals(False)

    @staticmethod
    def fixed(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 1:
            self.regrouped.emit(self.ids[item.row()], item.text().strip())

    def on_selection_changed(self) -> None:
        rows = {index.row() for index in self.selectedIndexes()}
        self.chosen.emit(self.ids[min(rows)] if rows else None)
