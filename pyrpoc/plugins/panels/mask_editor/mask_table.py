"""``MaskTable``: the drawn masks, one row each, with the one being drawn selected."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHeaderView, QSizePolicy, QTableWidget, QTableWidgetItem, QWidget

from pyrpoc.core.mask import Mask
from pyrpoc.shared.widgets.table import horizontal_header, vertical_header


class MaskTable(QTableWidget):
    """A name and a size per mask. Selecting a row chooses the mask to draw on; editing a
    name renames it."""

    # The mask chosen to draw on, by id, or "" for none.
    chosen = pyqtSignal(str)
    renamed = pyqtSignal(str, str)

    def __init__(self, parent: QWidget):
        super().__init__(0, 2, parent)
        self.ids: list[str] = []
        self.setHorizontalHeaderLabels(["Mask", "Size"])
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
        horizontal_header(self).setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        horizontal_header(self).setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        vertical_header(self).setVisible(False)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.itemChanged.connect(self.on_item_changed)

    def show_masks(self, masks: tuple[Mask, ...], current: str) -> None:
        """List ``masks``, selecting the one whose id is ``current``."""
        self.blockSignals(True)
        self.ids = [mask.id for mask in masks]
        self.setRowCount(len(masks))
        for row, mask in enumerate(masks):
            name = QTableWidgetItem(mask.label)
            name.setToolTip("Double-click to rename")
            self.setItem(row, 0, name)
            height, width = mask.array.shape
            size = QTableWidgetItem(f"{width} × {height}")
            size.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.setItem(row, 1, size)
        self.clearSelection()
        if current in self.ids:
            self.selectRow(self.ids.index(current))
        self.fit_height()
        self.blockSignals(False)

    def fit_height(self) -> None:
        """Size the table to its rows, up to four."""
        header = horizontal_header(self).height()
        row_height = vertical_header(self).defaultSectionSize()
        self.setFixedHeight(header + min(max(self.rowCount(), 1), 4) * row_height + 4)

    def on_selection_changed(self) -> None:
        rows = self.selectionModel()
        selected = rows.selectedRows() if rows is not None else []
        self.chosen.emit(self.ids[selected[0].row()] if selected else "")

    def on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self.renamed.emit(self.ids[item.row()], item.text().strip())
