"""The data library panel: every dataset this session has open.

Size is shown because a dataset keeps every frame, so a long continuous run is
what can exhaust a machine; this makes that visible while it happens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QHeaderView, QLabel, QPushButton, QVBoxLayout

from pyrpoc.structs.data import Dataset
from pyrpoc.structs.panel import Panel

from ..components.table import ListTable, horizontal_header

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.app.model.application import Application

TIME, NAME, OUTPUT, SIZE = range(4)
COLUMNS = ["Time", "Name", "Output", "Size"]

# 1024-based, to match the number a task manager shows.
UNITS = ("B", "KB", "MB", "GB", "TB")


def format_size(nbytes: int) -> str:
    """Bytes in at most three significant figures, for a narrow column."""
    if nbytes <= 0:
        return "-"
    size = float(nbytes)
    unit = 0
    while size >= 1024.0 and unit < len(UNITS) - 1:
        size /= 1024.0
        unit += 1
    if unit == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {UNITS[unit]}" if size < 10.0 else f"{size:.0f} {UNITS[unit]}"


class DataLibraryPanel(Panel):
    display_name = "Data Library"

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        # Datasets in table order, so a row number maps back to a dataset.
        self.rows: list[Dataset] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 8)
        root.setSpacing(6)
        self.empty_label = QLabel("No acquisitions yet. Data appears here as it arrives.", self)
        self.empty_label.setStyleSheet("color: palette(mid); font-style: italic;")
        self.empty_label.setWordWrap(True)
        # Nothing expands once the table is hidden, so pin the label to the top.
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.empty_label)

        self.table = ListTable(COLUMNS, self)
        header = horizontal_header(self.table)
        for column in (TIME, OUTPUT, SIZE):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(NAME, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)
        root.addLayout(self.build_actions_row())

        self.table.itemSelectionChanged.connect(self.refresh_actions)
        self.app.library.subscribe(self.rebuild)
        self.app.runs.dataset_changed.connect(self.on_dataset_changed)
        self.rebuild()

    def build_actions_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.total_label = QLabel("", self)
        self.total_label.setToolTip("Memory held by every open acquisition together.")
        self.total_label.setStyleSheet("color: palette(mid);")
        row.addWidget(self.total_label)
        row.addStretch(1)
        self.close_btn = QPushButton("Close", self)
        self.close_btn.setToolTip(
            "Drop the selected acquisition from memory. Files already saved stay on disk."
        )
        self.close_btn.clicked.connect(self.close_selected)
        row.addWidget(self.close_btn)
        return row

    def rebuild(self) -> None:
        """Redraw every row, keeping the selection where the row survives.
        Membership changes are rare enough for a full rebuild."""
        chosen = self.selected_dataset()
        self.rows = list(reversed(self.app.library.all()))
        self.table.setRowCount(len(self.rows))
        for row, dataset in enumerate(self.rows):
            self.table.set_cell(row, TIME, dataset.started_time)
            name = self.table.set_cell(row, NAME, dataset.name)
            name.setToolTip(f"{dataset.name} · {dataset.spec.name}")
            self.table.set_cell(row, OUTPUT, dataset.output)
            self.table.set_cell(row, SIZE, format_size(dataset.nbytes), right=True)

        self.table.setVisible(bool(self.rows))
        self.empty_label.setVisible(not self.rows)
        if chosen in self.rows:
            self.table.selectRow(self.rows.index(chosen))
        self.refresh_actions()

    def on_dataset_changed(self, dataset: Dataset) -> None:
        """Update one cell and the total, not the whole table: a running
        acquisition appends several times a second, and a rebuild would drop the selection."""
        if dataset in self.rows:
            row = self.rows.index(dataset)
            self.table.set_cell(row, SIZE, format_size(dataset.nbytes), right=True)
            self.refresh_total()

    def refresh_total(self) -> None:
        total = self.app.library.nbytes
        self.total_label.setText(f"{format_size(total)} in memory" if total else "")

    def selected_dataset(self) -> Dataset | None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        return self.rows[rows.pop()] if len(rows) == 1 else None

    def close_selected(self) -> None:
        dataset = self.selected_dataset()
        if dataset is not None:
            self.app.library.close(dataset)

    def refresh_actions(self) -> None:
        """An empty panel shows only the hint, with no button to grey out."""
        self.refresh_total()
        self.total_label.setVisible(bool(self.rows))
        self.close_btn.setVisible(bool(self.rows))
        self.close_btn.setEnabled(self.selected_dataset() is not None)
