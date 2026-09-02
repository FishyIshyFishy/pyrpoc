"""The data panel: every acquisition this session has open.

Shares the Views dock rather than holding one of its own. What data exists and
what is drawing it are two halves of one question -- take an image, take three
more, decide which one a display is showing -- and behind separate tabs you
could only ever see one half at a time.

A table, not the row of concatenated text it replaces. Acquisitions differ in
four small ways, and four narrow columns let one be picked out at a glance
where "simulation #3 · intensity  -  10 frames  -  2D Image" had to be read.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.data.dataset import Dataset

from .app import Application

TIME, NAME, STREAM, FRAMES = range(4)
COLUMNS = ["Time", "Name", "Stream", "Frames"]


class DataPanel(QWidget):
    def __init__(self, app: Application, parent: QWidget | None = None):
        super().__init__(parent)
        self.app = app
        #: Datasets in table order, so a row number maps back to a dataset.
        self.rows: list[Dataset] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 8)
        root.setSpacing(6)

        self.empty_label = QLabel("No acquisitions yet. Data appears here as it arrives.", self)
        self.empty_label.setStyleSheet("color: palette(mid); font-style: italic;")
        self.empty_label.setWordWrap(True)
        root.addWidget(self.empty_label)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # Grid lines and row numbers draw a box around every cell in a table
        # whose whole job is to be skimmed. Alternating bands separate the rows
        # with no ink of their own.
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setCornerButtonEnabled(False)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setHighlightSections(False)
        for column in (TIME, STREAM, FRAMES):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(NAME, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addStretch(1)
        self.close_btn = QPushButton("Close", self)
        self.close_btn.setToolTip(
            "Drop the selected acquisition from memory. Files already saved stay on disk."
        )
        actions.addWidget(self.close_btn)
        root.addLayout(actions)

        self.close_btn.clicked.connect(self.close_selected)
        self.table.itemSelectionChanged.connect(self.refresh_actions)
        self.app.library.subscribe(self.rebuild)
        self.app.bridge.dataset_changed.connect(self.on_dataset_changed)
        self.rebuild()

    # -- rows ---------------------------------------------------------------- #

    def rebuild(self) -> None:
        """Redraw every row, keeping the selection where the row survives.

        Membership changes are rare -- one per run, one per close -- which is
        why this can afford to be a full rebuild and ``on_dataset_changed``
        cannot.
        """
        chosen = self.selected_dataset()
        self.rows = list(reversed(self.app.library.all()))
        self.table.setRowCount(len(self.rows))
        for row, dataset in enumerate(self.rows):
            self.set_cell(row, TIME, dataset.started_time or "-")
            self.set_cell(row, NAME, dataset.name)
            self.set_cell(row, STREAM, dataset.stream)
            self.set_cell(row, FRAMES, str(len(dataset)), right=True)
            self.table.item(row, NAME).setToolTip(f"{dataset.name} · {dataset.spec.name}")

        self.table.setVisible(bool(self.rows))
        self.empty_label.setVisible(not self.rows)
        if chosen is not None and chosen in self.rows:
            self.table.selectRow(self.rows.index(chosen))
        self.refresh_actions()

    def on_dataset_changed(self, dataset: Dataset, index: int) -> None:
        """One cell, not the whole table.

        A continuous run appends several frames a second, and rebuilding on
        each of them would drop the selection out from under whoever is
        clicking.
        """
        del index
        for row, existing in enumerate(self.rows):
            if existing is dataset:
                self.set_cell(row, FRAMES, str(len(dataset)), right=True)
                return

    def set_cell(self, row: int, column: int, text: str, *, right: bool = False) -> None:
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, column, item)
        item.setText(text)
        if right:
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    # -- the selection -------------------------------------------------------- #

    def selected_dataset(self) -> Dataset | None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def close_selected(self) -> None:
        dataset = self.selected_dataset()
        if dataset is not None:
            self.app.bridge.release(dataset)

    def refresh_actions(self) -> None:
        self.close_btn.setEnabled(self.selected_dataset() is not None)
