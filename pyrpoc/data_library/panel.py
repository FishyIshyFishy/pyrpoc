"""The data library panel: every dataset this session has open.

Size is shown against the library's limit because a dataset keeps every
frame, so a long continuous run is what can exhaust a machine. Past the limit,
auto-purge closes the oldest finished entries; with it off, new acquisitions
are refused until entries are closed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt, QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from pyrpoc.qt_components.table import ListTable, horizontal_header, viewport_of
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.plugins.data_panels import Panel

from .details import DetailsDialog
from .model import LibraryModel

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

    def __init__(self, library: LibraryModel, save_folder: Callable[[], Path]):
        """``save_folder`` is where Open… starts: the current save folder."""
        super().__init__()
        self.library = library
        self.save_folder = save_folder
        # Datasets in table order, so a row number maps back to a dataset.
        self.rows: list[Dataset] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 8)
        root.setSpacing(6)
        self.empty_label = QLabel(
            "Nothing open. Acquired data appears here as it arrives; Open… loads a saved "
            "recording.",
            self,
        )
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
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_row_menu)
        self.table.cellDoubleClicked.connect(lambda row, _column: self.show_details(self.rows[row]))
        self.library.subscribe(self.rebuild)
        self.library.auto_purge_changed.connect(self.auto_purge_check.setChecked)
        self.library.load_failed.connect(self.show_load_error)
        self.library.dataset_changed.connect(self.on_dataset_changed)
        self.rebuild()

    def build_actions_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.total_label = QLabel("", self)
        self.total_label.setToolTip("Memory held by every open entry, against the library's limit.")
        row.addWidget(self.total_label)
        self.auto_purge_check = QCheckBox("Auto-purge oldest", self)
        self.auto_purge_check.setToolTip(
            "Over the limit, close the oldest finished entries until back under it. "
            "Entries still recording and masks you drew are never closed."
        )
        self.auto_purge_check.setChecked(self.library.auto_purge)
        self.auto_purge_check.toggled.connect(self.library.set_auto_purge)
        row.addWidget(self.auto_purge_check)
        row.addStretch(1)
        open_btn = QPushButton("Open…", self)
        open_btn.setToolTip("Load a saved recording by its _meta.json file.")
        open_btn.clicked.connect(self.choose_recording)
        row.addWidget(open_btn)
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
        self.rows = list(reversed(self.library.all()))
        self.table.setRowCount(len(self.rows))
        for row, dataset in enumerate(self.rows):
            self.table.set_cell(row, TIME, dataset.started_time)
            name = self.table.set_cell(row, NAME, dataset.name)
            tooltip = f"{dataset.name} · {dataset.spec.name}"
            name.setToolTip(f"{tooltip}\n\n{dataset.notes}" if dataset.notes else tooltip)
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
        library = self.library
        self.total_label.setText(
            f"{format_size(library.nbytes)} of {format_size(library.limit_bytes)}"
        )
        # The accent, so the warning follows the theme like everything else.
        warning = "color: palette(highlight); font-weight: bold;"
        self.total_label.setStyleSheet(warning if library.over_limit else "color: palette(mid);")

    def selected_dataset(self) -> Dataset | None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        return self.rows[rows.pop()] if len(rows) == 1 else None

    def close_selected(self) -> None:
        dataset = self.selected_dataset()
        if dataset is not None:
            self.library.close(dataset)

    def show_row_menu(self, position: QPoint) -> None:
        row = self.table.rowAt(position.y())
        if row < 0:
            return
        self.table.selectRow(row)
        dataset = self.rows[row]
        menu = QMenu(self)
        menu.addAction("Details…", lambda: self.show_details(dataset))
        folder = QAction("Show in folder", menu)
        folder.setEnabled(dataset.meta_path is not None)
        folder.triggered.connect(lambda: self.show_in_folder(dataset))
        menu.addAction(folder)
        menu.addSeparator()
        menu.addAction("Close", lambda: self.library.close(dataset))
        menu.exec(viewport_of(self.table).mapToGlobal(position))

    def show_details(self, dataset: Dataset) -> None:
        dialog = DetailsDialog(dataset, lambda notes: self.library.set_notes(dataset, notes), self)
        if dialog.exec() and dataset in self.rows:
            # The notes show in the name's tooltip.
            self.rebuild()

    def show_in_folder(self, dataset: Dataset) -> None:
        if dataset.meta_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(dataset.meta_path.parent)))

    def choose_recording(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open recording",
            str(self.save_folder()),
            "pyrpoc recordings (*_meta.json)",
        )
        if path:
            self.library.load(Path(path))

    def show_load_error(self, message: str) -> None:
        QMessageBox.warning(self, "Could Not Open Recording", message)

    def refresh_actions(self) -> None:
        """An empty panel shows the hint and what can fill it, with no Close
        button to grey out."""
        self.refresh_total()
        self.total_label.setVisible(bool(self.rows))
        self.close_btn.setVisible(bool(self.rows))
        self.close_btn.setEnabled(self.selected_dataset() is not None)
