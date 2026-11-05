"""The Data Library panel: every open recording and its outputs, under its Experiment."""

# Size is shown against the library's limit because a program output keeps every frame, so a long
# continuous run is what can exhaust a machine.

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt, QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from pyrpoc.core.panels import Panel
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import Recording
from pyrpoc.utils.widgets.empty_state import EmptyState
from pyrpoc.utils.widgets.table import tree_header, viewport_of

from .details import DetailsDialog
from .grouping import (
    ExperimentNode,
    SortBy,
    build_tree,
    program_display_name,
    recording_title,
    sort_tree,
)
from .model import LibraryModel

NAME, PROGRAM, TAKEN, SIZE = range(4)
COLUMNS = ["Name", "Program", "Taken", "Size"]
# The columns that sort, and the order a first click on each gives.
SORTS = {NAME: (SortBy.NAME, False), TAKEN: (SortBy.DATE, True)}
NODE_ROLE = Qt.ItemDataRole.UserRole

# 1024-based, to match the number a task manager shows.
UNITS = ("B", "KB", "MB", "GB", "TB")

Node = ExperimentNode | Recording | ProgramOutput


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


def taken(iso: str) -> str:
    """When a run started, in local time, to the minute."""
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


def total_bytes(node: Node) -> int:
    if isinstance(node, ExperimentNode):
        return sum(recording.nbytes for recording in node.recordings)
    return node.nbytes


class DataLibraryPanel(Panel):
    display_name = "Data Library"

    def __init__(self, library: LibraryModel, save_folder: Callable[[], Path | None]):
        """``save_folder`` is where Open… starts: the current save folder, if one is set."""
        super().__init__()
        self.library = library
        self.save_folder = save_folder
        # What each row shows, so a click maps back to it.
        # Each row holds its index in here.
        self.nodes: list[Node] = []
        self.output_items: dict[ProgramOutput, QTreeWidgetItem] = {}
        # Rows the user folded or unfolded away from the default, by key, kept across rebuilds.
        # Experiments start open and recordings folded, like folders of files.
        self.toggled: set[str] = set()
        self.sort_by, self.descending = SortBy.DATE, True

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 8)
        root.setSpacing(6)
        self.empty = EmptyState(
            "Nothing open",
            "Acquired data appears here as it arrives. Open… loads a saved recording or a "
            "whole folder.",
            self,
        )
        root.addWidget(self.empty)
        root.addWidget(self.build_tree_view(), 1)
        root.addStretch(0)
        root.addLayout(self.build_actions_row())

        self.library.subscribe(self.rebuild)
        self.library.auto_purge_changed.connect(self.auto_purge_check.setChecked)
        self.library.load_failed.connect(self.show_load_error)
        self.library.output_changed.connect(self.on_output_changed)
        self.rebuild()

    def build_tree_view(self) -> QTreeWidget:
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        header = tree_header(self.tree)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(NAME, QHeaderView.ResizeMode.Stretch)
        for column in (PROGRAM, TAKEN, SIZE):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self.on_header_clicked)
        self.show_sort_indicator()

        self.tree.itemSelectionChanged.connect(self.refresh_actions)
        self.tree.itemCollapsed.connect(lambda item: self.set_folded(item, True))
        self.tree.itemExpanded.connect(lambda item: self.set_folded(item, False))
        self.tree.itemDoubleClicked.connect(lambda item, _column: self.open_details(item))
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_row_menu)
        return self.tree

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
        open_menu = QMenu(open_btn)
        open_menu.addAction("Folder…", self.choose_experiment)
        open_menu.addAction("Recording…", self.choose_recording)
        open_btn.setMenu(open_menu)
        open_btn.setToolTip("Load every recording in a folder, or one recording by its _meta.json.")
        row.addWidget(open_btn)
        self.close_btn = QPushButton("Close", self)
        self.close_btn.setToolTip(
            "Drop the selection from memory: its recording, or a whole folder. "
            "Files already saved stay on disk."
        )
        self.close_btn.clicked.connect(self.close_selected)
        row.addWidget(self.close_btn)
        return row

    def rebuild(self) -> None:
        """Redraw the tree, keeping the selection and folding where the rows survive."""
        # Membership changes are rare enough for a full rebuild.
        chosen = self.selected_key()
        self.tree.clear()
        self.nodes.clear()
        self.output_items.clear()
        experiments = sort_tree(
            build_tree(self.library.recordings(), self.library.folders),
            self.sort_by,
            self.descending,
        )
        for experiment in experiments:
            self.add_experiment(experiment)
        if chosen is not None:
            for item in self.items():
                node = self.node_of(item)
                if node is not None and node_key(node) == chosen:
                    self.tree.setCurrentItem(item)
        has_rows = bool(experiments)
        self.tree.setVisible(has_rows)
        self.empty.setVisible(not has_rows)
        self.refresh_actions()

    def add_experiment(self, experiment: ExperimentNode) -> None:
        item = QTreeWidgetItem(self.tree, [experiment.label, "", "", ""])
        style = self.style()
        if style is not None:
            icon = QStyle.StandardPixmap.SP_DirIcon
            if experiment.folder is None:
                icon = QStyle.StandardPixmap.SP_FileDialogDetailedView
            item.setIcon(NAME, style.standardIcon(icon))
        font = item.font(NAME)
        font.setBold(True)
        item.setFont(NAME, font)
        item.setToolTip(
            NAME,
            "Runs not saved to disk" if experiment.folder is None else str(experiment.folder),
        )
        item.setText(TAKEN, taken(experiment.latest) if experiment.latest else "")
        self.attach(item, experiment)
        if not experiment.recordings:
            placeholder = QTreeWidgetItem(item, ["No recordings in this folder yet"])
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            font = placeholder.font(NAME)
            font.setItalic(True)
            placeholder.setFont(NAME, font)
        for recording in experiment.recordings:
            self.add_recording(item, recording)
        self.set_size(item, total_bytes(experiment))
        item.setExpanded(experiment.key not in self.toggled)

    def add_recording(self, parent: QTreeWidgetItem, recording: Recording) -> None:
        item = QTreeWidgetItem(
            parent,
            [
                recording_title(recording),
                program_display_name(recording.program_key),
                taken(recording.started_at),
            ],
        )
        where = str(recording.location) if recording.location is not None else "Not saved"
        note = recording.note
        item.setToolTip(NAME, f"{where}\n\n{note}" if note else where)
        self.attach(item, recording)
        for program_output in recording.outputs.values():
            child = QTreeWidgetItem(item, [program_output.name, program_output.type_key])
            self.attach(child, program_output)
            self.output_items[program_output] = child
            self.set_size(child, program_output.nbytes)
        self.set_size(item, total_bytes(recording))
        item.setExpanded(node_key(recording) in self.toggled)

    def attach(self, item: QTreeWidgetItem, node: Node) -> None:
        item.setData(NAME, NODE_ROLE, len(self.nodes))
        self.nodes.append(node)

    def node_of(self, item: QTreeWidgetItem) -> Node | None:
        """What a row shows; None for a placeholder row."""
        index = item.data(NAME, NODE_ROLE)
        return None if index is None else self.nodes[index]

    def items(self) -> list[QTreeWidgetItem]:
        """Every row, depth first."""
        found: list[QTreeWidgetItem] = []
        pending = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while pending:
            item = pending.pop(0)
            if item is not None:
                found.append(item)
                pending.extend(item.child(i) for i in range(item.childCount()))
        return found

    def set_size(self, item: QTreeWidgetItem, nbytes: int) -> None:
        item.setText(SIZE, format_size(nbytes))
        item.setTextAlignment(SIZE, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def set_folded(self, item: QTreeWidgetItem, folded: bool) -> None:
        node = self.node_of(item)
        if node is None or isinstance(node, ProgramOutput):
            return
        if folded != isinstance(node, Recording):
            self.toggled.add(node_key(node))
        else:
            self.toggled.discard(node_key(node))

    def on_header_clicked(self, column: int) -> None:
        """Sort by name or by date; a second click on the same column reverses it."""
        if column in SORTS:
            by, first_descending = SORTS[column]
            self.descending = not self.descending if by is self.sort_by else first_descending
            self.sort_by = by
            self.rebuild()
        self.show_sort_indicator()

    def show_sort_indicator(self) -> None:
        column = next(col for col, (by, _) in SORTS.items() if by is self.sort_by)
        order = Qt.SortOrder.DescendingOrder if self.descending else Qt.SortOrder.AscendingOrder
        tree_header(self.tree).setSortIndicator(column, order)

    def on_output_changed(self, program_output: ProgramOutput) -> None:
        """Update one row's size and the total, not the whole tree: a running acquisition
        appends several times a second, and a rebuild would flicker."""
        item = self.output_items.get(program_output)
        if item is None:
            return
        self.set_size(item, program_output.nbytes)
        parent = item.parent()
        while parent is not None:
            node = self.node_of(parent)
            if node is not None:
                self.set_size(parent, total_bytes(node))
            parent = parent.parent()
        self.refresh_total()

    def refresh_total(self) -> None:
        library = self.library
        self.total_label.setText(
            f"{format_size(library.nbytes)} of {format_size(library.limit_bytes)}"
        )
        # The accent, so the warning follows the theme like everything else.
        warning = "color: palette(highlight); font-weight: bold;"
        self.total_label.setStyleSheet(warning if library.over_limit else "color: palette(mid);")

    def selected_node(self) -> Node | None:
        items = self.tree.selectedItems()
        return self.node_of(items[0]) if len(items) == 1 else None

    def selected_key(self) -> str | None:
        node = self.selected_node()
        return None if node is None else node_key(node)

    def close_selected(self) -> None:
        node = self.selected_node()
        if node is not None:
            self.close_node(node)

    def close_node(self, node: Node) -> None:
        """Close what a row shows: a folder, every unsaved recording, or one recording, which
        closes with all its outputs."""
        if isinstance(node, ExperimentNode):
            if node.folder is not None:
                self.library.close_experiment(node.folder)
                return
            for recording in node.recordings:
                self.library.close(recording)
            return
        self.library.close(node if isinstance(node, Recording) else node.recording)

    def show_row_menu(self, position: QPoint) -> None:
        item = self.tree.itemAt(position)
        node = self.node_of(item) if item is not None else None
        if item is None or node is None:
            return
        self.tree.setCurrentItem(item)
        menu = QMenu(self)
        if not isinstance(node, ExperimentNode):
            menu.addAction("Details…", lambda: self.open_details(item))
        folder = folder_of(node)
        show = QAction("Show in folder", menu)
        show.setEnabled(folder is not None)
        show.triggered.connect(lambda: self.show_in_folder(folder))
        menu.addAction(show)
        menu.addSeparator()
        menu.addAction(close_label(node), lambda: self.close_node(node))
        menu.exec(viewport_of(self.tree).mapToGlobal(position))

    def open_details(self, item: QTreeWidgetItem) -> None:
        node = self.node_of(item)
        if node is None or isinstance(node, ExperimentNode):
            return
        recording = node if isinstance(node, Recording) else node.recording
        dialog = DetailsDialog(recording, lambda note: self.library.set_note(recording, note), self)
        if dialog.exec():
            # The note shows in the recording's tooltip.
            self.rebuild()

    def show_in_folder(self, folder: Path | None) -> None:
        if folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def start_folder(self) -> str:
        folder = self.save_folder()
        return str(folder) if folder is not None else ""

    def choose_recording(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", self.start_folder(), "pyrpoc recordings (*_meta.json)"
        )
        if path:
            self.library.load(Path(path))

    def choose_experiment(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open folder", self.start_folder())
        if folder:
            self.library.load_experiment(Path(folder))

    def show_load_error(self, message: str) -> None:
        QMessageBox.warning(self, "Could Not Open Recording", message)

    def refresh_actions(self) -> None:
        """An empty panel shows the hint and what can fill it, with no Close
        button to grey out."""
        has_rows = bool(self.nodes)
        self.refresh_total()
        self.total_label.setVisible(has_rows)
        self.close_btn.setVisible(has_rows)
        self.close_btn.setEnabled(self.selected_node() is not None)


def node_key(node: Node) -> str:
    """What names a row across rebuilds: its folder, or the open object itself."""
    # An open recording or output is the same object for as long as it is open.
    return node.key if isinstance(node, ExperimentNode) else str(id(node))


def folder_of(node: Node) -> Path | None:
    """The Experiment folder a row's files sit in, if they are on disk."""
    if isinstance(node, ExperimentNode):
        return node.folder
    return node.folder if isinstance(node, Recording) else node.recording.folder


def close_label(node: Node) -> str:
    if isinstance(node, ExperimentNode):
        return "Close all unsaved" if node.folder is None else "Close folder"
    return "Close recording"
