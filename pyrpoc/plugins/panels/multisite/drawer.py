"""``Drawer``: the queue and the units beside the dish, opened and closed with Tab.
Both name units by key, so choosing in one chooses in the other and on the dish."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QHeaderView, QLabel, QSplitter, QVBoxLayout, QWidget

from pyrpoc.utils.widgets.table import ListTable, horizontal_header, viewport_of

from .state.pending import Pending
from .state.standing import QueueRow, Snapshot, Standing, UnitRow
from .state.wording import MEANING, next_text, standing_text, when_text

QUEUE_HELP = (
    "What the loop runs, top first, one routine on one unit at a time. A unit asked to run "
    "next goes to the front; the rest run as they fall due. Paused units are not queued."
)
UNITS_HELP = (
    "Every unit and what it does next. Right-click a unit here, in the queue or on the dish "
    "to run it next, pause it or remove it. Each acts on all of that unit's routines."
)
# Narrower than this, the tables' columns no longer fit their words.
MIN_WIDTH = 320
# Rows of units that have nothing to run are shown muted.
QUIET = (Standing.PAUSED, Standing.DONE, Standing.REFUSED)


class UnitTable(ListTable):
    """A list whose rows each name one unit by key."""

    chose = pyqtSignal(object)
    menu_asked = pyqtSignal(str, QPoint)
    locate = pyqtSignal(str)

    def __init__(self, columns: list[str], parent: QWidget):
        super().__init__(columns, parent)
        header = horizontal_header(self)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.ask_menu)
        self.itemSelectionChanged.connect(self.on_selected)
        self.cellDoubleClicked.connect(lambda row, _column: self.locate.emit(self.keys[row]))
        self.keys: list[str] = []
        self.chosen: str | None = None

    def fill(self, rows: Sequence[tuple[str, list[str], bool, str]]) -> None:
        """Each row's unit key, cells, whether it is muted, and its tooltip; the choice stays
        on the same unit."""
        self.blockSignals(True)
        self.setRowCount(len(rows))
        muted = self.palette().color(QPalette.ColorRole.Mid)
        plain = self.palette().color(QPalette.ColorRole.Text)
        for index, (_key, cells, quiet, tip) in enumerate(rows):
            for column, text in enumerate(cells):
                item = self.set_cell(index, column, text)
                item.setForeground(muted if quiet else plain)
                item.setToolTip(tip)
        self.keys = [key for key, _cells, _quiet, _tip in rows]
        self.show_choice()
        self.blockSignals(False)

    def choose(self, key: str | None) -> None:
        self.chosen = key
        self.blockSignals(True)
        self.show_choice()
        self.blockSignals(False)

    def show_choice(self) -> None:
        if self.chosen in self.keys:
            self.selectRow(self.keys.index(self.chosen))
        else:
            self.clearSelection()

    def on_selected(self) -> None:
        rows = {index.row() for index in self.selectedIndexes()}
        self.chosen = self.keys[min(rows)] if rows else None
        self.chose.emit(self.chosen)

    def ask_menu(self, at: QPoint) -> None:
        row = self.rowAt(at.y())
        if 0 <= row < len(self.keys):
            self.selectRow(row)
            self.menu_asked.emit(self.keys[row], viewport_of(self).mapToGlobal(at))


def queue_cells(row: QueueRow, snapshot: Snapshot) -> list[str]:
    place = "now" if row.place == 0 else str(row.place)
    return [place, f"Unit {row.unit}", row.routine, when_text(row, snapshot)]


def unit_cells(row: UnitRow, snapshot: Snapshot, pending: Pending) -> list[str]:
    change = None if row.number is None else pending.of(row.number)
    group = row.group or "—"
    return [row.name, group, standing_text(row, change), next_text(row, snapshot)]


def section(title: str, help_text: str, table: UnitTable, parent: QWidget) -> QWidget:
    box = QWidget(parent)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    heading = QLabel(title, box)
    face = heading.font()
    face.setBold(True)
    heading.setFont(face)
    caption = QLabel(help_text, box)
    caption.setWordWrap(True)
    caption.setStyleSheet("color: palette(mid);")
    for widget in (heading, caption, table):
        layout.addWidget(widget)
    layout.setStretchFactor(table, 1)
    return box


class Drawer(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setMinimumWidth(MIN_WIDTH)
        self.queue = UnitTable(["#", "Unit", "Routine", "When"], self)
        self.units = UnitTable(["Unit", "Group", "State", "Next"], self)
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(section("Queue", QUEUE_HELP, self.queue, splitter))
        splitter.addWidget(section("Units", UNITS_HELP, self.units, splitter))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.addWidget(splitter)

    def show_loop(
        self,
        snapshot: Snapshot,
        queue: tuple[QueueRow, ...],
        units: tuple[UnitRow, ...],
        pending: Pending,
    ) -> None:
        self.queue.fill([(row.key, queue_cells(row, snapshot), False, "") for row in queue])
        self.units.fill(
            [
                (
                    row.key,
                    unit_cells(row, snapshot, pending),
                    row.standing in QUIET,
                    MEANING[row.standing],
                )
                for row in units
            ]
        )

    def choose(self, key: str | None) -> None:
        self.queue.choose(key)
        self.units.choose(key)
