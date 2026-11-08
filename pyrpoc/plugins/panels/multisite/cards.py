"""The overlay's glass cards: the queue and the units, each a list whose rows name one unit by
key, so choosing in one chooses in the other and on the dish."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPalette, QPen
from PyQt6.QtWidgets import QFrame, QHeaderView, QLabel, QVBoxLayout, QWidget

from pyrpoc.utils.widgets.table import ListTable, horizontal_header, viewport_of

from . import look
from .state.pending import Pending
from .state.standing import QueueRow, Snapshot, Standing, UnitRow
from .state.wording import MEANING, next_text, standing_text, when_text

QUEUE_HELP = (
    "What the loop runs, top first, one routine on one unit at a time. A unit asked to run "
    "next goes to the front; the rest run as they fall due. Paused units are not queued."
)
UNITS_HELP = (
    "Every unit and what it does next. Right-click a unit here or on the dish to run it next, "
    "pause it or remove it. Each acts on all of that unit's routines."
)
RADIUS = 12.0
# Rows of units that have nothing to run are shown muted.
QUIET = (Standing.PAUSED, Standing.DONE, Standing.REFUSED)


def glass_sheet(accent: QColor) -> str:
    """The table on glass: no fill of its own, light text, the chosen row in the accent."""
    chosen = f"rgba({accent.red()}, {accent.green()}, {accent.blue()}, 90)"
    return (
        "QTableWidget { background: transparent; border: none; "
        f"color: {look.TEXT.name()}; selection-background-color: {chosen}; "
        f"selection-color: {look.TEXT.name()}; }}"
        "QTableWidget::item { padding: 2px 6px; border: none; }"
        f"QTableWidget::item:selected {{ background: {chosen}; }}"
        "QHeaderView, QHeaderView::section { background: transparent; border: none; "
        f"color: {look.TEXT_DIM.name()}; padding: 2px 6px; font-weight: bold; }}"
        "QTableCornerButton::section { background: transparent; border: none; }"
    )


class UnitTable(ListTable):
    """A list on glass whose rows each name one unit by key."""

    chose = pyqtSignal(object)
    menu_asked = pyqtSignal(str, QPoint)
    locate = pyqtSignal(str)

    def __init__(self, columns: list[str], parent: QWidget):
        super().__init__(columns, parent)
        self.setAlternatingRowColors(False)
        self.setStyleSheet(glass_sheet(self.palette().color(QPalette.ColorRole.Highlight)))
        viewport_of(self).setAutoFillBackground(False)
        header = horizontal_header(self)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
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
        for index, (_key, cells, quiet, tip) in enumerate(rows):
            for column, text in enumerate(cells):
                item = self.set_cell(index, column, text)
                item.setForeground(look.TEXT_DIM if quiet else look.TEXT)
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


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


class GlassCard(QFrame):
    """A see-through card on one side of the overlay: a heading, a line saying what it shows
    and how to act on it, and its table."""

    def __init__(self, parent: QWidget, title: str, help_text: str, table: UnitTable, side: Side):
        super().__init__(parent)
        self.side, self.table = side, table
        # The theme fills every widget; on glass only the card itself is drawn.
        self.setObjectName("glassCard")
        self.setStyleSheet(
            "#glassCard { background: transparent; border: none; }"
            "#glassCard QLabel, #glassCard QScrollBar { background: transparent; }"
        )
        table.setParent(self)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(6)
        heading = QLabel(title.upper(), self)
        face = heading.font()
        face.setBold(True)
        face.setPointSizeF(face.pointSizeF() * 0.9)
        heading.setFont(face)
        heading.setStyleSheet(f"color: {look.TEXT_DIM.name()}; letter-spacing: 1.5px;")
        caption = QLabel(help_text, self)
        caption.setWordWrap(True)
        caption.setStyleSheet(f"color: {look.TEXT_DIM.name()};")
        for widget in (heading, caption, table):
            root.addWidget(widget)
        root.setStretchFactor(table, 1)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(look.GLASS_EDGE, 1.0))
        painter.setBrush(look.GLASS)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
        painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        # A click on a card's glass is the card's: it neither reaches the dish nor closes the
        # overlay.
        if event is not None:
            event.accept()


def queue_cells(row: QueueRow, snapshot: Snapshot) -> list[str]:
    place = "now" if row.place == 0 else str(row.place)
    return [place, f"Unit {row.unit}", row.routine, when_text(row, snapshot)]


def unit_cells(row: UnitRow, snapshot: Snapshot, pending: Pending) -> list[str]:
    change = None if row.number is None else pending.of(row.number)
    group = row.group or "—"
    return [row.name, group, standing_text(row, change), next_text(row, snapshot)]


class QueueCard(GlassCard):
    def __init__(self, parent: QWidget):
        table = UnitTable(["#", "Unit", "Routine", "When"], parent)
        super().__init__(parent, "Queue", QUEUE_HELP, table, Side.LEFT)

    def show_loop(self, snapshot: Snapshot, queue: tuple[QueueRow, ...]) -> None:
        self.table.fill([(row.key, queue_cells(row, snapshot), False, "") for row in queue])


class UnitsCard(GlassCard):
    def __init__(self, parent: QWidget):
        table = UnitTable(["Unit", "Group", "State", "Next"], parent)
        super().__init__(parent, "Units", UNITS_HELP, table, Side.RIGHT)

    def show_loop(self, snapshot: Snapshot, units: tuple[UnitRow, ...], pending: Pending) -> None:
        self.table.fill(
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
