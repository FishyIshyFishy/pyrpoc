"""``QueueCard``: the routine queue, and the changes asked of it: pause or resume a unit, run it
next, remove it after asking, and reorder the routines that are due."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFrame, QLabel, QScrollArea, QWidget

from .. import look
from ..feed import Steer
from ..state.pending import Change, Pending
from ..state.reorder import movable, moved
from ..state.routines import RoutineKey
from ..state.run_order import QueueRow, Section
from ..state.snapshot import Snapshot, snapshot_of
from .card import Anchor, Card
from .queue_list import ROW_H, QueueList

WIDTH = 380


def hint(label: str, key: str, on: bool) -> str:
    colour = look.TEXT if on else look.faded(look.TEXT_DIM, 110)
    return f"<span style='color:{colour.name()}'><b>{key}</b> {label}</span>"


def hints_for(row: QueueRow | None, rows: tuple[QueueRow, ...], routines: int) -> str:
    """The keys that work on ``row``, those that do not greyed, and what pause will touch."""
    if row is None:
        return ""
    paused = row.section is Section.PAUSED
    unit = row.status.unit
    touches = f"unit {unit}" if routines == 1 else f"unit {unit}, all {routines} routines"
    parts = [
        hint("move", "↑↓", True),
        hint("reorder", "⇧↑↓", movable(rows, row.key)),
        hint("locate", "⏎", True),
        hint(f"{'resume' if paused else 'pause'} {touches}", "P", True),
        hint("run next", "R", row.section not in (Section.NOW, Section.RUN_NOW)),
        hint("remove", "Del", True),
    ]
    return " &nbsp;·&nbsp; ".join(parts)


class QueueCard(Card):
    # The unit whose routine is chosen, or None.
    chosen_unit = pyqtSignal(object)
    # The routine chosen, or None.
    chosen_routine = pyqtSignal(object)
    locate = pyqtSignal()

    def __init__(self, parent: QWidget, steer: Steer):
        super().__init__(parent, "Queue", Anchor.LEFT, WIDTH, 1)
        self.steer = steer
        self.pending = Pending()
        self.snapshot: Snapshot | None = None
        self.removing: int | None = None
        self.list = QueueList(self)
        self.scroller = QScrollArea(self)
        self.scroller.setWidget(self.list)
        self.scroller.setWidgetResizable(True)
        self.scroller.setFrameShape(QFrame.Shape.NoFrame)
        self.scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroller.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        self.empty = self.label("No routines are running.", look.TEXT_DIM)
        self.confirm = self.label("", look.TEXT)
        self.confirm.setStyleSheet(
            f"color: {look.TEXT.name()}; border: 1px solid {look.WARNING.name()};"
            "border-radius: 8px; padding: 8px;"
        )
        self.hints = self.label("", look.TEXT_DIM)
        self.hints.setFont(look.font(8.0, False))
        for widget in (self.empty, self.scroller, self.confirm, self.hints):
            self.content.addWidget(widget)
        self.content.setStretchFactor(self.scroller, 1)
        self.confirm.hide()
        self.connect_list()

    def label(self, text: str, colour: QColor) -> QLabel:
        label = QLabel(text, self)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setStyleSheet(f"color: {colour.name()};")
        return label

    def connect_list(self) -> None:
        self.list.chosen.connect(self.on_chosen)
        self.list.locate.connect(self.locate)
        self.list.pause.connect(self.toggle_pause)
        self.list.run_now.connect(self.run_next)
        self.list.remove.connect(self.ask_remove_chosen)
        self.list.reorder_by.connect(self.reorder)
        self.list.answered.connect(self.answer)

    def focus_widget(self) -> QWidget:
        return self.list

    def show_snapshot(self, snapshot: Snapshot) -> None:
        self.pending.settle(snapshot.view, snapshot.statuses, snapshot.now_s)
        self.snapshot = snapshot
        self.list.show_rows(snapshot, self.pending)
        self.empty.setVisible(not snapshot.rows)
        if self.removing is not None and snapshot.unit(self.removing) is None:
            self.end_confirm()
        self.refresh_hints()

    def refresh_hints(self) -> None:
        row = self.list.chosen_row()
        routines = 0 if row is None else sum(1 for r in self.list.rows if r.key[0] == row.key[0])
        rows = () if self.snapshot is None else self.snapshot.rows
        self.hints.setText(hints_for(row, rows, routines))

    def on_chosen(self, key: RoutineKey | None) -> None:
        rect = None if key is None else self.list.row_rect(key)
        if rect is not None:
            self.scroller.ensureVisible(0, int(rect.center().y()), 0, ROW_H)
        self.chosen_unit.emit(None if key is None else key[0])
        self.chosen_routine.emit(key)
        self.refresh_hints()

    def choose_unit(self, unit: int | None) -> None:
        if unit is not None:
            self.list.choose_unit(unit)

    def toggle_pause(self) -> None:
        row, snapshot = self.list.chosen_row(), self.snapshot
        if row is None or snapshot is None:
            return
        paused = row.section is Section.PAUSED
        self.steer.pause(row.status.unit, not paused)
        change = Change.RESUME if paused else Change.PAUSE
        self.pending.send(row.status.unit, change, snapshot.now_s)
        self.list.update()

    def run_next(self) -> None:
        row, snapshot = self.list.chosen_row(), self.snapshot
        if row is None or snapshot is None or row.section in (Section.NOW, Section.RUN_NOW):
            return
        self.steer.run_now(row.status.unit)
        self.pending.send(row.status.unit, Change.RUN_NOW, snapshot.now_s)
        self.list.update()

    def reorder(self, step: int) -> None:
        row, snapshot = self.list.chosen_row(), self.snapshot
        if row is None or snapshot is None:
            return
        order = moved(snapshot.rows, snapshot.order, row.key, step)
        if order is None:
            return
        self.steer.reorder(order)
        # Shown at once from the new order, so a second move builds on the first.
        view, statuses, now_s = snapshot.view, snapshot.statuses, snapshot.now_s
        self.show_snapshot(snapshot_of(view, statuses, now_s, order, snapshot.events))

    def ask_remove_chosen(self) -> None:
        row = self.list.chosen_row()
        if row is not None:
            self.ask_remove(row.status.unit)

    def ask_remove(self, unit: int) -> None:
        """Ask in the card, not in a dialog, whether to take ``unit`` out of the loop."""
        self.list.choose_unit(unit)
        routines = sum(1 for r in self.list.rows if r.key[0] == unit)
        self.removing = unit
        self.list.confirming = True
        self.confirm.setText(
            f"<b>Remove unit {unit} from the loop?</b><br>"
            f"Its {routines} routine{'s' if routines != 1 else ''} stop, and it cannot be "
            "added back; a unit drawn again is a new unit.<br>"
            f"<span style='color:{look.TEXT_DIM.name()}'><b>⏎</b> remove &nbsp;·&nbsp; "
            "<b>Esc</b> keep</span>"
        )
        self.confirm.show()
        self.list.setFocus()

    def answer(self, remove: bool) -> None:
        unit, snapshot = self.removing, self.snapshot
        self.end_confirm()
        if remove and unit is not None and snapshot is not None:
            self.steer.remove(unit)
            self.pending.send(unit, Change.REMOVE, snapshot.now_s)

    def end_confirm(self) -> None:
        self.removing = None
        self.list.confirming = False
        self.confirm.hide()
