"""``QueueList``: the routines in the order they will run, section by section, chosen and moved
with the keyboard or the mouse."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .. import keys, look
from ..state.narration import row_text
from ..state.pending import Pending
from ..state.phase import phase_of
from ..state.reorder import movable
from ..state.routines import RoutineKey
from ..state.run_order import QueueRow, Section
from ..state.snapshot import Snapshot

HEADING_H, ROW_H = 28, 44
# A press must travel this far before it drags a row.
DRAG_PX = 6
TITLE = look.font(10.0, True)
DETAIL = look.font(8.5, False)
HEADING = look.font(8.0, True)


@dataclass(frozen=True)
class Heading:
    section: Section
    count: int


Entry = Heading | QueueRow


@dataclass
class Drag:
    key: RoutineKey
    pressed_y: float
    # The row of its section the dragged row would land before, once it is moving.
    landing: int | None


class QueueList(QWidget):
    chosen = pyqtSignal(object)
    locate = pyqtSignal()
    pause = pyqtSignal()
    run_now = pyqtSignal()
    remove = pyqtSignal()
    # Move the chosen routine this many places within its section.
    reorder_by = pyqtSignal(int)
    # The answer to "remove this unit?": True to remove.
    answered = pyqtSignal(bool)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.entries: list[Entry] = []
        self.tops: list[int] = []
        self.snapshot: Snapshot | None = None
        self.pending = Pending()
        self.selected: RoutineKey | None = None
        self.confirming = False
        self.drag: Drag | None = None

    @property
    def rows(self) -> list[QueueRow]:
        return [entry for entry in self.entries if isinstance(entry, QueueRow)]

    def chosen_row(self) -> QueueRow | None:
        return next((row for row in self.rows if row.key == self.selected), None)

    def show_rows(self, snapshot: Snapshot, pending: Pending) -> None:
        """Show ``snapshot``'s queue, keeping the choice on the same routine where it is still
        open, and on the row in its place where not."""
        before = [row.key for row in self.rows]
        self.snapshot, self.pending = snapshot, pending
        self.entries = []
        for section in Section:
            members = [row for row in snapshot.rows if row.section is section]
            if members:
                self.entries += [Heading(section, len(members)), *members]
        self.tops = self.layout_rows()
        self.setFixedHeight(self.tops[-1] if self.tops else 0)
        keys_now = [row.key for row in self.rows]
        if self.drag is not None and self.drag.key not in keys_now:
            # The routine ended or its unit left while it was being dragged.
            self.drag = None
        if self.selected not in keys_now:
            at = before.index(self.selected) if self.selected in before else 0
            self.choose(keys_now[min(at, len(keys_now) - 1)] if keys_now else None)
        self.update()

    def layout_rows(self) -> list[int]:
        """Each entry's top, then the bottom of the last."""
        tops, y = [], 0
        for entry in self.entries:
            tops.append(y)
            y += HEADING_H if isinstance(entry, Heading) else ROW_H
        return [*tops, y]

    def choose(self, key: RoutineKey | None) -> None:
        if key == self.selected:
            return
        self.selected = key
        self.chosen.emit(key)
        self.update()

    def choose_unit(self, unit: int) -> None:
        """Choose the unit's first routine, unless one of its routines is chosen already."""
        if self.selected is not None and self.selected[0] == unit:
            return
        first = next((row.key for row in self.rows if row.key[0] == unit), None)
        if first is not None:
            self.choose(first)

    def step_choice(self, step: int) -> None:
        keys_now = [row.key for row in self.rows]
        if not keys_now:
            return
        at = keys_now.index(self.selected) if self.selected in keys_now else 0
        self.choose(keys_now[max(0, min(len(keys_now) - 1, at + step))])

    def row_rect(self, key: RoutineKey) -> QRectF | None:
        for index, entry in enumerate(self.entries):
            if isinstance(entry, QueueRow) and entry.key == key:
                return QRectF(0, self.tops[index], self.width(), ROW_H)
        return None

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        key, shifted = event.key(), bool(event.modifiers() & keys.REORDER)
        if self.confirming:
            self.answer(key)
        elif key in (keys.UP, keys.DOWN):
            step = -1 if key == keys.UP else 1
            if shifted:
                self.reorder_by.emit(step)
            else:
                self.step_choice(step)
        elif key == keys.PAUSE:
            self.pause.emit()
        elif key == keys.RUN_NOW:
            self.run_now.emit()
        elif key in keys.REMOVE:
            self.remove.emit()
        elif key in keys.LOCATE:
            self.locate.emit()
        else:
            event.ignore()

    def answer(self, key: int) -> None:
        """While asking whether to remove: Enter removes, Esc keeps, other keys do nothing."""
        if key in keys.LOCATE:
            self.answered.emit(True)
        elif key == keys.CLOSE:
            self.answered.emit(False)

    def entry_at(self, y: float) -> int | None:
        for index in range(len(self.entries)):
            if self.tops[index] <= y < self.tops[index + 1]:
                return index
        return None

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        index = self.entry_at(event.position().y())
        entry = None if index is None else self.entries[index]
        if isinstance(entry, QueueRow):
            self.choose(entry.key)
            self.drag = Drag(entry.key, event.position().y(), None)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        drag = self.drag
        if event is None or drag is None:
            return
        y = event.position().y()
        if drag.landing is None and abs(y - drag.pressed_y) < DRAG_PX:
            return
        if self.snapshot is not None and movable(self.snapshot.rows, drag.key):
            drag.landing = self.landing_at(drag.key, y)
            self.update()

    def landing_at(self, key: RoutineKey, y: float) -> int:
        """The row of ``key``'s section that a drop at ``y`` lands before, counted in the
        section."""
        (row,) = [r for r in self.rows if r.key == key]
        mates = [r for r in self.rows if r.section is row.section]
        middles = [rect.center().y() for r in mates if (rect := self.row_rect(r.key)) is not None]
        return sum(1 for middle in middles if y > middle)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        drag, self.drag = self.drag, None
        if drag is None or drag.landing is None:
            return
        (row,) = [r for r in self.rows if r.key == drag.key]
        mates = [r.key for r in self.rows if r.section is row.section]
        at = mates.index(drag.key)
        # Landing before a row below this one means taking this one out first.
        step = drag.landing - at - (1 if drag.landing > at else 0)
        if step:
            self.reorder_by.emit(step)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.locate.emit()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        snapshot = self.snapshot
        if snapshot is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, entry in enumerate(self.entries):
            box = QRectF(0, self.tops[index], self.width(), self.tops[index + 1] - self.tops[index])
            if isinstance(entry, Heading):
                self.paint_heading(painter, box, entry)
            else:
                self.paint_row(painter, box, entry, snapshot)
        self.paint_landing(painter)

    def paint_heading(self, painter: QPainter, box: QRectF, heading: Heading) -> None:
        painter.setFont(HEADING)
        painter.setPen(look.TEXT_DIM)
        text = f"{heading.section.value.upper()}   {heading.count}"
        painter.drawText(box.adjusted(4, 8, 0, 0), Qt.AlignmentFlag.AlignLeft, text)

    def paint_row(self, painter: QPainter, box: QRectF, row: QueueRow, snapshot: Snapshot) -> None:
        status = row.status
        if row.key == self.selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(look.SELECTED_ROW)
            painter.drawRoundedRect(box.adjusted(0, 2, 0, -2), 8, 8)
        dim = row.section is Section.PAUSED
        colour = self.row_colour(row, snapshot)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(look.faded(colour, 120 if dim else 255))
        painter.drawEllipse(QPointF(box.left() + 18, box.center().y()), 11, 11)
        painter.setFont(TITLE)
        painter.setPen(QColor(12, 14, 18))
        disc = QRectF(box.left() + 7, box.center().y() - 11, 22, 22)
        painter.drawText(disc, Qt.AlignmentFlag.AlignCenter, str(status.unit))
        text_left = box.left() + 38
        painter.setPen(look.TEXT_DIM if dim else look.TEXT)
        title = QRectF(text_left, box.top() + 5, box.width() - text_left - 8, 18)
        painter.drawText(
            title, Qt.AlignmentFlag.AlignLeft, f"Unit {status.unit} · {status.routine}"
        )
        painter.setFont(DETAIL)
        painter.setPen(look.WARNING if status.lost else look.TEXT_DIM)
        detail = QRectF(text_left, box.top() + 23, box.width() - text_left - 8, 16)
        painter.drawText(
            detail, Qt.AlignmentFlag.AlignLeft, row_text(row.section, status, snapshot)
        )
        change = self.pending.of(status.unit)
        if change is not None:
            painter.setPen(look.ACCENT)
            painter.drawText(detail, Qt.AlignmentFlag.AlignRight, f"{change.value}…")

    def row_colour(self, row: QueueRow, snapshot: Snapshot) -> QColor:
        doing = snapshot.doing
        if row.section is Section.NOW and doing is not None:
            return look.PHASE[phase_of(doing)]
        unit = snapshot.unit(row.status.unit)
        return look.TEXT if unit is None else look.STATE[unit.state]

    def paint_landing(self, painter: QPainter) -> None:
        """A line where a dragged row would land."""
        drag = self.drag
        if drag is None or drag.landing is None:
            return
        (row,) = [r for r in self.rows if r.key == drag.key]
        mates = [r for r in self.rows if r.section is row.section]
        rects = [rect for r in mates if (rect := self.row_rect(r.key)) is not None]
        y = rects[drag.landing].top() if drag.landing < len(rects) else rects[-1].bottom()
        painter.setPen(QPen(look.ACCENT, 2.0))
        painter.drawLine(QPointF(4, y), QPointF(self.width() - 4, y))
