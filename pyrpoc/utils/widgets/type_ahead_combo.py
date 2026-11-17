"""A dropdown that drops its list below the box and jumps to entries as the user types, like a
web form's country select."""

from __future__ import annotations

import time
from enum import Enum

from PyQt6.QtCore import QEvent, QModelIndex, QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import (
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStyleOptionViewItem,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)


class Kind(Enum):
    HEADING = "heading"
    ENTRY = "entry"
    ACTION = "action"


_KIND_ROLE = Qt.ItemDataRole.UserRole + 1
# What typing is matched against: the label without its indent, in lower case.
_MATCH_ROLE = Qt.ItemDataRole.UserRole + 2
# What the closed box shows while the entry is current.
_SHOWN_ROLE = Qt.ItemDataRole.UserRole + 3

_MAX_LIST_HEIGHT = 360
_PAGE = 8
# Longer than Qt's default keyboard interval: a whole name like "united s" is typed in one go,
# as in a browser's select.
_TYPING_PAUSE_S = 1.0


class _RowDelegate(QStyledItemDelegate):
    """Draws the highlighted row as selected, headings as enabled text, and the action quietly
    under a rule."""

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self.highlighted = -1

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        assert painter is not None
        # Headings and the action are disabled only so the closed box's keys and wheel skip them.
        option.state |= QStyle.StateFlag.State_Enabled
        option.state &= ~QStyle.StateFlag.State_MouseOver
        if index.row() == self.highlighted:
            # Active too: the theme fades a selection in a view without focus.
            option.state |= QStyle.StateFlag.State_Selected | QStyle.StateFlag.State_Active
        if index.data(_KIND_ROLE) is not Kind.ACTION:
            super().paint(painter, option, index)
            return
        placeholder = option.palette.color(QPalette.ColorRole.PlaceholderText)
        option.palette.setColor(QPalette.ColorRole.Text, placeholder)
        super().paint(painter, option, index)
        # A rule above sets it apart from the entries.
        painter.save()
        painter.setPen(option.palette.color(QPalette.ColorRole.Mid))
        painter.drawLine(option.rect.topLeft(), option.rect.topRight())
        painter.restore()


class _ListPopup(QFrame):
    """The open list, dropped below the box. Typing moves the highlight to the first entry that
    starts with what was typed."""

    def __init__(self, combo: TypeAheadComboBox):
        super().__init__(combo, Qt.WindowType.Popup)
        self.combo = combo
        self.typed = ""
        self.typed_at = 0.0
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListView(self)
        self.list.setModel(combo.entries)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list.setSelectionMode(QListView.SelectionMode.NoSelection)
        self.list.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setMouseTracking(True)
        self.delegate = _RowDelegate(self.list)
        self.list.setItemDelegate(self.delegate)
        layout.addWidget(self.list)
        viewport = self.list.viewport()
        assert viewport is not None
        viewport.installEventFilter(self)

    def open(self) -> None:
        """Show the list with the current entry highlighted in its middle."""
        self.typed = ""
        self.place()
        self.show()
        self.setFocus()
        current = self.combo.currentIndex()
        self.set_highlight(current)
        self.list.scrollTo(
            self.combo.entries.index(current, 0), QListView.ScrollHint.PositionAtCenter
        )

    def kind(self, row: int) -> Kind:
        return self.combo.entries.index(row, 0).data(_KIND_ROLE)

    def pickable(self, row: int) -> bool:
        return 0 <= row < self.combo.count() and self.kind(row) is not Kind.HEADING

    def set_highlight(self, row: int) -> None:
        self.delegate.highlighted = row
        viewport = self.list.viewport()
        assert viewport is not None
        viewport.update()
        # The first entry of a group brings its heading into view with it.
        if row > 0 and self.kind(row - 1) is Kind.HEADING:
            self.list.scrollTo(self.combo.entries.index(row - 1, 0))
        self.list.scrollTo(self.combo.entries.index(row, 0))

    def move_highlight(self, step: int) -> None:
        """Move ``step`` pickable rows down, or up when negative, stopping at either end."""
        row = self.delegate.highlighted
        direction = 1 if step > 0 else -1
        for _ in range(abs(step)):
            following = row + direction
            while 0 <= following < self.combo.count() and not self.pickable(following):
                following += direction
            if not self.pickable(following):
                break
            row = following
        self.set_highlight(row)

    def type_ahead(self, text: str) -> None:
        """Add ``text`` to what has been typed, starting afresh after a pause, and highlight the
        first entry it begins."""
        now = time.monotonic()
        if now - self.typed_at > _TYPING_PAUSE_S:
            self.typed = ""
        self.typed_at = now
        self.typed += text.lower()
        match = self.first_starting(self.typed, 0)
        # Pressing one letter again and again steps through the entries starting with it.
        if match < 0 and self.typed == self.typed[0] * len(self.typed):
            match = self.first_starting(self.typed[0], self.delegate.highlighted + 1)
        if match >= 0:
            self.set_highlight(match)

    def first_starting(self, prefix: str, start: int) -> int:
        """The first entry from row ``start`` on, wrapping round, whose label starts with
        ``prefix``; -1 if none does."""
        count = self.combo.count()
        for offset in range(count):
            row = (max(start, 0) + offset) % count
            label = self.combo.entries.index(row, 0).data(_MATCH_ROLE)
            if self.kind(row) is Kind.ENTRY and label.startswith(prefix):
                return row
        return -1

    def place(self) -> None:
        """Size the list to its entries and drop it below the box, or above when there is no
        room beneath."""
        rows = range(self.combo.count())
        content = sum(self.list.sizeHintForRow(row) for row in rows) + 2 * self.list.frameWidth()
        self.list.setFixedHeight(min(content, _MAX_LIST_HEIGHT))
        width = max(self.combo.width(), self.list.sizeHintForColumn(0) + 32)
        self.setFixedWidth(width)
        self.adjustSize()
        position = self.combo.mapToGlobal(QPoint(0, self.combo.height()))
        screen = self.combo.screen()
        if screen is not None:
            area = screen.availableGeometry()
            if position.y() + self.height() > area.bottom():
                position.setY(self.combo.mapToGlobal(QPoint(0, 0)).y() - self.height())
            position.setX(max(area.left(), min(position.x(), area.right() - width)))
        self.move(position)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is None or a1.type() not in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        ):
            return super().eventFilter(a0, a1)
        assert isinstance(a1, QMouseEvent)
        row = self.list.indexAt(a1.position().toPoint()).row()
        if a1.type() == QEvent.Type.MouseMove and self.pickable(row):
            self.set_highlight(row)
        if a1.type() == QEvent.Type.MouseButtonRelease and self.pickable(row):
            self.combo.pick(row)
        # The view's own handling would select and scroll; the highlight stands in for both.
        return a1.type() != QEvent.Type.MouseMove

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        assert a0 is not None
        key = a0.key()
        steps = {
            Qt.Key.Key_Up.value: -1,
            Qt.Key.Key_Down.value: 1,
            Qt.Key.Key_PageUp.value: -_PAGE,
            Qt.Key.Key_PageDown.value: _PAGE,
            Qt.Key.Key_Home.value: -self.combo.count(),
            Qt.Key.Key_End.value: self.combo.count(),
        }
        if key in steps:
            self.move_highlight(steps[key])
        elif key in (Qt.Key.Key_Return.value, Qt.Key.Key_Enter.value, Qt.Key.Key_Tab.value):
            if self.pickable(self.delegate.highlighted):
                self.combo.pick(self.delegate.highlighted)
        elif key == Qt.Key.Key_Escape.value:
            self.hide()
        elif typed_text(a0):
            self.type_ahead(a0.text())


def typed_text(event: QKeyEvent) -> bool:
    """Whether ``event`` types a character rather than giving a shortcut."""
    shortcut = event.modifiers() & (
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
    )
    return not shortcut and event.text().isprintable() and event.text() != ""


class TypeAheadComboBox(QComboBox):
    """A combo box whose list drops below it; typing jumps to the first entry starting with what
    was typed, the arrow keys walk the list and Enter picks. Entries may sit under unselectable
    headings, and an action entry at the foot may close the list."""

    # The action entry at the foot of the list was picked.
    action_clicked = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.entries = QStandardItemModel(self)
        self.setModel(self.entries)
        # Whether entries added now sit under a heading, and so are indented.
        self.under_heading = False
        self.popup = _ListPopup(self)

    def clear_entries(self) -> None:
        self.entries.clear()
        self.under_heading = False

    def add_heading(self, text: str) -> QStandardItem:
        """A bold row the entries added after it sit under; returned so a caller can add a
        tooltip."""
        item = QStandardItem(text)
        item.setData(Kind.HEADING, _KIND_ROLE)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = self.font()
        font.setBold(True)
        item.setFont(font)
        self.entries.appendRow(item)
        self.under_heading = True
        return item

    def add_entry(self, label: str, data: object, shown: str) -> None:
        """An entry listed as ``label``, holding ``data``, that the closed box shows as
        ``shown``."""
        item = QStandardItem(f"    {label}" if self.under_heading else label)
        item.setData(data, Qt.ItemDataRole.UserRole)
        item.setData(Kind.ENTRY, _KIND_ROLE)
        item.setData(label.lower(), _MATCH_ROLE)
        item.setData(shown, _SHOWN_ROLE)
        self.entries.appendRow(item)
        # A heading becomes current when it is the first row, and is never a choice.
        if self.currentData(_KIND_ROLE) is not Kind.ENTRY:
            self.setCurrentIndex(self.count() - 1)

    def add_action(self, label: str) -> None:
        """An entry at the foot of the list that emits ``action_clicked`` instead of being
        chosen."""
        item = QStandardItem(label)
        item.setData(Kind.ACTION, _KIND_ROLE)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = self.font()
        font.setItalic(True)
        item.setFont(font)
        self.entries.appendRow(item)
        self.under_heading = False

    def pick(self, row: int) -> None:
        self.popup.hide()
        if self.entries.index(row, 0).data(_KIND_ROLE) is Kind.ACTION:
            self.action_clicked.emit()
        else:
            self.setCurrentIndex(row)

    def showPopup(self) -> None:
        self.popup.open()

    def hidePopup(self) -> None:
        self.popup.hide()

    def keyPressEvent(self, e: QKeyEvent | None) -> None:
        """Typing on the closed box opens the list and jumps to what was typed."""
        assert e is not None
        if typed_text(e) and e.text() != " ":
            self.popup.open()
            self.popup.type_ahead(e.text())
            return
        super().keyPressEvent(e)

    def paintEvent(self, e: QPaintEvent | None) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self.currentData(_SHOWN_ROLE)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
