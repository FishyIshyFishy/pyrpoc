"""A dropdown whose open list is filtered by typing and walked with the arrow keys."""

from __future__ import annotations

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
    QLineEdit,
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
# What typing is matched against: an entry's label and its heading, in lower case.
_SEARCH_ROLE = Qt.ItemDataRole.UserRole + 2
# What the closed box shows while the entry is current.
_SHOWN_ROLE = Qt.ItemDataRole.UserRole + 3

_MAX_LIST_HEIGHT = 360
_PAGE = 8


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
            # Active too: the filter field holds focus, and the theme fades an inactive selection.
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


class _FilterPopup(QFrame):
    """The open list: a filter field over the entries, dropped below the box."""

    def __init__(self, combo: FilterComboBox):
        super().__init__(combo, Qt.WindowType.Popup)
        self.combo = combo
        # Whether the list opened above the box, so refitting it after filtering keeps it there.
        self.above = False
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Type to filter")
        self.search.setClearButtonEnabled(True)
        self.list = QListView(self)
        self.list.setModel(combo.entries)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list.setSelectionMode(QListView.SelectionMode.NoSelection)
        self.list.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setMouseTracking(True)
        self.delegate = _RowDelegate(self.list)
        self.list.setItemDelegate(self.delegate)
        layout.addWidget(self.search)
        layout.addWidget(self.list)

        self.search.textChanged.connect(self.apply_filter)
        self.search.installEventFilter(self)
        viewport = self.list.viewport()
        assert viewport is not None
        viewport.installEventFilter(self)

    def open(self, text: str) -> None:
        """Show every entry with the current one highlighted, then filter by ``text``."""
        self.search.clear()
        self.apply_filter("")
        self.place()
        self.show()
        self.set_highlight(self.combo.currentIndex())
        self.list.scrollTo(
            self.combo.entries.index(self.combo.currentIndex(), 0),
            QListView.ScrollHint.PositionAtCenter,
        )
        self.search.setFocus()
        self.search.setText(text)

    def kind(self, row: int) -> Kind:
        return self.combo.entries.index(row, 0).data(_KIND_ROLE)

    def pickable(self, row: int) -> bool:
        return (
            0 <= row < self.combo.count()
            and not self.list.isRowHidden(row)
            and self.kind(row) is not Kind.HEADING
        )

    def apply_filter(self, text: str) -> None:
        """Show the entries matching ``text``, and the headings over them; highlight the first
        match."""
        query = text.strip().lower()
        # Walked from the bottom so each heading knows whether any entry beneath it is shown.
        shown_below = False
        first_match = -1
        for row in reversed(range(self.combo.count())):
            kind = self.kind(row)
            if kind is Kind.HEADING:
                hidden = not shown_below
                shown_below = False
            elif kind is Kind.ACTION:
                hidden = False
            else:
                hidden = query not in self.combo.entries.index(row, 0).data(_SEARCH_ROLE)
                shown_below |= not hidden
                first_match = row if not hidden else first_match
            self.list.setRowHidden(row, hidden)
        if self.isVisible():
            self.place()
        if query or not self.pickable(self.delegate.highlighted):
            self.set_highlight(first_match)

    def set_highlight(self, row: int) -> None:
        self.delegate.highlighted = row
        viewport = self.list.viewport()
        assert viewport is not None
        viewport.update()
        if row < 0:
            return
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

    def place(self) -> None:
        """Size the list to its shown rows and drop it below the box, or above when there is
        no room beneath; once open it stays on the side it opened on."""
        shown = [row for row in range(self.combo.count()) if not self.list.isRowHidden(row)]
        content = sum(self.list.sizeHintForRow(row) for row in shown) + 2 * self.list.frameWidth()
        self.list.setFixedHeight(min(content, _MAX_LIST_HEIGHT))
        width = max(self.combo.width(), self.list.sizeHintForColumn(0) + 32)
        self.setFixedWidth(width)
        self.adjustSize()
        below = self.combo.mapToGlobal(QPoint(0, self.combo.height()))
        top = self.combo.mapToGlobal(QPoint(0, 0)).y() - self.height()
        screen = self.combo.screen()
        if not self.isVisible():
            fits = (
                screen is None or below.y() + self.height() <= screen.availableGeometry().bottom()
            )
            self.above = not fits
        position = QPoint(below.x(), top if self.above else below.y())
        if screen is not None:
            area = screen.availableGeometry()
            position.setX(max(area.left(), min(position.x(), area.right() - width)))
        self.move(position)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is None:
            return False
        if a0 is self.search and a1.type() == QEvent.Type.KeyPress:
            assert isinstance(a1, QKeyEvent)
            return self.on_key(a1.key())
        if a1.type() in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        ):
            assert isinstance(a1, QMouseEvent)
            row = self.list.indexAt(a1.position().toPoint()).row()
            if a1.type() == QEvent.Type.MouseMove and self.pickable(row):
                self.set_highlight(row)
            if a1.type() == QEvent.Type.MouseButtonRelease and self.pickable(row):
                self.combo.pick(row)
            # The view's own handling would select and scroll; the highlight stands in for both.
            return a1.type() != QEvent.Type.MouseMove
        return super().eventFilter(a0, a1)

    def on_key(self, key: int) -> bool:
        """Handle the keys that move or pick; the rest edit the filter."""
        steps = {
            Qt.Key.Key_Up.value: -1,
            Qt.Key.Key_Down.value: 1,
            Qt.Key.Key_PageUp.value: -_PAGE,
            Qt.Key.Key_PageDown.value: _PAGE,
        }
        if key in steps:
            self.move_highlight(steps[key])
        elif key in (Qt.Key.Key_Return.value, Qt.Key.Key_Enter.value):
            if self.pickable(self.delegate.highlighted):
                self.combo.pick(self.delegate.highlighted)
        elif key == Qt.Key.Key_Escape.value:
            self.hide()
        else:
            return False
        return True


class FilterComboBox(QComboBox):
    """A combo box whose list drops below it with a filter field: typing narrows the entries,
    the arrow keys walk them and Enter picks one. Entries may sit under unselectable headings,
    and an action entry may close the list."""

    # The action entry at the foot of the list was picked.
    action_clicked = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.entries = QStandardItemModel(self)
        self.setModel(self.entries)
        # The heading the next entries sit under, if any.
        self.heading = ""
        self.popup = _FilterPopup(self)

    def clear_entries(self) -> None:
        self.entries.clear()
        self.heading = ""

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
        self.heading = text
        return item

    def add_entry(self, label: str, data: object, shown: str) -> None:
        """An entry listed as ``label``, holding ``data``, that the closed box shows as
        ``shown``."""
        item = QStandardItem(f"    {label}" if self.heading else label)
        item.setData(data, Qt.ItemDataRole.UserRole)
        item.setData(Kind.ENTRY, _KIND_ROLE)
        item.setData(f"{self.heading} {label}".lower(), _SEARCH_ROLE)
        item.setData(shown, _SHOWN_ROLE)
        self.entries.appendRow(item)
        # A heading becomes current when it is the first row, and is never a choice.
        if self.currentData(_KIND_ROLE) is not Kind.ENTRY:
            self.setCurrentIndex(self.count() - 1)

    def add_action(self, label: str) -> None:
        """An entry at the foot of the list that emits ``action_clicked`` instead of being
        chosen; it is shown whatever the filter."""
        item = QStandardItem(label)
        item.setData(Kind.ACTION, _KIND_ROLE)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = self.font()
        font.setItalic(True)
        item.setFont(font)
        self.entries.appendRow(item)
        self.heading = ""

    def pick(self, row: int) -> None:
        self.popup.hide()
        if self.entries.index(row, 0).data(_KIND_ROLE) is Kind.ACTION:
            self.action_clicked.emit()
        else:
            self.setCurrentIndex(row)

    def showPopup(self) -> None:
        self.popup.open("")

    def hidePopup(self) -> None:
        self.popup.hide()

    def keyPressEvent(self, e: QKeyEvent | None) -> None:
        """Typing on the closed box opens the list filtered by what was typed."""
        assert e is not None
        text = e.text()
        plain = not e.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
        )
        if plain and text.isprintable() and text.strip():
            self.popup.open(text)
            return
        super().keyPressEvent(e)

    def paintEvent(self, e: QPaintEvent | None) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self.currentData(_SHOWN_ROLE)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
