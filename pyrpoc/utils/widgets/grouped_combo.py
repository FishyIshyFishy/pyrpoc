"""A combo box whose entries sit under unselectable group headings."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QModelIndex, QObject, Qt, pyqtSignal
from PyQt6.QtGui import (
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStyleOptionViewItem,
    QStylePainter,
    QWidget,
)

_GROUP_ROLE = Qt.ItemDataRole.UserRole + 1
# Set only on headings: whether their entries are currently hidden.
_COLLAPSED_ROLE = Qt.ItemDataRole.UserRole + 2
_HEADING_ROLE = Qt.ItemDataRole.UserRole + 3
# Set only on the action entry, which is clicked rather than chosen.
_ACTION_ROLE = Qt.ItemDataRole.UserRole + 4


class _HeadingDelegate(QStyledItemDelegate):
    """Draws headings as enabled text, and the action entry quietly."""

    # They are disabled only so they cannot be chosen, and the theme would otherwise dim them past
    # reading.

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        if index.flags() == Qt.ItemFlag.NoItemFlags:
            option.state |= QStyle.StateFlag.State_Enabled
        if not index.data(_ACTION_ROLE):
            super().paint(painter, option, index)
            return
        placeholder = option.palette.color(QPalette.ColorRole.PlaceholderText)
        option.palette.setColor(QPalette.ColorRole.Text, placeholder)
        super().paint(painter, option, index)
        assert painter is not None
        # A rule above sets it apart from the last group's entries.
        painter.save()
        painter.setPen(option.palette.color(QPalette.ColorRole.Mid))
        painter.drawLine(option.rect.topLeft(), option.rect.topRight())
        painter.restore()


class GroupedComboBox(QComboBox):
    """A combo box whose entries sit under headings, showing the current entry with its
    heading."""

    # An entry's label need only be unique within its group.

    # The action entry at the foot of the list was clicked.
    action_clicked = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.entries = QStandardItemModel(self)
        self.setModel(self.entries)
        # Kept as a QListView, the view type that can hide rows. Set before the
        # delegate, which goes on whichever view is current.
        self.list_view = QListView(self)
        self.setView(self.list_view)
        self.setItemDelegate(_HeadingDelegate(self))
        viewport = self.list_view.viewport()
        assert viewport is not None
        viewport.installEventFilter(self)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is not None and a1.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        ):
            assert isinstance(a1, QMouseEvent)
            index = self.list_view.indexAt(a1.position().toPoint())
            if index.isValid() and index.data(_ACTION_ROLE):
                if a1.type() == QEvent.Type.MouseButtonRelease:
                    self.hidePopup()
                    self.action_clicked.emit()
                return True
            if index.isValid() and index.data(_COLLAPSED_ROLE) is not None:
                if a1.type() == QEvent.Type.MouseButtonRelease:
                    self._toggle_group(index.row())
                return True
        return super().eventFilter(a0, a1)

    def _toggle_group(self, header_row: int) -> None:
        header = self.entries.index(header_row, 0)
        collapsed = not header.data(_COLLAPSED_ROLE)
        self.entries.setData(header, collapsed, _COLLAPSED_ROLE)
        self.entries.setData(
            header,
            f"{'▸' if collapsed else '▾'} {header.data(_HEADING_ROLE)}",
            Qt.ItemDataRole.DisplayRole,
        )
        row = header_row + 1
        while row < self.entries.rowCount() and self._is_entry(row):
            self.list_view.setRowHidden(row, collapsed)
            row += 1

    def _is_entry(self, row: int) -> bool:
        index = self.entries.index(row, 0)
        return index.data(_COLLAPSED_ROLE) is None and not index.data(_ACTION_ROLE)

    def clear_entries(self) -> None:
        """Remove every heading, entry and action."""
        self.entries.clear()

    def add_action(self, label: str) -> None:
        """An entry at the foot of the open list that emits ``action_clicked`` instead of
        being chosen."""
        item = QStandardItem(label)
        item.setData(True, _ACTION_ROLE)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = self.font()
        font.setItalic(True)
        item.setFont(font)
        self.entries.appendRow(item)

    def add_group(self, heading: str, entries: list[tuple[str, str]]) -> None:
        """One heading, then its entries as (label, data). Clicking the heading
        in the open list collapses its entries."""
        header = QStandardItem(f"▾ {heading}")
        header.setData(False, _COLLAPSED_ROLE)
        header.setData(heading, _HEADING_ROLE)
        header.setFlags(Qt.ItemFlag.NoItemFlags)
        font = self.font()
        font.setBold(True)
        header.setFont(font)
        self.entries.appendRow(header)
        for label, data in entries:
            item = QStandardItem(f"    {label}")
            item.setData(data, Qt.ItemDataRole.UserRole)
            item.setData(f"{heading} › {label}", _GROUP_ROLE)
            self.entries.appendRow(item)
        # Adding the first row selects it, and a heading is never a choice.
        if self.currentData() is None:
            self.setCurrentIndex(1)

    def paintEvent(self, e: QPaintEvent | None) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self.currentData(_GROUP_ROLE)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
