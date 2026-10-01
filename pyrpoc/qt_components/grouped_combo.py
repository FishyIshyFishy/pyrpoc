"""A combo box whose entries sit under unselectable group headings."""

from __future__ import annotations

from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtGui import QPainter, QPaintEvent, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStyleOptionViewItem,
    QStylePainter,
    QWidget,
)

_GROUP_ROLE = Qt.ItemDataRole.UserRole + 1


class _HeadingDelegate(QStyledItemDelegate):
    """Draws headings as enabled text: they are disabled only so they cannot
    be chosen, and the theme would otherwise dim them past reading."""

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        if index.flags() == Qt.ItemFlag.NoItemFlags:
            option.state |= QStyle.StateFlag.State_Enabled
        super().paint(painter, option, index)


class GroupedComboBox(QComboBox):
    """Closed, it shows the current entry with its heading, since an entry's
    label need only be unique within its group."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.entries = QStandardItemModel(self)
        self.setModel(self.entries)
        self.setItemDelegate(_HeadingDelegate(self))

    def add_group(self, heading: str, entries: list[tuple[str, str]]) -> None:
        """One heading, then its entries as (label, data)."""
        header = QStandardItem(heading)
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
