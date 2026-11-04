"""The Mask Editor's sidebar lists: the drawn masks, and the regions of the one being drawn."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap
from PyQt6.QtWidgets import QAbstractItemView, QLabel, QListWidget, QListWidgetItem, QWidget

from pyrpoc.plugins.program_io.inputs import Mask, thumbnail

from .canvas import MaskRoi, color_for_index

ICON_PX = 28
# About three rows, so a list stays usable when the sidebar is squeezed.
LIST_MIN_HEIGHT = 96


def section_title(text: str, parent: QWidget) -> QLabel:
    """A section's heading: its name in small bold capitals."""
    label = QLabel(text.upper(), parent)
    font = QFont(label.font())
    font.setBold(True)
    font.setPointSizeF(font.pointSizeF() * 0.85)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
    label.setFont(font)
    return label


def hint(text: str, parent: QWidget) -> QLabel:
    """A muted line saying what to do when a list is empty."""
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setEnabled(False)
    return label


class MaskPicker(QListWidget):
    """The drawn masks, each its thumbnail and name. The selected one is drawn on; a
    double-click renames it."""

    # The mask chosen to draw on, by id, or "" for none.
    chosen = pyqtSignal(str)
    renamed = pyqtSignal(str, str)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.ids: list[str] = []
        self.setIconSize(QSize(ICON_PX, ICON_PX))
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.setUniformItemSizes(True)
        self.setMinimumHeight(LIST_MIN_HEIGHT)
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.itemChanged.connect(self.on_item_changed)

    def show_masks(self, masks: tuple[Mask, ...], current: str) -> None:
        """List ``masks``, selecting the one whose id is ``current``."""
        self.blockSignals(True)
        self.clear()
        self.ids = [mask.id for mask in masks]
        for mask in masks:
            item = QListWidgetItem(QIcon(thumbnail(mask, ICON_PX)), mask.label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(f"{mask.describe()}. Double-click to rename.")
            self.addItem(item)
        if current in self.ids:
            self.setCurrentRow(self.ids.index(current))
        self.blockSignals(False)

    def on_selection_changed(self) -> None:
        rows = [index.row() for index in self.selectedIndexes()]
        self.chosen.emit(self.ids[rows[0]] if rows else "")

    def on_item_changed(self, item: QListWidgetItem) -> None:
        self.renamed.emit(self.ids[self.row(item)], item.text().strip())


class RegionList(QListWidget):
    """The regions of the mask being drawn, each in the colour it is outlined in."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setToolTip("Right-click a region to change its thresholds or delete it")
        self.setMinimumHeight(LIST_MIN_HEIGHT)

    def show_regions(self, rois: list[MaskRoi]) -> None:
        self.clear()
        for index, roi in enumerate(rois):
            self.addItem(QListWidgetItem(swatch(color_for_index(index)), region_text(index, roi)))


def swatch(color: QColor) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(color)
    return QIcon(pixmap)


def region_text(index: int, roi: MaskRoi) -> str:
    """A region's number and what it keeps: its whole shape, or a range on some channels."""
    threshold = roi.threshold
    if threshold is None:
        return f"Region {index + 1} · whole shape"
    channels = ", ".join(f"C{i + 1}" for i, on in enumerate(threshold.active_channels) if on)
    return f"Region {index + 1} · {threshold.low:.0f}–{threshold.high:.0f} on {channels or 'none'}"
