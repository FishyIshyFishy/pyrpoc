"""Authoring a mask from an acquired dataset.

A finished mask is filed in the library as a ``Mask2D`` dataset, which is how
it reaches a program: the parameter that consumes masks asks the library by
kind, so neither side names the other. ``Save mask...`` only exports a PNG.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from pyrpoc.plugins.data_panels.mask_editor.range_slider import RangeSlider
from pyrpoc.plugins.data_panels.mask_editor.transforms import normalize_channels
from pyrpoc.qt_components.table import horizontal_header, vertical_header
from pyrpoc.structs.data import Image2D, Mask2D
from pyrpoc.structs.dataset import Dataset, Origin, Provenance, utc_now
from pyrpoc.structs.library import Library
from pyrpoc.structs.panel import DataPanel, data_panel_registry

from .canvas import MaskImageView, MaskRoi
from .dialog import RoiThresholdDialog

CHANNEL_COLORS = tuple(
    np.array(rgb, dtype=np.float32)
    for rgb in (
        (255, 64, 64),
        (64, 180, 255),
        (255, 180, 64),
        (180, 64, 255),
        (64, 255, 160),
        (255, 64, 160),
        (200, 255, 64),
        (64, 160, 255),
    )
)


def write_mask(path: str, mask: np.ndarray) -> Path:
    """Export a 2-D mask to an image file; nothing here reads it back."""
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(resolved), np.asarray(mask, dtype=np.uint8)):
        raise OSError(f"failed to write a mask to '{resolved}'")
    return resolved


@data_panel_registry.register("mask_editor")
class MaskEditorPanel(DataPanel):
    """Draw thresholded polygon ROIs over an acquired image and file the mask."""

    display_name = "Mask Editor"
    renders = [Image2D]

    def __init__(self, library: Library):
        super().__init__(library)
        self.setObjectName("maskEditorRoot")
        self.setStyleSheet(
            "#maskEditorRoot, #maskEditorRoot QWidget { background: transparent; }"
            "#maskEditorRoot QGraphicsView, #maskEditorRoot QTableWidget"
            " { background: transparent; }"
        )
        self._rois: list[MaskRoi] = []
        self.apply_new_data(None)

        column = QVBoxLayout(self.body)
        column.addLayout(self.build_channels_row())
        column.addLayout(self.build_threshold_row())
        self.scene = QGraphicsScene(self)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.image_view = MaskImageView(self.scene, self)
        self.image_view.setMinimumSize(480, 320)
        column.addWidget(self.image_view, 1)
        self.roi_table = self.build_roi_table()
        column.addWidget(self.roi_table)
        column.addLayout(self.build_name_row())
        column.addLayout(self.build_button_row())

        self.rebuild_channel_boxes()
        self.reset_threshold_controls()
        self.update_view_image()
        self.connect_source()

    def build_channels_row(self) -> QHBoxLayout:
        self.channels_row = QHBoxLayout()
        self.channels_row.addWidget(QLabel("Channels:", self))
        self.channel_boxes: list[QCheckBox] = []
        return self.channels_row

    def build_threshold_row(self) -> QHBoxLayout:
        """One span read left to right: each number is the handle beside it,
        and the accented groove between them is what is kept."""
        self.low_spin = QSpinBox(self)
        self.high_spin = QSpinBox(self)
        for spin in (self.low_spin, self.high_spin):
            spin.setFixedWidth(74)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.valueChanged.connect(self.on_threshold_changed)
        self.threshold_slider = RangeSlider(self)
        self.threshold_slider.setToolTip("Pixels inside the accented span count toward the mask.")
        self.threshold_slider.values_changed.connect(self.on_slider_changed)

        row = QHBoxLayout()
        row.addWidget(self.low_spin)
        row.addWidget(self.threshold_slider, 1)
        row.addWidget(self.high_spin)
        return row

    def build_name_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Name:", self))
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("what this mask is of")
        self.name_edit.returnPressed.connect(self.add_to_library)
        row.addWidget(self.name_edit, 1)
        add_btn = QPushButton("Add to library", self)
        add_btn.setToolTip(
            "File this mask as data. It then appears in the Modulation table's "
            "mask list, and in the data panel."
        )
        add_btn.clicked.connect(self.add_to_library)
        row.addWidget(add_btn)
        return row

    def build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        preview_btn = QPushButton("Preview", self)
        preview_btn.clicked.connect(self.preview_mask)
        save_btn = QPushButton("Save mask...", self)
        save_btn.setToolTip("Export a PNG. Not needed to use the mask here.")
        save_btn.clicked.connect(self.save_mask)
        row.addWidget(preview_btn)
        row.addWidget(save_btn)
        row.addStretch(1)
        return row

    def build_roi_table(self) -> QTableWidget:
        """Read-only: thresholds live on the ROI and change through the row's
        context menu, so an editable cell would promise an edit that does nothing."""
        table = QTableWidget(0, 3, self)
        table.setHorizontalHeaderLabels(["Low", "High", "Channels"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setCornerButtonEnabled(False)
        horizontal_header(table).setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        horizontal_header(table).setHighlightSections(False)
        vertical_header(table).setHighlightSections(False)
        table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self.show_roi_menu)
        return table

    def fit_roi_table_height(self) -> None:
        """Height the table to its rows, up to five, so it neither swallows the
        image nor leaves an empty slab under it."""
        header = horizontal_header(self.roi_table).height()
        row_height = vertical_header(self.roi_table).defaultSectionSize()
        rows = min(max(self.roi_table.rowCount(), 1), 5)
        self.roi_table.setFixedHeight(header + rows * row_height + 4)

    def apply_new_data(self, image_data: np.ndarray | None) -> None:
        """Take a ``(C, H, W)`` frame, or a blank one when nothing is bound."""
        if image_data is None:
            self._data = np.zeros((1, 256, 256), dtype=np.float32)
        else:
            self._data = np.asarray(image_data, dtype=np.float32)
        self._h = int(self._data.shape[1])
        self._w = int(self._data.shape[2])
        self._channel_visibility = [True] * int(self._data.shape[0])
        self._data_min = float(np.min(self._data))
        self._data_max = max(float(np.max(self._data)), self._data_min + 1e-9)

    def rebuild_channel_boxes(self) -> None:
        clear_layout_after(self.channels_row, 1)
        self.channel_boxes = []
        for idx in range(self._data.shape[0]):
            box = QCheckBox(f"C{idx + 1}", self)
            box.setChecked(True)
            box.toggled.connect(lambda checked, i=idx: self.on_channel_toggled(i, checked))
            self.channel_boxes.append(box)
            self.channels_row.addWidget(box)
        self.channels_row.addStretch(1)

    def default_thresholds(self) -> tuple[int, int]:
        low = int(round(self._data_min + 0.2 * (self._data_max - self._data_min)))
        high = int(round(self._data_min + 0.8 * (self._data_max - self._data_min)))
        return low, max(high, low)

    def reset_threshold_controls(self) -> None:
        int_min = int(np.floor(self._data_min))
        int_max = int(np.ceil(self._data_max))
        for control in (self.low_spin, self.high_spin, self.threshold_slider):
            control.blockSignals(True)
            control.setRange(int_min, int_max)
            control.blockSignals(False)
        self.write_thresholds(*self.default_thresholds())

    def show_frame(self, dataset: Dataset, frame: np.ndarray) -> None:
        """ROIs survive a same-shape frame, so the editor stays usable during a
        live run and across runs; a new shape resets it."""
        del dataset
        scaled = normalize_channels(frame) * 255.0
        if self._data.shape == scaled.shape:
            self.apply_new_data(scaled)
            self.update_view_image()
            return
        self.reset_to(scaled)

    def clear(self) -> None:
        self.reset_to(None)

    def reset_to(self, scaled: np.ndarray | None) -> None:
        """Start over on ``scaled``: no ROIs, fresh channels and thresholds."""
        self.apply_new_data(scaled)
        self._rois.clear()
        self.sync_rois()
        self.rebuild_channel_boxes()
        self.reset_threshold_controls()
        self.update_view_image()

    def clamp_scene_point(self, point: QPointF) -> QPointF:
        x = min(max(point.x(), 0.0), float(self._w - 1))
        y = min(max(point.y(), 0.0), float(self._h - 1))
        return QPointF(x, y)

    def on_channel_toggled(self, idx: int, checked: bool) -> None:
        self._channel_visibility[idx] = checked
        self.update_view_image()

    def write_thresholds(self, low: int, high: int) -> None:
        """Set every threshold control at once, with signals blocked so none
        echoes the change back to the one that caused it."""
        controls = (self.low_spin, self.high_spin, self.threshold_slider)
        for control in controls:
            control.blockSignals(True)
        self.low_spin.setValue(low)
        self.high_spin.setValue(high)
        self.threshold_slider.setValues(low, high)
        for control in controls:
            control.blockSignals(False)

    def on_threshold_changed(self, _value: int) -> None:
        self.write_thresholds(*self.coerced_thresholds())
        self.update_view_image()

    def on_slider_changed(self, low: int, high: int) -> None:
        self.write_thresholds(low, high)
        self.update_view_image()

    def coerced_thresholds(self) -> tuple[int, int]:
        low, high = self.low_spin.value(), self.high_spin.value()
        return (high, low) if high < low else (low, high)

    def update_view_image(self) -> None:
        display = np.zeros((self._h, self._w, 3), dtype=np.float32)
        low, high = self.coerced_thresholds()
        active = np.zeros((self._h, self._w), dtype=bool)
        for idx, channel in enumerate(self._data):
            if not self._channel_visibility[idx]:
                continue
            span = max(float(np.max(channel) - np.min(channel)), 1e-9)
            norm = (channel - float(np.min(channel))) / span
            display += norm[..., None] * CHANNEL_COLORS[idx % len(CHANNEL_COLORS)] * 0.65
            active |= (channel >= low) & (channel <= high)
        display = np.clip(display, 0, 255)
        display[active] = 255

        rgb = display.astype(np.uint8)
        qimg = QImage(rgb.tobytes(), self._w, self._h, 3 * self._w, QImage.Format.Format_RGB888)
        self._display_qimage = qimg.copy()
        self.image_item.setPixmap(QPixmap.fromImage(self._display_qimage))
        self.scene.setSceneRect(QRectF(self._display_qimage.rect()))

    def add_roi(self, points: list[tuple[float, float]]) -> None:
        """File a drawn polygon. A click without a drag gives fewer than three
        points, which is not a region."""
        if len(points) < 3:
            return
        low, high = self.coerced_thresholds()
        self._rois.append(
            MaskRoi(
                points=points,
                threshold_low=float(low),
                threshold_high=float(high),
                active_channels=self._channel_visibility.copy(),
            )
        )
        self.sync_rois()

    def sync_rois(self) -> None:
        """Rebuild the table and the drawn outlines together: both are numbered
        by position, so editing either in place would put them out of step."""
        self.roi_table.setRowCount(len(self._rois))
        for row, roi in enumerate(self._rois):
            channels = ",".join(
                str(i + 1) for i, active in enumerate(roi.active_channels) if active
            )
            cells = (f"{roi.threshold_low:.1f}", f"{roi.threshold_high:.1f}", channels or "-")
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.roi_table.setItem(row, col, item)
        self.fit_roi_table_height()
        self.image_view.set_rois(self._rois)

    def show_roi_menu(self, pos: QPoint) -> None:
        row = self.roi_table.rowAt(pos.y())
        viewport = self.roi_table.viewport()
        if row < 0 or viewport is None:
            return
        self.roi_table.selectRow(row)
        menu = QMenu(self.roi_table)
        # Connected rather than read from exec()'s return: Qt hides the menu
        # before it emits, so the dialog opens with the menu already gone.
        change = QAction("Change thresholds...", menu)
        change.triggered.connect(lambda _checked=False, r=row: self.change_roi_thresholds(r))
        delete = QAction("Delete", menu)
        delete.triggered.connect(lambda _checked=False, r=row: self.delete_roi(r))
        menu.addActions([change, delete])
        menu.exec(viewport.mapToGlobal(pos))

    def change_roi_thresholds(self, row: int) -> None:
        dialog = RoiThresholdDialog(self, row)
        # A frame of a new shape during the modal dialog clears the ROIs.
        if dialog.exec() != QDialog.DialogCode.Accepted or row >= len(self._rois):
            return
        low, high = dialog.values()
        self._rois[row] = replace(
            self._rois[row], threshold_low=float(low), threshold_high=float(high)
        )
        self.sync_rois()

    def delete_roi(self, row: int) -> None:
        # A frame of a new shape while the menu was open clears the ROIs.
        if row >= len(self._rois):
            return
        del self._rois[row]
        self.sync_rois()

    def rois(self) -> list[MaskRoi]:
        return self._rois

    def data_min(self) -> float:
        return self._data_min

    def data_max(self) -> float:
        return self._data_max

    def generate_mask(self, rois: list[MaskRoi]) -> np.ndarray | None:
        """The mask ``rois`` produce, or None for no ROIs. Taking the list lets
        the threshold dialog preview an edit before it is written."""
        if not rois:
            return None
        final_mask = np.zeros((self._h, self._w), dtype=np.uint8)
        for roi in rois:
            polygon = np.array(
                [[int(round(x)), int(round(y))] for x, y in roi.points], dtype=np.int32
            ).reshape(-1, 1, 2)
            roi_mask = np.zeros((self._h, self._w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [polygon], 255)
            active = np.zeros((self._h, self._w), dtype=bool)
            for channel, enabled in zip(self._data, roi.active_channels, strict=False):
                if enabled:
                    active |= (channel >= roi.threshold_low) & (channel <= roi.threshold_high)
            final_mask[(roi_mask == 255) & active] = 255
        return final_mask

    def preview_mask(self) -> None:
        mask = self.generate_mask(self._rois)
        if mask is None:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI before previewing.")
            return
        qimg = QImage(
            mask.tobytes(), self._w, self._h, self._w, QImage.Format.Format_Grayscale8
        ).copy()
        dlg = QDialog(self)
        dlg.setWindowTitle("Mask Preview")
        layout = QVBoxLayout(dlg)
        label = QLabel(dlg)
        label.setPixmap(QPixmap.fromImage(qimg))
        layout.addWidget(label)
        dlg.resize(max(320, self._w), max(240, self._h))
        dlg.exec()

    def add_to_library(self) -> None:
        """File the drawn mask as a ``Mask2D`` dataset: how a mask gets used."""
        mask = self.generate_mask(self._rois)
        if mask is None:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI before adding.")
            return
        dataset = Dataset(
            output="mask",
            spec=Mask2D,
            provenance=Provenance(
                program_key="mask_editor",
                started_at=utc_now(),
                name=self.name_edit.text().strip(),
            ),
            origin=Origin.AUTHORED,
        )
        dataset.append(mask)
        self.library.add(dataset)
        self.name_edit.clear()

    def save_mask(self) -> None:
        mask = self.generate_mask(self._rois)
        if mask is None:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", "", "PNG (*.png);;TIFF (*.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        # A file boundary: an unwritable path or an unknown extension.
        try:
            written = write_mask(path, mask)
        except (OSError, cv2.error) as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to save mask to {path}: {exc}")
            return
        QMessageBox.information(self, "Mask Saved", f"Wrote a mask to {written}.")


def clear_layout_after(layout: QLayout, keep: int) -> None:
    """Delete every item in ``layout`` past the first ``keep``."""
    while layout.count() > keep:
        item = layout.takeAt(keep)
        widget = item.widget() if item is not None else None
        if widget is not None:
            widget.deleteLater()
