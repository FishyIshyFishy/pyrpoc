"""Authoring a mask from an acquired dataset.

Was ``gui/main_widgets/opto_control_mgr/mask_editor.py``. It is a panel now: it
reads a bound dataset rather than reaching into a display widget's
``_data_chw``.

A finished mask leaves by being added to the dataset library as a ``Mask2D``
entry, which is the whole of how it reaches a modality. This panel does not
know what a modality is, and the Modulation parameter that consumes masks does
not know this panel exists -- it asks the library for entries matching
``Mask2D`` the same way every dataset panel asks for its own sources. The
alternative, and the reason this is worth stating, was for one of the two to
name the other.

``Save mask...`` stays, and is now only what it says: writing a PNG for
something outside this application to read. It is not how a mask gets used.

The threshold and polygon-ROI machinery is unchanged, split across two
sibling files: ``canvas.py`` for the ROI data model and the drawing surface,
``dialog.py`` for the re-threshold popup. What is left here is the panel
itself: the controls around those two, the mask/channel state they read and
write, and the two ways a finished mask leaves.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
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
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.src.structs.data import Dataset, Image2D, Mask2D, Provenance, utc_now
from pyrpoc.src.structs.panel import Panel, panel_registry

from ..components.range_slider import RangeSlider
from ..components.source_picker import SourcePicker
from ..components.table import horizontal_header, vertical_header
from ..components.transforms import normalize_channels
from .canvas import MaskImageView, MaskRoi
from .dialog import RoiThresholdDialog

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.src.app.library import DataLibrary


def write_mask(path: Path | str, mask: np.ndarray) -> Path:
    """Write a 2-D mask to disk. Returns the path written.

    An export, not a step in using a mask -- nothing in this application reads
    the file back. It lives here because this panel is the only thing that
    authors a mask at all.
    """
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={array.shape}")
    resolved = Path(str(path)).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(resolved), array):
        raise OSError(f"failed to write a mask to '{resolved}'")
    return resolved


@panel_registry.register("mask_editor")
class MaskEditorPanel(Panel):
    """Draw thresholded polygon ROIs over an acquired image and file the mask."""

    display_name = "Mask Editor"
    renders = [Image2D]

    dirty_state_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.setObjectName("maskEditorRoot")
        self.setStyleSheet(
            "#maskEditorRoot, #maskEditorRoot QWidget { background: transparent; }"
            "#maskEditorRoot QGraphicsView, #maskEditorRoot QTableWidget"
            " { background: transparent; }"
        )
        self._rois: list[MaskRoi] = []
        self._dirty = False
        self._display_qimage: QImage | None = None

        self._data = np.zeros((1, 1, 1), dtype=np.float32)
        self._h = 1
        self._w = 1
        self._channel_visibility = [True]
        self._data_min = 0.0
        self._data_max = 1.0
        self.apply_new_data(None)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self.source = SourcePicker(self.renders, self)
        self.source.changed.connect(self.refresh)
        root.addWidget(self.source)
        self.body = QWidget(self)
        root.addWidget(self.body, 1)

        self.build_ui()
        self.rebuild_channel_boxes()
        self.reset_threshold_controls()
        self.update_view_image()

    # -- binding ------------------------------------------------------------ #

    def attach_library(self, library: DataLibrary) -> None:
        self.source.attach_library(library)

    def dataset(self) -> Dataset | None:
        return self.source.current()

    def library(self) -> DataLibrary | None:
        return self.source.library()

    # -- layout ---------------------------------------------------------------- #

    def build_ui(self) -> None:
        # A single column. The ROI table used to be a second column beside the
        # image, which sized it to the tallest thing in the panel rather than to
        # the handful of rows it holds -- mostly empty, and it took width the
        # image wanted.
        column = QVBoxLayout(self.body)
        self.channels_row = QHBoxLayout()
        self.channels_row.addWidget(QLabel("Channels:", self))
        self.channel_boxes: list[QCheckBox] = []
        column.addLayout(self.channels_row)

        # One span, read left to right: the number at each end is the handle
        # beside it, and the accented groove between them is what is kept. No
        # "Low"/"High" labels, because the arrangement already says it.
        threshold_row = QHBoxLayout()
        int_min = int(np.floor(self._data_min))
        int_max = int(np.ceil(self._data_max))
        low_default, high_default = self.default_thresholds()

        self.low_spin = QSpinBox(self)
        self.high_spin = QSpinBox(self)
        for spin in (self.low_spin, self.high_spin):
            spin.setRange(int_min, int_max)
            spin.setFixedWidth(74)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.low_spin.setValue(low_default)
        self.high_spin.setValue(high_default)
        self.low_spin.valueChanged.connect(self.on_threshold_changed)
        self.high_spin.valueChanged.connect(self.on_threshold_changed)

        self.threshold_slider = RangeSlider(self)
        self.threshold_slider.setRange(int_min, int_max)
        self.threshold_slider.setValues(low_default, high_default)
        self.threshold_slider.setToolTip("Pixels inside the accented span count toward the mask.")
        self.threshold_slider.values_changed.connect(self.on_slider_changed)

        threshold_row.addWidget(self.low_spin)
        threshold_row.addWidget(self.threshold_slider, 1)
        threshold_row.addWidget(self.high_spin)
        column.addLayout(threshold_row)

        self.scene = QGraphicsScene(self)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.image_view = MaskImageView(self.scene, self)
        self.image_view.setMinimumSize(480, 320)
        column.addWidget(self.image_view, 1)

        self.roi_table = self.build_roi_table()
        column.addWidget(self.roi_table)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:", self))
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("what this mask is of")
        self.name_edit.returnPressed.connect(self.add_to_library)
        name_row.addWidget(self.name_edit, 1)
        add_btn = QPushButton("Add to library", self)
        add_btn.setToolTip(
            "File this mask as data. It then appears in the Modulation table's "
            "mask list, and in the data panel."
        )
        add_btn.clicked.connect(self.add_to_library)
        name_row.addWidget(add_btn)
        column.addLayout(name_row)

        button_row = QHBoxLayout()
        preview_btn = QPushButton("Preview", self)
        save_btn = QPushButton("Save mask...", self)
        save_btn.setToolTip("Export a PNG. Not needed to use the mask here.")
        preview_btn.clicked.connect(self.preview_mask)
        save_btn.clicked.connect(self.save_mask)
        button_row.addWidget(preview_btn)
        button_row.addWidget(save_btn)
        button_row.addStretch(1)
        column.addLayout(button_row)

    def build_roi_table(self) -> QTableWidget:
        """The ROI table: something to read, not something to type into.

        Nothing here is editable in place. A cell that accepts a caret is
        promising an edit that was never wired up -- typing a new Low into the
        table did nothing, because the thresholds live on the ROI and are
        changed through the row's context menu.
        """
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
        # Editing an ROI is done on the row that names it, rather than through
        # a button elsewhere that acts on whatever happens to be selected.
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self.show_roi_menu)
        return table

    def fit_roi_table_height(self) -> None:
        """Height the table to its rows, up to a few, then let it scroll.

        A table that keeps its own height is what stops it from either
        swallowing the image or leaving an empty slab under it.
        """
        header = horizontal_header(self.roi_table).height()
        row_height = vertical_header(self.roi_table).defaultSectionSize()
        rows = min(max(self.roi_table.rowCount(), 1), 5)
        self.roi_table.setFixedHeight(header + rows * row_height + 4)

    # -- data -------------------------------------------------------------------- #

    def apply_new_data(self, image_data: np.ndarray | None) -> None:
        self._data = self.coerce_input_data(image_data)
        self._h = int(self._data.shape[1])
        self._w = int(self._data.shape[2])
        self._channel_visibility = [True] * int(self._data.shape[0])
        self._data_min = float(np.min(self._data))
        self._data_max = float(np.max(self._data))
        if self._data_max <= self._data_min:
            self._data_max = self._data_min + 1e-9

    def rebuild_channel_boxes(self) -> None:
        while self.channels_row.count() > 1:
            item = self.channels_row.takeAt(1)
            if item is None:
                continue
            child = item.widget()
            if child is not None:
                child.deleteLater()
        self.channel_boxes = []
        for idx in range(self._data.shape[0]):
            cb = QCheckBox(f"C{idx + 1}", self)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, i=idx: self.on_channel_toggled(i, checked))
            self.channel_boxes.append(cb)
            self.channels_row.addWidget(cb)
        self.channels_row.addStretch(1)

    def default_thresholds(self) -> tuple[int, int]:
        low = int(round(self._data_min + 0.2 * (self._data_max - self._data_min)))
        high = int(round(self._data_min + 0.8 * (self._data_max - self._data_min)))
        return low, max(high, low)

    def reset_threshold_controls(self) -> None:
        int_min = int(np.floor(self._data_min))
        int_max = int(np.ceil(self._data_max))
        low_default, high_default = self.default_thresholds()
        self.low_spin.setRange(int_min, int_max)
        self.high_spin.setRange(int_min, int_max)
        self.threshold_slider.setRange(int_min, int_max)
        self.write_thresholds(low_default, high_default)

    def refresh(self) -> None:
        """Re-read the bound dataset.

        Editing state survives a same-shape frame: rebuilding the ROI table on
        every published frame would make the editor unusable during a live run,
        so it only resets when the image changes shape.
        """
        dataset = self.dataset()
        latest = dataset.latest() if dataset is not None else None
        scaled = None
        if latest is not None:
            normalized = normalize_channels(latest)
            if normalized is not None:
                scaled = normalized * 255.0

        if scaled is not None and self._data is not None and self._data.shape == scaled.shape:
            self.apply_new_data(scaled)
            self.update_view_image()
            return
        self.set_image_data(scaled)

    def clear(self) -> None:
        self.set_image_data(None)

    def set_image_data(self, image_data: np.ndarray | None) -> None:
        self.apply_new_data(image_data)
        self._rois.clear()
        self.sync_rois()
        self.rebuild_channel_boxes()
        self.reset_threshold_controls()
        self.set_dirty(False)
        self.update_view_image()

    def coerce_input_data(self, image_data: np.ndarray | None) -> np.ndarray:
        if image_data is None:
            return self.generate_default_data()
        arr = np.asarray(image_data, dtype=np.float32)
        if arr.ndim == 2:
            return arr[None, ...]
        if arr.ndim == 3:
            if arr.shape[0] <= 8:
                return arr
            if arr.shape[-1] <= 8:
                return np.moveaxis(arr, -1, 0)
        raise ValueError("image_data must be [H,W], [C,H,W], or [H,W,C] with <= 8 channels")

    def generate_default_data(self) -> np.ndarray:
        """A blank frame when nothing is bound.

        v3.0 synthesised a plausible-looking microscope image here, which is
        indistinguishable from real data at a glance. Blank is honest.
        """
        return np.zeros((1, 256, 256), dtype=np.float32)

    def clamp_scene_point(self, point: QPointF) -> QPointF:
        x = min(max(point.x(), 0.0), float(self._w - 1))
        y = min(max(point.y(), 0.0), float(self._h - 1))
        return QPointF(x, y)

    def on_channel_toggled(self, idx: int, checked: bool) -> None:
        if idx < 0 or idx >= len(self._channel_visibility):
            return
        self._channel_visibility[idx] = bool(checked)
        self.update_view_image()

    # -- thresholds ---------------------------------------------------------------- #

    def write_thresholds(self, low: int, high: int) -> None:
        """Push one pair of values into every threshold control at once.

        Each control would otherwise echo the change back to the one that
        caused it, so they are all written with their signals blocked and the
        redraw is done once, here.
        """
        controls = (self.low_spin, self.high_spin, self.threshold_slider)
        for control in controls:
            control.blockSignals(True)
        self.low_spin.setValue(low)
        self.high_spin.setValue(high)
        self.threshold_slider.setValues(low, high)
        for control in controls:
            control.blockSignals(False)

    def on_threshold_changed(self, _value: int) -> None:
        low, high = self.coerced_thresholds()
        self.write_thresholds(low, high)
        self.update_view_image()

    def on_slider_changed(self, low: int, high: int) -> None:
        self.write_thresholds(low, high)
        self.update_view_image()

    def coerced_thresholds(self) -> tuple[int, int]:
        low = int(self.low_spin.value())
        high = int(self.high_spin.value())
        if high < low:
            low, high = high, low
        return low, high

    def update_view_image(self) -> None:
        display = np.zeros((self._h, self._w, 3), dtype=np.float32)
        low, high = self.coerced_thresholds()
        active = np.zeros((self._h, self._w), dtype=bool)
        channel_colors = (
            np.array([255.0, 64.0, 64.0], dtype=np.float32),
            np.array([64.0, 180.0, 255.0], dtype=np.float32),
            np.array([255.0, 180.0, 64.0], dtype=np.float32),
            np.array([180.0, 64.0, 255.0], dtype=np.float32),
            np.array([64.0, 255.0, 160.0], dtype=np.float32),
            np.array([255.0, 64.0, 160.0], dtype=np.float32),
            np.array([200.0, 255.0, 64.0], dtype=np.float32),
            np.array([64.0, 160.0, 255.0], dtype=np.float32),
        )

        for idx, channel in enumerate(self._data):
            if idx >= len(self._channel_visibility) or not self._channel_visibility[idx]:
                continue
            span = max(float(np.max(channel) - np.min(channel)), 1e-9)
            norm = (channel - float(np.min(channel))) / span
            color = channel_colors[idx % len(channel_colors)]
            display += norm[..., None] * color * 0.65
            active |= (channel >= low) & (channel <= high)

        display = np.clip(display, 0, 255)
        display[active] = 255

        rgb = display.astype(np.uint8)
        qimg = QImage(rgb.tobytes(), self._w, self._h, 3 * self._w, QImage.Format.Format_RGB888)
        self._display_qimage = qimg.copy()
        self.image_item.setPixmap(QPixmap.fromImage(self._display_qimage))
        self.scene.setSceneRect(QRectF(self._display_qimage.rect()))

    # -- ROIs ------------------------------------------------------------------- #

    def add_roi(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            return
        low, high = self.coerced_thresholds()
        self._rois.append(
            MaskRoi(
                points=[(float(x), float(y)) for x, y in points],
                threshold_low=float(low),
                threshold_high=float(high),
                active_channels=self._channel_visibility.copy(),
            )
        )
        self.sync_rois()
        self.set_dirty(True)

    def sync_rois(self) -> None:
        """Rebuild the table and the drawn outlines from ``self._rois``.

        Both are numbered by position, so both are rebuilt together rather
        than edited in place: a table row is numbered by Qt's vertical header,
        which is the row index, and the label on the image is that same index.
        """
        self.roi_table.setRowCount(len(self._rois))
        for row, roi in enumerate(self._rois):
            channels = ",".join(
                str(i + 1) for i, active in enumerate(roi.active_channels) if active
            )
            for col, text in enumerate(
                (
                    f"{roi.threshold_low:.1f}",
                    f"{roi.threshold_high:.1f}",
                    channels if channels else "-",
                )
            ):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.roi_table.setItem(row, col, item)
        self.fit_roi_table_height()
        self.image_view.set_rois(self._rois)

    def show_roi_menu(self, pos) -> None:
        row = self.roi_table.rowAt(pos.y())
        if row < 0 or row >= len(self._rois):
            return
        self.roi_table.selectRow(row)
        viewport = self.roi_table.viewport()
        if viewport is None:
            return
        menu = QMenu(self.roi_table)
        # Connected rather than compared against exec()'s return value: Qt
        # hides the menu before it emits triggered, so the dialog opens with
        # the menu already gone.
        change = QAction("Change thresholds...", menu)
        change.triggered.connect(lambda _checked=False, r=row: self.change_roi_thresholds(r))
        delete = QAction("Delete", menu)
        delete.triggered.connect(lambda _checked=False, r=row: self.delete_roi(r))
        menu.addActions([change, delete])
        menu.exec(viewport.mapToGlobal(pos))

    def change_roi_thresholds(self, row: int) -> None:
        if row < 0 or row >= len(self._rois):
            return
        dialog = RoiThresholdDialog(self, row)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if row >= len(self._rois):
            return
        low, high = dialog.values()
        self._rois[row] = replace(
            self._rois[row], threshold_low=float(low), threshold_high=float(high)
        )
        self.sync_rois()
        self.set_dirty(True)

    def delete_roi(self, row: int) -> None:
        if row < 0 or row >= len(self._rois):
            return
        del self._rois[row]
        self.sync_rois()
        self.set_dirty(True)

    def rois(self) -> list[MaskRoi]:
        return self._rois

    def data_min(self) -> float:
        return self._data_min

    def data_max(self) -> float:
        return self._data_max

    def has_rois(self) -> bool:
        return len(self._rois) > 0

    def is_dirty(self) -> bool:
        return self._dirty

    def set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        self._dirty = dirty
        self.dirty_state_changed.emit(dirty)

    # -- generating and using the mask -------------------------------------------- #

    def generate_mask(self, rois: list[MaskRoi] | None = None) -> np.ndarray | None:
        """The mask *rois* would produce, defaulting to the drawn ones.

        Taking the list is what lets the threshold dialog preview an edit
        without writing it first: it passes a copy with one ROI substituted.
        """
        rois = self._rois if rois is None else rois
        if not rois:
            return None
        final_mask = np.zeros((self._h, self._w), dtype=np.uint8)
        for roi in rois:
            if len(roi.points) < 3:
                continue
            polygon = np.array(
                [[int(round(x)), int(round(y))] for x, y in roi.points], dtype=np.int32
            ).reshape(-1, 1, 2)
            roi_mask = np.zeros((self._h, self._w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [polygon], 255)

            active = np.zeros((self._h, self._w), dtype=bool)
            for idx, channel in enumerate(self._data):
                if idx >= len(roi.active_channels) or not roi.active_channels[idx]:
                    continue
                active |= (channel >= roi.threshold_low) & (channel <= roi.threshold_high)

            final_mask[(roi_mask == 255) & active] = 255
        return final_mask

    def preview_mask(self) -> None:
        mask = self.generate_mask()
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
        """File the drawn mask as a ``Mask2D`` dataset. The way a mask is used.

        A dataset with no run behind it: ``Provenance`` needs only a
        ``program_key``, and the ``run_id`` of 0 it defaults to is what says
        nothing acquired this. One ``append`` and it is done -- a mask is drawn
        once, not streamed.
        """
        library = self.library()
        if library is None:
            QMessageBox.warning(
                self, "No Library", "This panel is not attached to the open data yet."
            )
            return
        mask = self.generate_mask()
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
        )
        try:
            dataset.append(mask)
        except Exception as exc:  # noqa: BLE001 - reported, not raised: Qt slot
            QMessageBox.critical(self, "Add Failed", f"Could not file that mask: {exc}")
            return
        library.add(dataset)
        self.set_dirty(False)
        self.name_edit.clear()

    def save_mask(self) -> None:
        mask = self.generate_mask()
        if mask is None:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            "",
            "PNG (*.png);;TIFF (*.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        try:
            written = write_mask(path, mask)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to save mask to {path}: {exc}")
            return
        self.set_dirty(False)
        QMessageBox.information(self, "Mask Saved", f"Wrote a mask to {written}.")
