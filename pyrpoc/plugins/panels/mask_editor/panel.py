"""The Mask Editor: draw masks over an acquired image. It holds the list of drawn masks,
which programs read."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QAction, QImage, QPalette, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.plugins.panels.mask_editor.range_slider import RangeSlider
from pyrpoc.plugins.panels.mask_editor.transforms import normalize_channels
from pyrpoc.plugins.program_io.inputs import MASKS, Mask
from pyrpoc.plugins.program_io.outputs import ImageOutput
from pyrpoc.utils.file_formats.png import read_png, write_png
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry
from pyrpoc.utils.widgets.stage_link import StageLink

from .canvas import MaskImageView, MaskRoi, Threshold
from .dialog import RoiThresholdDialog
from .preview import MaskPreview
from .sidebar import MaskPicker, RegionList, hint, section_title

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

# The sidebar's width beside the views, and the narrowest the two views may get beside it.
SIDEBAR_WIDTH = 260
MIN_VIEWS_WIDTH = 320
# The smallest either view may get, so a small dock still shows both.
MIN_VIEW_SIDE = 120
# Below this width, a sidebar under the image stacks its two columns.
TWO_COLUMN_WIDTH = 480
# While the sidebar is under the image: the tallest a list may get, and the share of the
# panel's height the sidebar may take before it scrolls.
LIST_HEIGHT_BELOW = 140
SIDEBAR_SHARE_BELOW = 0.45
# A loaded picture is lit where brighter than mid-grey.
LIT_ABOVE = 127


@data_panel_registry.register("mask_editor")
class MaskEditorPanel(DataPanel[ImageOutput]):
    """Draw thresholded polygon ROIs over an acquired image, one set per mask, beside a preview
    of the mask being drawn. Every edit rewrites that mask in the drawn masks, which programs
    read."""

    display_name = "Mask Editor"
    group = "Mask editors"
    order = 30
    misc = True
    renders = ImageOutput

    def __init__(self, library: Library, inputs: ProgramInputs, stage: StageLink):
        super().__init__(library, inputs, stage)
        # The ROIs of each mask, by id, and the mask being drawn on: "" until one is chosen.
        self.drafts: dict[str, list[MaskRoi]] = {}
        self.current = ""
        self._rois: list[MaskRoi] = []
        self.apply_new_data(None)

        self.scene = QGraphicsScene(self)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.image_view = MaskImageView(self.scene, self)
        self.preview = MaskPreview(self)
        self.sidebar = QWidget()
        self.columns = QBoxLayout(QBoxLayout.Direction.TopToBottom, self.sidebar)
        self.columns.setContentsMargins(0, 0, 0, 0)
        self.columns.setSpacing(16)
        self.columns.addWidget(self.build_masks_column(), 1)
        self.columns.addWidget(self.build_details_column(), 1)
        self.split = QBoxLayout(QBoxLayout.Direction.LeftToRight, self.body)
        self.split.setContentsMargins(0, 4, 0, 0)
        self.split.setSpacing(12)
        self.split.addWidget(self.build_views(), 1)
        self.split.addWidget(self.build_sidebar_scroll())

        self.rebuild_channel_boxes()
        self.reset_threshold_controls()
        self.update_view_image()
        inputs.subscribe(self.on_input_changed)
        self.show_masks()
        self.connect_source()
        self.arrange()

    def build_views(self) -> QWidget:
        """The image to draw on and the preview of the mask, each under its title, sharing the
        space equally."""
        views = QWidget(self.body)
        self.pair = QBoxLayout(QBoxLayout.Direction.LeftToRight, views)
        self.pair.setContentsMargins(0, 0, 0, 0)
        self.pair.setSpacing(12)
        self.preview_title = section_title("Mask", views)
        for title, view in (
            (section_title("Draw", views), self.image_view),
            (self.preview_title, self.preview),
        ):
            view.setMinimumSize(MIN_VIEW_SIDE, MIN_VIEW_SIDE)
            column = QVBoxLayout()
            column.setSpacing(4)
            column.addWidget(title)
            column.addWidget(view, 1)
            self.pair.addLayout(column, 1)
        return views

    def build_sidebar_scroll(self) -> QScrollArea:
        """The sidebar in a frameless scroll area, so a small dock scrolls it rather than
        refusing to shrink."""
        self.sidebar_scroll = QScrollArea(self.body)
        self.sidebar_scroll.setWidget(self.sidebar)
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Painted by the panel behind it, so the sidebar sits on the panel's own background.
        self.sidebar_scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        return self.sidebar_scroll

    def build_masks_column(self) -> QWidget:
        """The drawn masks, with a row after them to add another."""
        column = QWidget(self.sidebar)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(section_title("Masks", column))
        self.mask_picker = MaskPicker(column)
        self.mask_picker.chosen.connect(self.choose_mask)
        self.mask_picker.renamed.connect(self.rename_mask)
        self.mask_picker.new_requested.connect(self.new_mask)
        self.mask_picker.load_requested.connect(self.load_mask)
        self.mask_picker.export_requested.connect(self.save_mask)
        self.mask_picker.delete_requested.connect(self.delete_mask)
        layout.addWidget(self.mask_picker, 1)
        self.masks_hint = hint("Draw on the image to start a mask.", column)
        layout.addWidget(self.masks_hint)
        layout.addStretch(0)
        return column

    def build_details_column(self) -> QWidget:
        """The selected mask's regions, then how new regions are drawn."""
        column = QWidget(self.sidebar)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(section_title("Regions", column))
        self.region_list = RegionList(column)
        self.region_list.customContextMenuRequested.connect(self.show_roi_menu)
        layout.addWidget(self.region_list, 1)
        self.regions_hint = hint("Drag on the image to add a region.", column)
        layout.addWidget(self.regions_hint)
        layout.addSpacing(12)
        layout.addWidget(section_title("Drawing", column))
        layout.addLayout(self.build_channels_row(column))
        layout.addLayout(self.build_threshold_rows(column))
        layout.addStretch(0)
        return column

    def build_channels_row(self, parent: QWidget) -> QHBoxLayout:
        self.channels_row = QHBoxLayout()
        self.channels_row.addWidget(QLabel("Show", parent))
        self.channel_boxes: list[QCheckBox] = []
        return self.channels_row

    def build_threshold_rows(self, parent: QWidget) -> QVBoxLayout:
        """A checkbox turning thresholds on, the span between two handles, and the span's ends
        as numbers."""
        # Off by default, so a drawn shape is the mask unless a threshold is asked for.
        self.apply_thresholds = QCheckBox("Keep only pixels in range", parent)
        self.apply_thresholds.setToolTip(
            "On: a region drawn now keeps only the pixels inside the span, on the shown "
            "channels. Off: it keeps every pixel of the drawn shape."
        )
        self.apply_thresholds.toggled.connect(self.on_apply_thresholds_toggled)
        self.threshold_slider = RangeSlider(parent)
        self.threshold_slider.setToolTip("Pixels inside the accented span count toward the mask.")
        self.threshold_slider.values_changed.connect(self.on_slider_changed)
        self.low_spin = QSpinBox(parent)
        self.high_spin = QSpinBox(parent)
        ends = QHBoxLayout()
        for label, spin in (("Low", self.low_spin), ("High", self.high_spin)):
            spin.valueChanged.connect(self.on_threshold_changed)
            ends.addWidget(QLabel(label, parent))
            ends.addWidget(spin, 1)
        self.enable_threshold_controls(False)
        rows = QVBoxLayout()
        rows.addWidget(self.apply_thresholds)
        rows.addWidget(self.threshold_slider)
        rows.addLayout(ends)
        return rows

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.arrange()

    def arrange(self) -> None:
        """Put the sidebar beside the views or under them, and the two views side by side or
        one above the other: whichever leaves the views largest."""
        # The panel's size, not the body's, which is laid out only after this runs.
        width, height = self.width(), self.height() - self.source.height()
        beside_area = (width - SIDEBAR_WIDTH, height)
        below_area = (width, height * (1 - SIDEBAR_SHARE_BELOW))
        roomier = view_side(*beside_area) >= view_side(*below_area)
        beside = width >= SIDEBAR_WIDTH + MIN_VIEWS_WIDTH and roomier
        across = side_by_side(*(beside_area if beside else below_area))
        self.pair.setDirection(
            QBoxLayout.Direction.LeftToRight if across else QBoxLayout.Direction.TopToBottom
        )
        self.arrange_sidebar(beside, width, height)

    def arrange_sidebar(self, beside: bool, width: int, height: int) -> None:
        """A sidebar beside the views is one tall column; under them, two columns side by side
        when wide enough, kept to a share of the height."""
        if beside:
            self.split.setDirection(QBoxLayout.Direction.LeftToRight)
            self.columns.setDirection(QBoxLayout.Direction.TopToBottom)
            self.sidebar_scroll.setFixedWidth(SIDEBAR_WIDTH)
            self.sidebar_scroll.setMaximumHeight(16_777_215)
        else:
            self.split.setDirection(QBoxLayout.Direction.TopToBottom)
            two = width >= TWO_COLUMN_WIDTH
            direction = (
                QBoxLayout.Direction.LeftToRight if two else QBoxLayout.Direction.TopToBottom
            )
            self.columns.setDirection(direction)
            self.sidebar_scroll.setMinimumWidth(0)
            self.sidebar_scroll.setMaximumWidth(16_777_215)
            fits = self.sidebar.sizeHint().height()
            self.sidebar_scroll.setMaximumHeight(min(fits, int(height * SIDEBAR_SHARE_BELOW)))
        list_height = 16_777_215 if beside else LIST_HEIGHT_BELOW
        self.mask_picker.setMaximumHeight(list_height)
        self.region_list.setMaximumHeight(list_height)

    def enable_threshold_controls(self, enabled: bool) -> None:
        for control in (self.low_spin, self.high_spin, self.threshold_slider):
            control.setEnabled(enabled)

    def on_apply_thresholds_toggled(self, checked: bool) -> None:
        self.enable_threshold_controls(checked)
        self.update_view_image()

    def masks(self) -> tuple[Mask, ...]:
        return self.inputs.get(MASKS.key)

    def show_masks(self) -> None:
        masks = self.masks()
        self.mask_picker.show_masks(masks, self.current)
        self.masks_hint.setVisible(not masks)

    def on_input_changed(self, key: str) -> None:
        """Follow the drawn masks, which another Mask Editor may have changed."""
        if key != MASKS.key:
            return
        ids = {mask.id for mask in self.masks()}
        self.drafts = {mask_id: rois for mask_id, rois in self.drafts.items() if mask_id in ids}
        if self.current and self.current not in ids:
            self.choose_mask("")
        self.show_masks()
        self.update_view_image()

    def choose_mask(self, mask_id: str) -> None:
        """Draw on mask ``mask_id`` from now on; "" draws on none until an ROI starts one."""
        self.current = mask_id
        self._rois = self.drafts.setdefault(mask_id, []) if mask_id else []
        self.sync_rois()
        self.update_view_image()
        self.state_changed.emit()

    def new_mask(self) -> None:
        """Add an empty mask on the image's grid, and draw on it."""
        mask = Mask(np.zeros((self._h, self._w), dtype=np.uint8), self.free_label(), Mask.new_id())
        # Chosen before it is written, so the list selects it when it hears of it.
        self.drafts[mask.id] = []
        self.choose_mask(mask.id)
        self.inputs.set(MASKS.key, (*self.masks(), mask))

    def free_label(self) -> str:
        taken = {mask.label for mask in self.masks()}
        return next(f"mask {n}" for n in itertools.count(1) if f"mask {n}" not in taken)

    def delete_mask(self, gone: str) -> None:
        self.inputs.set(MASKS.key, tuple(mask for mask in self.masks() if mask.id != gone))

    def rename_mask(self, mask_id: str, label: str) -> None:
        masks = tuple(replace(m, label=label) if m.id == mask_id else m for m in self.masks())
        self.inputs.set(MASKS.key, masks)

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
            box = QCheckBox(f"C{idx + 1}", self.sidebar)
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

    def draw(self, program_output: ImageOutput) -> None:
        """Show its newest image, keeping the ROIs if it has the same shape and dropping them
        if not."""
        # So the editor stays usable during a live run and across runs.
        if not len(program_output):
            return
        scaled = normalize_channels(program_output[-1]) * 255.0
        if self._data.shape == scaled.shape:
            self.apply_new_data(scaled)
            self.update_view_image()
            return
        self.reset_to(scaled)

    def clear(self) -> None:
        self.reset_to(None)

    def reset_to(self, scaled: np.ndarray | None) -> None:
        """Start over on ``scaled``: fresh channels and thresholds, keeping only the ROIs of
        masks drawn on a grid of its shape."""
        # An ROI is in pixels of the image it was drawn on, so on another grid it is misplaced.
        self.apply_new_data(scaled)
        on_grid = {m.id for m in self.masks() if m.array.shape == (self._h, self._w)}
        self.drafts = {mask_id: rois for mask_id, rois in self.drafts.items() if mask_id in on_grid}
        self._rois = self.drafts.setdefault(self.current, []) if self.current else []
        self.sync_rois()
        self.rebuild_channel_boxes()
        self.reset_threshold_controls()
        self.update_view_image()
        self.image_view.fit()

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
        """Draw the shown channels, whiten what a threshold keeps, and tint the pixels of the
        mask being drawn."""
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
        self.show_preview()
        # Whitening shows what a threshold keeps; with none, the image is left plain.
        if self.apply_thresholds.isChecked():
            display[active] = 255
        lit = self.current_pixels()
        if lit is not None:
            # The mask itself, not just the outlines, so a thresholded region shows what it keeps.
            tint = self.palette().color(QPalette.ColorRole.Highlight)
            display[lit] = display[lit] * 0.45 + np.array(tint.getRgb()[:3]) * 0.55

        rgb = np.ascontiguousarray(display.astype(np.uint8))
        qimg = QImage(rgb.tobytes(), self._w, self._h, 3 * self._w, QImage.Format.Format_RGB888)
        self._display_qimage = qimg.copy()
        self.image_item.setPixmap(QPixmap.fromImage(self._display_qimage))
        self.scene.setSceneRect(QRectF(self._display_qimage.rect()))

    def show_preview(self) -> None:
        """Show the mask being drawn as it is saved, or the image's grid blank when there is
        none."""
        mask = self.current_mask()
        self.preview_title.setText(f"MASK · {mask.label}".upper() if mask else "MASK")
        blank = np.zeros((self._h, self._w), dtype=np.uint8)
        self.preview.show_image(blank if mask is None else mask.as_image())

    def current_mask(self) -> Mask | None:
        return next((m for m in self.masks() if m.id == self.current), None)

    def current_pixels(self) -> np.ndarray | None:
        """The lit pixels of the mask being drawn, if it is on this image's grid."""
        mask = self.current_mask()
        if mask is None or mask.array.shape != (self._h, self._w):
            return None
        return mask.array > 0

    def add_roi(self, points: list[tuple[float, float]]) -> None:
        """File a drawn polygon. A click without a drag gives fewer than three
        points, which is not a region."""
        if len(points) < 3:
            return
        if not self.current:
            self.new_mask()
        threshold = None
        if self.apply_thresholds.isChecked():
            low, high = self.coerced_thresholds()
            threshold = Threshold(float(low), float(high), self._channel_visibility.copy())
        self._rois.append(MaskRoi(points=points, threshold=threshold, pixels=None))
        self.sync_rois()
        self.put_mask()

    def sync_rois(self) -> None:
        """Rebuild the region list and the drawn outlines together."""
        # Both are numbered by position, so editing either in place would put them out of step.
        self.region_list.show_regions(self._rois)
        self.region_list.setVisible(bool(self._rois))
        self.regions_hint.setVisible(not self._rois)
        self.image_view.set_rois(self._rois)

    def show_roi_menu(self, pos: QPoint) -> None:
        item = self.region_list.itemAt(pos)
        viewport = self.region_list.viewport()
        if item is None or viewport is None:
            return
        row = self.region_list.row(item)
        self.region_list.setCurrentRow(row)
        menu = QMenu(self.region_list)
        # Connected rather than read from exec()'s return: Qt hides the menu
        # before it emits, so the dialog opens with the menu already gone.
        change = QAction("Change thresholds...", menu)
        threshold = self._rois[row].threshold
        if threshold is None:
            # A shape drawn without thresholds has none to change.
            change.setEnabled(False)
        else:
            change.triggered.connect(
                lambda _checked=False, r=row, t=threshold: self.change_roi_thresholds(r, t)
            )
        delete = QAction("Delete", menu)
        delete.triggered.connect(lambda _checked=False, r=row: self.delete_roi(r))
        menu.addActions([change, delete])
        menu.exec(viewport.mapToGlobal(pos))

    def change_roi_thresholds(self, row: int, threshold: Threshold) -> None:
        dialog = RoiThresholdDialog(self, row, threshold)
        # A frame of a new shape during the modal dialog clears the ROIs.
        if dialog.exec() != QDialog.DialogCode.Accepted or row >= len(self._rois):
            return
        low, high = dialog.values()
        changed = replace(threshold, low=float(low), high=float(high))
        self._rois[row] = replace(self._rois[row], threshold=changed)
        self.sync_rois()
        self.put_mask()

    def delete_roi(self, row: int) -> None:
        # A frame of a new shape while the menu was open clears the ROIs.
        if row >= len(self._rois):
            return
        del self._rois[row]
        self.sync_rois()
        self.put_mask()

    def rois(self) -> list[MaskRoi]:
        return self._rois

    def data_min(self) -> float:
        return self._data_min

    def data_max(self) -> float:
        return self._data_max

    def generate_mask(self, rois: list[MaskRoi]) -> np.ndarray | None:
        """The mask ``rois`` produce, or None for no ROIs."""
        # Taking the list lets the threshold dialog preview an edit before it is written.
        if not rois:
            return None
        return mask_from_rois(rois, self._data)

    def put_mask(self) -> None:
        """Rewrite the mask being drawn from its ROIs; with none, it is left empty."""
        # Called on the user's edits only, never on a new frame, so a live run's frames do not
        # rewrite the masks under it.
        drawn = self.generate_mask(self._rois)
        array = np.zeros((self._h, self._w), dtype=np.uint8) if drawn is None else drawn
        masks = self.masks()
        self.inputs.set(
            MASKS.key, tuple(replace(m, array=array) if m.id == self.current else m for m in masks)
        )

    def detach(self) -> None:
        super().detach()
        self.inputs.unsubscribe(self.on_input_changed)

    def save_mask(self, mask_id: str) -> None:
        mask = next(m for m in self.masks() if m.id == mask_id)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mask", f"{mask.label}.png", "PNG (*.png)"
        )
        if not path:
            return
        written = Path(path).expanduser()
        if written.suffix.lower() != ".png":
            written = written.with_name(f"{written.name}.png")
        # A file boundary: an unwritable path.
        try:
            written.parent.mkdir(parents=True, exist_ok=True)
            write_png(written, mask.as_image())
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to save mask to {written}: {exc}")
            return
        QMessageBox.information(self, "Mask Exported", f"Wrote a mask to {written}.")

    def load_mask(self) -> None:
        """Add a mask saved as a picture, fitted to the image's grid as one region, and draw
        on it."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Mask", "", "PNG (*.png);;All Files (*)")
        if not path:
            return
        # A file boundary: a missing or unreadable picture.
        try:
            lit = read_png(Path(path)) > LIT_ABOVE
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Load Failed", f"Could not load a mask from {path}: {exc}")
            return
        loaded = Mask(lit, Path(path).stem, Mask.new_id())
        pixels = loaded.on_grid(self._h, self._w)
        if not pixels.any():
            QMessageBox.warning(self, "Nothing to Load", f"{Path(path).name} has no lit pixels.")
            return
        region = MaskRoi(points=[], threshold=None, pixels=pixels)
        mask = replace(loaded, array=mask_from_rois([region], self._data))
        # Chosen before it is written, so the list selects it when it hears of it.
        self.drafts[mask.id] = [region]
        self.choose_mask(mask.id)
        self.inputs.set(MASKS.key, (*self.masks(), mask))


def mask_from_rois(rois: list[MaskRoi], data: np.ndarray) -> np.ndarray:
    """The ``(H, W)`` mask ``rois`` draw over a ``(C, H, W)`` frame: each ROI's
    filled shape, cut to its threshold when it has one."""
    height, width = data.shape[1:]
    mask = np.zeros((height, width), dtype=np.uint8)
    for roi in rois:
        inside = roi_shape(roi, height, width)
        if roi.threshold is not None:
            inside &= in_range(data, roi.threshold)
        mask[inside] = 255
    return mask


def roi_shape(roi: MaskRoi, height: int, width: int) -> np.ndarray:
    """The pixels inside a region's shape, before any threshold: its loaded pixels, or its
    polygon filled."""
    if roi.pixels is not None:
        return roi.pixels.copy()
    polygon = np.array(
        [[int(round(x)), int(round(y))] for x, y in roi.points], dtype=np.int32
    ).reshape(-1, 1, 2)
    shape = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(shape, [polygon], 255)
    return shape == 255


def view_side(width: float, height: float) -> float:
    """The side of the largest square each of two views gets in an area, laid out the better
    way."""
    return max(min(width / 2, height), min(width, height / 2))


def side_by_side(width: float, height: float) -> bool:
    """Whether two views get more room side by side in an area than one above the other."""
    return min(width / 2, height) >= min(width, height / 2)


def in_range(data: np.ndarray, threshold: Threshold) -> np.ndarray:
    """The pixels of a ``(C, H, W)`` frame within ``threshold`` on any of its
    active channels."""
    active = np.zeros(data.shape[1:], dtype=bool)
    for channel, enabled in zip(data, threshold.active_channels, strict=False):
        if enabled:
            active |= (channel >= threshold.low) & (channel <= threshold.high)
    return active


def clear_layout_after(layout: QLayout, keep: int) -> None:
    """Delete every item in ``layout`` past the first ``keep``."""
    while layout.count() > keep:
        item = layout.takeAt(keep)
        widget = item.widget() if item is not None else None
        if widget is not None:
            widget.hide()
            widget.deleteLater()
