"""Re-thresholding one ROI against a live preview of the resulting mask."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.plugins.data_panels.mask_editor.range_slider import RangeSlider

from .canvas import MaskRoi

if TYPE_CHECKING:  # pragma: no cover
    from .panel import MaskEditorPanel


class MaskPreviewLabel(QLabel):
    """Shows a mask scaled to fit, re-fitting on every resize: the layout
    stretches the label after the mask is set, which would clip a pixmap
    scaled only once."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_source(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self.rescale()

    def rescale(self) -> None:
        if self._source is None:
            return
        # Nearest-neighbour: smoothing would invent greys a two-value mask lacks.
        super().setPixmap(
            self._source.scaled(
                self.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.rescale()


class RoiThresholdDialog(QDialog):
    """Re-threshold one ROI, previewing the whole mask: a threshold is chosen
    for how the finished mask looks, with the other ROIs as context. The stored
    ROI is untouched until accepted, so Cancel needs no undo."""

    def __init__(self, editor: MaskEditorPanel, index: int):
        super().__init__(editor)
        self.editor = editor
        self.index = index
        self.setWindowTitle(f"ROI {index + 1} thresholds")

        layout = QVBoxLayout(self)
        self.preview_label = MaskPreviewLabel(self)
        layout.addWidget(self.preview_label, 1)
        layout.addLayout(self.build_threshold_row(editor.rois()[index]))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(420, 520)
        self.update_preview()

    def build_threshold_row(self, roi: MaskRoi) -> QHBoxLayout:
        int_min = int(np.floor(self.editor.data_min()))
        int_max = int(np.ceil(self.editor.data_max()))
        low = max(int_min, min(int_max, int(round(roi.threshold_low))))
        high = max(low, min(int_max, int(round(roi.threshold_high))))

        self.low_spin = QSpinBox(self)
        self.high_spin = QSpinBox(self)
        for spin, value in ((self.low_spin, low), (self.high_spin, high)):
            spin.setRange(int_min, int_max)
            spin.setFixedWidth(74)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.setValue(value)
            spin.valueChanged.connect(self.on_spin_changed)
        self.slider = RangeSlider(self)
        self.slider.setRange(int_min, int_max)
        self.slider.setValues(low, high)
        self.slider.values_changed.connect(self.on_slider_changed)

        row = QHBoxLayout()
        row.addWidget(self.low_spin)
        row.addWidget(self.slider, 1)
        row.addWidget(self.high_spin)
        return row

    def values(self) -> tuple[int, int]:
        low, high = self.low_spin.value(), self.high_spin.value()
        return (high, low) if high < low else (low, high)

    def write_values(self, low: int, high: int) -> None:
        controls = (self.low_spin, self.high_spin, self.slider)
        for control in controls:
            control.blockSignals(True)
        self.low_spin.setValue(low)
        self.high_spin.setValue(high)
        self.slider.setValues(low, high)
        for control in controls:
            control.blockSignals(False)

    def on_spin_changed(self, _value: int) -> None:
        self.write_values(*self.values())
        self.update_preview()

    def on_slider_changed(self, low: int, high: int) -> None:
        self.write_values(low, high)
        self.update_preview()

    def previewed_rois(self) -> list[MaskRoi]:
        low, high = self.values()
        rois = list(self.editor.rois())
        # A frame of a new shape while the dialog is open clears the ROIs.
        if self.index < len(rois):
            rois[self.index] = replace(
                rois[self.index], threshold_low=float(low), threshold_high=float(high)
            )
        return rois

    def update_preview(self) -> None:
        mask = self.editor.generate_mask(self.previewed_rois())
        if mask is None:
            self.preview_label.setText("No ROI to preview.")
            return
        height, width = mask.shape
        qimg = QImage(mask.tobytes(), width, height, width, QImage.Format.Format_Grayscale8).copy()
        self.preview_label.set_source(QPixmap.fromImage(qimg))
