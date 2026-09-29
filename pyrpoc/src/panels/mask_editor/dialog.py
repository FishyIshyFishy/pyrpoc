"""Re-thresholding one ROI against a live preview of the resulting mask.

Its own file because it is a real dialog with its own state machine (spin
boxes and a slider that must never echo each other's change back), not a
detail of how the panel lays itself out. ``MaskPreviewLabel`` lives here
rather than in its own file because nothing else uses it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
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

from ..components.range_slider import RangeSlider
from .canvas import MaskRoi

if TYPE_CHECKING:  # pragma: no cover
    from .panel import MaskEditorPanel


class MaskPreviewLabel(QLabel):
    """Shows a mask scaled to fit, and keeps fitting it as it is resized.

    Scaling once when the mask changes is not enough: the label is stretched
    by its layout after that, and a pixmap sized to the old geometry is
    silently clipped rather than re-fitted.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap
        self.rescale()

    def rescale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        # Nearest-neighbour: a mask is two values, and smoothing invents
        # greys that no pixel of it has.
        super().setPixmap(
            self._source.scaled(
                self.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.rescale()


class RoiThresholdDialog(QDialog):
    """Re-threshold one ROI against a live preview of the resulting mask.

    The preview is the same array ``Preview`` shows -- the whole mask, every
    ROI -- recomputed on each change. Showing only the edited ROI's own pixels
    would answer a question nobody asked: a threshold is chosen for how the
    finished mask looks, and the other ROIs are the context that decision is
    made in.

    The editor's stored ROI is untouched until the dialog is accepted, so the
    preview runs against a substituted copy and Cancel needs no undo.
    """

    def __init__(self, editor: MaskEditorPanel, index: int):
        super().__init__(editor)
        self.editor = editor
        self.index = index
        roi = editor.rois()[index]
        self.setWindowTitle(f"ROI {index + 1} thresholds")

        int_min = int(np.floor(editor.data_min()))
        int_max = int(np.ceil(editor.data_max()))
        low = max(int_min, min(int_max, int(round(roi.threshold_low))))
        high = max(low, min(int_max, int(round(roi.threshold_high))))

        layout = QVBoxLayout(self)

        self.preview_label = MaskPreviewLabel(self)
        layout.addWidget(self.preview_label, 1)

        threshold_row = QHBoxLayout()
        self.low_spin = QSpinBox(self)
        self.high_spin = QSpinBox(self)
        for spin in (self.low_spin, self.high_spin):
            spin.setRange(int_min, int_max)
            spin.setFixedWidth(74)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.low_spin.setValue(low)
        self.high_spin.setValue(high)
        self.low_spin.valueChanged.connect(self.on_spin_changed)
        self.high_spin.valueChanged.connect(self.on_spin_changed)

        self.slider = RangeSlider(self)
        self.slider.setRange(int_min, int_max)
        self.slider.setValues(low, high)
        self.slider.values_changed.connect(self.on_slider_changed)

        threshold_row.addWidget(self.low_spin)
        threshold_row.addWidget(self.slider, 1)
        threshold_row.addWidget(self.high_spin)
        layout.addLayout(threshold_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(420, 520)
        self.update_preview()

    def values(self) -> tuple[int, int]:
        low = int(self.low_spin.value())
        high = int(self.high_spin.value())
        if high < low:
            low, high = high, low
        return low, high

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
        low, high = self.values()
        self.write_values(low, high)
        self.update_preview()

    def on_slider_changed(self, low: int, high: int) -> None:
        self.write_values(low, high)
        self.update_preview()

    def previewed_rois(self) -> list[MaskRoi]:
        low, high = self.values()
        rois = list(self.editor.rois())
        if 0 <= self.index < len(rois):
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
