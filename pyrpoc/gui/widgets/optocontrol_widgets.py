from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from pyrpoc.gui.services.display_service import DisplayService
    from pyrpoc.optocontrols.base import BaseOptoControl
    from pyrpoc.optocontrols.mask import MaskOptoControl


class BaseOptoControlWidget(QWidget):
    """Base Qt widget for a single optocontrol instance."""

    def __init__(self, control: "BaseOptoControl", on_change: Callable[[], None] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.control = control
        self.on_change = on_change

    def refresh_from_model(self) -> None:
        return None

    def request_persist(self) -> None:
        if self.on_change is not None:
            self.on_change()


class MaskOptoControlWidget(BaseOptoControlWidget):
    def __init__(
        self,
        control: "MaskOptoControl",
        on_change: Callable[[], None] | None = None,
        display_service: "DisplayService | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(control, on_change=on_change, parent=parent)
        self.control = control
        self._display_service = display_service
        self._editor = None
        self.setObjectName("maskOptoControlBody")
        self.setStyleSheet("#maskOptoControlBody, #maskOptoControlBody QWidget { background: transparent; }")

        root = QVBoxLayout(self)

        form = QFormLayout()
        self.port_spin = QSpinBox(self)
        self.port_spin.setRange(0, 1024)
        self.port_spin.setValue(int(self.control.daq_port))
        self.line_spin = QSpinBox(self)
        self.line_spin.setRange(0, 1024)
        self.line_spin.setValue(int(self.control.daq_line))
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("Mask file path")
        self.path_edit.setText(self.control.mask_path)
        form.addRow("DAQ Port", self.port_spin)
        form.addRow("DAQ Line", self.line_spin)
        form.addRow("Mask Path", self.path_edit)
        root.addLayout(form)

        button_row = QHBoxLayout()
        self.load_mask_btn = QPushButton("Load Mask", self)
        self.create_mask_btn = QPushButton("Open mask Editor", self)
        button_row.addWidget(self.load_mask_btn)
        button_row.addWidget(self.create_mask_btn)
        button_row.addStretch(1)
        root.addLayout(button_row)

        self.editor_container = QWidget(self)
        self.editor_container.setVisible(False)
        editor_layout = QVBoxLayout(self.editor_container)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source Display:", self.editor_container))
        self.source_combo = QComboBox(self.editor_container)
        source_row.addWidget(self.source_combo, 1)
        editor_layout.addLayout(source_row)
        self.editor_body_layout = QVBoxLayout()
        editor_layout.addLayout(self.editor_body_layout)
        root.addWidget(self.editor_container)
        root.addStretch(1)

        self.port_spin.valueChanged.connect(self.on_port_changed)
        self.line_spin.valueChanged.connect(self.on_line_changed)
        self.path_edit.textChanged.connect(self.on_path_changed)
        self.load_mask_btn.clicked.connect(self.on_load_mask_clicked)
        self.create_mask_btn.clicked.connect(self.on_create_mask_clicked)
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)

        self._display_signals_connected = False
        self.set_display_service(display_service)
        self.refresh_display_sources()

    def refresh_from_model(self) -> None:
        self.port_spin.blockSignals(True)
        self.line_spin.blockSignals(True)
        self.path_edit.blockSignals(True)
        self.port_spin.setValue(int(self.control.daq_port))
        self.line_spin.setValue(int(self.control.daq_line))
        self.path_edit.setText(self.control.mask_path)
        self.port_spin.blockSignals(False)
        self.line_spin.blockSignals(False)
        self.path_edit.blockSignals(False)

    def emit_change(self) -> None:
        self.request_persist()

    def on_port_changed(self, value: int) -> None:
        self.control.daq_port = int(value)
        self.emit_change()

    def on_line_changed(self, value: int) -> None:
        self.control.daq_line = int(value)
        self.emit_change()

    def on_path_changed(self, value: str) -> None:
        self.control.mask_path = value.strip()
        self.emit_change()

    def on_load_mask_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mask File",
            self.path_edit.text().strip(),
            "Image Files (*.png *.tif *.tiff *.bmp);;All Files (*)",
        )
        if not path:
            return
        image = cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE)
        if image is None or image.ndim != 2:
            QMessageBox.critical(self, "Load Failed", f"Failed to load mask from '{path}'.")
            return
        self.path_edit.setText(path)
        self.control.mask_data = image.astype(np.uint8, copy=True)
        self.emit_change()

    def on_create_mask_clicked(self) -> None:
        self.ensure_editor()
        self.set_editor_visible(True)
        self.refresh_display_sources()
        self.load_selected_display_data(warn=False)

    def ensure_editor(self) -> None:
        if self._editor is not None:
            return
        from pyrpoc.gui.panels.optocontrols.mask_editor import MaskEditorWidget

        self._editor = MaskEditorWidget(parent=self.editor_container)
        self._editor.create_mask_requested.connect(self.on_editor_create_mask)
        self._editor.mask_saved.connect(self.on_editor_mask_saved)
        self._editor.cancel_requested.connect(self.on_editor_cancel)
        self.editor_body_layout.addWidget(self._editor)

    def on_editor_cancel(self) -> None:
        self.set_editor_visible(False)

    def on_editor_create_mask(self, mask: object) -> None:
        arr = np.asarray(mask, dtype=np.uint8)
        if arr.ndim != 2:
            QMessageBox.critical(self, "Mask Error", "Generated mask must be a 2D array.")
            return
        fd, temp_path = tempfile.mkstemp(prefix="pyrpoc_mask_", suffix=".png")
        os.close(fd)
        ok = cv2.imwrite(str(Path(temp_path)), arr)
        if not ok:
            QMessageBox.critical(self, "Save Failed", f"Failed to save mask to '{temp_path}'.")
            return
        self.path_edit.setText(temp_path)
        self.control.mask_data = arr.copy()
        self.set_editor_visible(False)
        self.emit_change()

    def on_editor_mask_saved(self, path: object, mask: object) -> None:
        path_str = str(path).strip()
        if not path_str:
            return
        arr = np.asarray(mask, dtype=np.uint8)
        if arr.ndim != 2:
            return
        self.path_edit.setText(path_str)
        self.control.mask_data = arr.copy()
        self.set_editor_visible(False)
        self.emit_change()

    def set_editor_visible(self, visible: bool) -> None:
        self.editor_container.setVisible(visible)
        self.create_mask_btn.setEnabled(not visible)

    def on_display_inventory_changed(self, _state: object) -> None:
        if not self.editor_container.isVisible():
            return
        self.refresh_display_sources()

    def refresh_display_sources(self) -> None:
        current_display_id = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        if self._display_service is not None:
            for row in self._display_service.list_instances():
                display_id = row.get("display_id")
                if not isinstance(display_id, int):
                    continue
                name = str(row.get("name", f"Display {display_id}"))
                self.source_combo.addItem(name, display_id)
        if self.source_combo.count() == 0:
            self.source_combo.addItem("No displays available", None)
        if isinstance(current_display_id, int):
            idx = self.source_combo.findData(current_display_id)
            if idx >= 0:
                self.source_combo.setCurrentIndex(idx)
        self.source_combo.blockSignals(False)

    def on_source_changed(self, _index: int) -> None:
        self.load_selected_display_data(warn=True)

    def load_selected_display_data(self, warn: bool) -> None:
        if self._editor is None or self._display_service is None:
            return
        display_id = self.source_combo.currentData()
        if not isinstance(display_id, int):
            self._editor.set_image_data(None)
            return

        display = self._display_service.get_display_by_id(display_id)
        if display is None:
            self._editor.set_image_data(None)
            if warn:
                QMessageBox.warning(self, "Display Unavailable", "Selected display is no longer available.")
            return

        getter = getattr(display, "get_normalized_data_3d", None)
        if not callable(getter):
            self._editor.set_image_data(None)
            if warn:
                QMessageBox.warning(self, "No Data", "Selected display does not provide source data.")
            return

        data = getter()
        if data is None:
            self._editor.set_image_data(None)
            if warn:
                QMessageBox.warning(self, "No Data", "Selected display has no image data yet.")
            return

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 3:
            self._editor.set_image_data(None)
            if warn:
                QMessageBox.warning(self, "Invalid Data", "Display data must have shape [C,H,W].")
            return
        self._editor.set_image_data(np.clip(arr, 0.0, 1.0) * 255.0)

    def set_display_service(self, display_service: "DisplayService | None") -> None:
        self._display_service = display_service
        if self._display_service is not None and not self._display_signals_connected:
            self._display_service.display_added.connect(self.on_display_inventory_changed)
            self._display_service.display_removed.connect(self.on_display_inventory_changed)
            self._display_service.display_changed.connect(self.on_display_inventory_changed)
            self._display_signals_connected = True


def build_mask_widget(control, parent, on_change, display_service) -> BaseOptoControlWidget:
    return MaskOptoControlWidget(control, on_change=on_change, display_service=display_service, parent=parent)


optocontrol_widget_builders: dict[str, Any] = {
    "mask": build_mask_widget,
}


def build_optocontrol_widget(control, parent: QWidget | None = None, on_change=None, display_service=None) -> BaseOptoControlWidget:
    builder = optocontrol_widget_builders.get(control.optocontrol_key)
    if builder is None:
        return BaseOptoControlWidget(control, on_change=on_change, parent=parent)
    return builder(control, parent, on_change, display_service)
