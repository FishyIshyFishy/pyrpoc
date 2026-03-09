from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget

from pyrpoc.gui.main_widgets.acquisition_mgr.handlers import (
    acquire_single_frame,
    handle_modality_selected,
    on_parameter_values_changed,
    handle_requirements_changed,
    on_continuous_clicked,
    on_modality_selected,
    on_parameter_widgets_changed,
    on_service_error,
    on_start_clicked,
    on_stop_clicked,
    populate_modalities,
)
from pyrpoc.gui.main_widgets.acquisition_mgr.state import AcquisitionManagerState
from pyrpoc.gui.main_widgets.acquisition_mgr.ui import build_acquisition_manager_ui
from pyrpoc.services.modality_service import ModalityService


class AcquisitionManagerWidget(QWidget):
    def __init__(self, modality_service: ModalityService, parent: QWidget | None = None):
        super().__init__(parent)
        self.modality_service = modality_service
        self.state = AcquisitionManagerState()
        self.ui = build_acquisition_manager_ui(self)
        self.state.continuous_timer = QTimer(self)
        self.state.continuous_timer.setInterval(250)
        self.state.continuous_timer.timeout.connect(self._acquire_single_frame)

        # Compatibility aliases for existing callers.
        self.modality_combo = self.ui.modality_combo
        self.refresh_btn = self.ui.refresh_btn
        self.start_btn = self.ui.start_btn
        self.continuous_btn = self.ui.continuous_btn
        self.stop_btn = self.ui.stop_btn
        self.status_label = self.ui.status_label
        self.params_container = self.ui.params_container
        self.params_layout = self.ui.params_layout
        self.param_widgets = self.state.param_widgets
        self.param_defs = self.state.param_defs
        self._continuous_timer = self.state.continuous_timer

        self._wire_signals()
        self._set_acquiring_ui(False)
        self._populate_modalities()

    def _wire_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._populate_modalities)
        self.modality_combo.currentTextChanged.connect(self._on_modality_selected)
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.continuous_btn.clicked.connect(self._on_continuous_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)

        self.modality_service.modality_selected.connect(self._handle_modality_selected)
        self.modality_service.modality_params_changed.connect(self._on_parameter_values_changed)
        self.modality_service.requirements_changed.connect(self._handle_requirements_changed)
        self.modality_service.acq_started.connect(self._on_acq_started)
        self.modality_service.acq_stopped.connect(self._on_acq_stopped)
        self.modality_service.acq_error.connect(self._on_service_error)

    def _populate_modalities(self) -> None:
        populate_modalities(self)

    def _on_modality_selected(self, key: str) -> None:
        on_modality_selected(self, key)

    def _handle_modality_selected(self, key: str) -> None:
        handle_modality_selected(self, key)

    def _on_start_clicked(self) -> None:
        on_start_clicked(self)

    def _on_continuous_clicked(self) -> None:
        on_continuous_clicked(self)

    def _acquire_single_frame(self) -> None:
        acquire_single_frame(self)

    def _on_stop_clicked(self) -> None:
        on_stop_clicked(self)

    def _handle_requirements_changed(self, ok: bool, missing_names: list[str]) -> None:
        handle_requirements_changed(self, ok, missing_names)

    def _on_service_error(self, message: str) -> None:
        on_service_error(self, message)
        self._set_acquiring_ui(False)

    def _on_parameter_widgets_changed(self) -> None:
        on_parameter_widgets_changed(self)

    def _on_parameter_values_changed(self, values: object) -> None:
        on_parameter_values_changed(self, values)

    def _on_acq_started(self) -> None:
        self.status_label.setText("Status: acquiring")
        self._set_acquiring_ui(True)

    def _on_acq_stopped(self) -> None:
        self.status_label.setText("Status: stopped")
        self._set_acquiring_ui(False)

    def _set_acquiring_ui(self, acquiring: bool) -> None:
        self.start_btn.setEnabled(not acquiring)
        self.continuous_btn.setEnabled(not acquiring)
        self.stop_btn.setEnabled(acquiring)
