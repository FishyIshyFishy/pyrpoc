from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from pyrpoc.gui.main_widgets.acquisition_mgr.handlers import (
    handle_modality_selected,
    on_parameter_values_changed,
    handle_requirements_changed,
    on_continuous_clicked,
    on_modality_selected,
    on_parameter_widgets_changed,
    on_service_error,
    on_service_warning,
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
        # Compatibility aliases for existing callers.
        self.modality_combo = self.ui.modality_combo
        self.start_btn = self.ui.start_btn
        self.continuous_btn = self.ui.continuous_btn
        self.stop_btn = self.ui.stop_btn
        self.status_label = self.ui.status_label
        self.params_container = self.ui.params_container
        self.params_layout = self.ui.params_layout
        self.param_widgets = self.state.param_widgets
        self.param_defs = self.state.param_defs

        self.wire_signals()
        self.set_acquiring_ui(False)
        self.populate_modalities()

    def wire_signals(self) -> None:
        self.modality_combo.currentTextChanged.connect(self.on_modality_selected)
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.continuous_btn.clicked.connect(self.on_continuous_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)

        self.modality_service.modality_selected.connect(self.handle_modality_selected)
        self.modality_service.modality_params_changed.connect(self.on_parameter_values_changed)
        self.modality_service.requirements_changed.connect(self.handle_requirements_changed)
        self.modality_service.acq_started.connect(self.on_acq_started)
        self.modality_service.acq_stopped.connect(self.on_acq_stopped)
        self.modality_service.acq_error.connect(self.on_service_error)
        self.modality_service.acq_warning.connect(self.on_service_warning)

    def populate_modalities(self) -> None:
        populate_modalities(self)

    def on_modality_selected(self, key: str) -> None:
        on_modality_selected(self, key)

    def handle_modality_selected(self, key: str) -> None:
        handle_modality_selected(self, key)

    def on_start_clicked(self) -> None:
        on_start_clicked(self)

    def on_continuous_clicked(self) -> None:
        on_continuous_clicked(self)

    def on_stop_clicked(self) -> None:
        on_stop_clicked(self)

    def handle_requirements_changed(self, ok: bool, missing_names: list[str]) -> None:
        handle_requirements_changed(self, ok, missing_names)

    def on_service_error(self, message: str) -> None:
        on_service_error(self, message)
        self.set_acquiring_ui(False)

    def on_service_warning(self, message: str) -> None:
        on_service_warning(self, message)

    def on_parameter_widgets_changed(self) -> None:
        on_parameter_widgets_changed(self)

    def on_parameter_values_changed(self, values: object) -> None:
        on_parameter_values_changed(self, values)

    def on_acq_started(self) -> None:
        self.status_label.setText("Status: acquiring")
        self.set_acquiring_ui(True)

    def on_acq_stopped(self) -> None:
        self.status_label.setText("Status: stopped")
        self.set_acquiring_ui(False)

    def set_acquiring_ui(self, acquiring: bool) -> None:
        self.start_btn.setEnabled(not acquiring)
        self.continuous_btn.setEnabled(not acquiring)
        self.stop_btn.setEnabled(acquiring)
