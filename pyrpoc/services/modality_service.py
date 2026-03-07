from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter_utils import ParameterValidationError, coerce_parameter_values
from pyrpoc.domain.app_state import AppState, ParameterValue
from pyrpoc.instruments.base_instrument import BaseInstrument
from pyrpoc.modalities.base_modality import BaseModality
from pyrpoc.modalities.mod_registry import modality_registry
from .instrument_service import InstrumentService


class ModalityService(QObject):
    modality_selected = pyqtSignal(str)
    requirements_changed = pyqtSignal(bool, list)
    acq_started = pyqtSignal()
    data_ready = pyqtSignal(object)
    acq_stopped = pyqtSignal()
    acq_error = pyqtSignal(str)

    def __init__(self, instrument_service: InstrumentService, app_state: AppState, parent=None):
        super().__init__(parent)
        self.instrument_service = instrument_service
        self.app_state = app_state

    def list_available(self) -> list[dict[str, Any]]:
        return modality_registry.describe_all()

    def select_modality(self, key: str) -> None:
        if self.app_state.modality.running:
            self.stop()

        try:
            cls = modality_registry.get_class(key)
        except KeyError as exc:
            self.app_state.modality.selected_key = None
            self.app_state.modality.selected_class = None
            self.app_state.modality.instance = None
            self.app_state.modality.configured_params = []
            self.acq_error.emit(str(exc))
            raise

        self.app_state.modality.selected_key = key
        self.app_state.modality.selected_class = cls
        self.app_state.modality.instance = cls()
        self.app_state.modality.configured_params = []
        self.modality_selected.emit(key)
        self.validate_required_instruments()

    def get_selected_parameters(self) -> dict[str, list]:
        if self.app_state.modality.selected_class is None:
            return {}
        return self.app_state.modality.selected_class.PARAMETERS

    def get_selected_contract(self) -> dict[str, Any]:
        '''
        read the selected contract from app_state
        '''
        if self.app_state.modality.selected_class is None:
            return {}
        return self.app_state.modality.selected_class.get_contract()

    def validate_required_instruments(self) -> tuple[bool, list[type[BaseInstrument]]]:
        if self.app_state.modality.selected_class is None:
            missing: list[type[BaseInstrument]] = []
            self.requirements_changed.emit(True, [])
            return True, missing

        missing = []
        for required_cls in self.app_state.modality.selected_class.REQUIRED_INSTRUMENTS:
            connected = self.instrument_service.get_connected_by_class(required_cls)
            if not connected:
                missing.append(required_cls)

        ok = len(missing) == 0
        self.requirements_changed.emit(ok, [cls.__name__ for cls in missing])
        if self.app_state.modality.running and not ok:
            self.acq_error.emit("required instrument disconnected during acquisition")
            self.stop()
        return ok, missing

    def configure(self, raw_params: dict[str, Any]) -> None:
        if self.app_state.modality.selected_class is None:
            raise RuntimeError("no modality selected")

        ok, missing = self.validate_required_instruments()
        if not ok:
            missing_names = ", ".join(cls.__name__ for cls in missing)
            msg = f"missing required instruments: {missing_names}"
            self.acq_error.emit(msg)
            raise RuntimeError(msg)

        if self.app_state.modality.instance is None:
            self.app_state.modality.instance = self.app_state.modality.selected_class()

        try:
            cleaned_params = coerce_parameter_values(self.app_state.modality.selected_class.PARAMETERS, raw_params)
        except ParameterValidationError as exc:
            self.acq_error.emit(str(exc))
            raise

        bound: dict[type[BaseInstrument], BaseInstrument] = {}
        for required_cls in self.app_state.modality.selected_class.REQUIRED_INSTRUMENTS:
            bound[required_cls] = self.instrument_service.get_connected_by_class(required_cls)[0]
        for optional_cls in self.app_state.modality.selected_class.OPTIONAL_INSTRUMENTS:
            connected = self.instrument_service.get_connected_by_class(optional_cls)
            if connected:
                bound[optional_cls] = connected[0]

        self.app_state.modality.instance.configure(cleaned_params, bound)
        self.app_state.modality.configured_params = [
            ParameterValue(label=k, value=v) for k, v in cleaned_params.items()
        ]

    def start(self) -> None:
        if self.app_state.modality.instance is None:
            raise RuntimeError("no modality selected")
        if not self.app_state.modality.configured_params:
            raise RuntimeError("modality is not configured")

        ok, missing = self.validate_required_instruments()
        if not ok:
            missing_names = ", ".join(cls.__name__ for cls in missing)
            msg = f"cannot start acquisition, missing required instruments: {missing_names}"
            self.acq_error.emit(msg)
            raise RuntimeError(msg)

        try:
            self.app_state.modality.instance.start()
            self.app_state.modality.running = True
            self.acq_started.emit()
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise

    def acquire_once(self) -> BaseData:
        if self.app_state.modality.instance is None:
            raise RuntimeError("no modality selected")
        if not self.app_state.modality.running:
            raise RuntimeError("acquisition is not running")

        try:
            data = self.app_state.modality.instance.acquire_once()
            self.data_ready.emit(data)
            return data
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise

    def stop(self) -> None:
        if self.app_state.modality.instance is None:
            return
        try:
            if self.app_state.modality.running:
                self.app_state.modality.instance.stop()
                self.app_state.modality.running = False
            self.acq_stopped.emit()
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise
