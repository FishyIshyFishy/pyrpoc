from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter_utils import ParameterValidationError, coerce_parameter_values
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

    def __init__(self, instrument_service: InstrumentService, parent=None):
        super().__init__(parent)
        self.instrument_service = instrument_service
        self._selected_key: str | None = None
        self._selected_class: type[BaseModality] | None = None
        self._instance: BaseModality | None = None
        self._last_params: dict[str, Any] = {}
        self._running = False

    def list_available(self) -> list[dict[str, Any]]:
        return modality_registry.describe_all()

    def select_modality(self, key: str) -> None:
        if self._running:
            self.stop()

        try:
            cls = modality_registry.get_class(key)
        except KeyError as exc:
            self._selected_key = None
            self._selected_class = None
            self._instance = None
            self._last_params = {}
            self.acq_error.emit(str(exc))
            raise

        self._selected_key = key
        self._selected_class = cls
        self._instance = cls()
        self._last_params = {}
        self.modality_selected.emit(key)
        self.validate_required_instruments()

    def get_selected_parameters(self) -> dict[str, list]:
        if self._selected_class is None:
            return {}
        return self._selected_class.PARAMETERS

    def get_selected_contract(self) -> dict[str, Any]:
        if self._selected_class is None:
            return {}
        return self._selected_class.get_contract()

    def validate_required_instruments(self) -> tuple[bool, list[type[BaseInstrument]]]:
        if self._selected_class is None:
            missing: list[type[BaseInstrument]] = []
            self.requirements_changed.emit(True, [])
            return True, missing

        missing = []
        for required_cls in self._selected_class.REQUIRED_INSTRUMENTS:
            connected = self.instrument_service.get_connected_by_class(required_cls)
            if not connected:
                missing.append(required_cls)

        ok = len(missing) == 0
        self.requirements_changed.emit(ok, [cls.__name__ for cls in missing])
        if self._running and not ok:
            self.acq_error.emit("required instrument disconnected during acquisition")
            self.stop()
        return ok, missing

    def configure(self, raw_params: dict[str, Any]) -> None:
        if self._selected_class is None:
            raise RuntimeError("no modality selected")

        ok, missing = self.validate_required_instruments()
        if not ok:
            missing_names = ", ".join(cls.__name__ for cls in missing)
            msg = f"missing required instruments: {missing_names}"
            self.acq_error.emit(msg)
            raise RuntimeError(msg)

        if self._instance is None:
            self._instance = self._selected_class()

        try:
            cleaned_params = coerce_parameter_values(self._selected_class.PARAMETERS, raw_params)
        except ParameterValidationError as exc:
            self.acq_error.emit(str(exc))
            raise

        bound: dict[type[BaseInstrument], BaseInstrument] = {}
        for required_cls in self._selected_class.REQUIRED_INSTRUMENTS:
            bound[required_cls] = self.instrument_service.get_connected_by_class(required_cls)[0]
        for optional_cls in self._selected_class.OPTIONAL_INSTRUMENTS:
            connected = self.instrument_service.get_connected_by_class(optional_cls)
            if connected:
                bound[optional_cls] = connected[0]

        self._instance.configure(cleaned_params, bound)
        self._last_params = cleaned_params

    def start(self) -> None:
        if self._instance is None:
            raise RuntimeError("no modality selected")
        if not self._last_params:
            raise RuntimeError("modality is not configured")

        ok, missing = self.validate_required_instruments()
        if not ok:
            missing_names = ", ".join(cls.__name__ for cls in missing)
            msg = f"cannot start acquisition, missing required instruments: {missing_names}"
            self.acq_error.emit(msg)
            raise RuntimeError(msg)

        try:
            self._instance.start()
            self._running = True
            self.acq_started.emit()
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise

    def acquire_once(self) -> BaseData:
        if self._instance is None:
            raise RuntimeError("no modality selected")
        if not self._running:
            raise RuntimeError("acquisition is not running")

        try:
            data = self._instance.acquire_once()
            self.data_ready.emit(data)
            return data
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise

    def stop(self) -> None:
        if self._instance is None:
            return
        try:
            if self._running:
                self._instance.stop()
                self._running = False
            self.acq_stopped.emit()
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise
