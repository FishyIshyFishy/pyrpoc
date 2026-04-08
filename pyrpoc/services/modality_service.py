from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.array_contracts import matches_array_contract
from pyrpoc.backend_utils.parameter_utils import coerce_parameter_values
from pyrpoc.domain.app_state import AppState, ParameterValue
from pyrpoc.instruments.base_instrument import BaseInstrument
from pyrpoc.modalities.base_modality import BaseModality
from pyrpoc.modalities.mod_registry import modality_registry
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from .instrument_service import InstrumentService


class ModalityService(QObject):
    modality_selected = pyqtSignal(str)
    modality_params_changed = pyqtSignal(object)
    requirements_changed = pyqtSignal(bool, list)
    acq_started = pyqtSignal()
    data_ready = pyqtSignal(object)
    acq_stopped = pyqtSignal()
    acq_error = pyqtSignal(str)
    acq_warning = pyqtSignal(str)

    def __init__(self, instrument_service: InstrumentService, app_state: AppState, parent=None):
        super().__init__(parent)
        self.instrument_service = instrument_service
        self.app_state = app_state
        self._active_frame_limit: int | None = None
        self._active_output_contract: str = ""
        self._active_modality_instance: BaseModality | None = None
        self._frames_emitted: int = 0
        self._acq_warned_messages: set[str] = set()
        self.acquisition_thread: threading.Thread | None = None
        self.acquisition_stop_requested = threading.Event()
        self.acquisition_lock = threading.Lock()

    def list_available(self) -> list[dict[str, Any]]:
        return modality_registry.describe_all()

    def select_modality(self, key: str) -> None:
        if self.app_state.modality.running:
            self.stop()

        try:
            cls = modality_registry.get_class(key)
        except KeyError:
            self.app_state.modality.selected_key = None
            self.app_state.modality.selected_class = None
            self.app_state.modality.instance = None
            self.app_state.modality.configured_params = []
            self.acq_error.emit(f"unknown modality '{key}'")
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
        if self.app_state.modality.selected_class is None:
            return {}
        return self.app_state.modality.selected_class.get_contract()

    def validate_required_instruments(self) -> tuple[bool, list[type[BaseInstrument]]]:
        if self.app_state.modality.selected_class is None:
            missing: list[type[BaseInstrument]] = []
            self.requirements_changed.emit(True, [])
            return True, missing

        missing: list[type[BaseInstrument]] = []
        for required_cls in self.app_state.modality.selected_class.REQUIRED_INSTRUMENTS:
            instances = self.instrument_service.get_instances_by_class(required_cls)
            if not instances:
                missing.append(required_cls)

        ok = len(missing) == 0
        self.requirements_changed.emit(ok, [cls.__name__ for cls in missing])
        if self.app_state.modality.running and not ok:
            self.acq_error.emit("required instrument removed during acquisition")
            self.stop()
        return ok, missing

    def configure(self, raw_params: dict[str, Any]) -> None:
        instance = self.app_state.modality.instance
        selected_class = self.app_state.modality.selected_class
        if instance is None or selected_class is None:
            raise RuntimeError("no modality selected")

        ok, missing = self.validate_required_instruments()
        if not ok:
            missing_names = ", ".join(cls.__name__ for cls in missing)
            msg = f"missing required instruments: {missing_names}"
            self.acq_error.emit(msg)
            raise RuntimeError(msg)

        try:
            cleaned_params = coerce_parameter_values(selected_class.PARAMETERS, raw_params)
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise

        bound: dict[type[BaseInstrument], BaseInstrument] = {}
        for required_cls in selected_class.REQUIRED_INSTRUMENTS:
            bound_instances = self.instrument_service.get_instances_by_class(required_cls)
            if not bound_instances:
                raise RuntimeError(f"required instrument {required_cls.__name__} disappeared before configure")
            bound[required_cls] = bound_instances[0]
        for optional_cls in selected_class.OPTIONAL_INSTRUMENTS:
            optional_instances = self.instrument_service.get_instances_by_class(optional_cls)
            if optional_instances:
                bound[optional_cls] = optional_instances[0]

        allowed_types = tuple(selected_class.ALLOWED_OPTOCONTROLS)
        bound_opto: list[BaseOptoControl] = []
        for control in self.app_state.optocontrols:
            if not control.enabled:
                continue
            if not allowed_types:
                continue
            if not isinstance(control, allowed_types):
                continue
            control.prepare_for_acquisition()
            bound_opto.append(control)

        instance.configure(cleaned_params, bound, bound_opto)
        self.set_parameter_values(cleaned_params)

    def start(self, *, force_continuous: bool = False) -> None:
        with self.acquisition_lock:
            instance, contract, frame_limit = self._prepare_acquisition_start(force_continuous=force_continuous)
            self.acquisition_stop_requested = threading.Event()
            self._frames_emitted = 0
            self._active_frame_limit = frame_limit
            self._active_output_contract = contract
            self._active_modality_instance = instance
            try:
                self._acq_warned_messages = set()
                instance._warn_callback = self._emit_acq_warning
                instance.prepare_acquisition_storage(frame_limit=frame_limit)
                self.app_state.modality.running = True
                self.acq_started.emit()
                self.acquisition_thread = instance.acquire_continuous(
                    on_frame=self.handle_frame,
                    frame_limit=frame_limit,
                    should_stop=self.acquisition_stop_requested.is_set,
                    on_error=self.handle_acquisition_error,
                    on_finished=self.handle_acquisition_finished,
                )
            except Exception:
                self.app_state.modality.running = False
                self._active_modality_instance = None
                self._active_output_contract = ""
                raise

    def handle_frame(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"modality returned {type(data).__name__}, expected numpy.ndarray")

        contract = self._active_output_contract
        if contract and not matches_array_contract(data, contract):
            raise ValueError(
                f"modality returned payload not matching contract '{contract}', "
                f"shape={data.shape}, dtype={data.dtype}"
            )

        instance = self._require_active_instance()
        instance.save_acquired_frame(data, frame_index=self._frames_emitted)
        self._frames_emitted += 1
        self.data_ready.emit(data)

    def _emit_acq_warning(self, message: str) -> None:
        if message not in self._acq_warned_messages:
            self._acq_warned_messages.add(message)
            self.acq_warning.emit(message)

    def handle_acquisition_error(self, error: Exception) -> None:
        self.acq_error.emit(str(error))

    def handle_acquisition_finished(self, frame_count: int, error: Exception | None) -> None:
        try:
            self.app_state.modality.running = False
        finally:
            instance = self._active_modality_instance
            if instance is not None:
                try:
                    instance.finalize_acquisition_storage(
                        frame_count=frame_count,
                        frame_limit=self._active_frame_limit,
                        error=error,
                    )
                except Exception as finalize_exc:
                    self.acq_error.emit(str(finalize_exc))
            self.acq_stopped.emit()
            self.acquisition_thread = None
            self._active_modality_instance = None
            self._active_output_contract = ""

    def stop(self) -> None:
        self.acquisition_stop_requested.set()
        instance = self._active_modality_instance or self.app_state.modality.instance
        if instance is None:
            return
        try:
            if self.app_state.modality.running:
                instance.stop()
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise

    def set_parameter_values(self, raw_params: dict[str, Any]) -> None:
        self.app_state.modality.configured_params = [ParameterValue(label=k, value=v) for k, v in raw_params.items()]
        self.modality_params_changed.emit(dict(raw_params))

    def get_parameter_values(self) -> dict[str, Any]:
        return {entry.label: entry.value for entry in self.app_state.modality.configured_params}

    def _prepare_acquisition_start(self, *, force_continuous: bool) -> tuple[BaseModality, str, int | None]:
        selected_class = self.app_state.modality.selected_class
        instance = self.app_state.modality.instance
        if selected_class is None or instance is None:
            raise RuntimeError("no modality selected")
        if not self.app_state.modality.configured_params:
            raise RuntimeError("modality is not configured")
        if self.app_state.modality.running:
            raise RuntimeError("acquisition already running")

        ok, missing = self.validate_required_instruments()
        if not ok:
            missing_names = ", ".join(cls.__name__ for cls in missing)
            msg = f"cannot start acquisition, missing required instruments: {missing_names}"
            self.acq_error.emit(msg)
            raise RuntimeError(msg)

        frame_limit = None if force_continuous else instance.get_frame_limit()
        contract = str(getattr(selected_class, "OUTPUT_DATA_CONTRACT", ""))
        return instance, contract, frame_limit

    def _require_active_instance(self) -> BaseModality:
        instance = self._active_modality_instance
        if instance is None:
            raise RuntimeError("acquisition callback received without active modality instance")
        return instance
