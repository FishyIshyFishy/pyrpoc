from __future__ import annotations

import threading
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.structs.app_state import AppState
from pyrpoc.gui.services.instrument_service import InstrumentService
from pyrpoc.acquisition.executor import Executor
from pyrpoc.presets.base import preset_registry
from pyrpoc.acquisition.storage import FrameStorage
from pyrpoc.instruments.base import BaseInstrument
from pyrpoc.structs.acquired_data import AcquiredData
from pyrpoc.structs.parameters import ParameterValue, coerce_parameter_values

# Ensure presets + handlers register on import.
import pyrpoc.presets  # noqa: F401,E402


class AcquisitionService(QObject):
    """Drives the acquisition Executor for the selected preset.

    Keeps the GUI-facing method/signal names of the former ModalityService, so
    the acquisition panel needs no behavioural changes; internally it composes a
    preset's command source (+ optocontrol decorators) and runs the Executor on
    a background thread, emitting one AcquiredData per result.
    """

    modality_selected = pyqtSignal(str)
    modality_params_changed = pyqtSignal(object)
    requirements_changed = pyqtSignal(bool, list)
    acq_started = pyqtSignal()
    data_emitted = pyqtSignal(object)
    acq_stopped = pyqtSignal()
    acq_error = pyqtSignal(str)
    acq_warning = pyqtSignal(str)

    def __init__(self, instrument_service: InstrumentService, app_state: AppState, parent=None):
        super().__init__(parent)
        self.instrument_service = instrument_service
        self.app_state = app_state
        self.acquisition_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.warned_messages: set[str] = set()

    def list_available(self) -> list[dict[str, Any]]:
        return preset_registry.describe_all()

    def select_modality(self, key: str) -> None:
        if self.app_state.preset.running:
            self.stop()
        try:
            cls = preset_registry.get_class(key)
        except KeyError:
            self.app_state.preset.selected_key = None
            self.app_state.preset.selected_class = None
            self.app_state.preset.instance = None
            self.acq_error.emit(f"unknown preset '{key}'")
            raise
        self.app_state.preset.selected_key = key
        self.app_state.preset.selected_class = cls
        self.app_state.preset.instance = cls()
        self.modality_selected.emit(key)
        self.validate_required_instruments()

    def get_selected_parameters(self) -> dict[str, list]:
        cls = self.app_state.preset.selected_class
        return cls.parameter_groups if cls is not None else {}

    def get_selected_contract(self) -> dict[str, Any]:
        cls = self.app_state.preset.selected_class
        return cls.get_contract() if cls is not None else {}

    def validate_required_instruments(self) -> tuple[bool, list[type[BaseInstrument]]]:
        cls = self.app_state.preset.selected_class
        if cls is None:
            self.requirements_changed.emit(True, [])
            return True, []
        missing: list[type[BaseInstrument]] = []
        for required_cls in cls.required_instruments:
            if not self.instrument_service.get_instances_by_class(required_cls):
                missing.append(required_cls)
        ok = len(missing) == 0
        self.requirements_changed.emit(ok, [c.__name__ for c in missing])
        if self.app_state.preset.running and not ok:
            self.acq_error.emit("required instrument removed during acquisition")
            self.stop()
        return ok, missing

    def configure(self, raw_params: dict[str, Any]) -> None:
        instance = self.app_state.preset.instance
        cls = self.app_state.preset.selected_class
        if instance is None or cls is None:
            raise RuntimeError("no preset selected")
        ok, missing = self.validate_required_instruments()
        if not ok:
            msg = f"missing required instruments: {', '.join(c.__name__ for c in missing)}"
            self.acq_error.emit(msg)
            raise RuntimeError(msg)
        try:
            cleaned = coerce_parameter_values(cls.parameter_groups, raw_params)
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise
        self.set_parameter_values(cleaned)

    def start(self, *, force_continuous: bool = False) -> None:
        with self.lock:
            cls = self.app_state.preset.selected_class
            instance = self.app_state.preset.instance
            if cls is None or instance is None:
                raise RuntimeError("no preset selected")
            if not self.app_state.preset.configured_params:
                raise RuntimeError("preset is not configured")
            if self.app_state.preset.running:
                raise RuntimeError("acquisition already running")
            ok, missing = self.validate_required_instruments()
            if not ok:
                msg = f"cannot start, missing required instruments: {', '.join(c.__name__ for c in missing)}"
                self.acq_error.emit(msg)
                raise RuntimeError(msg)

            params = self.get_parameter_values()
            frame_limit = None if force_continuous else int(params.get("num_frames", 1))

            instruments: dict[type, BaseInstrument] = {}
            for required_cls in cls.required_instruments:
                found = self.instrument_service.get_instances_by_class(required_cls)
                if found:
                    instruments[required_cls] = found[0]

            source, setup = instance.build_source_and_setup(
                params=params, instruments=instruments, frame_limit=frame_limit
            )

            allowed = tuple(cls.allowed_optocontrols)
            for control in self.app_state.optocontrols:
                if not control.enabled:
                    continue
                if allowed and not isinstance(control, allowed):
                    continue
                source = control.build_decorator(source)

            storage = FrameStorage(
                save_enabled=bool(params.get("save_enabled", False)),
                save_path=str(params.get("save_path", "acquisition")),
                preset_key=self.app_state.preset.selected_key or "",
                parameters=dict(params),
            )
            storage.prepare(frame_limit=frame_limit)

            self.stop_event = threading.Event()
            self.warned_messages = set()
            self.app_state.preset.running = True
            self.acq_started.emit()
            self.acquisition_thread = threading.Thread(
                target=self.run_worker, args=(source, setup, storage), daemon=True
            )
            self.acquisition_thread.start()

    def run_worker(self, source, setup, storage: FrameStorage) -> None:
        def on_results(results: list[AcquiredData]) -> None:
            try:
                storage.save(results)
            except Exception as exc:
                self.acq_error.emit(f"save failed: {exc}")
            for result in results:
                warning = result.metadata.get("warning")
                if warning:
                    self.emit_acq_warning(str(warning))
                self.data_emitted.emit(result)

        def on_finished(error: Exception | None) -> None:
            try:
                storage.finalize(error)
            except Exception:
                pass
            self.app_state.preset.running = False
            if error is not None:
                self.acq_error.emit(str(error))
            self.acquisition_thread = None
            self.acq_stopped.emit()

        Executor().run(
            source=source,
            setup=setup,
            on_results=on_results,
            should_stop=self.stop_event.is_set,
            on_finished=on_finished,
        )

    def emit_acq_warning(self, message: str) -> None:
        if message not in self.warned_messages:
            self.warned_messages.add(message)
            self.acq_warning.emit(message)

    def stop(self) -> None:
        self.stop_event.set()

    def set_parameter_values(self, raw_params: dict[str, Any]) -> None:
        key = self.app_state.preset.selected_key
        if key is not None:
            self.app_state.preset.params_by_preset[key] = [
                ParameterValue(label=k, value=v) for k, v in raw_params.items()
            ]
        self.modality_params_changed.emit(dict(raw_params))

    def get_parameter_values(self) -> dict[str, Any]:
        return {entry.label: entry.value for entry in self.app_state.preset.configured_params}
