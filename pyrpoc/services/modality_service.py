from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
import tifffile

from pyrpoc.backend_utils.array_contracts import matches_array_contract
from pyrpoc.backend_utils.parameter_utils import ParameterValidationError, coerce_parameter_values
from pyrpoc.domain.app_state import AppState, ParameterValue
from pyrpoc.instruments.base_instrument import BaseInstrument
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

    def __init__(self, instrument_service: InstrumentService, app_state: AppState, parent=None):
        super().__init__(parent)
        self.instrument_service = instrument_service
        self.app_state = app_state
        self._save_root_path: Path | None = None
        self._save_json_path: Path | None = None
        self._save_channel_paths: dict[str, Path] = {}
        self._save_channel_labels: list[str] = []
        self._auxiliary_payload_buffers: dict[str, list[np.ndarray]] = {}
        self._auxiliary_paths: dict[str, Path] = {}
        self._frame_limit: int | None = 1
        self._saved_frame_count = 0
        self._acquisition_id = 0
        self._acquisition_start_time = datetime.now(timezone.utc).isoformat()
        self._run_in_progress = False
        self._acquisition_thread: threading.Thread | None = None
        self._acquisition_stop_requested = threading.Event()
        self._active_channel_labels: list[str] = []
        self._active_frame_limit: int | None = 1
        self._acquisition_lock = threading.Lock()

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

        self._configure_acquisition_plan(cleaned_params)

        bound: dict[type[BaseInstrument], BaseInstrument] = {}
        for required_cls in self.app_state.modality.selected_class.REQUIRED_INSTRUMENTS:
            bound_instances = self.instrument_service.get_instances_by_class(required_cls)
            if not bound_instances:
                raise RuntimeError(f"required instrument {required_cls.__name__} disappeared before configure")
            bound[required_cls] = bound_instances[0]
        for optional_cls in self.app_state.modality.selected_class.OPTIONAL_INSTRUMENTS:
            optional_instances = self.instrument_service.get_instances_by_class(optional_cls)
            if optional_instances:
                bound[optional_cls] = optional_instances[0]

        allowed_types = tuple(self.app_state.modality.selected_class.ALLOWED_OPTOCONTROLS)
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

        self.app_state.modality.instance.configure(cleaned_params, bound, bound_opto)
        self._active_channel_labels = list(self.app_state.modality.instance.get_active_channel_labels())
        self.set_parameter_values(cleaned_params)

    def should_stop_acquisition(self) -> bool:
        if self._active_frame_limit is None:
            return False
        return self._saved_frame_count >= self._active_frame_limit

    def _configure_acquisition_plan(self, cleaned_params: dict[str, Any]) -> None:
        self._frame_limit = 1
        self._saved_frame_count = 0
        self._run_in_progress = False
        self._save_root_path = None
        self._save_json_path = None
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._auxiliary_payload_buffers = {}
        self._auxiliary_paths = {}

        raw_num_frames = cleaned_params.get("num_frames")
        if raw_num_frames is not None:
            limit = int(raw_num_frames)
            if limit < 1:
                raise ParameterValidationError(
                    "parameter validation failed",
                    {"num_frames": "must be >= 1"},
                )
            self._frame_limit = limit

        if not bool(cleaned_params.get("save_enabled", False)):
            return

        raw_save_path = cleaned_params.get("save_path")
        if raw_save_path is None:
            raise ParameterValidationError(
                "parameter validation failed",
                {"save_path": "value is required when save_enabled is true"},
            )
        path = Path(raw_save_path).expanduser() if not isinstance(raw_save_path, Path) else raw_save_path.expanduser()
        if path.suffix.lower() in {".tif", ".tiff"}:
            path = path.with_suffix("")
        self._save_root_path = path

        self._save_json_path = self._build_json_path(path)

    def _build_json_path(self, root_path: Path) -> Path:
        root_name = root_path.with_name(f"{root_path.name}_meta.json")
        return root_name

    def _build_channel_paths(self, labels: list[str], root_path: Path) -> dict[str, Path]:
        return {
            label: root_path.with_name(f"{root_path.name}_{label}.tiff")
            for label in labels
        }

    def _resolve_channel_labels(self, channel_count: int) -> list[str]:
        if self._active_channel_labels and len(self._active_channel_labels) == channel_count:
            return list(self._active_channel_labels)
        return [f"channel_{index}" for index in range(channel_count)]

    def _prepare_save_paths(self, channel_count: int) -> None:
        if self._save_root_path is None:
            return
        if self._save_channel_paths and len(self._save_channel_paths) == channel_count:
            return

        labels = self._resolve_channel_labels(channel_count)
        self._save_channel_labels = labels
        self._save_channel_paths = self._build_channel_paths(labels, self._save_root_path)

        self._save_root_path.parent.mkdir(parents=True, exist_ok=True)
        for path in self._save_channel_paths.values():
            if path.exists():
                path.unlink()

    def _prepare_auxiliary_paths(self, labels: list[str]) -> None:
        if self._save_root_path is None:
            return
        if self._auxiliary_paths:
            return
        self._auxiliary_paths = {
            label: self._save_root_path.with_name(f"{self._save_root_path.name}_{label}.npz")
            for label in labels
        }
        self._save_root_path.parent.mkdir(parents=True, exist_ok=True)
        for path in self._auxiliary_paths.values():
            if path.exists():
                path.unlink()

    def _consume_auxiliary_payload(self) -> dict[str, np.ndarray] | None:
        instance = self.app_state.modality.instance
        if instance is None:
            return None
        getter = getattr(instance, "consume_auxiliary_payload", None)
        if getter is None or not callable(getter):
            return None
        payload = getter()
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise TypeError(
                f"modality {type(instance).__name__} auxiliary payload must be a dict, got {type(payload).__name__}"
            )
        if not payload:
            return None
        normalized: dict[str, np.ndarray] = {}
        for key, value in payload.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"auxiliary payload '{key}' must be a numpy array")
            normalized[str(key)] = np.asarray(value, dtype=np.float32, copy=False)
        return normalized

    def _append_auxiliary(self) -> None:
        if self._save_root_path is None:
            return

        payload = self._consume_auxiliary_payload()
        if not payload:
            return

        labels = list(payload.keys())
        self._prepare_auxiliary_paths(labels)

        for label, data in payload.items():
            frames = self._auxiliary_payload_buffers.setdefault(label, [])
            frames.append(data)

    def _flush_auxiliary_payloads(self) -> None:
        if self._save_root_path is None:
            return
        if not self._auxiliary_paths:
            return

        self._save_root_path.parent.mkdir(parents=True, exist_ok=True)
        configured = {entry.label: entry.value for entry in self.app_state.modality.configured_params}
        for label, path in self._auxiliary_paths.items():
            frames = self._auxiliary_payload_buffers.get(label, [])
            if not frames:
                continue
            payload = np.stack(frames, axis=0)
            np.savez_compressed(
                str(path),
                frames=payload,
                acquisition_parameters=np.asarray(configured, dtype=object),
                frame_indices=np.arange(payload.shape[0], dtype=np.int32),
            )

    def _start_acquisition(self) -> None:
        self._saved_frame_count = 0
        self._run_in_progress = True
        self._acquisition_id += 1
        self._acquisition_start_time = datetime.now(timezone.utc).isoformat()
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._auxiliary_payload_buffers = {}
        self._auxiliary_paths = {}
        if self._save_root_path is None:
            return

        self._save_root_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_json_path = self._build_json_path(self._save_root_path)
        self._write_metadata(None)

    def _append_tiff(self, data: np.ndarray) -> None:
        if self._save_root_path is None:
            return

        if not self._save_channel_paths:
            channel_data = self._split_channels(data)
            self._prepare_save_paths(len(channel_data))
        channel_data = self._split_channels(data)
        if len(channel_data) != len(self._save_channel_paths):
            raise ValueError("frame channel count does not match configured save layout")

        for (label, path), frame in zip(self._save_channel_paths.items(), channel_data):
            with tifffile.TiffWriter(str(path), append=True) as writer:
                writer.write(frame.astype(np.float32))
        self._append_auxiliary()

    def _split_channels(self, data: np.ndarray) -> list[np.ndarray]:
        if data.ndim == 2:
            return [data]
        if data.ndim == 3:
            return [data[index] for index in range(data.shape[0])]
        raise ValueError(f"unsupported frame dimensions {data.ndim}")

    def _write_metadata(self, last_frame_error: str | None) -> None:
        if self._save_json_path is None:
            return

        configured = {entry.label: entry.value for entry in self.app_state.modality.configured_params}
        payload = {
            "acquisition_id": self._acquisition_id,
            "started": self._acquisition_start_time,
            "modality_key": self.app_state.modality.selected_key,
            "save_root_path": str(self._save_root_path),
            "save_json_path": str(self._save_json_path),
            "tiff_paths": {label: str(path) for label, path in self._save_channel_paths.items()},
            "auxiliary_paths": {label: str(path) for label, path in self._auxiliary_paths.items()},
            "frames_saved": self._saved_frame_count,
            "frame_limit": self._active_frame_limit,
            "parameters": configured,
            "last_error": last_frame_error,
        }
        self._save_json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def start(self, continuous: bool = False) -> None:
        with self._acquisition_lock:
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

            if self.app_state.modality.running:
                raise RuntimeError("acquisition already running")

            self._acquisition_stop_requested = threading.Event()
            frame_limit: int | None = None if continuous else self._frame_limit
            self._active_frame_limit = frame_limit
            self._start_acquisition()
            self.app_state.modality.running = True
            self.acq_started.emit()
            self._acquisition_thread = self.app_state.modality.instance.run_acquisition_threaded(
                on_frame=self._handle_frame,
                frame_limit=frame_limit,
                should_stop=self._acquisition_stop_requested.is_set,
                on_error=self._handle_acquisition_error,
                on_finished=self._handle_acquisition_finished,
            )

    def _handle_frame(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"modality returned {type(data).__name__}, expected numpy.ndarray")

        contract = str(
            getattr(
                self.app_state.modality.selected_class,
                "OUTPUT_DATA_CONTRACT",
                "",
            )
        )
        if contract and not matches_array_contract(data, contract):
            raise ValueError(
                f"modality returned payload not matching contract '{contract}', "
                f"shape={data.shape}, dtype={data.dtype}"
            )

        if self._save_root_path is not None:
            self._append_tiff(data)

        self._saved_frame_count += 1
        if self._save_root_path is not None:
            self._write_metadata(None)
        self.data_ready.emit(data)

    def _handle_acquisition_error(self, error: Exception) -> None:
        msg = str(error)
        self._write_metadata(msg)
        self.acq_error.emit(msg)

    def _handle_acquisition_finished(self, _frame_count: int, _error: Exception | None) -> None:
        try:
            self.app_state.modality.running = False
        finally:
            self._run_in_progress = False
            self._flush_auxiliary_payloads()
            if self._save_root_path is not None:
                if _error is None:
                    self._write_metadata(None)
                else:
                    self._write_metadata(str(_error))
            self.acq_stopped.emit()
            self._acquisition_thread = None

    def stop(self) -> None:
        self._acquisition_stop_requested.set()
        if self.app_state.modality.instance is None:
            return
        try:
            if self.app_state.modality.running:
                self.app_state.modality.instance.stop()
        except Exception as exc:
            self.acq_error.emit(str(exc))
            raise

    def set_parameter_values(self, raw_params: dict[str, Any]) -> None:
        self.app_state.modality.configured_params = [
            ParameterValue(label=k, value=v) for k, v in raw_params.items()
        ]
        self.modality_params_changed.emit(dict(raw_params))

    def get_parameter_values(self) -> dict[str, Any]:
        return {entry.label: entry.value for entry in self.app_state.modality.configured_params}
