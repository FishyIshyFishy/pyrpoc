from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.parameter_utils import (
    ParameterValidationError,
    coerce_action_values,
    coerce_parameter_values,
)
from pyrpoc.domain.app_state import AppState, InstrumentState, ParameterValue
from pyrpoc.instruments.base_instrument import BaseInstrument
from pyrpoc.instruments.instrument_registry import instrument_registry


class InstrumentService(QObject):
    inventory_changed = pyqtSignal()
    connection_changed = pyqtSignal(object, bool)
    instrument_error = pyqtSignal(object, str)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state

    def list_available(self) -> list[dict[str, Any]]:
        return instrument_registry.describe_all()

    def create_instrument(self, key: str) -> InstrumentState:
        cls = instrument_registry.get_class(key)
        instance = cls(alias=key)
        state = InstrumentState(type_key=key, instance=instance)
        self.app_state.instruments.append(state)
        self.inventory_changed.emit()
        return state

    def remove_instrument(self, state: InstrumentState) -> None:
        if state not in self.app_state.instruments:
            return

        if state.connected:
            state.instance.disconnect()
            state.connected = False
            self.connection_changed.emit(state, False)

        self.app_state.instruments.remove(state)
        self.inventory_changed.emit()

    def connect(self, state: InstrumentState, raw_config: dict[str, Any]) -> None:
        self._require_state(state)
        parameter_groups = state.instance.__class__.CONFIG_PARAMETERS

        try:
            config = coerce_parameter_values(parameter_groups, raw_config)
            state.instance.connect(config)
            state.connected = state.instance.is_connected()
            state.config_values = [ParameterValue(label=k, value=v) for k, v in config.items()]
            self.connection_changed.emit(state, state.connected)
        except (ParameterValidationError, Exception) as exc:
            state.last_error = str(exc)
            self.instrument_error.emit(state, str(exc))
            raise

    def disconnect(self, state: InstrumentState) -> None:
        self._require_state(state)
        try:
            state.instance.disconnect()
            state.connected = False
            self.connection_changed.emit(state, False)
        except Exception as exc:
            state.last_error = str(exc)
            self.instrument_error.emit(state, str(exc))
            raise

    def run_action(self, state: InstrumentState, action_label: str, raw_args: dict[str, Any]) -> None:
        self._require_state(state)
        if not state.connected:
            msg = "instrument is not connected"
            self.instrument_error.emit(state, msg)
            raise RuntimeError(msg)

        actions = state.instance.__class__.ACTIONS
        action = next((candidate for candidate in actions if candidate.label == action_label), None)
        if action is None:
            msg = f"instrument has no action '{action_label}'"
            self.instrument_error.emit(state, msg)
            raise KeyError(msg)

        try:
            args = coerce_action_values(action, raw_args)
            state.instance.execute_action(action.method_name, args)
        except (ParameterValidationError, Exception) as exc:
            state.last_error = str(exc)
            self.instrument_error.emit(state, str(exc))
            raise

    def get_connected_by_class(self, cls: type[BaseInstrument]) -> list[BaseInstrument]:
        return [
            entry.instance
            for entry in self.app_state.instruments
            if isinstance(entry.instance, cls) and entry.connected
        ]

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state in self.app_state.instruments:
            key = state.type_key
            cls = instrument_registry.get_class(key)
            rows.append(
                {
                    "state": state,
                    "key": key,
                    "name": getattr(cls, "DISPLAY_NAME", key),
                    "connected": state.connected,
                    "status": state.instance.get_status(),
                }
            )
        return rows

    def get_instance(self, state: InstrumentState) -> BaseInstrument:
        self._require_state(state)
        return state.instance

    def get_instance_key(self, state: InstrumentState) -> str:
        self._require_state(state)
        return state.type_key

    def clear_all(self) -> None:
        for state in list(self.app_state.instruments):
            self.remove_instrument(state)

    def _require_state(self, state: InstrumentState) -> None:
        if state not in self.app_state.instruments:
            raise KeyError("instrument state is not registered")
