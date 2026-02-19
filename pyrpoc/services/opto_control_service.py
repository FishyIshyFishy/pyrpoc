from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.parameter_utils import (
    ParameterValidationError,
    coerce_action_values,
    coerce_parameter_values,
)
from pyrpoc.domain.app_state import AppState, OptoControlState, ParameterValue
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.opto_control_registry import opto_control_registry


class OptoControlService(QObject):
    inventory_changed = pyqtSignal()
    connection_changed = pyqtSignal(object, bool)
    opto_control_error = pyqtSignal(object, str)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state

    def list_available(self) -> list[dict[str, Any]]:
        return opto_control_registry.describe_all()

    def create_opto_control(self, key: str) -> OptoControlState:
        cls = opto_control_registry.get_class(key)
        instance = cls(alias=key)
        state = OptoControlState(type_key=key, instance=instance)
        self.app_state.optocontrols.append(state)
        self.inventory_changed.emit()
        return state

    def remove_opto_control(self, state: OptoControlState) -> None:
        if state not in self.app_state.optocontrols:
            return

        if state.connected:
            state.instance.disconnect()
            state.connected = False
            self.connection_changed.emit(state, False)

        self.app_state.optocontrols.remove(state)
        self.inventory_changed.emit()

    def connect(self, state: OptoControlState, raw_config: dict[str, Any]) -> None:
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
            self.opto_control_error.emit(state, str(exc))
            raise

    def disconnect(self, state: OptoControlState) -> None:
        self._require_state(state)
        try:
            state.instance.disconnect()
            state.connected = False
            self.connection_changed.emit(state, False)
        except Exception as exc:
            state.last_error = str(exc)
            self.opto_control_error.emit(state, str(exc))
            raise

    def run_action(self, state: OptoControlState, action_label: str, raw_args: dict[str, Any]) -> None:
        self._require_state(state)
        if not state.connected:
            msg = "opto-control is not connected"
            self.opto_control_error.emit(state, msg)
            raise RuntimeError(msg)

        actions = state.instance.__class__.ACTIONS
        action = next((candidate for candidate in actions if candidate.label == action_label), None)
        if action is None:
            msg = f"opto-control has no action '{action_label}'"
            self.opto_control_error.emit(state, msg)
            raise KeyError(msg)

        try:
            args = coerce_action_values(action, raw_args)
            state.instance.execute_action(action.method_name, args)
        except (ParameterValidationError, Exception) as exc:
            state.last_error = str(exc)
            self.opto_control_error.emit(state, str(exc))
            raise

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state in self.app_state.optocontrols:
            key = state.type_key
            cls = opto_control_registry.get_class(key)
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

    def get_instance(self, state: OptoControlState) -> BaseOptoControl:
        self._require_state(state)
        return state.instance

    def get_instance_key(self, state: OptoControlState) -> str:
        self._require_state(state)
        return state.type_key

    def set_mask_data(
        self,
        state: OptoControlState,
        mask_data: np.ndarray,
        source_path: str | None = None,
    ) -> None:
        self._require_state(state)
        setter = getattr(state.instance, "set_mask_data", None)
        if not callable(setter):
            raise RuntimeError("opto-control does not support in-memory masks")
        setter(mask_data, source_path=source_path)
        if source_path is not None:
            updated = False
            for entry in state.config_values:
                if entry.label == "Mask Path":
                    entry.value = Path(source_path)
                    updated = True
                    break
            if not updated:
                state.config_values.append(ParameterValue(label="Mask Path", value=Path(source_path)))
        state.connected = state.instance.is_connected()
        self.connection_changed.emit(state, state.connected)

    def _require_state(self, state: OptoControlState) -> None:
        if state not in self.app_state.optocontrols:
            raise KeyError("opto-control state is not registered")

    def clear_all(self) -> None:
        for state in list(self.app_state.optocontrols):
            self.remove_opto_control(state)
