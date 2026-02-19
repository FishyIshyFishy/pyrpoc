from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.parameter_utils import (
    ParameterValidationError,
    coerce_action_values,
    coerce_parameter_values,
)
from pyrpoc.domain.app_state import AppState, OptoControlState, ParameterValue
from pyrpoc.optocontrols.opto_control_registry import opto_control_registry
from .opto_control_types import ActionExecutionResult, OptoInstanceRow, OptoInstanceSchema, OptoStatusSnapshot


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

    def list_instance_rows(self, contract: dict[str, Any] | None) -> list[OptoInstanceRow]:
        rows: list[OptoInstanceRow] = []
        for state in self.app_state.optocontrols:
            key = state.type_key
            cls = opto_control_registry.get_class(key)
            rows.append(
                OptoInstanceRow(
                    state=state,
                    key=key,
                    display_name=getattr(cls, "DISPLAY_NAME", key),
                    connected=state.connected,
                    compatible_with_selected_modality=self._is_instance_compatible(key, contract),
                    status=self.get_status_snapshot(state),
                )
            )
        return rows

    def get_instance_schema(self, state: OptoControlState) -> OptoInstanceSchema:
        self._require_state(state)
        key = state.type_key
        cls = opto_control_registry.get_class(key)
        return OptoInstanceSchema(
            type_key=key,
            display_name=getattr(cls, "DISPLAY_NAME", key),
            config_parameters=getattr(cls, "CONFIG_PARAMETERS", {}),
            actions=tuple(getattr(cls, "ACTIONS", [])),
            editor_key=getattr(cls, "EDITOR_KEY", None),
            editor_anchor_param=getattr(cls, "EDITOR_ANCHOR_PARAM", None),
            editor_apply_method=getattr(cls, "EDITOR_APPLY_METHOD", None),
        )

    def ensure_connected(self, state: OptoControlState, raw_config: dict[str, Any]) -> None:
        self._require_state(state)
        if state.connected:
            return
        config = raw_config
        if not config:
            config = {entry.label: entry.value for entry in state.config_values}
        self.connect(state, config)

    def run_action_with_auto_connect(
        self,
        state: OptoControlState,
        action_label: str,
        raw_args: dict[str, Any],
        raw_config: dict[str, Any],
    ) -> ActionExecutionResult:
        self.ensure_connected(state, raw_config)
        self.run_action(state, action_label, raw_args)
        return ActionExecutionResult(
            state=state,
            action_label=action_label,
            status=self.get_status_snapshot(state),
        )

    def set_enabled(self, state: OptoControlState, enabled: bool, raw_config: dict[str, Any]) -> OptoStatusSnapshot:
        self._require_state(state)
        self.ensure_connected(state, raw_config)
        state.enabled = bool(enabled)
        try:
            state.instance.on_enabled_changed(state.enabled)
        except Exception as exc:
            state.last_error = str(exc)
            self.opto_control_error.emit(state, str(exc))
            raise
        self.connection_changed.emit(state, state.connected)
        return self.get_status_snapshot(state)

    def apply_editor_payload(
        self,
        state: OptoControlState,
        payload: object,
        raw_config: dict[str, Any],
    ) -> OptoStatusSnapshot:
        self._require_state(state)
        schema = self.get_instance_schema(state)
        method_name = schema.editor_apply_method
        if not method_name:
            raise RuntimeError(f"opto-control '{schema.type_key}' does not support editor payload apply")

        self.ensure_connected(state, raw_config)
        method = getattr(state.instance, method_name, None)
        if method is None or not callable(method):
            raise RuntimeError(
                f"opto-control '{schema.type_key}' has invalid editor apply method '{method_name}'"
            )

        try:
            method(payload)
        except Exception as exc:
            state.last_error = str(exc)
            self.opto_control_error.emit(state, str(exc))
            raise

        self.connection_changed.emit(state, state.connected)
        return self.get_status_snapshot(state)

    def get_status_snapshot(self, state: OptoControlState) -> OptoStatusSnapshot:
        self._require_state(state)
        raw_status = state.instance.get_status()
        status = raw_status if isinstance(raw_status, dict) else {}
        return OptoStatusSnapshot(
            last_action=str(status.get("last_action", "idle")),
            enabled=bool(state.enabled),
            raw=status,
        )

    def _require_state(self, state: OptoControlState) -> None:
        if state not in self.app_state.optocontrols:
            raise KeyError("opto-control state is not registered")

    def clear_all(self) -> None:
        for state in list(self.app_state.optocontrols):
            self.remove_opto_control(state)

    def _is_instance_compatible(self, instance_key: str, contract: dict[str, Any] | None) -> bool:
        if not contract:
            return True

        allowed = contract.get("allowed_optocontrols", [])
        if not allowed:
            return False

        cls = opto_control_registry.get_class(instance_key)
        for allowed_cls in allowed:
            if isinstance(allowed_cls, type) and issubclass(cls, allowed_cls):
                return True
        return False
