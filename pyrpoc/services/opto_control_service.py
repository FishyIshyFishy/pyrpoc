from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.parameter_utils import (
    ParameterValidationError,
    coerce_action_values,
    coerce_parameter_values,
)
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.opto_control_registry import opto_control_registry


class OptoControlService(QObject):
    inventory_changed = pyqtSignal()
    connection_changed = pyqtSignal(str, bool)
    opto_control_error = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._instances: dict[str, BaseOptoControl] = {}
        self._instance_types: dict[str, str] = {}
        self._next_instance_index: int = 1

    def list_available(self) -> list[dict[str, Any]]:
        return opto_control_registry.describe_all()

    def create_opto_control(self, key: str) -> tuple[str, BaseOptoControl]:
        instance_id = self._allocate_instance_id()
        cls = opto_control_registry.get_class(key)
        instance = cls(alias=instance_id)
        self._instances[instance_id] = instance
        self._instance_types[instance_id] = key
        self.inventory_changed.emit()
        return instance_id, instance

    def remove_opto_control(self, instance_id: str) -> None:
        instance = self._instances.get(instance_id)
        if instance is None:
            return

        if instance.is_connected():
            instance.disconnect()
            self.connection_changed.emit(instance_id, False)

        del self._instances[instance_id]
        del self._instance_types[instance_id]
        self.inventory_changed.emit()

    def connect(self, instance_id: str, raw_config: dict[str, Any]) -> None:
        instance = self._require_instance(instance_id)
        parameter_groups = instance.__class__.CONFIG_PARAMETERS

        try:
            config = coerce_parameter_values(parameter_groups, raw_config)
            instance.connect(config)
            self.connection_changed.emit(instance_id, instance.is_connected())
        except (ParameterValidationError, Exception) as exc:
            self.opto_control_error.emit(instance_id, str(exc))
            raise

    def disconnect(self, instance_id: str) -> None:
        instance = self._require_instance(instance_id)
        try:
            instance.disconnect()
            self.connection_changed.emit(instance_id, False)
        except Exception as exc:
            self.opto_control_error.emit(instance_id, str(exc))
            raise

    def run_action(self, instance_id: str, action_label: str, raw_args: dict[str, Any]) -> None:
        instance = self._require_instance(instance_id)
        if not instance.is_connected():
            msg = f"opto-control '{instance_id}' is not connected"
            self.opto_control_error.emit(instance_id, msg)
            raise RuntimeError(msg)

        actions = instance.__class__.ACTIONS
        action = next((candidate for candidate in actions if candidate.label == action_label), None)
        if action is None:
            msg = f"opto-control '{instance_id}' has no action '{action_label}'"
            self.opto_control_error.emit(instance_id, msg)
            raise KeyError(msg)

        try:
            args = coerce_action_values(action, raw_args)
            instance.execute_action(action.method_name, args)
        except (ParameterValidationError, Exception) as exc:
            self.opto_control_error.emit(instance_id, str(exc))
            raise

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for instance_id in sorted(self._instances.keys()):
            instance = self._instances[instance_id]
            key = self._instance_types[instance_id]
            cls = opto_control_registry.get_class(key)
            rows.append(
                {
                    "instance_id": instance_id,
                    "key": key,
                    "name": getattr(cls, "DISPLAY_NAME", key),
                    "connected": instance.is_connected(),
                    "status": instance.get_status(),
                }
            )
        return rows

    def get_instance(self, instance_id: str) -> BaseOptoControl:
        return self._require_instance(instance_id)

    def get_instance_key(self, instance_id: str) -> str:
        self._require_instance(instance_id)
        return self._instance_types[instance_id]

    def set_mask_data(
        self,
        instance_id: str,
        mask_data: np.ndarray,
        source_path: str | None = None,
    ) -> None:
        instance = self._require_instance(instance_id)
        setter = getattr(instance, "set_mask_data", None)
        if not callable(setter):
            raise RuntimeError(f"opto-control '{instance_id}' does not support in-memory masks")
        setter(mask_data, source_path=source_path)
        self.connection_changed.emit(instance_id, instance.is_connected())

    def _require_instance(self, instance_id: str) -> BaseOptoControl:
        if instance_id not in self._instances:
            raise KeyError(f"opto-control instance '{instance_id}' is not registered")
        return self._instances[instance_id]

    def _allocate_instance_id(self) -> str:
        while True:
            candidate = f"optocontrol_{self._next_instance_index}"
            self._next_instance_index += 1
            if candidate not in self._instances:
                return candidate
