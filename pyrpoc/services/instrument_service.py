from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.parameter_utils import (
    ParameterValidationError,
    coerce_action_values,
    coerce_parameter_values,
)
from pyrpoc.instruments.base_instrument import BaseInstrument
from pyrpoc.instruments.instrument_registry import instrument_registry


class InstrumentService(QObject):
    inventory_changed = pyqtSignal()
    connection_changed = pyqtSignal(str, bool)
    instrument_error = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._instances: dict[str, BaseInstrument] = {}
        self._instance_types: dict[str, str] = {}
        self._next_instance_index: int = 1

    def list_available(self) -> list[dict[str, Any]]:
        return instrument_registry.describe_all()

    def create_instrument(self, key: str) -> tuple[str, BaseInstrument]:
        instance_id = self._allocate_instance_id()
        cls = instrument_registry.get_class(key)
        instance = cls(alias=instance_id)
        self._instances[instance_id] = instance
        self._instance_types[instance_id] = key
        self.inventory_changed.emit()
        return instance_id, instance

    def remove_instrument(self, instance_id: str) -> None:
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
            self.instrument_error.emit(instance_id, str(exc))
            raise

    def disconnect(self, instance_id: str) -> None:
        instance = self._require_instance(instance_id)
        try:
            instance.disconnect()
            self.connection_changed.emit(instance_id, False)
        except Exception as exc:
            self.instrument_error.emit(instance_id, str(exc))
            raise

    def run_action(self, instance_id: str, action_label: str, raw_args: dict[str, Any]) -> None:
        instance = self._require_instance(instance_id)
        if not instance.is_connected():
            msg = f"instrument '{instance_id}' is not connected"
            self.instrument_error.emit(instance_id, msg)
            raise RuntimeError(msg)

        actions = instance.__class__.ACTIONS
        action = next((candidate for candidate in actions if candidate.label == action_label), None)
        if action is None:
            msg = f"instrument '{instance_id}' has no action '{action_label}'"
            self.instrument_error.emit(instance_id, msg)
            raise KeyError(msg)

        try:
            args = coerce_action_values(action, raw_args)
            instance.execute_action(action.method_name, args)
        except (ParameterValidationError, Exception) as exc:
            self.instrument_error.emit(instance_id, str(exc))
            raise

    def get_connected_by_class(self, cls: type[BaseInstrument]) -> list[BaseInstrument]:
        return [
            instance
            for instance in self._instances.values()
            if isinstance(instance, cls) and instance.is_connected()
        ]

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for instance_id in sorted(self._instances.keys()):
            instance = self._instances[instance_id]
            key = self._instance_types[instance_id]
            cls = instrument_registry.get_class(key)
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

    def get_instance(self, instance_id: str) -> BaseInstrument:
        return self._require_instance(instance_id)

    def get_instance_key(self, instance_id: str) -> str:
        self._require_instance(instance_id)
        return self._instance_types[instance_id]

    def _require_instance(self, instance_id: str) -> BaseInstrument:
        if instance_id not in self._instances:
            raise KeyError(f"instrument instance '{instance_id}' is not registered")
        return self._instances[instance_id]

    def _allocate_instance_id(self) -> str:
        while True:
            candidate = f"instrument_{self._next_instance_index}"
            self._next_instance_index += 1
            if candidate not in self._instances:
                return candidate
