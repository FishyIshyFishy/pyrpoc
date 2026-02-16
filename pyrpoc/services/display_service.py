from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter_utils import ParameterValidationError, coerce_parameter_values
from pyrpoc.displays.base_display import BaseDisplay
from pyrpoc.displays.display_registry import display_registry
from pyrpoc.rpoc.types import RPOCImageInput


class DisplayService(QObject):
    # declare all signals at the top
    display_added = pyqtSignal(str)
    display_removed = pyqtSignal(str)
    display_error = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._instances: dict[str, BaseDisplay] = {}
        self._display_keys: dict[str, str] = {}
        self._attached: set[str] = set()
        self._reported_incompatibilities: set[tuple[str, str]] = set()
        self._next_display_index: int = 1
        self._last_data: BaseData | None = None

    def list_available(self) -> list[dict[str, Any]]:
        return display_registry.describe_all()

    def list_compatible_with(self, data_type: type[BaseData]) -> list[str]:
        compatible: list[str] = []
        for key in display_registry.list_keys():
            display_cls = display_registry.get_class(key)
            if any(issubclass(data_type, accepted) for accepted in display_cls.ACCEPTED_DATA_TYPES):
                compatible.append(key)
        return compatible

    def create_display(self, key: str, raw_settings: dict[str, Any]) -> tuple[str, BaseDisplay]:
        display_id = self._allocate_display_id()

        display_cls = display_registry.get_class(key)
        settings_parameters = display_cls.DISPLAY_PARAMETERS

        try:
            settings = coerce_parameter_values(settings_parameters, raw_settings)
            widget = display_cls()
            widget.configure(settings)
        except (ParameterValidationError, Exception) as exc:
            self.display_error.emit(display_id, str(exc))
            raise

        self._instances[display_id] = widget
        self._display_keys[display_id] = key
        self._attached.add(display_id)
        self.display_added.emit(display_id)
        return display_id, widget

    def remove_display(self, display_id: str) -> None:
        widget = self._instances.get(display_id)
        if widget is None:
            return

        self._attached.discard(display_id)
        del self._instances[display_id]
        del self._display_keys[display_id]
        self.display_removed.emit(display_id)
        widget.deleteLater()

    def attach(self, display_id: str) -> None:
        if display_id not in self._instances:
            raise KeyError(f"display '{display_id}' does not exist")
        self._attached.add(display_id)

    def detach(self, display_id: str) -> None:
        self._attached.discard(display_id)

    def push_data(self, data: BaseData) -> None:
        self._last_data = data
        for display_id in list(self._attached):
            widget = self._instances.get(display_id)
            if widget is None:
                continue

            compatible = any(isinstance(data, accepted) for accepted in widget.ACCEPTED_DATA_TYPES)
            if not compatible:
                key = (display_id, type(data).__name__)
                if key not in self._reported_incompatibilities:
                    self._reported_incompatibilities.add(key)
                    self.display_error.emit(
                        display_id,
                        f"display cannot render data type {type(data).__name__}",
                    )
                continue

            try:
                widget.render(data)
            except Exception as exc:
                self.display_error.emit(display_id, str(exc))

    def get_latest_data(self) -> BaseData | None:
        return self._last_data

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for display_id in sorted(self._instances.keys()):
            key = self._display_keys[display_id]
            cls = display_registry.get_class(key)
            rows.append(
                {
                    "display_id": display_id,
                    "key": key,
                    "name": getattr(cls, "DISPLAY_NAME", key),
                    "attached": display_id in self._attached,
                }
            )
        return rows

    def get_widget(self, display_id: str) -> BaseDisplay:
        if display_id not in self._instances:
            raise KeyError(f"display '{display_id}' does not exist")
        return self._instances[display_id]

    def get_rpoc_input(self, display_id: str) -> RPOCImageInput | None:
        widget = self.get_widget(display_id)
        exporter = getattr(widget, "export_rpoc_input", None)
        if not callable(exporter):
            return None
        payload = exporter()
        if payload is None:
            return None
        if not isinstance(payload, RPOCImageInput):
            raise TypeError(
                f"display '{display_id}' export_rpoc_input returned {type(payload).__name__}, "
                "expected RPOCImageInput or None"
            )
        return payload

    def _allocate_display_id(self) -> str:
        while True:
            candidate = f"display_{self._next_display_index}"
            self._next_display_index += 1
            if candidate not in self._instances:
                return candidate
