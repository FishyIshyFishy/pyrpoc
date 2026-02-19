from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter_utils import ParameterValidationError, coerce_parameter_values
from pyrpoc.domain.app_state import AppState, DisplayState, ParameterValue
from pyrpoc.displays.base_display import BaseDisplay
from pyrpoc.displays.display_registry import display_registry
from pyrpoc.rpoc.types import RPOCImageInput


class DisplayService(QObject):
    # declare all signals at the top
    display_added = pyqtSignal(object)
    display_removed = pyqtSignal(object)
    display_changed = pyqtSignal(object)
    display_error = pyqtSignal(object, str)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._reported_incompatibilities: set[tuple[int, str]] = set()
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

    def create_display(self, key: str, raw_settings: dict[str, Any]) -> DisplayState:
        display_cls = display_registry.get_class(key)
        settings_parameters = display_cls.DISPLAY_PARAMETERS

        try:
            settings = coerce_parameter_values(settings_parameters, raw_settings)
            widget = display_cls()
            widget.configure(settings)
        except (ParameterValidationError, Exception) as exc:
            self.display_error.emit(None, str(exc))
            raise

        state = DisplayState(
            type_key=key,
            instance=widget,
            attached=True,
            config_values=[ParameterValue(label=k, value=v) for k, v in settings.items()],
        )
        self.app_state.displays.append(state)
        self.display_added.emit(state)
        return state

    def remove_display(self, state: DisplayState) -> None:
        if state not in self.app_state.displays:
            return

        self.app_state.displays.remove(state)
        self.display_removed.emit(state)
        state.instance.deleteLater()

    def attach(self, state: DisplayState) -> None:
        self._require_state(state)
        state.attached = True
        self.display_changed.emit(state)

    def detach(self, state: DisplayState) -> None:
        self._require_state(state)
        state.attached = False
        self.display_changed.emit(state)

    def push_data(self, data: BaseData) -> None:
        self._last_data = data
        for state in self.app_state.displays:
            if not state.attached:
                continue

            widget = state.instance
            compatible = any(isinstance(data, accepted) for accepted in widget.ACCEPTED_DATA_TYPES)
            if not compatible:
                key = (id(state), type(data).__name__)
                if key not in self._reported_incompatibilities:
                    self._reported_incompatibilities.add(key)
                    self.display_error.emit(
                        state,
                        f"display cannot render data type {type(data).__name__}",
                    )
                continue

            try:
                widget.render(data)
            except Exception as exc:
                state.last_error = str(exc)
                self.display_error.emit(state, str(exc))

    def get_latest_data(self) -> BaseData | None:
        return self._last_data

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state in self.app_state.displays:
            key = state.type_key
            cls = display_registry.get_class(key)
            rows.append(
                {
                    "state": state,
                    "key": key,
                    "name": getattr(cls, "DISPLAY_NAME", key),
                    "attached": state.attached,
                }
            )
        return rows

    def get_widget(self, state: DisplayState) -> BaseDisplay:
        self._require_state(state)
        return state.instance

    def get_rpoc_input(self, state: DisplayState) -> RPOCImageInput | None:
        widget = self.get_widget(state)
        exporter = getattr(widget, "export_rpoc_input", None)
        if not callable(exporter):
            return None
        payload = exporter()
        if payload is None:
            return None
        if not isinstance(payload, RPOCImageInput):
            raise TypeError(
                f"display export_rpoc_input returned {type(payload).__name__}, "
                "expected RPOCImageInput or None"
            )
        return payload

    def _require_state(self, state: DisplayState) -> None:
        if state not in self.app_state.displays:
            raise KeyError("display state does not exist")

    def clear_all(self) -> None:
        for state in list(self.app_state.displays):
            self.remove_display(state)
