from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter_utils import ParameterValidationError, coerce_parameter_values
from pyrpoc.domain.app_state import AppState, ParameterValue
from pyrpoc.displays.base_display import BaseDisplay
from pyrpoc.displays.display_registry import display_registry
from pyrpoc.rpoc.types import RPOCImageInput


class DisplayService(QObject):
    # from display manager actions / modality data stream
    # through service inventory + push_data routing
    # to live display widgets and UI refresh/autosave signals.
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

    def create_display(
        self,
        key: str,
        raw_settings: dict[str, Any],
        user_label: str | None = None,
    ) -> BaseDisplay:
        """
        Build one display instance, configure it, and register it in runtime inventory.

        Route:
        - DisplayManager add click
        - -> `display_mgr.handlers.on_add_clicked`
        - -> this method
        - -> `display_added` signal for tab/list rendering.
        """
        display_cls = display_registry.get_class(key)
        settings_parameters = display_cls.DISPLAY_PARAMETERS

        try:
            settings = coerce_parameter_values(settings_parameters, raw_settings)
            widget = display_cls()
            widget.configure(settings)
        except (ParameterValidationError, Exception) as exc:
            self.display_error.emit(None, str(exc))
            raise

        widget.attached = True
        widget.docked_visible = True
        widget.config_values = [ParameterValue(label=k, value=v) for k, v in settings.items()]
        widget.last_error = None
        widget.user_label = user_label if isinstance(user_label, str) and user_label.strip() else getattr(
            widget,
            "user_label",
            None,
        )

        self.app_state.displays.append(widget)
        self.display_added.emit(widget)
        return widget

    def remove_display(self, display: BaseDisplay) -> None:
        if display not in self.app_state.displays:
            return
        self.app_state.displays.remove(display)
        self.display_removed.emit(display)
        display.deleteLater()

    def attach(self, display: BaseDisplay) -> None:
        self._require_state(display)
        display.attached = True
        self.display_changed.emit(display)

    def detach(self, display: BaseDisplay) -> None:
        self._require_state(display)
        display.attached = False
        self.display_changed.emit(display)

    def set_dock_visibility(self, display: BaseDisplay, visible: bool) -> None:
        self._require_state(display)
        if bool(getattr(display, "docked_visible", True)) == bool(visible):
            return
        display.docked_visible = bool(visible)
        self.display_changed.emit(display)

    def push_data(self, data: BaseData) -> None:
        """
        Fan out one modality frame to all attached compatible displays.

        Route:
        - `AppController` wires `ModalityService.data_ready`
        - -> this method
        - -> per-display render calls + error signals.
        """
        self._last_data = data
        for display in self.app_state.displays:
            if not getattr(display, "attached", True):
                continue
            if not bool(getattr(display, "docked_visible", True)):
                continue

            compatible = any(isinstance(data, accepted) for accepted in display.ACCEPTED_DATA_TYPES)
            if not compatible:
                key = (id(display), type(data).__name__)
                if key not in self._reported_incompatibilities:
                    self._reported_incompatibilities.add(key)
                    self.display_error.emit(
                        display,
                        f"display cannot render data type {type(data).__name__}",
                    )
                continue

            try:
                display.render(data)
            except Exception as exc:
                display.last_error = str(exc)
                self.display_error.emit(display, str(exc))

    def get_latest_data(self) -> BaseData | None:
        return self._last_data

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for display in self.app_state.displays:
            key = display.type_key
            cls = display_registry.get_class(key)
            name = getattr(display, "user_label", None) or getattr(cls, "DISPLAY_NAME", key)
            rows.append(
                {
                    "state": display,
                    "display_id": id(display),
                    "key": key,
                    "name": name,
                    "attached": bool(getattr(display, "attached", True)),
                    "docked_visible": bool(getattr(display, "docked_visible", True)),
                }
            )
        return rows

    def get_display_by_id(self, display_id: int) -> BaseDisplay | None:
        for display in self.app_state.displays:
            if id(display) == display_id:
                return display
        return None

    def set_display_name(self, display: BaseDisplay, user_label: str) -> None:
        self._require_state(display)
        label = (user_label or "").strip()
        display.user_label = label or None
        self.display_changed.emit(display)

    def get_widget(self, display: BaseDisplay) -> BaseDisplay:
        self._require_state(display)
        return display

    def get_rpoc_input(self, display: BaseDisplay) -> RPOCImageInput | None:
        widget = self.get_widget(display)
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

    def _require_state(self, display: BaseDisplay) -> None:
        if display not in self.app_state.displays:
            raise KeyError("display does not exist")

    def clear_all(self) -> None:
        # from session reset/restore -> through clear_all -> to empty display inventory
        for display in list(self.app_state.displays):
            self.remove_display(display)
