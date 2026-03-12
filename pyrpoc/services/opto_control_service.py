from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget

from pyrpoc.domain.app_state import AppState
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.opto_control_registry import opto_control_registry


class OptoControlService(QObject):
    inventory_changed = pyqtSignal()
    control_state_changed = pyqtSignal(object, bool)
    control_changed = pyqtSignal(object)
    opto_control_error = pyqtSignal(object, str)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state

    def list_available(self) -> list[dict[str, Any]]:
        return opto_control_registry.describe_all()

    def create_opto_control(
        self,
        key: str,
        *,
        instance_id: str | None = None,
        persisted_state: dict[str, Any] | None = None,
        user_label: str | None = None,
        enabled: bool = False,
        connected: bool = False,
    ) -> BaseOptoControl:
        '''Create a control instance from dropdown selection and add it to app state.

        Called by `opto_control_mgr.handlers.on_add_clicked`, then rendered by
        `refresh_instances` in the same file.
        '''
        cls = opto_control_registry.get_class(key)
        instance = cls(alias=key)
        if instance_id:
            instance.instance_id = str(instance_id)
        instance.user_label = user_label
        instance.enabled = bool(enabled)
        instance.connected = bool(connected)
        if isinstance(persisted_state, dict):
            instance.import_persistence_state(dict(persisted_state))
        self.app_state.optocontrols.append(instance)
        self.inventory_changed.emit()
        return instance

    def remove_opto_control(self, control: BaseOptoControl) -> None:
        '''Remove a control and release any resources it owns.

        Called from the manager remove button handler. This keeps persistence and UI list
        in sync by deleting the same object referenced by `AppState`.
        '''
        if control not in self.app_state.optocontrols:
            return
        self.app_state.optocontrols.remove(control)
        control.cleanup()
        self.control_state_changed.emit(control, False)
        self.inventory_changed.emit()

    def clear_all(self) -> None:
        for state in list(self.app_state.optocontrols):
            self.remove_opto_control(state)

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for control in self.app_state.optocontrols:
            key = control.type_key
            cls = opto_control_registry.get_class(key)
            rows.append(
                {
                    "state": control,
                    "key": key,
                    "name": getattr(cls, "DISPLAY_NAME", key),
                    "enabled": control.enabled,
                }
            )
        return rows

    def get_widget(
        self,
        control: BaseOptoControl,
        parent: QWidget | None = None,
        display_service: Any | None = None,
        on_change=None,
    ) -> QWidget:
        '''Return the concrete control widget for list rows and editors.

        Called by `refresh_instances` during UI rebuild, after which signals from the
        card bind user actions back into the service.
        '''
        self._require_control(control)
        return control.get_widget(parent=parent, display_service=display_service, on_change=on_change)

    def set_enabled(self, control: BaseOptoControl, enabled: bool) -> None:
        '''Track enabled state in the control instance and notify persistence/autosave path.

        Called when the row checkbox changes.
        '''
        self._require_control(control)
        control.enabled = enabled
        self.control_state_changed.emit(control, enabled)
        self.control_changed.emit(control)

    def collect_data_for_acquisition(self) -> list[Any]:
        '''Collect enabled control payloads in UI order.

        This is the handoff point from GUI config to modality-specific execution.
        '''
        rows: list[Any] = []
        for control in self.app_state.optocontrols:
            if not control.enabled:
                continue
            try:
                rows.append(control.prepare_for_acquisition())
            except Exception as exc:
                control.last_error = str(exc)
                self.opto_control_error.emit(control, str(exc))
                raise
        return rows

    def _require_control(self, control: BaseOptoControl) -> None:
        if control not in self.app_state.optocontrols:
            raise KeyError("opto-control is not registered")

    def mark_control_changed(self, control: BaseOptoControl) -> None:
        self._require_control(control)
        self.control_changed.emit(control)
