from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget

from pyrpoc.domain.app_state import AppState, OptoControlState
from pyrpoc.optocontrols.opto_control_registry import opto_control_registry


class OptoControlService(QObject):
    inventory_changed = pyqtSignal()
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
        self.app_state.optocontrols.remove(state)
        self.inventory_changed.emit()

    def clear_all(self) -> None:
        for state in list(self.app_state.optocontrols):
            self.remove_opto_control(state)

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
                }
            )
        return rows

    def get_widget(self, state: OptoControlState, parent: QWidget | None = None) -> QWidget:
        self._require_state(state)
        return state.instance.get_widget(parent=parent)

    def collect_data_for_acquisition(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state in self.app_state.optocontrols:
            try:
                rows.append(
                    {
                        "type_key": state.type_key,
                        "alias": state.instance.alias,
                        "data": state.instance.prepare_data_for_acquisition(),
                    }
                )
            except Exception as exc:
                state.last_error = str(exc)
                self.opto_control_error.emit(state, str(exc))
                raise
        return rows

    def _require_state(self, state: OptoControlState) -> None:
        if state not in self.app_state.optocontrols:
            raise KeyError("opto-control state is not registered")
