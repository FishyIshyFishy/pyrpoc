from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget

from pyrpoc.structs.app_state import AppState
from pyrpoc.instruments.base import BaseInstrument
from pyrpoc.instruments.registry import instrument_registry


class InstrumentService(QObject):
    """Manages active instrument instances as an instance-first inventory."""

    inventory_changed = pyqtSignal()
    instance_changed = pyqtSignal(object)

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state

    def list_available(self) -> list[dict[str, Any]]:
        return instrument_registry.describe_all()

    def create_instrument(
        self,
        key: str,
        *,
        instance_id: str | None = None,
        persisted_state: dict[str, Any] | None = None,
        user_label: str | None = None,
        connected: bool = False,
    ) -> BaseInstrument:
        cls = instrument_registry.get_class(key)
        instance = cls(alias=key)
        if instance_id:
            instance.instance_id = str(instance_id)
        instance.user_label = user_label
        instance.connected = bool(connected)
        if isinstance(persisted_state, dict):
            instance.import_persistence_state(dict(persisted_state))
        self.app_state.instruments.append(instance)
        self.inventory_changed.emit()
        return instance

    def remove_instrument(self, instrument: BaseInstrument) -> None:
        if instrument not in self.app_state.instruments:
            return
        self.app_state.instruments.remove(instrument)
        self.inventory_changed.emit()

    def get_instances_by_class(self, cls: type[BaseInstrument]) -> list[BaseInstrument]:
        return [instance for instance in self.app_state.instruments if isinstance(instance, cls)]

    def get_connected_by_class(self, cls: type[BaseInstrument]) -> list[BaseInstrument]:
        return self.get_instances_by_class(cls)

    def list_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for instrument in self.app_state.instruments:
            key = instrument.type_key
            cls = instrument_registry.get_class(key)
            rows.append({"state": instrument, "key": key, "name": getattr(cls, "display_name", key)})
        return rows

    def get_widget(self, instrument: BaseInstrument, parent: QWidget | None = None, on_change=None):
        self.require_instrument(instrument)
        from pyrpoc.gui.widgets.instrument_widgets import build_instrument_widget
        return build_instrument_widget(instrument, parent=parent, on_change=on_change)

    def get_instance(self, instrument: BaseInstrument) -> BaseInstrument:
        self.require_instrument(instrument)
        return instrument

    def get_instance_key(self, instrument: BaseInstrument) -> str:
        self.require_instrument(instrument)
        return instrument.type_key

    def clear_all(self) -> None:
        for instrument in list(self.app_state.instruments):
            self.remove_instrument(instrument)

    def mark_instance_changed(self, instrument: BaseInstrument) -> None:
        self.require_instrument(instrument)
        self.instance_changed.emit(instrument)

    def require_instrument(self, instrument: BaseInstrument) -> None:
        if instrument not in self.app_state.instruments:
            raise KeyError("instrument is not registered")
