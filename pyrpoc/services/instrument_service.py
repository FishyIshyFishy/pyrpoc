from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget

from pyrpoc.domain.app_state import AppState, InstrumentState
from pyrpoc.instruments.base_instrument import BaseInstrument, BaseInstrumentWidget
from pyrpoc.instruments.instrument_registry import instrument_registry


class InstrumentService(QObject):
    """Service managing active instrument instances as a lightweight inventory."""

    inventory_changed = pyqtSignal()

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state

    def list_available(self) -> list[dict[str, Any]]:
        """
        List all registered instruments for dropdown population.

        Call flow:
        - `InstrumentManagerWidget._refresh_available` asks for this list
        - dropdown is populated from returned rows so UI never imports instrument classes.
        """
        return instrument_registry.describe_all()

    def create_instrument(self, key: str) -> InstrumentState:
        """
        Instantiate instrument class from registry key and register in app state.

        Call flow:
        - user clicks Add in `InstrumentManagerWidget`
        - widget handler calls this method
        - emitted `inventory_changed` repopulates cards in the manager.
        """
        cls = instrument_registry.get_class(key)
        instance = cls(alias=key)
        state = InstrumentState(type_key=key, instance=instance)
        self.app_state.instruments.append(state)
        self.inventory_changed.emit()
        return state

    def remove_instrument(self, state_or_instance: InstrumentState | BaseInstrument) -> None:
        """
        Remove an instrument from app state and drop it from UI list.

        Call flow:
        - card "Remove" button -> handler -> this method
        - card is recreated after `inventory_changed`.
        """
        state = self._resolve_state(state_or_instance)
        if state is None:
            return
        self.app_state.instruments.remove(state)
        self.inventory_changed.emit()

    def get_instances_by_class(self, cls: type[BaseInstrument]) -> list[BaseInstrument]:
        """
        Return all current instances for a required/injected instrument class.

        Call flow:
        - `ModalityService.validate_required_instruments` asks for this list
        - modality gating checks only instance presence now, no connection state.
        """
        return [
            entry.instance
            for entry in self.app_state.instruments
            if isinstance(entry.instance, cls)
        ]

    def get_connected_by_class(self, cls: type[BaseInstrument]) -> list[BaseInstrument]:
        """
        Compatibility alias retained for existing modality logic.
        """
        return self.get_instances_by_class(cls)

    def list_instances(self) -> list[dict[str, Any]]:
        """
        Return card-friendly rows for inventory UI rendering.
        """
        rows: list[dict[str, Any]] = []
        for state in self.app_state.instruments:
            key = state.type_key
            cls = instrument_registry.get_class(key)
            rows.append(
                {
                    "state": state,
                    "key": key,
                    "name": getattr(cls, "DISPLAY_NAME", key),
                    "connected": state.connected,
                }
            )
        return rows

    def get_widget(
        self,
        state_or_instance: InstrumentState | BaseInstrument,
        parent: QWidget | None = None,
        on_change=None,
    ) -> BaseInstrumentWidget:
        """
        Resolve instance and return its expanded-card widget.

        Call flow:
        - manager toggles card expand -> handler -> this method
        - concrete instance supplies its own widget class via BaseInstrument.get_widget.
        """
        state = self._resolve_state(state_or_instance)
        if state is None:
            raise KeyError("instrument state is not registered")
        return state.instance.get_widget(parent=parent, on_change=on_change)

    def get_instance(self, state: InstrumentState) -> BaseInstrument:
        self._require_state(state)
        return state.instance

    def get_instance_key(self, state: InstrumentState) -> str:
        self._require_state(state)
        return state.type_key

    def clear_all(self) -> None:
        for state in list(self.app_state.instruments):
            self.remove_instrument(state)

    def _require_state(self, state: InstrumentState) -> None:
        if state not in self.app_state.instruments:
            raise KeyError("instrument state is not registered")

    def _resolve_state(self, state_or_instance: InstrumentState | BaseInstrument) -> InstrumentState | None:
        """
        Resolve either a state object or concrete instrument into a registered state.
        """
        if isinstance(state_or_instance, InstrumentState):
            if state_or_instance in self.app_state.instruments:
                return state_or_instance
            return None

        for state in self.app_state.instruments:
            if state.instance is state_or_instance:
                return state
        return None
