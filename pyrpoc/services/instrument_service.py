from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget

from pyrpoc.domain.app_state import AppState
from pyrpoc.instruments.base_instrument import BaseInstrument, BaseInstrumentWidget
from pyrpoc.instruments.instrument_registry import instrument_registry


class InstrumentService(QObject):
    """Service managing active instrument instances as an instance-first inventory.

    Flow overview:
    - from InstrumentManager add/remove/expand actions
    - through this service's inventory/get_widget methods
    - to `AppState.instruments` and concrete instrument widgets.
    """

    inventory_changed = pyqtSignal()

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state

    def list_available(self) -> list[dict[str, Any]]:
        """
        List all registered instruments for dropdown population.

        Route:
        - `InstrumentManagerWidget._refresh_available`
        - -> `InstrumentService.list_available`
        - -> registry descriptors for UI rendering.
        """
        return instrument_registry.describe_all()

    def create_instrument(self, key: str) -> BaseInstrument:
        """
        Instantiate one instrument and register it in runtime inventory.

        Route:
        - Add button click
        - -> `instrument_mgr.handlers.on_add_clicked`
        - -> this method
        - -> emit `inventory_changed` for list/card refresh.
        """
        cls = instrument_registry.get_class(key)
        instance = cls(alias=key)
        self.app_state.instruments.append(instance)
        self.inventory_changed.emit()
        return instance

    def remove_instrument(self, instrument: BaseInstrument) -> None:
        """
        Remove one instrument from runtime inventory.

        Route:
        - Remove button click on card
        - -> `instrument_mgr.handlers.on_remove_requested`
        - -> this method
        - -> emit `inventory_changed` for card diff refresh.
        """
        if instrument not in self.app_state.instruments:
            return
        self.app_state.instruments.remove(instrument)
        self.inventory_changed.emit()

    def get_instances_by_class(self, cls: type[BaseInstrument]) -> list[BaseInstrument]:
        """
        Return current runtime instances matching a required instrument class.

        Route:
        - `ModalityService.validate_required_instruments/configure`
        - -> this method
        - -> modality gating/binding.
        """
        return [instance for instance in self.app_state.instruments if isinstance(instance, cls)]

    def get_connected_by_class(self, cls: type[BaseInstrument]) -> list[BaseInstrument]:
        """
        Compatibility alias retained for existing modality logic.
        """
        return self.get_instances_by_class(cls)

    def list_instances(self) -> list[dict[str, Any]]:
        """
        Return card-friendly rows for inventory UI rendering.

        Each row keeps the stable `row["state"]` key for compatibility, now pointing
        directly to the concrete `BaseInstrument` instance.
        """
        rows: list[dict[str, Any]] = []
        for instrument in self.app_state.instruments:
            key = instrument.type_key
            cls = instrument_registry.get_class(key)
            rows.append(
                {
                    "state": instrument,
                    "key": key,
                    "name": getattr(cls, "DISPLAY_NAME", key),
                }
            )
        return rows

    def get_widget(
        self,
        instrument: BaseInstrument,
        parent: QWidget | None = None,
        on_change=None,
    ) -> BaseInstrumentWidget:
        """
        Return one concrete expanded-card widget for an instrument.

        Route:
        - card expand event
        - -> `instrument_mgr.handlers.on_expand_requested`
        - -> this method
        - -> concrete `BaseInstrument.get_widget`.
        """
        self._require_instrument(instrument)
        return instrument.get_widget(parent=parent, on_change=on_change)

    def get_instance(self, instrument: BaseInstrument) -> BaseInstrument:
        self._require_instrument(instrument)
        return instrument

    def get_instance_key(self, instrument: BaseInstrument) -> str:
        self._require_instrument(instrument)
        return instrument.type_key

    def clear_all(self) -> None:
        """
        Clear inventory during reset/restore.

        Route:
        - `SessionCoordinator.reset_session/restore_on_startup`
        - -> this method
        - -> per-item remove path with `inventory_changed` emissions.
        """
        for instrument in list(self.app_state.instruments):
            self.remove_instrument(instrument)

    def _require_instrument(self, instrument: BaseInstrument) -> None:
        if instrument not in self.app_state.instruments:
            raise KeyError("instrument is not registered")
