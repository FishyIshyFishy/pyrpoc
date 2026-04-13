from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from pyrpoc.gui.main_widgets.instrument_mgr.handlers import (
    on_add_clicked,
    refresh_available,
    refresh_instances,
)
from pyrpoc.gui.main_widgets.instrument_mgr.state import InstrumentManagerState
from pyrpoc.gui.main_widgets.instrument_mgr.ui import build_instrument_manager_ui
from pyrpoc.services.instrument_service import InstrumentService


class InstrumentManagerWidget(QWidget):
    def __init__(self, instrument_service: InstrumentService, parent: QWidget | None = None):
        super().__init__(parent)
        self.instrument_service = instrument_service
        self.state = InstrumentManagerState()
        self.ui = build_instrument_manager_ui(self)

        # Stable handles used by legacy-style handlers.
        self.type_combo = self.ui.type_combo
        self.add_btn = self.ui.add_btn
        self.instances_layout = self.ui.instances_layout

        self._wire_signals()
        self._refresh_available()
        self._refresh_instances()

    def _wire_signals(self) -> None:
        '''
        Widget signals are intentionally small:
        - Add button is the only direct user action here.
        - Service emits `inventory_changed` for any list refresh after add/remove.
        '''
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.instrument_service.inventory_changed.connect(self._refresh_instances)

    def _refresh_available(self) -> None:
        """
        Dropdown content should always come from service registry descriptors.
        """
        refresh_available(self)

    def _refresh_instances(self) -> None:
        """
        Keep existing cards when state objects are unchanged and only add/remove
        diffs, so expanded card widgets keep their QWidget owners.
        """
        refresh_instances(self)

    def _on_add_clicked(self) -> None:
        """
        Add button handler -> `InstrumentService.create_instrument`.
        """
        on_add_clicked(self)

    def _selected_type_key(self) -> str:
        data = self.type_combo.currentData()
        if isinstance(data, str):
            return data
        return self.type_combo.currentText().strip()
