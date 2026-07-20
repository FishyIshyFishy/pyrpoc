"""The instrument-manager panel: original widget, rewired to the new backend."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from pyrpoc_next.gui.panels.instruments.handlers import (
    on_add_clicked,
    refresh_available,
    refresh_instances,
)
from pyrpoc_next.gui.panels.instruments.state import InstrumentManagerState
from pyrpoc_next.gui.panels.instruments.ui import build_instrument_manager_ui
from pyrpoc_next.structs.keys import InstrumentKey


class InstrumentManagerWidget(QWidget):
    """Add/remove instruments as cards and test their connection.

    Takes the shared ``AppState`` directly: add/remove mutate ``state.instruments``.
    """

    def __init__(self, state, parent: QWidget | None = None):
        super().__init__(parent)
        self.app_state = state
        self.state = InstrumentManagerState()
        self.ui = build_instrument_manager_ui(self)

        self.type_combo = self.ui.type_combo
        self.add_btn = self.ui.add_btn
        self.instances_layout = self.ui.instances_layout

        self.add_btn.clicked.connect(lambda: on_add_clicked(self))
        refresh_available(self)
        refresh_instances(self)

    def selected_key(self) -> InstrumentKey | None:
        data = self.type_combo.currentData()
        return data if isinstance(data, InstrumentKey) else None
