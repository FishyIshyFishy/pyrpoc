"""The display-manager panel: original widget, rewired to the new backend."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from pyrpoc_next.gui.panels.displays.handlers import (
    on_add_clicked,
    refresh_available,
    refresh_instances,
)
from pyrpoc_next.gui.panels.displays.state import DisplayManagerState
from pyrpoc_next.gui.panels.displays.ui import build_display_manager_ui
from pyrpoc_next.structs.keys import DisplayKey


class DisplayManagerWidget(QWidget):
    """Add displays (each opens as its own dock) and list them as removable cards.

    Each card carries the attach toggle that gates the display's live updates.
    """

    def __init__(
        self,
        state,
        open_dock: Callable[[QWidget, str], object],
        close_dock: Callable[[object], None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.app_state = state
        self.open_dock = open_dock
        self.close_dock = close_dock
        self.docks: dict[object, object] = {}
        self.state = DisplayManagerState()
        self.ui = build_display_manager_ui(self)

        self.display_combo = self.ui.display_combo
        self.add_btn = self.ui.add_btn
        self.instances_layout = self.ui.instances_layout

        self.add_btn.clicked.connect(lambda: on_add_clicked(self))
        refresh_available(self)
        refresh_instances(self)

    def selected_key(self) -> DisplayKey | None:
        data = self.display_combo.currentData()
        return data if isinstance(data, DisplayKey) else None
