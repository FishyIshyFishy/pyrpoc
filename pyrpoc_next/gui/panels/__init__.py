"""The dockable panels: acquisition (settings), instruments, displays, routine editor."""

from __future__ import annotations

from pyrpoc_next.gui.panels.acquisition_panel import AcquisitionPanel
from pyrpoc_next.gui.panels.displays import DisplayManagerWidget
from pyrpoc_next.gui.panels.instruments import InstrumentManagerWidget
from pyrpoc_next.gui.panels.routine_editor import RoutineEditor

__all__ = ["AcquisitionPanel", "DisplayManagerWidget", "InstrumentManagerWidget", "RoutineEditor"]
