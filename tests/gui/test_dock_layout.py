from __future__ import annotations

import PyQt6Ads as qtads

import pyrpoc.displays  # noqa: F401  -- registers display types
import pyrpoc.modalities  # noqa: F401  -- registers modalities

from pyrpoc.domain.app_state import AppState
from pyrpoc.gui.main_gui import DockKey, MainGUI
from pyrpoc.gui.styles.theme_manager import ThemeController
from pyrpoc.services.display_service import DisplayService
from pyrpoc.services.instrument_service import InstrumentService
from pyrpoc.services.modality_service import ModalityService
from pyrpoc.services.opto_control_service import OptoControlService


def build_gui(qapp) -> MainGUI:
    app_state = AppState()
    instruments = InstrumentService(app_state)
    modality = ModalityService(instruments, app_state)
    displays = DisplayService(app_state)
    opto = OptoControlService(app_state)
    theme = ThemeController(qapp)
    return MainGUI(instruments, modality, displays, opto, theme)


def test_save_dock_layout_returns_str_or_none(qapp):
    gui = build_gui(qapp)
    try:
        layout = gui.save_dock_layout()
        assert layout is None or isinstance(layout, str)
    finally:
        gui.deleteLater()


def test_restore_dock_layout_is_robust(qapp):
    gui = build_gui(qapp)
    try:
        # None / empty / garbage must all be safe no-ops, never raising.
        gui.restore_dock_layout(None)
        gui.restore_dock_layout("")
        gui.restore_dock_layout("!!not-valid-base64!!")
        layout = gui.save_dock_layout()
        if layout:
            gui.restore_dock_layout(layout)  # genuine round-trip must not raise
        # The guard flag must always be reset, even if restoreState misbehaved.
        assert gui.restoring_layout is False
    finally:
        gui.deleteLater()


def test_dock_layout_actually_round_trips(qapp):
    """The real regression test: a rearranged layout must restore byte-identically
    in a fresh GUI. This only holds when default docks set objectName *before*
    addDockWidget; with the old (after-add) order, restoreState silently no-ops.
    """
    gui1 = build_gui(qapp)
    # Rearrange: pull the Instruments dock out of the left tab group into the right area.
    moved = gui1.dock_by_key[DockKey.INSTRUMENTS]
    gui1.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, moved)
    saved = gui1.save_dock_layout()
    gui1.deleteLater()

    assert saved  # a real, non-empty layout was captured

    gui2 = build_gui(qapp)
    try:
        gui2.restore_dock_layout(saved)
        # Re-saving the restored layout reproduces the exact saved state.
        assert gui2.save_dock_layout() == saved
    finally:
        gui2.deleteLater()
