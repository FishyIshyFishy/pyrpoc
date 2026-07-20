"""Headless GUI smoke tests: the window builds and a sim run reaches a real display."""

from __future__ import annotations

from pyrpoc_next.core import AppState, Controller
from pyrpoc_next.gui.displays import StreamedDisplay, display_registry
from pyrpoc_next.instruments import SimulatedDAQ
from pyrpoc_next.structs.keys import DisplayKey, ModalityKey
from pyrpoc_next.structs.parameters import ParameterValue
from pyrpoc_next.structs.routine import Routine, RoutineBlock


def simulated_routine():
    block = RoutineBlock(
        modality=ModalityKey.simulated,
        values=[ParameterValue("X Pixels", 8), ParameterValue("Y Pixels", 8),
                ParameterValue("Active AI Channels", [0]), ParameterValue("Frames", 2)],
    )
    return Routine(blocks=[block], active_index=0)


def test_display_registry_has_all(qapp):
    assert set(display_registry.available()) == {
        DisplayKey.streamed, DisplayKey.tiled, DisplayKey.multichannel, DisplayKey.flim,
    }


def test_theme_loads(qapp):
    from pyrpoc_next.gui.styles.theme import dark_stylesheet

    assert len(dark_stylesheet()) > 1000  # the breeze qss resource is present


def test_main_window_builds_and_seeds_a_block(qapp):
    from pyrpoc_next.gui import MainWindow

    window = MainWindow(Controller())
    assert window.dock_manager is not None
    # the routine editor seeds one block, so the acquisition tab renders one block card
    assert len(window.controller.state.routine.blocks) == 1
    assert len(window.acquisition.block_cards) == 1
    window.close()


def test_instruments_panel_adds_instrument(qapp):
    from pyrpoc_next.gui.panels import InstrumentsPanel

    state = AppState()
    panel = InstrumentsPanel(state)
    panel.add()
    assert len(state.instruments) == 1


def test_simulated_run_reaches_a_real_display(qapp):
    display = StreamedDisplay()
    seen = []
    display.parcel_received.connect(lambda parcel: seen.append(parcel))
    state = AppState(instruments=[SimulatedDAQ()], displays=[display], routine=simulated_routine())
    controller = Controller(state)

    report = controller.play()
    assert not report.blocked
    controller.runner.thread.join(timeout=5)
    for _ in range(20):
        qapp.processEvents()

    assert len(seen) == 2  # both frames marshaled to the GUI thread and drawn
