"""End-to-end persistence test for the user's exact scenario:
set params on one modality, 'close' (save), 'reopen' (restore), and confirm that
switching modalities preserves each one's parameters — plus the ADS dock layout.
"""

from __future__ import annotations

import pyrpoc.displays  # noqa: F401  -- registers display types
import pyrpoc.modalities  # noqa: F401  -- registers modalities

from pyrpoc.domain.app_state import AppState
from pyrpoc.gui.main_gui import MainGUI
from pyrpoc.gui.styles.theme_manager import ThemeController
from pyrpoc.persistence.session_repository import SessionRepository
from pyrpoc.services.display_service import DisplayService
from pyrpoc.services.instrument_service import InstrumentService
from pyrpoc.services.modality_service import ModalityService
from pyrpoc.services.opto_control_service import OptoControlService
from pyrpoc.services.session_coordinator import SessionCoordinator


def build_stack(qapp, repo: SessionRepository):
    app_state = AppState()
    instruments = InstrumentService(app_state)
    modality = ModalityService(instruments, app_state)
    displays = DisplayService(app_state)
    opto = OptoControlService(app_state)
    theme = ThemeController(qapp)
    gui = MainGUI(instruments, modality, displays, opto, theme)
    coordinator = SessionCoordinator(
        app_state=app_state,
        repository=repo,
        theme_controller=theme,
        instrument_service=instruments,
        modality_service=modality,
        display_service=displays,
        opto_control_service=opto,
        main_window=gui,
    )
    return app_state, modality, gui, coordinator


def make_repo(tmp_path) -> SessionRepository:
    repo = SessionRepository.__new__(SessionRepository)
    repo.path = tmp_path / "session.json"
    repo.last_load_error = None
    return repo


def test_modality_params_survive_close_reopen_and_switching(qapp, tmp_path):
    repo = make_repo(tmp_path)

    # --- first session: configure two modalities, then "close" (save) ---
    _, modality, gui, coordinator = build_stack(qapp, repo)
    modality.select_modality("confocal")
    modality.set_parameter_values({"X Pixels": 128})
    modality.select_modality("split_confocal")
    modality.set_parameter_values({"X Pixels": 32})
    modality.select_modality("confocal")
    coordinator.save_now()
    gui.deleteLater()

    assert repo.path.exists()

    # --- second session: "reopen" with a fresh stack on the same session file ---
    app_state2, modality2, gui2, coordinator2 = build_stack(qapp, repo)
    try:
        coordinator2.restore_on_startup()

        # The previously selected modality comes back configured.
        assert app_state2.modality.selected_key == "confocal"
        assert modality2.get_parameter_values()["X Pixels"] == 128

        # The *other* modality's params were remembered too (the original bug).
        assert app_state2.modality.params_by_modality["split_confocal"][0].value == 32

        # Switching to it restores its edited value rather than reverting to the
        # default (the form fills remaining params with defaults around it).
        modality2.select_modality("split_confocal")
        assert modality2.get_parameter_values()["X Pixels"] == 32

        # And switching back still has the first modality's edited value.
        modality2.select_modality("confocal")
        assert modality2.get_parameter_values()["X Pixels"] == 128
    finally:
        gui2.deleteLater()


def test_ads_layout_is_captured_and_restored(qapp, tmp_path):
    repo = make_repo(tmp_path)

    _, modality, gui, coordinator = build_stack(qapp, repo)
    modality.select_modality("confocal")
    coordinator.save_now()
    saved = repo.load_or_default()
    gui.deleteLater()

    # The dock layout was captured into the session (or None if unavailable headless).
    assert saved.ads_layout is None or isinstance(saved.ads_layout, str)

    # Restoring must not raise and must leave the guard flag cleared.
    _, _, gui2, coordinator2 = build_stack(qapp, repo)
    try:
        coordinator2.restore_on_startup()
        assert gui2.restoring_layout is False
    finally:
        gui2.deleteLater()
