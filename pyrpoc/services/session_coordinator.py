from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, QTimer

from pyrpoc.domain.app_state import AppState, ParameterValue
from pyrpoc.domain.session_state import (
    DisplaySessionState,
    ModalitySessionState,
    OptoControlSessionState,
    SessionState,
)
from pyrpoc.gui.main_gui import MainGUI
from pyrpoc.gui.styles.theme_manager import ThemeController
from pyrpoc.persistence.session_repository import SessionRepository
from .display_service import DisplayService
from .instrument_service import InstrumentService
from .modality_service import ModalityService
from .opto_control_service import OptoControlService


class SessionCoordinator(QObject):
    def __init__(
        self,
        app_state: AppState,
        repository: SessionRepository,
        theme_controller: ThemeController,
        instrument_service: InstrumentService,
        modality_service: ModalityService,
        display_service: DisplayService,
        opto_control_service: OptoControlService,
        main_window: MainGUI,
        parent=None,
    ):
        super().__init__(parent)
        self.app_state = app_state
        self.repository = repository
        self.theme_controller = theme_controller
        self.instrument_service = instrument_service
        self.modality_service = modality_service
        self.display_service = display_service
        self.opto_control_service = opto_control_service
        self.main_window = main_window

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self.save_now)
        self._wire_autosave_signals()

    def _wire_autosave_signals(self) -> None:
        self.instrument_service.inventory_changed.connect(self.autosave_debounced)
        self.opto_control_service.inventory_changed.connect(self.autosave_debounced)
        self.opto_control_service.control_state_changed.connect(self.autosave_debounced)
        self.display_service.display_added.connect(lambda *_: self.autosave_debounced())
        self.display_service.display_removed.connect(lambda *_: self.autosave_debounced())
        self.display_service.display_changed.connect(lambda *_: self.autosave_debounced())
        self.modality_service.modality_selected.connect(lambda *_: self.autosave_debounced())

    def autosave_debounced(self) -> None:
        self._save_timer.start()

    def _values_to_raw(self, values: list[ParameterValue]) -> dict[str, Any]:
        return {entry.label: entry.value for entry in values}

    def capture_snapshot(self) -> SessionState:
        '''Collect runtime app state into session payload for persistence.

        Called by autosave and explicit Save; the optocontrols here are serialized as
        `OptoControlSessionState` rows and restored later via `restore_on_startup`.
        '''
        modality_state = None
        if self.app_state.modality.selected_key is not None:
            modality_state = ModalitySessionState(
                selected_key=self.app_state.modality.selected_key,
                configured_params=list(self.app_state.modality.configured_params),
            )

        return SessionState(
            theme_mode=self.theme_controller.get_saved_mode(),
            # Temporary migration policy: do not persist instrument rows while we
            # refactor toward a minimal add/remove+widget model.
            instruments=[],
            optocontrols=[
                OptoControlSessionState(
                    type_key=row.type_key,
                    connected=False,
                    enabled=row.enabled,
                    config_values=[],
                    user_label=row.user_label,
                )
                for row in self.app_state.optocontrols
            ],
            displays=[
                DisplaySessionState(
                    type_key=row.type_key,
                    attached=row.attached,
                    config_values=list(row.config_values),
                    user_label=row.user_label,
                )
                for row in self.app_state.displays
            ],
            modality=modality_state,
            gui_layout=self.main_window.capture_layout_state(),
        )

    def save_now(self) -> None:
        self.repository.save(self.capture_snapshot())

    def reset_session(self) -> None:
        self.display_service.clear_all()
        self.opto_control_service.clear_all()
        self.instrument_service.clear_all()
        self.modality_service.stop()
        self.app_state.modality.selected_key = None
        self.app_state.modality.selected_class = None
        self.app_state.modality.instance = None
        self.app_state.modality.configured_params = []
        self.main_window.restore_default_layout()
        self.save_now()

    def restore_on_startup(self) -> None:
        '''Rebuild app state from saved rows with a simplified instrument phase.

        Opto-controls and displays are restored as before. Instrument rows are
        intentionally not reconstructed here while the add/remove model is in
        migration.
        '''
        session = self.repository.load_or_default()
        self.theme_controller.apply(session.theme_mode)

        self.display_service.clear_all()
        self.opto_control_service.clear_all()
        self.instrument_service.clear_all()

        # Legacy instrument rows are intentionally ignored during this migration.
        # This avoids rehydration/auto-connect behavior and keeps startup behavior
        # stable while the simplified inventory contract is being built out.
        for _row in getattr(session, "instruments", []):
            continue

        for row in session.optocontrols:
            state = self.opto_control_service.create_opto_control(row.type_key)
            state.user_label = row.user_label
            state.enabled = row.enabled

        if session.modality and session.modality.selected_key:
            try:
                self.modality_service.select_modality(session.modality.selected_key)
                if session.modality.configured_params:
                    self.modality_service.configure(self._values_to_raw(session.modality.configured_params))
            except Exception:
                pass

        for row in session.displays:
            try:
                state = self.display_service.create_display(row.type_key, self._values_to_raw(row.config_values))
                state.user_label = row.user_label
                if not row.attached:
                    self.display_service.detach(state)
            except Exception:
                pass

        self.main_window.restore_layout_state(session.gui_layout)
