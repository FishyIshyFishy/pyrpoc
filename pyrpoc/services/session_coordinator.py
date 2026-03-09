from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QMessageBox

from pyrpoc.domain.app_state import AppState, ParameterValue
from pyrpoc.domain.session_state import (
    InstrumentSessionState,
    ModalitySessionState,
    DisplaySessionState,
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
        self._restore_in_progress = False

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self.save_now)
        self._wire_autosave_signals()

    def _wire_autosave_signals(self) -> None:
        """
        Autosave wiring flow:
        - from inventory/state mutation signals
        - through debounce timer
        - to `save_now` snapshot persistence.
        """
        self.instrument_service.inventory_changed.connect(self.autosave_debounced)
        self.instrument_service.instance_changed.connect(self.autosave_debounced)
        self.opto_control_service.inventory_changed.connect(self.autosave_debounced)
        self.opto_control_service.control_state_changed.connect(self.autosave_debounced)
        self.opto_control_service.control_changed.connect(self.autosave_debounced)
        self.display_service.display_added.connect(lambda *_: self.autosave_debounced())
        self.display_service.display_removed.connect(lambda *_: self.autosave_debounced())
        self.display_service.display_changed.connect(lambda *_: self.autosave_debounced())
        self.modality_service.modality_selected.connect(lambda *_: self.autosave_debounced())
        self.modality_service.modality_params_changed.connect(lambda *_: self.autosave_debounced())

    def autosave_debounced(self) -> None:
        if self._restore_in_progress:
            return
        self._save_timer.start()

    def _values_to_raw(self, values: list[ParameterValue]) -> dict[str, Any]:
        return {entry.label: entry.value for entry in values}

    def capture_snapshot(self) -> SessionState:
        """
        Collect runtime app state into a session payload.

        Route:
        - autosave timer timeout or explicit Save action
        - -> this method
        - -> repository JSON encode/write.
        """
        modality_state = None
        if self.app_state.modality.selected_key is not None:
            modality_state = ModalitySessionState(
                selected_key=self.app_state.modality.selected_key,
                configured_params=list(self.app_state.modality.configured_params),
            )

        return SessionState(
            theme_mode=self.theme_controller.get_saved_mode(),
            instruments=[
                InstrumentSessionState(
                    type_key=instrument.type_key,
                    instance_id=str(getattr(instrument, "instance_id", "")),
                    connected=bool(getattr(instrument, "connected", False)),
                    persisted_state=instrument.export_persistence_state(),
                    config_values=list(getattr(instrument, "config_values", [])),
                    user_label=getattr(instrument, "user_label", None),
                )
                for instrument in self.app_state.instruments
            ],
            displays=[
                DisplaySessionState(
                    type_key=display.type_key,
                    instance_id=str(getattr(display, "instance_id", "")),
                    attached=bool(getattr(display, "attached", True)),
                    dock_visible=bool(getattr(display, "docked_visible", True)),
                    persisted_state=display.export_persistence_state(),
                    config_values=list(getattr(display, "config_values", [])),
                    user_label=getattr(display, "user_label", None),
                )
                for display in self.app_state.displays
            ],
            optocontrols=[
                OptoControlSessionState(
                    type_key=row.type_key,
                    instance_id=str(getattr(row, "instance_id", "")),
                    connected=bool(getattr(row, "connected", False)),
                    enabled=bool(getattr(row, "enabled", False)),
                    persisted_state=row.export_persistence_state(),
                    config_values=list(getattr(row, "config_values", [])),
                    user_label=getattr(row, "user_label", None),
                )
                for row in self.app_state.optocontrols
            ],
            modality=modality_state,
        )

    def save_now(self) -> None:
        if self._restore_in_progress:
            return
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
        self.save_now()

    def restore_on_startup(self) -> None:
        """
        Rebuild runtime state from persisted session.

        Route:
        - app startup or Open action
        - -> this method
        - -> clear runtime services
        - -> recreate instruments/optocontrols/displays
        - -> restore modality+layout.
        """
        self._restore_in_progress = True
        try:
            session = self.repository.load_or_default()
            if self.repository.last_load_error:
                self._show_restore_warning(self.repository.last_load_error)
            self.theme_controller.apply(session.theme_mode)

            self.display_service.clear_all()
            self.opto_control_service.clear_all()
            self.instrument_service.clear_all()
            self.modality_service.stop()
            self.app_state.modality.selected_key = None
            self.app_state.modality.selected_class = None
            self.app_state.modality.instance = None
            self.app_state.modality.configured_params = []

            for row in session.instruments:
                try:
                    instrument = self.instrument_service.create_instrument(
                        row.type_key,
                        instance_id=row.instance_id,
                        user_label=row.user_label,
                        persisted_state=row.persisted_state,
                        connected=False,
                    )
                    self._restore_instrument_connection(instrument)
                except Exception:
                    pass

            for row in session.optocontrols:
                try:
                    self.opto_control_service.create_opto_control(
                        row.type_key,
                        instance_id=row.instance_id,
                        user_label=row.user_label,
                        enabled=row.enabled,
                        connected=row.connected,
                        persisted_state=row.persisted_state,
                    )
                except Exception:
                    continue

            for row in session.displays:
                try:
                    settings = self._values_to_raw(list(row.config_values))
                    self.display_service.create_display(
                        row.type_key,
                        settings,
                        user_label=row.user_label,
                        instance_id=row.instance_id,
                        persisted_state=row.persisted_state,
                        attached=bool(row.attached),
                        dock_visible=bool(row.dock_visible),
                    )
                except Exception:
                    continue

            restored_modality = False
            if session.modality and session.modality.selected_key:
                try:
                    self.modality_service.select_modality(session.modality.selected_key)
                    if session.modality.configured_params:
                        self.modality_service.configure(self._values_to_raw(session.modality.configured_params))
                    restored_modality = True
                except Exception:
                    restored_modality = False

            if not restored_modality:
                rows = self.modality_service.list_available()
                if rows:
                    try:
                        self.modality_service.select_modality(str(rows[0]["key"]))
                    except Exception:
                        pass

        finally:
            self._restore_in_progress = False

    def _show_restore_warning(self, detail: str) -> None:
        QMessageBox.warning(
            self.main_window,
            "Session Restore Warning",
            "Session restore failed and defaults were loaded.\n\n"
            f"{detail}",
        )

    def _restore_instrument_connection(self, instrument: object) -> None:
        if not hasattr(instrument, "connect"):
            return
        name = str(getattr(instrument, "user_label", "") or getattr(instrument, "type_key", "instrument"))
        try:
            connected = bool(instrument.connect())
        except Exception:
            connected = False
        instrument.connected = connected
        if connected:
            return
        QMessageBox.warning(
            self.main_window,
            "Instrument Connection Warning",
            f"Instrument '{name}' failed to connect and will remain disconnected.",
        )
