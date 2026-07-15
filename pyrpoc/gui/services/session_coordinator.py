from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QMessageBox

from pyrpoc.gui.services.acquisition_service import AcquisitionService
from pyrpoc.structs.app_state import AppState
from pyrpoc.gui.services.display_service import DisplayService
from pyrpoc.gui.services.instrument_service import InstrumentService
from pyrpoc.gui.services.opto_control_service import OptoControlService
from pyrpoc.instruments.base import BaseInstrument
from pyrpoc.utils.session_store import SessionRepository
from pyrpoc.gui.window import MainGUI
from pyrpoc.gui.theme.theme_manager import ThemeController
from pyrpoc.structs.parameters import ParameterValue
from pyrpoc.structs.session import (
    DisplaySessionState,
    InstrumentSessionState,
    OptoControlSessionState,
    PresetSessionState,
    SessionState,
)


class SessionCoordinator(QObject):
    def __init__(
        self,
        app_state: AppState,
        repository: SessionRepository,
        theme_controller: ThemeController,
        instrument_service: InstrumentService,
        acquisition_service: AcquisitionService,
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
        self.acquisition_service = acquisition_service
        self.display_service = display_service
        self.opto_control_service = opto_control_service
        self.main_window = main_window
        self.restore_in_progress = False

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(300)
        self.save_timer.timeout.connect(self.save_now)
        self.wire_autosave_signals()

    def wire_autosave_signals(self) -> None:
        self.instrument_service.inventory_changed.connect(self.autosave_debounced)
        self.instrument_service.instance_changed.connect(self.autosave_debounced)
        self.opto_control_service.inventory_changed.connect(self.autosave_debounced)
        self.opto_control_service.control_state_changed.connect(self.autosave_debounced)
        self.opto_control_service.control_changed.connect(self.autosave_debounced)
        self.display_service.display_added.connect(lambda *_: self.autosave_debounced())
        self.display_service.display_removed.connect(lambda *_: self.autosave_debounced())
        self.display_service.display_changed.connect(lambda *_: self.autosave_debounced())
        self.acquisition_service.modality_selected.connect(lambda *_: self.autosave_debounced())
        self.acquisition_service.modality_params_changed.connect(lambda *_: self.autosave_debounced())

    def autosave_debounced(self) -> None:
        if self.restore_in_progress:
            return
        self.save_timer.start()

    def values_to_raw(self, values: list[ParameterValue]) -> dict[str, Any]:
        return {entry.label: entry.value for entry in values}

    def capture_snapshot(self) -> SessionState:
        preset_state = PresetSessionState(
            selected_key=self.app_state.preset.selected_key,
            params_by_preset={
                key: list(values)
                for key, values in self.app_state.preset.params_by_preset.items()
            },
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
            preset=preset_state,
            ads_layout=self.main_window.save_dock_layout(),
        )

    def save_now(self) -> None:
        if self.restore_in_progress:
            return
        self.repository.save(self.capture_snapshot())

    def reset_preset_state(self) -> None:
        self.app_state.preset.selected_key = None
        self.app_state.preset.selected_class = None
        self.app_state.preset.instance = None
        self.app_state.preset.params_by_preset = {}

    def reset_session(self) -> None:
        self.display_service.clear_all()
        self.opto_control_service.clear_all()
        self.instrument_service.clear_all()
        self.acquisition_service.stop()
        self.reset_preset_state()
        self.save_now()

    def restore_on_startup(self) -> None:
        self.restore_in_progress = True
        try:
            session = self.repository.load_or_default()
            if self.repository.last_load_error:
                self.show_restore_warning(self.repository.last_load_error)
            self.theme_controller.apply(session.theme_mode)

            self.display_service.clear_all()
            self.opto_control_service.clear_all()
            self.instrument_service.clear_all()
            self.acquisition_service.stop()
            self.reset_preset_state()

            for row in session.instruments:
                try:
                    instrument = self.instrument_service.create_instrument(
                        row.type_key,
                        instance_id=row.instance_id,
                        user_label=row.user_label,
                        persisted_state=row.persisted_state,
                        connected=False,
                    )
                    self.restore_instrument_connection(instrument)
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
                    settings = self.values_to_raw(list(row.config_values))
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

            if session.preset is not None:
                self.app_state.preset.params_by_preset = {
                    key: list(values) for key, values in session.preset.params_by_preset.items()
                }

            restored = False
            if session.preset and session.preset.selected_key:
                try:
                    self.acquisition_service.select_modality(session.preset.selected_key)
                    remembered = self.acquisition_service.get_parameter_values()
                    if remembered:
                        self.acquisition_service.configure(remembered)
                    restored = True
                except Exception:
                    restored = False

            if not restored:
                rows = self.acquisition_service.list_available()
                if rows:
                    try:
                        self.acquisition_service.select_modality(str(rows[0]["key"]))
                    except Exception:
                        pass

            self.main_window.restore_dock_layout(session.ads_layout)
        finally:
            self.restore_in_progress = False

    def show_restore_warning(self, detail: str) -> None:
        QMessageBox.warning(
            self.main_window,
            "Session Restore Warning",
            f"Session restore failed and defaults were loaded.\n\n{detail}",
        )

    def restore_instrument_connection(self, instrument: BaseInstrument) -> None:
        name = str(instrument.user_label or instrument.type_key or "instrument")
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
