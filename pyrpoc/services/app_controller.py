from __future__ import annotations

from PyQt6.QtCore import QObject

# Ensure decorators run and registries are populated.
import pyrpoc.displays  # noqa: F401
import pyrpoc.instruments  # noqa: F401
import pyrpoc.modalities  # noqa: F401
import pyrpoc.optocontrols  # noqa: F401

from pyrpoc.domain.app_state import AppState
from pyrpoc.gui.main_gui import MainGUI
from pyrpoc.gui.styles.theme_manager import ThemeController
from pyrpoc.persistence.session_repository import SessionRepository
from .display_service import DisplayService
from .instrument_service import InstrumentService
from .modality_service import ModalityService
from .opto_control_service import OptoControlService
from .session_coordinator import SessionCoordinator


class AppController(QObject):
    def __init__(self, theme_controller: ThemeController, parent=None):
        super().__init__(parent)
        self.app_state = AppState()

        self.instrument_service = InstrumentService(self.app_state, self)
        self.modality_service = ModalityService(self.instrument_service, self.app_state, self)
        self.display_service = DisplayService(self.app_state, self)
        self.opto_control_service = OptoControlService(self.app_state, self)

        self.modality_service.data_ready.connect(self.display_service.push_data)
        self.instrument_service.connection_changed.connect(
            lambda *_: self.modality_service.validate_required_instruments()
        )

        self.main_window = MainGUI(
            instrument_service=self.instrument_service,
            modality_service=self.modality_service,
            display_service=self.display_service,
            opto_control_service=self.opto_control_service,
            theme_controller=theme_controller,
        )
        self.session_coordinator = SessionCoordinator(
            app_state=self.app_state,
            repository=SessionRepository(),
            theme_controller=theme_controller,
            instrument_service=self.instrument_service,
            modality_service=self.modality_service,
            display_service=self.display_service,
            opto_control_service=self.opto_control_service,
            main_window=self.main_window,
            parent=self,
        )
        self.main_window.menubar.new_requested.connect(self.session_coordinator.reset_session)
        self.main_window.menubar.open_requested.connect(self.session_coordinator.restore_on_startup)
        self.main_window.menubar.save_requested.connect(self.session_coordinator.save_now)
        self.main_window.menubar.save_as_requested.connect(self.session_coordinator.save_now)

    def show(self) -> None:
        self.session_coordinator.restore_on_startup()
        self.main_window.show()
