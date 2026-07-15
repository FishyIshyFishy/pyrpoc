from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QStandardPaths

# Ensure registries populate: presets (which import instruments + optocontrols
# + engine handlers) and displays.
import pyrpoc.presets  # noqa: F401
import pyrpoc.gui.displays  # noqa: F401

from pyrpoc.gui.services.routing import AcquisitionInterpreter
from pyrpoc.gui.services.acquisition_service import AcquisitionService
from pyrpoc.structs.app_state import AppState
from pyrpoc.gui.services.display_service import DisplayService
from pyrpoc.gui.services.instrument_service import InstrumentService
from pyrpoc.gui.services.opto_control_service import OptoControlService
from pyrpoc.gui.services.session_coordinator import SessionCoordinator
from pyrpoc.utils.session_store import SessionRepository
from pyrpoc.gui.window import MainGUI
from pyrpoc.gui.theme.theme_manager import ThemeController


def session_path() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    root = Path(base) if base else Path(".")
    root.mkdir(parents=True, exist_ok=True)
    return root / "session.json"


class AppController(QObject):
    def __init__(self, theme_controller: ThemeController, parent=None):
        super().__init__(parent)
        self.app_state = AppState()

        self.instrument_service = InstrumentService(self.app_state, self)
        self.acquisition_service = AcquisitionService(self.instrument_service, self.app_state, self)
        self.display_service = DisplayService(self.app_state, self)
        self.opto_control_service = OptoControlService(self.app_state, self)

        self.interpreter = AcquisitionInterpreter(self.acquisition_service, self.app_state, self)

        self.instrument_service.inventory_changed.connect(
            lambda *_: self.acquisition_service.validate_required_instruments()
        )

        self.main_window = MainGUI(
            instrument_service=self.instrument_service,
            acquisition_service=self.acquisition_service,
            display_service=self.display_service,
            opto_control_service=self.opto_control_service,
            theme_controller=theme_controller,
        )
        self.session_coordinator = SessionCoordinator(
            app_state=self.app_state,
            repository=SessionRepository(session_path()),
            theme_controller=theme_controller,
            instrument_service=self.instrument_service,
            acquisition_service=self.acquisition_service,
            display_service=self.display_service,
            opto_control_service=self.opto_control_service,
            main_window=self.main_window,
            parent=self,
        )
        self.main_window.menubar.new_requested.connect(self.session_coordinator.reset_session)
        self.main_window.menubar.open_requested.connect(self.session_coordinator.restore_on_startup)
        self.main_window.menubar.save_requested.connect(self.session_coordinator.save_now)
        self.main_window.menubar.save_as_requested.connect(self.session_coordinator.save_now)
        self.main_window.closing.connect(self.session_coordinator.save_now)

    def show(self) -> None:
        self.main_window.show()
        self.session_coordinator.restore_on_startup()
