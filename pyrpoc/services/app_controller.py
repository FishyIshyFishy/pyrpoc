from __future__ import annotations

from PyQt6.QtCore import QObject

# Ensure decorators run and registries are populated.
import pyrpoc.displays  # noqa: F401
import pyrpoc.instruments  # noqa: F401
import pyrpoc.modalities  # noqa: F401

from pyrpoc.gui.main_gui import MainGUI
from .display_service import DisplayService
from .instrument_service import InstrumentService
from .modality_service import ModalityService


class AppController(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.instrument_service = InstrumentService(self)
        self.modality_service = ModalityService(self.instrument_service, self)
        self.display_service = DisplayService(self)

        self.modality_service.data_ready.connect(self.display_service.push_data)
        self.instrument_service.connection_changed.connect(
            lambda _alias, _connected: self.modality_service.validate_required_instruments()
        )

        self.main_window = MainGUI(
            instrument_service=self.instrument_service,
            modality_service=self.modality_service,
            display_service=self.display_service,
        )

    def show(self) -> None:
        self.main_window.show()
