from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject

from pyrpoc.structs.app_state import AppState
from pyrpoc.structs.acquired_data import AcquiredData

if TYPE_CHECKING:
    from pyrpoc.gui.services.acquisition_service import AcquisitionService


class AcquisitionInterpreter(QObject):
    """Routes AcquiredData from the acquisition service to compatible displays.

    Connects to data_emitted while acquisition is live; the acquisition and
    display services stay agnostic to this routing.
    """

    def __init__(self, acquisition_service: "AcquisitionService", app_state: AppState, parent: QObject | None = None):
        super().__init__(parent)
        self.acquisition_service = acquisition_service
        self.app_state = app_state
        acquisition_service.acq_started.connect(self.on_acq_started)
        acquisition_service.acq_stopped.connect(self.on_acq_stopped)

    def on_acq_started(self) -> None:
        self.acquisition_service.data_emitted.connect(self.route)

    def on_acq_stopped(self) -> None:
        try:
            self.acquisition_service.data_emitted.disconnect(self.route)
        except (RuntimeError, TypeError):
            pass

    def route(self, acquired: AcquiredData) -> None:
        for display in self.app_state.displays:
            if not display.attached or not display.docked_visible:
                continue
            if acquired.kind in display.accepted_kinds:
                try:
                    display.render(acquired)
                except Exception as exc:
                    display.last_error = str(exc)
