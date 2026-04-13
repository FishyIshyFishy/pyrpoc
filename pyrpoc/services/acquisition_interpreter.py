from __future__ import annotations

from PyQt6.QtCore import QObject

from pyrpoc.backend_utils.acquired_data import AcquiredData
from pyrpoc.domain.app_state import AppState

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrpoc.services.modality_service import ModalityService


class AcquisitionInterpreter(QObject):
    """Routes AcquiredData objects from the modality layer to compatible displays.

    Wired per acquisition session: connects to modality_service.data_emitted at
    acquisition start and disconnects at acquisition stop. modality_service and
    display_service remain agnostic to this routing logic.
    """

    def __init__(
        self,
        modality_service: ModalityService,
        app_state: AppState,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._modality_service = modality_service
        self._app_state = app_state

        modality_service.acq_started.connect(self._on_acq_started)
        modality_service.acq_stopped.connect(self._on_acq_stopped)

    def _on_acq_started(self) -> None:
        self._modality_service.data_emitted.connect(self._route)

    def _on_acq_stopped(self) -> None:
        try:
            self._modality_service.data_emitted.disconnect(self._route)
        except RuntimeError:
            pass  # already disconnected

    def _route(self, acquired: AcquiredData) -> None:
        for display in self._app_state.displays:
            if not display.attached or not display.docked_visible:
                continue
            if acquired.kind in display.ACCEPTED_KINDS:
                try:
                    display.render(acquired)
                except Exception as exc:
                    display.last_error = str(exc)
