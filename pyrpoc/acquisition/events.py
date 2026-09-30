"""Qt in front of the executor: starting runs and reporting them on the GUI thread.

Runs report on their worker threads; the signals here are what carries that
onto the GUI thread. This is the only subscriber to each dataset a run opens,
because ``Dataset.append`` runs on the worker thread. Nothing here knows which
program is selected or which runner asked.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.acquisition.claims import DeviceBusy
from pyrpoc.acquisition.executor import Executor, Run, RunCallbacks
from pyrpoc.acquisition.recording import Series
from pyrpoc.data_library.store import LibraryFull, LibraryStore
from pyrpoc.structs.device import Device, MissingDevice
from pyrpoc.structs.params import BlockStore, ParameterError
from pyrpoc.structs.program import Program
from pyrpoc.structs.saving import SaveTarget


class RunEvents(QObject):
    """Starts runs and reports them. Each signal carries its ``Run``."""

    run_started = pyqtSignal(object)
    run_status = pyqtSignal(object, str)
    run_failed = pyqtSignal(object, str)
    run_finished = pyqtSignal(object)
    # Refused before any run existed: a missing or busy device, a bad parameter,
    # or a full library.
    start_refused = pyqtSignal(str)

    def __init__(self, library: LibraryStore, parent: QObject):
        super().__init__(parent)
        self.executor = Executor(
            library,
            RunCallbacks(
                on_status=self.run_status.emit,
                on_failed=self.run_failed.emit,
                on_finished=self.run_finished.emit,
            ),
        )

    def start(
        self,
        program: Program,
        key: str,
        blocks: BlockStore,
        devices: list[Device],
        save: SaveTarget,
        series: Series | None,
        claim: Callable[[Run], None],
    ) -> None:
        """Start ``program``, or report why not. Reported rather than raised,
        since every caller is a Qt slot. ``claim`` gets the run before
        ``run_started`` announces it, so whoever started it has already
        recorded it by the time anyone asks."""
        try:
            run = self.executor.start(
                program, blocks, devices, program_key=key, save=save, series=series
            )
        except (MissingDevice, DeviceBusy, ParameterError, LibraryFull) as exc:
            self.start_refused.emit(str(exc))
            return
        claim(run)
        self.run_started.emit(run)

    def end_series(self, series: Series) -> None:
        self.executor.end_series(series)
