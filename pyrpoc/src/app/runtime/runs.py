"""Qt in front of the executor: starting runs and reporting them on the GUI thread.

Runs report on their worker threads; the signals here are what carries that
onto the GUI thread. This is the only subscriber to each dataset a run opens,
because ``Dataset.append`` runs on the worker thread. Nothing here knows which
program is selected or which runner asked.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.src.structs.data import Dataset, SaveTarget
from pyrpoc.src.structs.device import Device, MissingDevice
from pyrpoc.src.structs.params import BlockStore, ParameterError
from pyrpoc.src.structs.program import Program

from .claims import DeviceBusy
from .executor import Executor, Run, RunCallbacks
from .library import DataLibrary


class Runs(QObject):
    """Starts runs and reports them. Each signal carries its ``Run``."""

    run_started = pyqtSignal(object)
    run_status = pyqtSignal(object, str)
    run_failed = pyqtSignal(object, str)
    run_finished = pyqtSignal(object)
    # Refused before any run existed: a missing or busy device, a bad parameter.
    start_refused = pyqtSignal(str)
    dataset_changed = pyqtSignal(object)

    def __init__(self, library: DataLibrary, parent: QObject):
        super().__init__(parent)
        self.library = library
        self.executor = Executor(
            library,
            RunCallbacks(
                on_dataset=self.on_dataset,
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
        claim: Callable[[Run], None],
    ) -> None:
        """Start ``program``, or report why not. Reported rather than raised,
        since every caller is a Qt slot. ``claim`` gets the run before
        ``run_started`` announces it, so whoever started it has already
        recorded it by the time anyone asks."""
        try:
            run = self.executor.start(program, blocks, devices, program_key=key, save=save)
        except (MissingDevice, DeviceBusy, ParameterError) as exc:
            self.start_refused.emit(str(exc))
            return
        claim(run)
        self.run_started.emit(run)

    def on_dataset(self, dataset: Dataset) -> None:
        dataset.subscribe(self.on_dataset_changed)

    def on_dataset_changed(self, dataset: Dataset) -> None:
        """Called on the worker thread. The signal hops to the GUI thread."""
        self.dataset_changed.emit(dataset)

    def release(self, dataset: Dataset) -> None:
        dataset.unsubscribe(self.on_dataset_changed)
        self.library.remove(dataset)
