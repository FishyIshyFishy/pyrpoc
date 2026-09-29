"""Qt in front of the executor: worker-thread events onto the GUI thread.

It subscribes to each dataset the executor creates and re-emits as Qt signals,
which Qt queues to receivers on the GUI thread. It is the only subscriber to a
dataset, because ``Dataset.append`` runs on the worker thread.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.src.structs.data import Dataset
from pyrpoc.src.structs.device import Device, MissingDevice
from pyrpoc.src.structs.params import BlockStore, ParameterError
from pyrpoc.src.structs.program import Program

from .executor import Executor, RunCallbacks
from .library import DataLibrary
from .saving import SaveTarget


class RunBridge(QObject):
    run_started = pyqtSignal()
    run_status = pyqtSignal(str)
    dataset_opened = pyqtSignal(object)
    dataset_changed = pyqtSignal(object)
    run_finished = pyqtSignal()
    run_failed = pyqtSignal(str)

    def __init__(self, library: DataLibrary, parent: QObject):
        super().__init__(parent)
        self.library = library
        self.executor = Executor(self.library)
        self._subscribed: list[Dataset] = []

    @property
    def is_running(self) -> bool:
        return self.executor.is_running

    def start(
        self,
        program: Program,
        blocks: BlockStore,
        devices: list[Device],
        *,
        continuous: bool,
        program_key: str,
        save: SaveTarget,
    ) -> None:
        """Start a run. A missing device or a bad parameter is reported through
        ``run_failed`` rather than raised, since every caller is a Qt slot."""
        callbacks = RunCallbacks(
            on_status=self.run_status.emit,
            on_dataset=self.on_dataset,
            on_finished=self.run_finished.emit,
            on_failed=self.run_failed.emit,
        )
        try:
            self.executor.start(
                program,
                blocks,
                devices,
                continuous=continuous,
                program_key=program_key,
                save=save,
                callbacks=callbacks,
            )
        except (MissingDevice, ParameterError) as exc:
            self.run_failed.emit(str(exc))
            return
        self.run_started.emit()

    def stop(self) -> None:
        self.executor.stop()

    def on_dataset(self, dataset: Dataset) -> None:
        dataset.subscribe(self.on_dataset_changed)
        self._subscribed.append(dataset)
        self.dataset_opened.emit(dataset)

    def on_dataset_changed(self, dataset: Dataset) -> None:
        """Called on the worker thread. The signal hops to the GUI thread."""
        self.dataset_changed.emit(dataset)

    def release(self, dataset: Dataset) -> None:
        dataset.unsubscribe(self.on_dataset_changed)
        if dataset in self._subscribed:
            self._subscribed.remove(dataset)
        self.library.remove(dataset)
