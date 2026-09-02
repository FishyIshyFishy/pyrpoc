"""Qt in front of the runner.

``run/`` is pure Python so it can be tested with no QApplication. That leaves
one job for the shell: getting worker-thread events onto the GUI thread. This
does it by subscribing to each dataset the runner creates and re-emitting as Qt
signals -- emitting from any thread is safe, and Qt queues delivery to receivers
living in the GUI thread. Same guarantee v3.0's ``data_emitted`` pyqtSignal gave.

This is the only subscriber to a dataset's change notification. No view
subscribes directly, because ``Dataset.append`` runs on the worker thread.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.errors import MissingDevice, ParameterError
from pyrpoc.data.dataset import Dataset
from pyrpoc.data.io import SaveTarget
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.devices.base import Device
from pyrpoc.run.runner import Runner


class RunBridge(QObject):
    run_started = pyqtSignal()
    run_status = pyqtSignal(str)
    dataset_opened = pyqtSignal(object)
    dataset_changed = pyqtSignal(object, int)
    run_finished = pyqtSignal(int)
    run_failed = pyqtSignal(str)

    def __init__(self, library: DatasetLibrary | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self.library = library if library is not None else DatasetLibrary()
        self.runner = Runner(self.library)
        self._subscribed: list[Dataset] = []

    @property
    def is_running(self) -> bool:
        return self.runner.is_running

    def start(
        self,
        program: Any,
        params: Any,
        devices: list[Device],
        *,
        continuous: bool = False,
        program_key: str | None = None,
        save: SaveTarget | None = None,
    ):
        """Start a run. Raises MissingDevice or ParameterError before anything begins."""
        try:
            handle = self.runner.start(
                program,
                params,
                devices,
                continuous=continuous,
                program_key=program_key,
                save=save,
                on_status=self.run_status.emit,
                on_dataset=self.on_dataset,
                on_finished=self.run_finished.emit,
                on_failed=self.run_failed.emit,
            )
        except (MissingDevice, ParameterError) as exc:
            self.run_failed.emit(str(exc))
            raise
        self.run_started.emit()
        return handle

    def stop(self) -> None:
        self.runner.stop()

    def on_dataset(self, dataset: Dataset) -> None:
        dataset.subscribe(self.on_dataset_changed)
        self._subscribed.append(dataset)
        self.dataset_opened.emit(dataset)

    def on_dataset_changed(self, dataset: Dataset, index: int) -> None:
        """Called on the worker thread. The signal hops to the GUI thread."""
        self.dataset_changed.emit(dataset, index)

    def release(self, dataset: Dataset) -> None:
        dataset.unsubscribe(self.on_dataset_changed)
        if dataset in self._subscribed:
            self._subscribed.remove(dataset)
        self.library.remove(dataset)

    def release_all(self) -> None:
        for dataset in list(self._subscribed):
            self.release(dataset)
