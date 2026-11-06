"""``RunEvents``: starts runs and reports them on the GUI thread."""

# Runs report on their worker threads; these signals carry that onto the GUI thread. They are the
# only subscriber to each program output a run opens, because a program adds to its outputs on
# the worker thread.

from __future__ import annotations

from collections.abc import Callable, Mapping

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.device import Device
from pyrpoc.core.program import Program
from pyrpoc.core.program_input import ParameterError
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.engine.acquisition.claims import DeviceBusy
from pyrpoc.engine.acquisition.executor import Executor, Run, RunCallbacks
from pyrpoc.engine.acquisition.program_inputs import ProgramInputsStore
from pyrpoc.engine.acquisition.readiness import NotReady
from pyrpoc.engine.acquisition.requests import RequestsStore
from pyrpoc.engine.acquisition.save_target import SaveTarget
from pyrpoc.engine.data_library.earlier import EarlierOutput
from pyrpoc.engine.data_library.format import RecordingError
from pyrpoc.engine.data_library.store import LibraryFull, LibraryStore


class RunEvents(QObject):
    """Starts runs and reports them. Each signal carries its ``Run``."""

    run_started = pyqtSignal(object)
    run_status = pyqtSignal(object, str)
    run_failed = pyqtSignal(object, str)
    run_finished = pyqtSignal(object)
    # Refused before any run existed: something it needs is missing, a device is busy, a
    # parameter is bad, an earlier recording it reads is unreadable, or the library is full.
    start_refused = pyqtSignal(str)

    def __init__(self, library: LibraryStore, requests: RequestsStore, parent: QObject):
        super().__init__(parent)
        self.executor = Executor(
            library,
            requests,
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
        inputs: ProgramInputsStore,
        devices: list[Device],
        save: SaveTarget,
        note: str,
        earlier: Mapping[str, EarlierOutput],
        load: Callable[[EarlierOutput], ProgramOutput],
        repeat: bool,
        claim: Callable[[Run], None],
    ) -> Run | None:
        """Start ``program`` and return its run, or report through ``start_refused`` why it
        cannot start."""
        # Reported rather than raised, since every caller is a Qt slot. ``claim`` gets the run
        # before ``run_started`` announces it, so whoever started it has recorded it by then.
        try:
            run = self.executor.start(
                program,
                inputs,
                devices,
                program_key=key,
                save=save,
                note=note,
                earlier=earlier,
                load=load,
                repeat=repeat,
            )
        except (NotReady, DeviceBusy, ParameterError, RecordingError, LibraryFull) as exc:
            self.start_refused.emit(str(exc))
            return None
        claim(run)
        self.run_started.emit(run)
        return run
