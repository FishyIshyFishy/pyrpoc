"""``RunnerHost``: attaches the selected program's runners, each program through
its own ``RunnerSession``, and drops them again when another is selected."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.input_requests import InputRequest
from pyrpoc.core.program import Program
from pyrpoc.core.program_inputs import BlockMap
from pyrpoc.core.runner import Control, RunnerContext
from pyrpoc.engine.acquisition.executor import Run
from pyrpoc.engine.acquisition.recording import Series

if TYPE_CHECKING:  # pragma: no cover
    from .model import Acquisition

T = TypeVar("T")


class RunnerSession(RunnerContext):
    """The context one program's runners were attached with, and the runs they
    started. Once ``closed``, every request through it is refused."""

    def __init__(self, host: RunnerHost, key: str, program: type[Program]):
        self.host = host
        self.key = key
        self.program = program
        self.controls: list[Control] = []
        self.started: list[Callable[[], None]] = []
        self.ended: list[Callable[[bool], None]] = []
        self.runs: list[Run] = []
        self.failed: set[Run] = set()
        self.series: Series | None = None
        self.pending: InputRequest[Any] | None = None
        self.closed = False

    @property
    def running(self) -> bool:
        return any(run.running for run in self.runs)

    def execute(self) -> None:
        if not self.closed:
            self.host.start(self)

    def on_run_started(self, callback: Callable[[], None]) -> None:
        self.started.append(callback)

    def open_series(self) -> None:
        if not self.closed:
            self.series = Series()

    def close_series(self) -> None:
        series, self.series = self.series, None
        if series is not None:
            self.host.acquisition.events.end_series(series)

    def on_run_ended(self, callback: Callable[[bool], None]) -> None:
        self.ended.append(callback)

    @property
    def params(self) -> BlockMap:
        return self.host.acquisition.blocks.for_program(self.program.params)

    def params_written(self) -> None:
        if not self.closed:
            self.host.acquisition.params_written.emit()
            self.host.acquisition.edited.emit()

    @property
    def awaiting(self) -> bool:
        """Whether this context's input request is still open."""
        return self.pending is not None and self.pending.open

    def request(self, kind: type[T], prompt: str, on_answer: Callable[[T | None], None]) -> None:
        if self.closed:
            on_answer(None)
            return
        self.cancel_request()
        self.pending = self.host.acquisition.requests.request(kind, prompt, on_answer)
        self.host.status.emit(prompt)

    def cancel_request(self) -> None:
        pending, self.pending = self.pending, None
        if pending is not None:
            self.host.acquisition.requests.cancel(pending)

    def add_control(self, control: Control) -> None:
        if self.closed:
            return
        self.controls.append(control)
        self.host.controls_changed.emit()

    def status(self, text: str) -> None:
        self.host.status.emit(text)

    def blockers(self) -> list[str]:
        return self.host.acquisition.blockers()

    def claim(self, run: Run) -> None:
        """Record a run this slot started, then tell its runners."""
        self.runs.append(run)
        for callback in list(self.started):
            callback()

    def release(self, run: Run) -> None:
        """Drop a run that has ended, then tell its runners how it went."""
        self.runs.remove(run)
        completed = not run.cancel.is_set() and run not in self.failed
        self.failed.discard(run)
        for callback in list(self.ended):
            callback(completed)


class RunnerHost(QObject):
    """Attaches the selected program's runners and starts the runs they ask for."""

    # The control list changed; the view re-renders from ``controls``.
    controls_changed = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, acquisition: Acquisition):
        super().__init__(acquisition)
        self.acquisition = acquisition
        self.slot: RunnerSession | None = None

        acquisition.events.run_failed.connect(self.note_failure)
        acquisition.events.run_finished.connect(self.forget)
        acquisition.inventory.changed.connect(self.check_request_blockers)
        acquisition.save_changed.connect(self.check_request_blockers)

    @property
    def controls(self) -> list[Control]:
        return list(self.slot.controls) if self.slot is not None else []

    @property
    def running(self) -> bool:
        """Whether a run the selected program's runners started is going."""
        return self.slot is not None and self.slot.running

    @property
    def awaiting(self) -> bool:
        """Whether the selected program's runners are waiting on an input request."""
        return self.slot is not None and self.slot.awaiting

    def attach(self, key: str, program: type[Program]) -> None:
        """Replace whatever is attached with ``program``'s runners."""
        self.detach()
        slot = RunnerSession(self, key, program)
        self.slot = slot
        for runner in program.runners:
            runner.attach(slot)
        self.controls_changed.emit()

    def detach(self) -> None:
        slot, self.slot = self.slot, None
        if slot is None:
            return
        slot.cancel_request()
        slot.closed = True
        # Nothing else can reach these runs to stop them once the slot is gone.
        for run in slot.runs:
            run.stop()
        slot.close_series()
        self.controls_changed.emit()

    def start(self, slot: RunnerSession) -> None:
        """Start ``slot``'s program with what the workbench holds right now."""
        acquisition = self.acquisition
        acquisition.events.start(
            slot.program(),
            slot.key,
            acquisition.blocks,
            acquisition.inventory.devices,
            acquisition.save,
            slot.series,
            slot.claim,
        )

    def stop(self) -> None:
        """Stop every run the selected program's runners started."""
        if self.slot is not None:
            for run in self.slot.runs:
                run.stop()

    def note_failure(self, run: Run, message: str) -> None:
        del message
        if self.slot is not None and run in self.slot.runs:
            self.slot.failed.add(run)

    def forget(self, run: Run) -> None:
        """Drop a finished run, so closing its program outputs frees them."""
        if self.slot is not None and run in self.slot.runs:
            self.slot.release(run)

    def check_request_blockers(self) -> None:
        """An open request starts a run once answered, so it cannot outlive what
        the run needs."""
        slot = self.slot
        if slot is None or not slot.awaiting:
            return
        missing = self.acquisition.blockers()
        if missing:
            slot.cancel_request()
            self.status.emit("needs " + ", ".join(missing))
