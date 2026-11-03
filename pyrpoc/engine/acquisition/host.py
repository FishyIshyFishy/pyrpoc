"""``RunnerHost``: attaches the selected program's runners, each program through
its own ``RunnerSession``, and drops them again when another is selected."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.input_requests import InputRequest, InputRequests
from pyrpoc.core.program import Program
from pyrpoc.core.program_inputs import ProgramInputs
from pyrpoc.core.runner import Control, RunnerContext
from pyrpoc.engine.acquisition.executor import Run
from pyrpoc.engine.acquisition.recording import Series
from pyrpoc.engine.acquisition.runs import Runs

T = TypeVar("T")


class RunnerSession(RunnerContext):
    """The context one program's runners are attached with, and the runs they started; once
    ``closed``, it refuses every request."""

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
            self.host.runs.events.end_series(series)

    def on_run_ended(self, callback: Callable[[bool], None]) -> None:
        self.ended.append(callback)

    @property
    def inputs(self) -> ProgramInputs:
        return self.host.inputs

    @property
    def awaiting(self) -> bool:
        """Whether this context's input request is still open."""
        return self.pending is not None and self.pending.open

    def request(self, kind: type[T], prompt: str, on_answer: Callable[[T | None], None]) -> None:
        if self.closed:
            on_answer(None)
            return
        self.cancel_request()
        self.pending = self.host.requests.request(kind, prompt, on_answer)
        self.host.status.emit(prompt)

    def cancel_request(self) -> None:
        pending, self.pending = self.pending, None
        if pending is not None:
            self.host.requests.cancel(pending)

    def add_control(self, control: Control) -> None:
        if self.closed:
            return
        self.controls.append(control)
        self.host.controls_changed.emit()

    def status(self, text: str) -> None:
        self.host.status.emit(text)

    def blockers(self) -> list[str]:
        return self.host.runs.blockers(self.key)

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

    def __init__(self, runs: Runs, inputs: ProgramInputs, requests: InputRequests, parent: QObject):
        super().__init__(parent)
        self.runs = runs
        self.inputs = inputs
        self.requests = requests
        self.slot: RunnerSession | None = None

        runs.events.run_failed.connect(self.note_failure)
        runs.events.run_finished.connect(self.forget)
        runs.readiness_changed.connect(self.check_request_blockers)

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
        self.runs.start(slot.key, slot.series, slot.claim)

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
        """Drop a finished run."""
        # So closing its program outputs frees them.
        if self.slot is not None and run in self.slot.runs:
            self.slot.release(run)

    def check_request_blockers(self) -> None:
        """Withdraw an open request once something its run needs is missing."""
        # An answered request starts a run, so it cannot outlive what the run needs.
        slot = self.slot
        if slot is None or not slot.awaiting:
            return
        missing = self.runs.blockers(slot.key)
        if missing:
            slot.cancel_request()
            self.status.emit("needs " + ", ".join(missing))
