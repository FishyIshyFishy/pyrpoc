"""``RunnerHost``: attaches the selected program's runners, each program through
its own ``RunnerSession``, and drops them again when another is selected."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.program import Program
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.recording import Outcome
from pyrpoc.core.runner import Control
from pyrpoc.engine.acquisition.executor import Run
from pyrpoc.engine.acquisition.runs import Runs

from .earlier import earlier_inputs


class RunnerSession:
    """The context one program's runners are attached with, and the runs they started; once
    ``closed``, it starts nothing and tells its runners nothing."""

    def __init__(self, host: RunnerHost, program: type[Program]):
        self.host = host
        self.program = program
        self.controls: list[Control] = []
        self.started: list[Callable[[], None]] = []
        self.ended: list[Callable[[bool], None]] = []
        self.input_watchers: list[Callable[[str], None]] = []
        self.runs: list[Run] = []
        self.closed = False

    @property
    def running(self) -> bool:
        return any(run.running for run in self.runs)

    def execute(self, repeat: bool) -> None:
        if not self.closed:
            self.host.start(self, repeat)

    def on_run_started(self, callback: Callable[[], None]) -> None:
        self.started.append(callback)

    def wind_down(self) -> None:
        for run in self.runs:
            run.wind_down()

    def on_run_ended(self, callback: Callable[[bool], None]) -> None:
        self.ended.append(callback)

    @property
    def inputs(self) -> ProgramInputs:
        return self.host.inputs

    def on_input_changed(self, callback: Callable[[str], None]) -> None:
        self.input_watchers.append(callback)

    def input_changed(self, key: str) -> None:
        """Tell this context's runners that the program input ``key`` changed."""
        if self.closed:
            return
        for callback in list(self.input_watchers):
            callback(key)

    def add_control(self, control: Control) -> None:
        if self.closed:
            return
        self.controls.append(control)
        self.host.controls_changed.emit()

    def status(self, text: str) -> None:
        self.host.status.emit(text)

    def claim(self, run: Run) -> None:
        """Record a run this slot started, then tell its runners."""
        self.runs.append(run)
        for callback in list(self.started):
            callback()

    def release(self, run: Run) -> None:
        """Drop a run that has ended, then tell its runners how it went."""
        self.runs.remove(run)
        completed = run.recording.outcome is Outcome.COMPLETED
        for callback in list(self.ended):
            callback(completed)


class RunnerHost(QObject):
    """Attaches the selected program's runners and starts the runs they ask for."""

    # The control list changed; the view re-renders from ``controls``.
    controls_changed = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, runs: Runs, inputs: ProgramInputs, parent: QObject):
        super().__init__(parent)
        self.runs = runs
        self.inputs = inputs
        self.slot: RunnerSession | None = None

        runs.events.run_finished.connect(self.forget)
        inputs.subscribe(self.on_input_changed)

    @property
    def controls(self) -> list[Control]:
        return list(self.slot.controls) if self.slot is not None else []

    @property
    def running(self) -> bool:
        """Whether a run the selected program's runners started is going."""
        return self.slot is not None and self.slot.running

    def attach(self, program: type[Program]) -> None:
        """Replace whatever is attached with ``program``'s runners."""
        self.detach()
        if earlier_inputs(program.inputs):
            # The folder may have changed outside the app since it was last looked at.
            self.runs.refresh_earlier()
        slot = RunnerSession(self, program)
        self.slot = slot
        for runner in program.runners:
            runner.attach(slot)
        self.controls_changed.emit()

    def detach(self) -> None:
        slot, self.slot = self.slot, None
        if slot is None:
            return
        slot.closed = True
        # Nothing else can reach these runs to stop them once the slot is gone.
        for run in slot.runs:
            run.stop()
        self.controls_changed.emit()

    def start(self, slot: RunnerSession, repeat: bool) -> None:
        """Start ``slot``'s program with what the workbench holds right now."""
        self.runs.start(slot.program, repeat, slot.claim)

    def stop(self) -> None:
        """Stop every run the selected program's runners started."""
        if self.slot is not None:
            for run in self.slot.runs:
                run.stop()

    def forget(self, run: Run) -> None:
        """Drop a finished run."""
        # So closing its program outputs frees them.
        if self.slot is not None and run in self.slot.runs:
            self.slot.release(run)

    def on_input_changed(self, key: str) -> None:
        if self.slot is not None:
            self.slot.input_changed(key)
