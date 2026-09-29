"""The selected program's runners, as the screen sees them.

Selecting a program detaches the previous one's runners and attaches the new
one's, each through its own ``Slot``; detaching drops the slot, so its controls
and callbacks go with it, a pending pick is cancelled, and the runs it started
are stopped. A runner's request for a ``Pick`` goes out as ``pick_mode_changed``
to whatever displays are listening, and the first matching ``on_picked``
answers it. Pick state never reaches the saved workspace: a relaunch that came
back armed would point a hardware trigger at the next stray click.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.src.structs.params import BlockMap
from pyrpoc.src.structs.picks import Pick
from pyrpoc.src.structs.program import Program
from pyrpoc.src.structs.runner import Control, RunnerContext

from ..runtime.executor import Run

if TYPE_CHECKING:  # pragma: no cover
    from .application import Application

PickCallback = Callable[[Pick | None], None]


class Slot(RunnerContext):
    """The context one program's runners were attached with, and the runs they
    started. ``closed`` is set before the host lets go of it, so a runner
    holding a stale slot finds every request refused."""

    def __init__(self, host: Runners, key: str, program: type[Program]):
        self.host = host
        self.key = key
        self.program = program
        self.controls: list[Control] = []
        self.started: list[Callable[[], None]] = []
        self.runs: list[Run] = []
        self.closed = False

    @property
    def running(self) -> bool:
        return any(run.running for run in self.runs)

    def execute(self) -> None:
        if not self.closed:
            self.host.start(self)

    def on_run_started(self, callback: Callable[[], None]) -> None:
        self.started.append(callback)

    @property
    def params(self) -> BlockMap:
        return self.host.app.blocks.for_program(self.program.params)

    def params_written(self) -> None:
        if not self.closed:
            self.host.app.params_written.emit()
            self.host.app.state_changed.emit()

    def request(self, kind: type[Pick], callback: PickCallback) -> None:
        if self.closed:
            callback(None)
            return
        self.host.request_pick(kind, callback)

    def cancel_request(self) -> None:
        self.host.cancel_pick()

    def add_control(self, control: Control) -> None:
        if self.closed:
            return
        self.controls.append(control)
        self.host.controls_changed.emit()

    def status(self, text: str) -> None:
        self.host.status.emit(text)

    def blockers(self) -> list[str]:
        return self.host.app.blockers()

    def claim(self, run: Run) -> None:
        """Record a run this slot started, then tell its runners."""
        self.runs.append(run)
        for callback in list(self.started):
            callback()


class Runners(QObject):
    """Attaches the selected program's runners and routes what they ask for."""

    # The control list changed; the view re-renders from ``controls``.
    controls_changed = pyqtSignal()
    status = pyqtSignal(str)
    # A pick was requested (True) or ended (False).
    picking_changed = pyqtSignal(bool)
    # The kind of pick wanted, or None; displays offer it or stop offering.
    pick_mode_changed = pyqtSignal(object)

    def __init__(self, app: Application):
        super().__init__(app)
        self.app = app
        self.slot: Slot | None = None
        self._pick: tuple[type[Pick], PickCallback] | None = None

        app.runs.run_finished.connect(self.forget)
        app.devices_changed.connect(self.check_pick_blockers)
        app.save_changed.connect(self.check_pick_blockers)

    @property
    def controls(self) -> list[Control]:
        return list(self.slot.controls) if self.slot is not None else []

    @property
    def running(self) -> bool:
        """Whether a run the selected program's runners started is going."""
        return self.slot is not None and self.slot.running

    @property
    def picking(self) -> bool:
        return self._pick is not None

    @property
    def pick_mode(self) -> type[Pick] | None:
        """What a display added now should offer."""
        return self._pick[0] if self._pick is not None else None

    def attach(self, key: str, program: type[Program]) -> None:
        """Replace whatever is attached with ``program``'s runners."""
        self.detach()
        slot = Slot(self, key, program)
        self.slot = slot
        for runner in program.runners:
            runner.attach(slot)
        self.controls_changed.emit()

    def detach(self) -> None:
        slot, self.slot = self.slot, None
        if slot is None:
            return
        self.cancel_pick()
        slot.closed = True
        # Nothing else can reach these runs to stop them once the slot is gone.
        for run in slot.runs:
            run.stop()
        self.controls_changed.emit()

    def start(self, slot: Slot) -> None:
        """Start ``slot``'s program with what the workbench holds right now."""
        app = self.app
        app.runs.start(slot.program(), slot.key, app.blocks, app.devices, app.save, slot.claim)

    def stop(self) -> None:
        """Stop every run the selected program's runners started."""
        if self.slot is not None:
            for run in self.slot.runs:
                run.stop()

    def forget(self, run: Run) -> None:
        """Drop a finished run, so closing its datasets frees them."""
        if self.slot is not None and run in self.slot.runs:
            self.slot.runs.remove(run)

    def request_pick(self, kind: type[Pick], callback: PickCallback) -> None:
        self.cancel_pick()
        self._pick = (kind, callback)
        self.pick_mode_changed.emit(kind)
        self.status.emit("waiting for a pick on a display")
        self.picking_changed.emit(True)

    def cancel_pick(self) -> None:
        """End the pending request, if there is one. Its callback gets None."""
        pending = self.end_pick()
        if pending is not None:
            pending(None)

    def on_picked(self, pick: Pick) -> None:
        # A display may emit a kind of pick nobody asked for.
        if self._pick is None or not isinstance(pick, self._pick[0]):
            return
        callback = self.end_pick()
        if callback is not None:
            callback(pick)

    def end_pick(self) -> PickCallback | None:
        """Disarm every display before anything that can fail, so no path out
        of a pick leaves a live cursor behind."""
        if self._pick is None:
            return None
        _, callback = self._pick
        self._pick = None
        self.pick_mode_changed.emit(None)
        self.picking_changed.emit(False)
        return callback

    def check_pick_blockers(self) -> None:
        """A pending pick starts a run, so it cannot outlive what the run needs."""
        if self._pick is None:
            return
        missing = self.app.blockers()
        if missing:
            self.cancel_pick()
            self.status.emit("needs " + ", ".join(missing))
