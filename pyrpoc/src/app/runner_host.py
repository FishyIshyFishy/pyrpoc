"""Hosting the selected program's runners.

A program declares its entry points in ``Program.runners``; this is where they
are attached, and it is all the app knows about them. Nothing here names Start,
Continuous, a point or a scan: a runner asks for controls, runs, and picks
through a ``RunnerContext``, and the host carries those requests to the
executor, the view and the displays.

**Lifecycle.** Selecting a program detaches the previous one's runners and
attaches the new one's. Each attachment gets its own ``Slot`` -- the context
its runners were given -- and detaching drops the slot, so its controls and run
callbacks go with it and any pending pick is cancelled. An armed display never
outlives a program switch.

**Requests.** A runner asks for a kind of ``Pick`` and never learns who answers.
This host answers with displays, which opt in by duck typing: a panel with
``set_pick_mode`` and a ``picked`` signal is told what kind of pick is wanted,
or ``None``, and emits a ``Pick`` when the user makes one. One request at a
time; the host checks the pick is the kind asked for before handing it over.

Pick state is interaction, not configuration, and deliberately never reaches
the session file: a relaunch that came back armed would be pointing a hardware
trigger at the next stray click.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.src.structs.params import BlockMap
from pyrpoc.src.structs.picks import Pick
from pyrpoc.src.structs.program import Program
from pyrpoc.src.structs.runner import Control, RunnerContext

if TYPE_CHECKING:  # pragma: no cover
    from .application import Application

PickCallback = Callable[[Pick | None], None]


class Slot(RunnerContext):
    """The context one program's runners were attached with.

    Everything a runner registers lives here, so detaching is dropping this
    object. A runner holding on to a stale slot finds every request refused:
    ``closed`` is set before the host lets go of it.
    """

    def __init__(self, host: RunnerHost, program: type[Program]):
        self.host = host
        self.program = program
        self.controls: list[Control] = []
        self.started: list[Callable[[], None]] = []
        self.finished: list[Callable[[], None]] = []
        self.closed = False

    # -- running ----------------------------------------------------------- #

    def execute(self, continuous: bool = False) -> None:
        if not self.closed:
            self.host.execute(continuous=continuous)

    def stop(self) -> None:
        self.host.app.stop_run()

    @property
    def running(self) -> bool:
        return self.host.app.bridge.is_running

    def on_run_started(self, callback: Callable[[], None]) -> None:
        self.started.append(callback)

    def on_run_finished(self, callback: Callable[[], None]) -> None:
        self.finished.append(callback)

    # -- parameters -------------------------------------------------------- #

    @property
    def params(self) -> BlockMap:
        return self.host.app.blocks.for_program(self.program.params)

    def params_written(self) -> None:
        if not self.closed:
            self.host.app.params_written.emit()
            self.host.app.state_changed.emit()

    # -- requests ---------------------------------------------------------- #

    def request(self, kind: type[Pick], callback: PickCallback) -> None:
        """Answered by a display: that is this host's choice, not the runner's."""
        if self.closed:
            callback(None)
            return
        self.host.request_pick(kind, callback)

    def cancel_request(self) -> None:
        self.host.cancel_pick()

    # -- view -------------------------------------------------------------- #

    def add_control(self, control: Control) -> None:
        if self.closed:
            return
        self.controls.append(control)
        self.host.controls_changed.emit()

    def status(self, text: str) -> None:
        self.host.status.emit(text)

    def blockers(self) -> list[str]:
        return self.host.app.blockers()


class RunnerHost(QObject):
    """Attaches the selected program's runners and routes what they ask for."""

    # The control list changed: a program was attached or detached, or a
    # runner added a control. The view re-renders from ``controls``.
    controls_changed = pyqtSignal()
    # A runner (or the host on its behalf) has something to say.
    status = pyqtSignal(str)
    # A pick was requested (True) or ended (False), for anything that wants
    # to say "click a point" while one is pending.
    picking_changed = pyqtSignal(bool)

    def __init__(self, app: Application):
        super().__init__(app)
        self.app = app
        self.slot: Slot | None = None
        self.displays: list[Any] = []
        self._pick: tuple[type[Pick], PickCallback] | None = None

        app.bridge.run_started.connect(self.on_run_started)
        app.bridge.run_finished.connect(self.on_run_finished)
        app.devices_changed.connect(self.check_pick_blockers)
        app.save_changed.connect(self.check_pick_blockers)

    @property
    def controls(self) -> list[Control]:
        return list(self.slot.controls) if self.slot is not None else []

    @property
    def picking(self) -> bool:
        return self._pick is not None

    # -- lifecycle ---------------------------------------------------------- #

    def attach(self, program: type[Program]) -> None:
        """Replace whatever is attached with ``program``'s runners."""
        self.detach()
        slot = Slot(self, program)
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
        self.controls_changed.emit()

    # -- running ------------------------------------------------------------ #

    def execute(self, *, continuous: bool = False) -> None:
        """Start the selected program, reporting rather than raising.

        Every caller is a Qt slot -- a button press, or a display's mouse
        handler by way of a pick -- and an exception escaping one aborts the
        process instead of unwinding. A launch refused for a missing device or
        a bad parameter has already been announced through ``run_failed``.
        """
        try:
            self.app.start_run(continuous=continuous)
        except Exception as exc:  # noqa: BLE001 - reported, not raised: Qt slot
            self.status.emit(f"error - {exc}")

    def on_run_started(self) -> None:
        if self.slot is not None:
            for callback in list(self.slot.started):
                callback()

    def on_run_finished(self) -> None:
        if self.slot is not None:
            for callback in list(self.slot.finished):
                callback()

    # -- displays and picks ------------------------------------------------- #

    def attach_display(self, panel: Any) -> None:
        """Route picks to and from ``panel``, if it takes part in picking.

        A panel added while a pick is pending arrives already picking.
        """
        if not (hasattr(panel, "set_pick_mode") and hasattr(panel, "picked")):
            return
        if panel in self.displays:
            return
        self.displays.append(panel)
        panel.picked.connect(self.on_picked)
        self.tell_display(panel, self._pick[0] if self._pick is not None else None)

    def detach_display(self, panel: Any) -> None:
        if panel not in self.displays:
            return
        self.displays.remove(panel)
        # TypeError/RuntimeError: already gone with its widget.
        with contextlib.suppress(TypeError, RuntimeError):
            panel.picked.disconnect(self.on_picked)

    def request_pick(self, kind: type[Pick], callback: PickCallback) -> None:
        self.cancel_pick()
        self._pick = (kind, callback)
        self.set_pick_mode(kind)
        self.status.emit("waiting for a pick on a display")
        self.picking_changed.emit(True)

    def cancel_pick(self) -> None:
        """End the pending request, if there is one. Its callback gets None."""
        pending = self.end_pick()
        if pending is not None:
            pending(None)

    def on_picked(self, pick: Any) -> None:
        if self._pick is None or not isinstance(pick, self._pick[0]):
            return
        callback = self.end_pick()
        if callback is not None:
            callback(pick)

    def end_pick(self) -> PickCallback | None:
        """Disarm every display first, before anything that can fail, so no
        path out of a pick leaves a live cursor behind."""
        if self._pick is None:
            return None
        _, callback = self._pick
        self._pick = None
        self.set_pick_mode(None)
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

    def set_pick_mode(self, kind: type[Pick] | None) -> None:
        for panel in list(self.displays):
            self.tell_display(panel, kind)

    @staticmethod
    def tell_display(panel: Any, kind: type[Pick] | None) -> None:
        try:
            panel.set_pick_mode(kind)
        except Exception as exc:  # noqa: BLE001 - one bad panel must not stick
            panel.last_error = str(exc)
