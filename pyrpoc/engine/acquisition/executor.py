"""``Executor``: runs programs, one ``Run`` per start, each on its own worker thread and each
filling one recording, in parts."""

# No Qt, so runs can be driven without a QApplication. The executor owns what a program does not:
# the recording each run fills, the save policy, and which devices each run holds. Runs may
# overlap as long as they hold no device in common.

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pyrpoc.core.device import Device
from pyrpoc.core.program import Cancelled, Program, RunContext
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import EarlierRead, Recording, RecordingPart, utc_now
from pyrpoc.engine.acquisition.save_target import SaveTarget
from pyrpoc.engine.data_library.saving import RecordingSaver, save_location
from pyrpoc.engine.data_library.store import LibraryStore

from . import claims
from .inbox import RunInbox, device_state
from .preflight import NotReady, Preflight, Start


class Run:
    """One execution of one program: the recording it fills, where its files go and its saver,
    the earlier outputs it reads, its inbox, whether it repeats, and whether it is still
    going."""

    def __init__(
        self,
        program: Program,
        run_id: int,
        devices: dict[type[Device], Device],
        recording: Recording,
        root: Path | None,
        saver: RecordingSaver | None,
        earlier: dict[str, ProgramOutput],
        inbox: RunInbox,
        repeat: bool,
    ):
        self.program = program
        self.id = run_id
        self.devices = devices
        self.recording = recording
        self.root = root
        self.saver = saver
        self.earlier = earlier
        self.inbox = inbox
        self.repeat = repeat
        self.cancel = threading.Event()
        # Set to end a repeating run once the work it is doing is done.
        self.winding_down = threading.Event()
        self.running = True

    def stop(self) -> None:
        """Ask the program to stop at its next cancellation point."""
        # A blocking NI scan is one call, so a stop lands after it completes.
        self.cancel.set()

    def wind_down(self) -> None:
        """Let a repeating run finish the work it is doing, then end."""
        self.winding_down.set()


@dataclass(frozen=True)
class RunCallbacks:
    """How runs report back, each called on that run's worker thread."""

    on_status: Callable[[Run, str], None]
    on_failed: Callable[[Run, str], None]
    on_finished: Callable[[Run], None]


class Executor:
    def __init__(self, library: LibraryStore, callbacks: RunCallbacks):
        self.library = library
        self.callbacks = callbacks
        self.leases = claims.Leases()
        self._run_id = 0
        self._lock = threading.RLock()

    def preflight(self, start: Start) -> list[str]:
        """Everything ``start`` still needs; empty when it may start."""
        with self._lock:
            return Preflight(start, self.leases.held(), self.library).blockers()

    def start(self, start: Start, *, note: str, repeat: bool) -> Run:
        """Start the program on a worker thread of its own, filling a new recording that
        ``note`` goes to. Raises ``NotReady``, listing everything it needs, before anything
        opens."""
        # Judged under the lock, so no other run can take a device between judging and claiming.
        # The reads it loads are kept by ``start.load``, as the panel has judged it already.
        with self._lock:
            needs = self.preflight(start)
            if needs:
                raise NotReady(needs)
            program = start.program
            devices = claims.resolve(list(program.devices), start.inventory)
            read = {name: start.load(each) for name, each in start.chosen.items()}
            reads = {
                name: EarlierRead(each.meta_path.name, each.output)
                for name, each in start.chosen.items()
            }
            self._run_id += 1
            recording, root, saver = self.open_recording(start.program_key, reads, start.save, note)
            inbox = RunInbox(
                program, start.inputs, devices, None if saver is None else saver.log_taken
            )
            run = Run(program, self._run_id, devices, recording, root, saver, read, inbox, repeat)
            self.leases.take(run.id, devices.values())
            ctx = self.begin_part(run)
            self.launch(run, ctx)
        return run

    def open_recording(
        self, program_key: str, reads: dict[str, EarlierRead], save: SaveTarget, note: str
    ) -> tuple[Recording, Path | None, RecordingSaver | None]:
        """A new recording with no parts yet, where its files go, and its saver if saving is
        on."""
        root: Path | None = None
        meta_path: Path | None = None
        name = save.filename
        if save.enabled:
            # A recording still open in the library keeps its name even once its files have
            # left the folder, so no new recording is saved under it.
            taken = {each.location for each in self.library.recordings() if each.location}
            root, meta_path = save_location(save.root, taken)
            name = root.name
        recording = Recording(
            program_key=program_key,
            name=name,
            started_at=utc_now(),
            parts=[],
            reads=reads,
            location=meta_path,
            note=note,
            finished=False,
        )
        saver = None if meta_path is None else RecordingSaver(recording, meta_path)
        return recording, root, saver

    def begin_part(self, run: Run) -> RunContext:
        """Add a part to the run's recording, made with the settings the run has been given
        and holding a new output for each the program emits, and return the context its passes
        run in. The first part's outputs take their declared names; part n's add `` n``."""
        recording = run.recording
        number = len(recording.parts) + 1
        outputs = {
            name: output_type(
                name=name if number == 1 else f"{name} {number}",
                recording=recording,
                root=run.root,
            )
            for name, output_type in run.program.emits.items()
        }
        part = RecordingPart(
            started_at=utc_now(),
            parameters=run.inbox.parameters(),
            devices=device_state(run.devices),
            outputs=tuple(output.name for output in outputs.values()),
        )
        # Replaced whole, never changed in place: the GUI thread may be reading them.
        named = {output.name: output for output in outputs.values()}
        recording.outputs = {**recording.outputs, **named}
        recording.parts = [*recording.parts, part]
        self.announce(run, first=number == 1)
        return RunContext(
            inputs=run.inbox.given(),
            inbox=run.inbox,
            earlier=run.earlier,
            devices=run.devices,
            outputs=outputs,
            cancel=run.cancel,
            on_status=lambda text: self.callbacks.on_status(run, text),
        )

    def announce(self, run: Run, *, first: bool) -> None:
        """Put the recording's new part on disk, and the recording in the library the first
        time."""
        if run.saver is not None:
            if first:
                run.saver.begin()
            else:
                run.saver.part_begun()
        if first:
            self.library.add(run.recording)
        else:
            self.library.notify()

    def launch(self, run: Run, ctx: RunContext) -> None:
        thread = threading.Thread(
            target=self.worker,
            args=(run, ctx),
            name=f"pyrpoc-run-{run.id}",
            daemon=True,
        )
        thread.start()

    def worker(self, run: Run, ctx: RunContext) -> None:
        # The run boundary: whatever a program raises becomes a reported failure.
        error: Exception | None = None
        try:
            run.program.run(ctx)
            while run.repeat and not run.winding_down.is_set():
                # Edits made during a pass are taken here, between passes, never mid-pass.
                if run.inbox.take_settings():
                    ctx = self.begin_part(run)
                run.program.run(ctx)
        except Cancelled:
            pass  # a clean stop, not a failure: what the Stop button does
        except Exception as exc:
            error = exc
            self.callbacks.on_failed(run, str(exc))
        finally:
            self.finalize(run, error)

    def finalize(self, run: Run, error: Exception | None) -> None:
        """Let the run's devices go, then close its recording."""
        # Reported rather than raised: this runs on the worker thread after the program. The
        # devices are let go first, since nothing of the run touches them any more.
        with self._lock:
            self.leases.release(run.id)
            run.running = False
        for message in close_recording(run.recording, run.saver, error):
            self.callbacks.on_failed(run, message)
        self.callbacks.on_finished(run)


def close_recording(
    recording: Recording, saver: RecordingSaver | None, error: Exception | None
) -> list[str]:
    """Close the recording's outputs, then its saver, returning every failure's message."""
    # Collected rather than raised: a failed save is still a failure, and the other closers
    # must still run. Closing writes files, which raise whatever their library raises.
    closers: list[Callable[[], None]] = [
        lambda output=output: output.close(error) for output in recording.outputs.values()
    ]
    if saver is not None:
        closers.append(lambda: saver.finalize(error))
    errors: list[str] = []
    for close in closers:
        try:
            close()
        except Exception as exc:
            errors.append(str(exc))
    recording.finished = True
    return errors
