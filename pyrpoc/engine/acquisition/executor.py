"""``Executor``: runs programs, one ``Run`` per start, each on its own worker thread and each
filling one recording."""

# No Qt, so runs can be driven without a QApplication. The executor owns what a program does not:
# the recording each run fills, the save policy, and which devices each run holds. Runs may
# overlap as long as they hold no device in common.

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyrpoc.core.device import Device
from pyrpoc.core.device_config import encode_config
from pyrpoc.core.program import Cancelled, Program, RunContext
from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import EarlierRead, Recording, utc_now
from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.acquisition.program_inputs import ProgramInputsStore
from pyrpoc.engine.data_library.earlier import EarlierOutput
from pyrpoc.engine.data_library.format import load_recording
from pyrpoc.engine.data_library.saving import RecordingSaver, save_location
from pyrpoc.engine.data_library.store import LibraryStore

from . import claims
from .live import LiveFeed
from .readiness import NotReady, missing
from .requests import RequestInbox, RequestsStore


class Run:
    """One execution of one program: the recording it fills and its saver, its live inputs and
    the requests posted to it, whether it repeats, and whether it is still going."""

    def __init__(
        self,
        program: Program,
        run_id: int,
        devices: dict[type[Device], Device],
        recording: Recording,
        saver: RecordingSaver | None,
        live: LiveFeed,
        inbox: RequestInbox,
        repeat: bool,
    ):
        self.program = program
        self.id = run_id
        self.devices = devices
        self.recording = recording
        self.saver = saver
        self.live = live
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


def device_state(devices: dict[type[Device], Device]) -> dict[str, Any]:
    return {cls.__name__: encode_config(device.config) for cls, device in devices.items()}


class Executor:
    def __init__(self, library: LibraryStore, requests: RequestsStore, callbacks: RunCallbacks):
        self.library = library
        self.requests = requests
        self.callbacks = callbacks
        self.leases = claims.Leases()
        self._run_id = 0
        self._lock = threading.RLock()

    def start(
        self,
        program: Program,
        inputs: ProgramInputsStore,
        inventory: list[Device],
        *,
        program_key: str,
        save: SaveTarget,
        note: str,
        earlier: Mapping[str, EarlierOutput],
        repeat: bool,
    ) -> Run:
        """Start ``program`` on a worker thread of its own, filling a new recording that
        ``note`` goes to. ``earlier`` is the output chosen for each name it reads.
        Raises ``NotReady``, ``DeviceBusy``, ``ParameterError``, ``RecordingError`` or
        ``LibraryFull`` before anything starts."""
        needs = missing(type(program), inventory, save, earlier)
        if needs:
            raise NotReady(needs)
        # Read outside the lock: loading files must not hold up another run starting.
        read = load_earlier(earlier, self.library)
        with self._lock:
            devices = claims.resolve(list(program.devices), inventory)
            self.leases.check(devices.values())
            values = self.resolve_inputs(program, inputs)
            self.library.check_room()
            self._run_id += 1
            recording, saver = self.open_recording(
                program,
                program_key=program_key,
                # One encoding, shared by the recording and the session file.
                parameters=inputs.encode(program.inputs),
                devices=device_state(devices),
                reads={
                    name: EarlierRead(each.meta_path.name, each.output)
                    for name, each in earlier.items()
                },
                save=save,
                note=note,
            )
            live = LiveFeed(inputs, program.live, saver.log_live if saver is not None else None)
            inbox = self.requests.open(
                program.requests, saver.log_request if saver is not None else None
            )
            run = Run(program, self._run_id, devices, recording, saver, live, inbox, repeat)
            self.leases.take(run.id, devices.values())
            self.launch(run, values, read)
        return run

    def open_recording(
        self,
        program: Program,
        *,
        program_key: str,
        parameters: dict[str, Any],
        devices: dict[str, Any],
        reads: dict[str, EarlierRead],
        save: SaveTarget,
        note: str,
    ) -> tuple[Recording, RecordingSaver | None]:
        """A new recording, listed in the library at once, and its saver if saving is on."""
        root: Path | None = None
        meta_path: Path | None = None
        if save.enabled:
            root, meta_path = save_location(save.root)
        recording = Recording(
            program_key=program_key,
            name=save.filename,
            started_at=utc_now(),
            parameters=parameters,
            devices=devices,
            reads=reads,
            location=meta_path,
            note=note,
            finished=False,
        )
        recording.outputs = {
            name: output_type(name=name, recording=recording, root=root)
            for name, output_type in program.emits.items()
        }
        saver = None
        if meta_path is not None:
            saver = RecordingSaver(recording, meta_path)
            saver.begin()
        self.library.add(recording)
        return recording, saver

    def resolve_inputs(self, program: Program, inputs: ProgramInputsStore) -> ProgramInputValues:
        """The values of the program's inputs, checked against their rules."""
        declared = program.inputs
        inputs.validate(declared)
        return inputs.values_for(declared)

    def launch(
        self, run: Run, values: ProgramInputValues, earlier: dict[str, ProgramOutput]
    ) -> None:
        ctx = RunContext(
            inputs=values,
            live=run.live,
            requests=run.inbox,
            earlier=earlier,
            devices=run.devices,
            outputs=run.recording.outputs,
            cancel=run.cancel,
            on_status=lambda text: self.callbacks.on_status(run, text),
        )
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


def load_earlier(
    chosen: Mapping[str, EarlierOutput], library: LibraryStore
) -> dict[str, ProgramOutput]:
    """Each chosen earlier output, the library's open copy when it holds one, else loaded."""
    # Loaded copies stay out of the library: reading one must not take a place there.
    open_at: dict[Path, Recording] = {
        recording.location: recording
        for recording in library.recordings()
        if recording.location is not None
    }
    read: dict[str, ProgramOutput] = {}
    for name, each in chosen.items():
        recording = open_at.get(each.meta_path) or load_recording(each.meta_path)
        read[name] = recording.outputs[each.output]
    return read
