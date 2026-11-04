"""``Executor``: runs programs, one ``Run`` per start, each on its own worker thread."""

# No Qt, so runs can be driven without a QApplication. The executor owns what a program does not:
# the recording each run writes into, the save policy, and which devices each run holds. Runs may
# overlap as long as they hold no device in common.

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from pyrpoc.core.device import Device
from pyrpoc.core.device_config import encode_config
from pyrpoc.core.program import Cancelled, Program, RunContext
from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.core.program_output import Origin, ProgramOutput, Provenance, utc_now
from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.acquisition.program_inputs import ProgramInputsStore
from pyrpoc.engine.acquisition.recording import Recording, RecordingKey, Series
from pyrpoc.engine.data_library.saving import RecordingSaver
from pyrpoc.engine.data_library.store import LibraryStore

from . import claims


class Run:
    """One execution of one program: what it was given, where it writes, and whether it is
    still going."""

    def __init__(
        self,
        program: Program,
        provenance: Provenance,
        devices: dict[type[Device], Device],
        recording: Recording,
        series: Series | None,
    ):
        self.program = program
        self.provenance = provenance
        self.devices = devices
        self.recording = recording
        self.series = series
        self.cancel = threading.Event()
        self.running = True

    @property
    def id(self) -> int:
        return self.provenance.run_id

    @property
    def outputs(self) -> dict[str, ProgramOutput[Any]]:
        return self.recording.outputs

    def stop(self) -> None:
        """Ask the program to stop at its next cancellation point."""
        # A blocking NI scan is one call, so a stop lands after it completes.
        self.cancel.set()


@dataclass(frozen=True)
class RunCallbacks:
    """How runs report back, each called on that run's worker thread."""

    on_status: Callable[[Run, str], None]
    on_failed: Callable[[Run, str], None]
    on_finished: Callable[[Run], None]


def device_state(devices: dict[type[Device], Device]) -> dict[str, Any]:
    return {cls.__name__: encode_config(device.config) for cls, device in devices.items()}


class Executor:
    def __init__(self, library: LibraryStore, callbacks: RunCallbacks):
        self.library = library
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
        series: Series | None,
    ) -> Run:
        """Start ``program`` on a worker thread of its own, continuing the
        recording of ``series`` if nothing it depends on changed. Raises
        ``MissingDevice``, ``DeviceBusy``, ``ParameterError`` or, for a new
        recording, ``LibraryFull`` before anything starts."""
        with self._lock:
            devices = claims.resolve(list(program.devices), inventory)
            self.leases.check(devices.values())
            values = self.resolve_inputs(program, inputs)
            self._run_id += 1
            key = RecordingKey(
                program_key=program_key,
                # One encoding, shared by the recording and the session file.
                parameters=inputs.encode(program.inputs()),
                devices=device_state(devices),
                # A copy: the app edits its save target in place.
                save=replace(save),
            )
            provenance = Provenance(
                program_key=program_key,
                started_at=utc_now(),
                name=save.filename,
                parameters=key.parameters,
                devices=key.devices,
                run_id=self._run_id,
            )
            recording, stale = self.recording_for(program, key, provenance, series)
            run = Run(program, provenance, devices, recording, series)
            recording.active += 1
            recording.last_run = run
            if recording.saver is not None:
                recording.saver.add_run(run.id, provenance.started_at)
            self.leases.take(run.id, devices.values())
            self.launch(run, values)
        if stale is not None:
            self.close(stale)
        return run

    def recording_for(
        self, program: Program, key: RecordingKey, provenance: Provenance, series: Series | None
    ) -> tuple[Recording, Recording | None]:
        """Where the run writes, and the series' old recording if this replaces one that is
        ready to close."""
        # Called under the lock.
        current = series.recording if series is not None else None
        if current is not None and current.key == key:
            return current, None
        self.library.check_room()
        recording = self.open_recording(program, key, provenance)
        if series is None:
            return recording, None
        series.recording = recording
        stale = current if current is not None and self.claim_close(current) else None
        return recording, stale

    def open_recording(
        self, program: Program, key: RecordingKey, provenance: Provenance
    ) -> Recording:
        saver = self.build_saver(program, provenance, key.save)
        outputs = self.open_outputs(program, provenance, saver)
        # Before the library sees them, so a saved output is listed in its Experiment at once.
        if saver is not None:
            saver.begin(outputs)
        for program_output in outputs.values():
            self.library.add(program_output)
        return Recording(key, outputs, saver)

    def end_series(self, series: Series) -> None:
        """Let no more runs join ``series``, closing its recording now if idle or else when
        its last run ends."""
        with self._lock:
            series.closed = True
            recording = series.recording
            closing = recording is not None and self.claim_close(recording)
        if closing and recording is not None:
            self.close(recording)

    def claim_close(self, recording: Recording) -> bool:
        """Whether the caller is the one to close ``recording``: it is idle and nobody has
        claimed it yet."""
        # Called under the lock.
        if recording.closed or recording.active > 0:
            return False
        recording.closed = True
        return True

    def close(self, recording: Recording) -> None:
        """Finalize ``recording``."""
        # Outside the lock: closing writes files, and must not hold up a run starting.
        for message in recording.finalize():
            if recording.last_run is not None:
                self.callbacks.on_failed(recording.last_run, message)

    def resolve_inputs(self, program: Program, inputs: ProgramInputsStore) -> ProgramInputValues:
        """The values of the program's inputs, checked against their rules."""
        declared = program.inputs()
        inputs.validate(declared)
        return inputs.values_for(declared)

    @staticmethod
    def build_saver(
        program: Program, provenance: Provenance, save: SaveTarget
    ) -> RecordingSaver | None:
        """The saver for this run, or None when saving is off."""
        if not save.enabled:
            return None
        saver = RecordingSaver(
            root=save.root,
            program_key=provenance.program_key,
            name=provenance.name,
            parameters=provenance.parameters,
            devices=provenance.devices,
            started_at=provenance.started_at,
        )
        saver.prepare(dict(program.emits))
        return saver

    def open_outputs(
        self, program: Program, provenance: Provenance, saver: RecordingSaver | None
    ) -> dict[str, ProgramOutput[Any]]:
        return {
            output: ProgramOutput(
                name=output,
                kind=kind,
                provenance=provenance,
                origin=Origin.ACQUIRED,
                writer=saver.writer_for(output) if saver is not None else None,
            )
            for output, kind in program.emits.items()
        }

    def launch(self, run: Run, values: ProgramInputValues) -> None:
        ctx = RunContext(
            inputs=values,
            devices=run.devices,
            outputs=run.outputs,
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
        except Cancelled:
            pass  # a clean stop, not a failure: what the Stop button does
        except Exception as exc:
            error = exc
            self.callbacks.on_failed(run, str(exc))
        finally:
            self.finalize(run, error)

    def finalize(self, run: Run, error: Exception | None) -> None:
        """Record the run's end, and close its recording if no run will write into it
        again."""
        # Reported rather than raised: this runs on the worker thread after the program. The
        # devices are let go before the recording closes, since nothing of the run touches them
        # any more.
        recording = run.recording
        if recording.saver is not None:
            try:
                recording.saver.end_run(run.id, error)
            except OSError as exc:
                self.callbacks.on_failed(run, str(exc))
        with self._lock:
            self.leases.release(run.id)
            run.running = False
            recording.active -= 1
            if error is not None:
                recording.error = error
            series = run.series
            done = series is None or series.closed or series.recording is not recording
            closing = done and self.claim_close(recording)
        if closing:
            self.close(recording)
        self.callbacks.on_finished(run)
