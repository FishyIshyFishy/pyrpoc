"""Executing programs: one ``Run`` per start, each on its own worker thread.

Pure Python, no Qt, so runs can be driven without a QApplication. The
executor never knows what a program does; it owns what a program deliberately
does not: the recording each run writes into (a dataset per declared output,
and its files; see ``recording.py``), the save policy, and which devices each
run holds. Runs may overlap as long as they hold no device in common.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from pyrpoc.structs import params as P
from pyrpoc.structs.data import Dataset, Origin, Provenance, SaveTarget, utc_now
from pyrpoc.structs.device import Device
from pyrpoc.structs.program import Cancelled, Program, RunContext

from . import claims
from .library import DataLibrary
from .recording import Recording, RecordingKey, Series
from .saving import RecordingSaver


class Run:
    """One execution of one program, and the handle for reaching it once it
    has started: what it was given, where it writes, and whether it is going."""

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
    def datasets(self) -> dict[str, Dataset]:
        return self.recording.datasets

    def stop(self) -> None:
        """Ask the program to stop at its next cancellation point. A blocking
        NI scan is one call, so a stop lands after it completes."""
        self.cancel.set()


@dataclass(frozen=True)
class RunCallbacks:
    """How runs report back. ``on_dataset`` is called while a run starts, on
    the starting thread; the rest on that run's worker thread."""

    on_dataset: Callable[[Dataset], None]
    on_status: Callable[[Run, str], None]
    on_failed: Callable[[Run, str], None]
    on_finished: Callable[[Run], None]


def device_state(devices: dict[type[Device], Device]) -> dict[str, Any]:
    return {cls.__name__: P.encode_block(device.config) for cls, device in devices.items()}


class Executor:
    def __init__(self, library: DataLibrary, callbacks: RunCallbacks):
        self.library = library
        self.callbacks = callbacks
        self.leases = claims.Leases()
        self._run_id = 0
        self._lock = threading.RLock()

    def start(
        self,
        program: Program,
        blocks: P.BlockStore,
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
            devices = claims.resolve(list(program.uses), inventory)
            self.leases.check(devices.values())
            params = self.resolve_params(program, blocks)
            self._run_id += 1
            key = RecordingKey(
                program_key=program_key,
                # One encoding, shared by the recording and the workspace file.
                parameters=blocks.to_dict(list(program.params)),
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
            self.launch(run, params)
        if stale is not None:
            self.close(stale)
        return run

    def recording_for(
        self, program: Program, key: RecordingKey, provenance: Provenance, series: Series | None
    ) -> tuple[Recording, Recording | None]:
        """Where the run writes, and the series' old recording if this replaces
        one that is ready to close. Called under the lock."""
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
        datasets = self.open_datasets(program, provenance, saver)
        if saver is not None:
            saver.begin(datasets)
        return Recording(key, datasets, saver)

    def end_series(self, series: Series) -> None:
        """No more runs join ``series``. Its recording closes now if idle,
        otherwise when its last run ends."""
        with self._lock:
            series.closed = True
            recording = series.recording
            closing = recording is not None and self.claim_close(recording)
        if closing and recording is not None:
            self.close(recording)

    def claim_close(self, recording: Recording) -> bool:
        """Whether the caller is the one to close ``recording``: it is idle and
        nobody has claimed it yet. Called under the lock."""
        if recording.closed or recording.active > 0:
            return False
        recording.closed = True
        return True

    def close(self, recording: Recording) -> None:
        """Finalize outside the lock: closing writes files, and must not hold
        up a run starting."""
        for message in recording.finalize():
            if recording.last_run is not None:
                self.callbacks.on_failed(recording.last_run, message)

    def resolve_params(self, program: Program, blocks: P.BlockStore) -> P.BlockMap:
        """The program's declared blocks, validated and resolved against the
        open data, so a mask that is no longer open stops the run here. Each
        is a copy, so edits made while the run goes do not reach it."""
        declared = list(program.params)
        blocks.validate(declared)
        resolved = blocks.for_program(declared).items()
        return P.BlockMap({cls: P.resolve_block(block, self.library) for cls, block in resolved})

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

    def open_datasets(
        self, program: Program, provenance: Provenance, saver: RecordingSaver | None
    ) -> dict[str, Dataset]:
        datasets = {
            output: Dataset(
                output=output,
                spec=spec,
                provenance=provenance,
                origin=Origin.ACQUIRED,
                writer=saver.writer_for(output) if saver is not None else None,
            )
            for output, spec in program.emits.items()
        }
        for dataset in datasets.values():
            self.library.add(dataset)
            self.callbacks.on_dataset(dataset)
        return datasets

    def launch(self, run: Run, params: P.BlockMap) -> None:
        ctx = RunContext(
            params=params,
            devices=run.devices,
            datasets=run.datasets,
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
        """Record the run's end, and close its recording if no run will write
        into it again. Reported rather than raised: this runs on the worker
        thread after the program. The devices are let go before the recording
        closes, since nothing of the run touches them any more."""
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
