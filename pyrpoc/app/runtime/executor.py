"""Executing programs: one ``Run`` per start, each on its own worker thread.

Pure Python, no Qt, so runs can be driven without a QApplication. The
executor never knows what a program does; it owns what a program deliberately
does not: a dataset per declared output, the save policy, and which devices
each run holds. Runs may overlap as long as they hold no device in common.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pyrpoc.structs import params as P
from pyrpoc.structs.data import Dataset, Provenance, SaveTarget, utc_now
from pyrpoc.structs.device import Device
from pyrpoc.structs.program import Cancelled, Program, RunContext

from . import claims
from .library import DataLibrary
from .saving import RunSaver


class Run:
    """One execution of one program, and the handle for reaching it once it
    has started: what it was given, what it holds, and whether it is going."""

    def __init__(
        self,
        program: Program,
        provenance: Provenance,
        devices: dict[type[Device], Device],
        datasets: dict[str, Dataset],
    ):
        self.program = program
        self.provenance = provenance
        self.devices = devices
        self.datasets = datasets
        self.cancel = threading.Event()
        self.running = True

    @property
    def id(self) -> int:
        return self.provenance.run_id

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
    ) -> Run:
        """Start ``program`` on a worker thread of its own. Raises
        ``MissingDevice``, ``DeviceBusy`` or ``ParameterError`` before anything
        starts."""
        with self._lock:
            devices = claims.resolve(list(program.uses), inventory)
            self.leases.check(devices.values())
            params = self.resolve_params(program, blocks)
            self._run_id += 1
            provenance = Provenance(
                program_key=program_key,
                started_at=utc_now(),
                name=save.filename,
                # One encoding, shared by the run metadata and the workspace file.
                parameters=blocks.to_dict(list(program.params)),
                devices=device_state(devices),
                run_id=self._run_id,
            )
            saver = self.build_saver(program, provenance, save)
            datasets = self.open_datasets(program, provenance, saver)
            run = Run(program, provenance, devices, datasets)
            self.leases.take(run.id, devices.values())
            self.launch(run, params, saver)
            return run

    def resolve_params(self, program: Program, blocks: P.BlockStore) -> P.BlockMap:
        """The program's declared blocks, validated and resolved against the
        open data, so a mask that is no longer open stops the run here. Each
        is a copy, so edits made while the run goes do not reach it."""
        declared = list(program.params)
        blocks.validate(declared)
        resolved = blocks.for_program(declared).items()
        return P.BlockMap({cls: P.resolve_block(block, self.library) for cls, block in resolved})

    @staticmethod
    def build_saver(program: Program, provenance: Provenance, save: SaveTarget) -> RunSaver | None:
        """The saver for this run, or None when saving is off."""
        if not save.enabled:
            return None
        saver = RunSaver(
            root=save.root,
            program_key=provenance.program_key,
            parameters=provenance.parameters,
            devices=provenance.devices,
            run_id=provenance.run_id,
            started_at=provenance.started_at,
        )
        saver.prepare(dict(program.emits))
        return saver

    def open_datasets(
        self, program: Program, provenance: Provenance, saver: RunSaver | None
    ) -> dict[str, Dataset]:
        datasets = {
            output: Dataset(
                output=output,
                spec=spec,
                provenance=provenance,
                writer=saver.writer_for(output) if saver is not None else None,
            )
            for output, spec in program.emits.items()
        }
        for dataset in datasets.values():
            self.library.add(dataset)
            self.callbacks.on_dataset(dataset)
        return datasets

    def launch(self, run: Run, params: P.BlockMap, saver: RunSaver | None) -> None:
        ctx = RunContext(
            params=params,
            devices=run.devices,
            datasets=run.datasets,
            cancel=run.cancel,
            on_status=lambda text: self.callbacks.on_status(run, text),
        )
        thread = threading.Thread(
            target=self.worker,
            args=(run, ctx, saver),
            name=f"pyrpoc-run-{run.id}",
            daemon=True,
        )
        thread.start()

    def worker(self, run: Run, ctx: RunContext, saver: RunSaver | None) -> None:
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
            self.finalize(run, saver, error)

    def finalize(self, run: Run, saver: RunSaver | None, error: Exception | None) -> None:
        """Close every writer, reporting rather than raising: this runs on the
        worker thread after the program, and a failed save is still a failure.
        The devices are let go last, once nothing of the run touches them."""
        closers: list[Callable[[], None]] = [
            lambda d=dataset: d.finalize(error) for dataset in run.datasets.values()
        ]
        if saver is not None:
            closers.append(lambda: saver.finalize(error))
        for close in closers:
            try:
                close()
            except Exception as exc:
                self.callbacks.on_failed(run, str(exc))
        with self._lock:
            self.leases.release(run.id)
            run.running = False
        self.callbacks.on_finished(run)
