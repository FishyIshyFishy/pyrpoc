"""Executing a program: the worker thread, cancellation, and dataset setup.

Pure Python, no Qt, so a run can be driven without a QApplication. The
executor never knows what a program does; it owns what a program deliberately
does not: a dataset per declared output, and the save policy.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pyrpoc.src.structs import params as P
from pyrpoc.src.structs.data import Dataset, Provenance, utc_now
from pyrpoc.src.structs.device import Device
from pyrpoc.src.structs.program import Cancelled, Program, RunContext

from . import claims
from .library import DataLibrary
from .saving import RunSaver, SaveTarget


@dataclass(frozen=True)
class RunCallbacks:
    """How a run reports back. Called on the worker thread."""

    on_status: Callable[[str], None]
    on_dataset: Callable[[Dataset], None]
    on_finished: Callable[[], None]
    on_failed: Callable[[str], None]


def device_state(devices: dict[type[Device], Device]) -> dict[str, Any]:
    return {cls.__name__: P.encode_block(device.config) for cls, device in devices.items()}


class Executor:
    def __init__(self, library: DataLibrary):
        self.library = library
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._run_id = 0
        self._lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        program: Program,
        blocks: P.BlockStore,
        inventory: list[Device],
        *,
        continuous: bool,
        program_key: str,
        save: SaveTarget,
        callbacks: RunCallbacks,
    ) -> None:
        """Execute one program on a worker thread. Raises ``MissingDevice`` or
        ``ParameterError`` before anything starts."""
        with self._lock:
            if self.is_running:
                raise RuntimeError("a run is already in progress")

            devices = claims.resolve(list(program.uses), inventory)
            params = self.resolve_params(program, blocks)
            self._run_id += 1
            self._cancel = threading.Event()
            provenance = Provenance(
                program_key=program_key,
                started_at=utc_now(),
                name=save.filename,
                # One encoding, shared by the run metadata and the session file.
                parameters=blocks.to_dict(list(program.params)),
                devices=device_state(devices),
                run_id=self._run_id,
            )
            saver = self.build_saver(program, provenance, save)
            datasets = self.open_datasets(program, provenance, saver, callbacks)
            ctx = RunContext(
                params=params,
                devices=devices,
                datasets=datasets,
                cancel=self._cancel,
                continuous=continuous,
                on_status=callbacks.on_status,
            )
            self.launch(program, ctx, saver, callbacks)

    def resolve_params(self, program: Program, blocks: P.BlockStore) -> P.BlockMap:
        """The program's declared blocks, validated and resolved against the
        open data, so a mask that is no longer open stops the run here."""
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
        self,
        program: Program,
        provenance: Provenance,
        saver: RunSaver | None,
        callbacks: RunCallbacks,
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
            callbacks.on_dataset(dataset)
        return datasets

    def launch(
        self, program: Program, ctx: RunContext, saver: RunSaver | None, callbacks: RunCallbacks
    ) -> None:
        thread = threading.Thread(
            target=self.worker,
            args=(program, ctx, saver, callbacks),
            name=f"pyrpoc-run-{self._run_id}",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def worker(
        self, program: Program, ctx: RunContext, saver: RunSaver | None, callbacks: RunCallbacks
    ) -> None:
        # The run boundary: whatever a program raises becomes a reported failure.
        error: Exception | None = None
        try:
            program.run(ctx)
        except Cancelled:
            pass  # a clean stop, not a failure: what the Stop button does
        except Exception as exc:
            error = exc
            callbacks.on_failed(str(exc))
        finally:
            self.finalize(ctx, saver, error, callbacks)

    def finalize(
        self,
        ctx: RunContext,
        saver: RunSaver | None,
        error: Exception | None,
        callbacks: RunCallbacks,
    ) -> None:
        """Close every writer, reporting rather than raising: this runs on the
        worker thread after the program, and a failed save is still a failure."""
        closers: list[Callable[[], None]] = [
            lambda d=dataset: d.finalize(error) for dataset in ctx.datasets.values()
        ]
        if saver is not None:
            closers.append(lambda: saver.finalize(error))
        for close in closers:
            try:
                close()
            except Exception as exc:
                callbacks.on_failed(str(exc))
        with self._lock:
            self._thread = None
        callbacks.on_finished()

    def stop(self) -> None:
        """Ask the running program to stop at its next cancellation point. A
        blocking NI scan is one call, so a stop lands after it completes."""
        self._cancel.set()
