"""Executing a program: the worker thread, cancellation, and dataset setup.

Pure Python, no Qt, so a runner test needs no QApplication -- which is section
12's "the test suite runs headless with no Qt application". The thread
marshalling that the GUI needs lives in ``shell/run_bridge.py``.

The runner never knows what any program does; it only knows how to execute one.
What it does own is everything section 8.1 lists as absent from the program:
creating a dataset per declared stream, attaching a save policy, and counting
frames.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable

from pyrpoc.core import params as P
from pyrpoc.core.errors import Cancelled
from pyrpoc.data.dataset import Dataset, Provenance
from pyrpoc.data.io import RunSaver, utc_now
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.devices.base import Device

from . import claims
from .program import Program, RunContext


def default_program_key(program: Program) -> str:
    """``SplitConfocal`` -> ``split_confocal``, ``FLIM`` -> ``flim``.

    run/ may not import programs/, so the runner cannot look a key up in the
    program registry. The shell passes the registry key explicitly; this is the
    fallback, and it reproduces the v3.0 modality keys that saved metadata uses.
    """
    name = type(program).__name__
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


class RunHandle:
    """What a caller gets back from ``start``: the run's identity and datasets."""

    def __init__(self, run_id: int, datasets: dict[str, Dataset], thread: threading.Thread):
        self.run_id = run_id
        self.datasets = datasets
        self.thread = thread


class Runner:
    def __init__(self, library: DatasetLibrary):
        self.library = library
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._run_id = 0
        self._lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # -- starting ----------------------------------------------------------- #

    def start(
        self,
        program: Program,
        params: Any,
        inventory: list[Device],
        *,
        continuous: bool = False,
        program_key: str | None = None,
        on_status: Callable[[str], None] | None = None,
        on_dataset: Callable[[Dataset], None] | None = None,
        on_finished: Callable[[int], None] | None = None,
        on_failed: Callable[[str], None] | None = None,
    ) -> RunHandle:
        with self._lock:
            if self.is_running:
                raise RuntimeError("a run is already in progress")

            devices = claims.resolve(list(program.uses), inventory)
            P.validate(params)
            key = program_key or default_program_key(program)

            self._run_id += 1
            run_id = self._run_id
            started_at = utc_now()
            self._cancel = threading.Event()

            saver = self.build_saver(
                program, params, devices, key, run_id=run_id, started_at=started_at,
                continuous=continuous,
            )
            provenance = Provenance(
                program_key=key,
                parameters=P.to_dict(params),
                devices=self.device_state(devices),
                started_at=started_at,
                run_id=run_id,
            )
            datasets = self.build_datasets(program, provenance, saver)

            for dataset in datasets.values():
                self.library.add(dataset)
                if on_dataset is not None:
                    on_dataset(dataset)

            ctx = RunContext(
                params=params,
                devices=devices,
                datasets=datasets,
                cancel=self._cancel,
                continuous=continuous,
                on_status=on_status,
            )

            thread = threading.Thread(
                target=self.worker,
                args=(program, ctx, datasets, saver, on_finished, on_failed),
                name=f"pyrpoc-run-{run_id}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return RunHandle(run_id, datasets, thread)

    def build_saver(
        self, program, params, devices, program_key, *, run_id, started_at, continuous
    ) -> RunSaver | None:
        save = getattr(params, "save", None)
        if save is None or not getattr(save, "save_enabled", False):
            return None
        num_frames = getattr(params, "num_frames", None)
        saver = RunSaver(
            root=P.resolved_save_root(save),
            program_key=program_key,
            parameters=P.to_dict(params),
            devices=self.device_state(devices),
            run_id=run_id,
            started_at=started_at,
            frame_limit=None if continuous else num_frames,
        )
        saver.prepare(dict(program.emits))
        return saver

    def build_datasets(self, program, provenance, saver) -> dict[str, Dataset]:
        return {
            stream: Dataset(
                stream=stream,
                spec=spec,
                provenance=provenance,
                writer=saver.writer_for(stream) if saver is not None else None,
            )
            for stream, spec in program.emits.items()
        }

    @staticmethod
    def device_state(devices: dict[type[Device], Device]) -> dict[str, Any]:
        return {
            cls.__name__: P.to_dict(device.config) if device.config is not None else {}
            for cls, device in devices.items()
        }

    # -- the worker --------------------------------------------------------- #

    def worker(self, program, ctx, datasets, saver, on_finished, on_failed) -> None:
        error: Exception | None = None
        try:
            program.run(ctx)
        except Cancelled:
            pass  # a clean stop, not a failure -- what the Stop button does
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            error = exc
            if on_failed is not None:
                on_failed(str(exc))
        finally:
            count = max((len(dataset) for dataset in datasets.values()), default=0)
            for dataset in datasets.values():
                try:
                    dataset.finalize(count, error)
                except Exception as finalize_exc:  # noqa: BLE001
                    if on_failed is not None:
                        on_failed(str(finalize_exc))
            if saver is not None:
                try:
                    saver.finalize(count, error)
                except Exception as finalize_exc:  # noqa: BLE001
                    if on_failed is not None:
                        on_failed(str(finalize_exc))
            with self._lock:
                self._thread = None
            if on_finished is not None:
                on_finished(count)

    # -- stopping ----------------------------------------------------------- #

    def stop(self) -> None:
        """Ask the running program to stop at its next cancellation point.

        A stop during a blocking scan is not observed until that frame
        completes, because the NI read is one blocking call for a whole frame.
        v3.0 behaved the same way; making it interruptible needs the incremental
        read that section 4 describes and section 11 defers.
        """
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
