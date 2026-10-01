"""Running programs headlessly through the real executor, as the app does."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from pyrpoc.acquisition.executor import Executor, Run, RunCallbacks
from pyrpoc.acquisition.recording import Series
from pyrpoc.data_library.store import LIBRARY_LIMIT_BYTES, LibraryStore
from pyrpoc.plugins.programs.components.param_groups import (
    FrameCountGroup,
    FrameGroup,
    PacingGroup,
)
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.saving import SaveTarget
from pyrpoc.structs.plugins.params import BlockStore
from pyrpoc.structs.plugins.programs.program import Program, RunContext


class SpectrumProgram(Program):
    """Three known spectra, saved through the NPZ writer."""

    display_name = "Spectrum fixture"
    emits = {"spectrum": Spectrum1D}

    def run(self, ctx: RunContext) -> None:
        ctx.describe("spectrum", units="counts")
        for index in range(3):
            spectrum = np.arange(8, dtype=np.float32)[None, :] * (index + 1)
            ctx.publish("spectrum", spectrum, channels=["ccd"])


def small_simulation_blocks() -> BlockStore:
    """Simulation parameters for a tiny, fast recording: 3 frames of 2x8x8."""
    blocks = BlockStore()
    blocks.get(FrameCountGroup).num_frames = 3
    frame = blocks.get(FrameGroup)
    frame.x_pixels = 8
    frame.y_pixels = 8
    frame.channels = 2
    blocks.get(PacingGroup).frame_interval_ms = 0
    return blocks


class Recorder:
    """An executor with a library, whose runs can be waited for."""

    def __init__(self, limit_bytes: int = LIBRARY_LIMIT_BYTES) -> None:
        self.library = LibraryStore(limit_bytes)
        self.failures: list[str] = []
        self.finished: list[Run] = []
        self._done = threading.Condition()
        self.executor = Executor(
            self.library,
            RunCallbacks(
                on_status=lambda _run, _text: None,
                on_failed=lambda _run, message: self.failures.append(message),
                on_finished=self.on_finished,
            ),
        )

    def on_finished(self, run: Run) -> None:
        with self._done:
            self.finished.append(run)
            self._done.notify_all()

    def wait(self, run: Run) -> None:
        with self._done:
            if not self._done.wait_for(lambda: run in self.finished, timeout=20):
                raise TimeoutError(f"run {run.id} did not finish")
        assert not self.failures, self.failures

    def record(
        self, program: Program, key: str, blocks: BlockStore, directory: Path, name: str
    ) -> list[Dataset]:
        """Run ``program`` to the end, saving as ``name`` into ``directory``."""
        run = self.start(program, key, blocks, directory, name, series=None)
        self.wait(run)
        return list(run.datasets.values())

    def start(
        self,
        program: Program,
        key: str,
        blocks: BlockStore,
        directory: Path,
        name: str,
        *,
        series: Series | None,
    ) -> Run:
        save = SaveTarget(name=name, directory=str(directory), enabled=True)
        return self.executor.start(program, blocks, [], program_key=key, save=save, series=series)
