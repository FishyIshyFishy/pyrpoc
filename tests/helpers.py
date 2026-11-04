"""Test programs and a way to run one to the end without Qt."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from pyrpoc.core.program import Program, RunContext
from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.acquisition.executor import Executor, Run, RunCallbacks
from pyrpoc.engine.acquisition.program_inputs import ProgramInputsStore
from pyrpoc.engine.data_library.store import LibraryStore
from pyrpoc.plugins.program_io.outputs import Image, Spectrum


class FrameProgram(Program):
    """Publishes one small image, as a confocal frame would."""

    display_name = "Test frame"
    emits = {"image": Image}

    def run(self, ctx: RunContext) -> None:
        ctx.publish("image", Image(np.ones((1, 4, 4), dtype=np.float32), ("ch0",)))


class SpectrumProgram(Program):
    """Publishes one small spectrum."""

    display_name = "Test spectrum"
    emits = {"spectrum": Spectrum}

    def run(self, ctx: RunContext) -> None:
        ctx.publish("spectrum", Spectrum(np.arange(8, dtype=np.float32)[None, :], ("ch0",)))


class Harness:
    """An executor over a fresh library, and the failures its runs reported."""

    def __init__(self) -> None:
        self.library = LibraryStore(1 << 30)
        self.inputs = ProgramInputsStore()
        self.failures: list[str] = []
        self.finished = threading.Event()
        self.executor = Executor(
            self.library,
            RunCallbacks(
                on_status=lambda _run, _text: None,
                on_failed=lambda _run, message: self.failures.append(message),
                on_finished=lambda _run: self.finished.set(),
            ),
        )

    def run(self, program: Program, key: str, save: SaveTarget, note: str = "") -> Run:
        """Start ``program`` and wait for it to end."""
        self.finished.clear()
        run = self.executor.start(
            program, self.inputs, [], program_key=key, save=save, note=note, series=None
        )
        assert self.finished.wait(10), "the run did not end"
        return run


def saving_into(folder: Path | None, name: str) -> SaveTarget:
    return SaveTarget(name=name, directory=str(folder) if folder is not None else "", enabled=True)
