"""Runs in a series add to one recording until something it depends on changes."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from pyrpoc.acquisition.recording import Series
from pyrpoc.data_library.format import META_SUFFIX, load_recording
from pyrpoc.plugins.programs.building_blocks.parameter_groups import FrameCountGroup
from pyrpoc.plugins.programs.simulation.program import Simulation
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.plugins.params import BlockStore
from pyrpoc.structs.plugins.programs.program import Program, RunContext

from .helpers import Recorder, small_simulation_blocks


def run_in_series(
    recorder: Recorder, blocks: BlockStore, directory: Path, series: Series, count: int
) -> None:
    for _ in range(count):
        run = recorder.start(Simulation(), "simulation", blocks, directory, "loop", series=series)
        recorder.wait(run)


def test_series_adds_to_one_recording(tmp_path: Path) -> None:
    recorder, series = Recorder(), Series()
    run_in_series(recorder, small_simulation_blocks(), tmp_path, series, count=3)

    (dataset,) = recorder.library.all()
    assert len(dataset) == 9
    assert not dataset.finished

    recorder.executor.end_series(series)
    assert dataset.finished
    meta_path = tmp_path / f"loop{META_SUFFIX}"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(raw["runs"]) == 3
    assert raw["outputs"]["intensity"]["frames"] == 9
    assert raw["ended_at"] is not None
    (loaded,) = load_recording(meta_path)
    assert len(loaded) == 9


def test_changed_parameters_start_a_new_recording(tmp_path: Path) -> None:
    recorder, series = Recorder(), Series()
    blocks = small_simulation_blocks()
    run_in_series(recorder, blocks, tmp_path, series, count=2)
    blocks.get(FrameCountGroup).num_frames = 1
    run_in_series(recorder, blocks, tmp_path, series, count=2)
    recorder.executor.end_series(series)

    first, second = recorder.library.all()
    assert (len(first), len(second)) == (6, 2)
    assert first.finished and second.finished
    assert first.meta_path == tmp_path / f"loop{META_SUFFIX}"
    assert second.meta_path == tmp_path / f"loop_2{META_SUFFIX}"


def test_runs_without_a_series_each_get_a_recording(tmp_path: Path) -> None:
    recorder = Recorder()
    for _ in range(2):
        recorder.record(Simulation(), "simulation", small_simulation_blocks(), tmp_path, "one")

    assert [len(dataset) for dataset in recorder.library.all()] == [3, 3]
    assert all(dataset.finished for dataset in recorder.library.all())


class GatedProgram(Program):
    """Publishes once, then waits for the test to let it finish."""

    display_name = "Gated"
    emits = {"spectrum": Spectrum1D}

    def __init__(self, gate: threading.Event):
        self.gate = gate

    def run(self, ctx: RunContext) -> None:
        ctx.publish("spectrum", np.array([[1.0, 2.0]], dtype=np.float32))
        self.gate.wait(timeout=20)


def test_closing_a_series_mid_run_closes_when_the_run_ends(tmp_path: Path) -> None:
    recorder, series, gate = Recorder(), Series(), threading.Event()
    run = recorder.start(GatedProgram(gate), "gated", BlockStore(), tmp_path, "g", series=series)

    recorder.executor.end_series(series)
    (dataset,) = recorder.library.all()
    assert not dataset.finished

    gate.set()
    recorder.wait(run)
    assert dataset.finished
    assert json.loads((tmp_path / f"g{META_SUFFIX}").read_text(encoding="utf-8"))["ended_at"]
