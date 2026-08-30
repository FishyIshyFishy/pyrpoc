"""The runner: threading, cancellation, dataset setup, saving, reporting.

No QApplication and no hardware. Programs here are fakes; the real programs are
exercised in tests/programs/ with their operation monkeypatched.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import numpy as np
import pytest

from pyrpoc.core.errors import MissingDevice, ParameterError
from pyrpoc.core.params import SaveGroup, ScanGroup, group, int_field
from pyrpoc.core.streams import Cube3D, Image2D
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.run.program import Program
from pyrpoc.run.runner import Runner, default_program_key


@dataclass
class FakeParams:
    scan: ScanGroup = group(ScanGroup, "Scan")
    save: SaveGroup = group(SaveGroup, "Save")
    num_frames: int = int_field("Frames", 3, minimum=1)


class Emitter(Program):
    uses = [Galvo, DAQ]
    params = FakeParams
    emits = {"intensity": Image2D}

    def run(self, ctx):
        for index in ctx.frames(ctx.params.num_frames):
            ctx.status(f"frame {index + 1}")
            ctx.publish("intensity", np.full((2, 3, 4), index, np.float32), channels=["a", "b"])


class TwoStreams(Program):
    uses = []
    params = FakeParams
    emits = {"intensity": Image2D, "histogram": Cube3D}

    def run(self, ctx):
        for _ in ctx.frames(ctx.params.num_frames):
            ctx.publish("histogram", np.zeros((3, 4, 5), np.float32))
            ctx.publish("intensity", np.zeros((1, 3, 4), np.float32))


class Exploder(Program):
    uses = []
    params = FakeParams
    emits = {"intensity": Image2D}

    def run(self, ctx):
        ctx.publish("intensity", np.zeros((1, 2, 2), np.float32))
        raise RuntimeError("NI-DAQ acquisition failed: boom")


class TeardownRecorder(Program):
    uses = []
    params = FakeParams
    emits = {"intensity": Image2D}

    def __init__(self):
        self.torn_down = False
        self.started = threading.Event()

    def run(self, ctx):
        try:
            for _ in ctx.frames(None):
                self.started.set()
                ctx.publish("intensity", np.zeros((1, 2, 2), np.float32))
                ctx.sleep(0.01)
        finally:
            self.torn_down = True


@pytest.fixture
def inventory():
    return [DAQ(), Galvo()]


@pytest.fixture
def runner():
    return Runner(DatasetLibrary())


def run_to_completion(runner, program, params, inventory=(), **kwargs):
    events = {"status": [], "datasets": [], "finished": [], "failed": []}
    handle = runner.start(
        program,
        params,
        list(inventory),
        on_status=events["status"].append,
        on_dataset=events["datasets"].append,
        on_finished=events["finished"].append,
        on_failed=events["failed"].append,
        **kwargs,
    )
    handle.thread.join(timeout=10)
    assert not handle.thread.is_alive(), "the run thread did not finish"
    return handle, events


# --- the happy path --------------------------------------------------------


def test_a_three_frame_run_produces_a_dataset_of_three(runner, inventory):
    handle, events = run_to_completion(runner, Emitter(), FakeParams(), inventory)
    assert len(handle.datasets["intensity"]) == 3
    assert events["finished"] == [3]
    assert events["failed"] == []


def test_datasets_are_created_before_the_program_runs(runner, inventory):
    """A view can be bound before any data arrives, which is the point of emits."""
    handle, events = run_to_completion(runner, Emitter(), FakeParams(), inventory)
    assert [d.stream for d in events["datasets"]] == ["intensity"]
    assert events["datasets"][0] is handle.datasets["intensity"]


def test_datasets_land_in_the_library(runner, inventory):
    handle, _ = run_to_completion(runner, Emitter(), FakeParams(), inventory)
    assert runner.library.get(handle.run_id, "intensity") is handle.datasets["intensity"]


def test_status_reaches_the_caller(runner, inventory):
    _, events = run_to_completion(runner, Emitter(), FakeParams(), inventory)
    assert events["status"] == ["frame 1", "frame 2", "frame 3"]


def test_run_ids_increment(runner, inventory):
    first, _ = run_to_completion(runner, Emitter(), FakeParams(), inventory)
    second, _ = run_to_completion(runner, Emitter(), FakeParams(), inventory)
    assert (first.run_id, second.run_id) == (1, 2)


def test_provenance_records_the_program_parameters_and_devices(runner, inventory):
    handle, _ = run_to_completion(runner, Emitter(), FakeParams(), inventory)
    provenance = handle.datasets["intensity"].provenance
    assert provenance.program_key == "emitter"
    assert provenance.parameters["num_frames"] == 3
    assert provenance.devices["DAQ"]["device_name"] == "Dev1"
    assert provenance.devices["Galvo"]["fast_ao"] == 0
    assert provenance.started_at


def test_one_dataset_per_declared_stream(runner):
    handle, events = run_to_completion(runner, TwoStreams(), FakeParams())
    assert sorted(handle.datasets) == ["histogram", "intensity"]
    assert len(events["datasets"]) == 2


def test_is_running_is_false_once_finished(runner, inventory):
    run_to_completion(runner, Emitter(), FakeParams(), inventory)
    assert runner.is_running is False


# --- claims ----------------------------------------------------------------


def test_missing_devices_are_named_and_nothing_starts(runner):
    with pytest.raises(MissingDevice) as excinfo:
        runner.start(Emitter(), FakeParams(), [])
    assert "Galvo" in str(excinfo.value) and "NI-DAQ" in str(excinfo.value)
    assert len(runner.library) == 0


def test_a_galvo_alone_is_not_enough_because_claims_propagate(runner):
    with pytest.raises(MissingDevice, match="NI-DAQ"):
        runner.start(Emitter(), FakeParams(), [Galvo()])


def test_parameters_are_validated_before_the_thread_starts(runner, inventory):
    params = FakeParams()
    params.scan.x_pixels = 2  # below the minimum of 8
    with pytest.raises(ParameterError):
        runner.start(Emitter(), params, inventory)
    assert len(runner.library) == 0


def test_two_runs_at_once_is_refused(runner):
    program = TeardownRecorder()
    handle = runner.start(program, FakeParams(), [])
    program.started.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="already in progress"):
            runner.start(Emitter(), FakeParams(), [])
    finally:
        runner.stop()
        handle.thread.join(timeout=10)


# --- cancellation ----------------------------------------------------------


def test_stop_ends_a_continuous_run_cleanly(runner):
    program = TeardownRecorder()
    events = {"finished": [], "failed": []}
    handle = runner.start(
        program,
        FakeParams(),
        [],
        continuous=True,
        on_finished=events["finished"].append,
        on_failed=events["failed"].append,
    )
    program.started.wait(timeout=5)
    runner.stop()
    handle.thread.join(timeout=10)

    assert events["failed"] == [], "a stop is not a failure"
    assert len(events["finished"]) == 1
    assert len(handle.datasets["intensity"]) >= 1


def test_cancellation_runs_the_programs_finally_block(runner):
    """Cancelled is an exception, so teardown is the program's finally."""
    program = TeardownRecorder()
    handle = runner.start(program, FakeParams(), [], continuous=True)
    program.started.wait(timeout=5)
    runner.stop()
    handle.thread.join(timeout=10)
    assert program.torn_down is True


def test_continuous_ignores_num_frames(runner):
    program = TeardownRecorder()
    params = FakeParams()
    params.num_frames = 1
    handle = runner.start(program, params, [], continuous=True)
    program.started.wait(timeout=5)
    for _ in range(200):
        if len(handle.datasets["intensity"]) > 1:
            break
        threading.Event().wait(0.01)
    runner.stop()
    handle.thread.join(timeout=10)
    assert len(handle.datasets["intensity"]) > 1


def test_a_bounded_run_stops_on_its_own(runner, inventory):
    handle, events = run_to_completion(runner, Emitter(), FakeParams(), inventory)
    assert events["finished"] == [3]
    assert runner.is_running is False


# --- failure ---------------------------------------------------------------


def test_a_failing_program_reports_and_still_finishes(runner):
    handle, events = run_to_completion(runner, Exploder(), FakeParams())
    assert events["failed"] and "NI-DAQ acquisition failed" in events["failed"][0]
    assert events["finished"] == [1]
    assert len(handle.datasets["intensity"]) == 1


def test_a_failure_is_recorded_in_the_saved_metadata(runner, tmp_path):
    params = FakeParams()
    params.save.save_enabled = True
    params.save.save_path = str(tmp_path / "acq")
    run_to_completion(runner, Exploder(), params)

    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert "NI-DAQ acquisition failed" in meta["last_error"]
    assert meta["frames_saved"] == 1


def test_the_runner_frees_itself_after_a_failure(runner):
    run_to_completion(runner, Exploder(), FakeParams())
    assert runner.is_running is False
    run_to_completion(runner, Exploder(), FakeParams())  # must be startable again


# --- saving ----------------------------------------------------------------


def test_saving_is_wired_from_the_save_group(runner, inventory, tmp_path):
    params = FakeParams()
    params.save.save_enabled = True
    params.save.save_path = str(tmp_path / "acq")
    run_to_completion(runner, Emitter(), params, inventory)

    assert (tmp_path / "acq_a.tiff").exists()
    assert (tmp_path / "acq_b.tiff").exists()
    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert meta["frames_saved"] == 3
    assert meta["frame_limit"] == 3
    assert meta["program_key"] == "emitter"


def test_saving_off_writes_nothing(runner, inventory, tmp_path):
    params = FakeParams()
    params.save.save_path = str(tmp_path / "acq")
    run_to_completion(runner, Emitter(), params, inventory)
    assert list(tmp_path.iterdir()) == []


def test_continuous_records_no_frame_limit(runner, tmp_path):
    program = TeardownRecorder()
    params = FakeParams()
    params.save.save_enabled = True
    params.save.save_path = str(tmp_path / "acq")
    handle = runner.start(program, params, [], continuous=True)
    program.started.wait(timeout=5)
    runner.stop()
    handle.thread.join(timeout=10)

    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert meta["frame_limit"] is None


def test_program_key_can_be_supplied_by_the_caller(runner, inventory, tmp_path):
    """The shell passes the registry key; run/ may not import programs/."""
    params = FakeParams()
    params.save.save_enabled = True
    params.save.save_path = str(tmp_path / "acq")
    run_to_completion(runner, Emitter(), params, inventory, program_key="confocal")

    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert meta["program_key"] == "confocal"
    assert meta["modality_key"] == "confocal"


@pytest.mark.parametrize(
    "class_name,expected",
    [("Confocal", "confocal"), ("SplitConfocal", "split_confocal"), ("FLIM", "flim")],
)
def test_the_fallback_key_reproduces_the_v30_modality_keys(class_name, expected):
    program = type(class_name, (Program,), {})()
    assert default_program_key(program) == expected
