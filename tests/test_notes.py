"""Notes on library entries reach the recording's metadata file."""

from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6.QtCore import QObject

from pyrpoc.app.runtime.recording import Series
from pyrpoc.data_library.format import META_SUFFIX, load_recording
from pyrpoc.data_library.model import LibraryModel
from pyrpoc.plugins.programs.simulation import Simulation

from .helpers import Recorder, small_simulation_blocks

FIXTURES = Path(__file__).parent / "fixtures" / "recordings" / "v1"


def model_for(recorder: Recorder, parent: QObject) -> LibraryModel:
    return LibraryModel(recorder.library, parent)


def test_notes_on_a_loaded_recording_are_written(qt_parent: QObject, tmp_path: Path) -> None:
    for path in FIXTURES.glob("spectrum_*"):
        shutil.copy(path, tmp_path / path.name)
    recorder = Recorder()
    model = model_for(recorder, qt_parent)
    model.load(tmp_path / f"spectrum{META_SUFFIX}")
    (dataset,) = model.all()

    assert model.set_notes(dataset, "bleached") is None
    (reloaded,) = load_recording(tmp_path / f"spectrum{META_SUFFIX}")
    assert reloaded.notes == "bleached"


def test_notes_typed_during_a_series_survive_its_end(qt_parent: QObject, tmp_path: Path) -> None:
    recorder, series = Recorder(), Series()
    model = model_for(recorder, qt_parent)
    run = recorder.start(
        Simulation(), "simulation", small_simulation_blocks(), tmp_path, "live", series=series
    )
    recorder.wait(run)
    (dataset,) = model.all()

    assert model.set_notes(dataset, "stage drifted") is None
    recorder.executor.end_series(series)

    (reloaded,) = load_recording(tmp_path / f"live{META_SUFFIX}")
    assert reloaded.notes == "stage drifted"


def test_a_failed_write_is_reported_and_the_notes_kept(qt_parent: QObject, tmp_path: Path) -> None:
    recorder = Recorder()
    model = model_for(recorder, qt_parent)
    (dataset,) = recorder.record(
        Simulation(), "simulation", small_simulation_blocks(), tmp_path, "gone"
    )
    (tmp_path / f"gone{META_SUFFIX}").unlink()

    problem = model.set_notes(dataset, "kept anyway")
    assert problem is not None and "gone" in problem
    assert dataset.notes == "kept anyway"
