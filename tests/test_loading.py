"""Opening saved recordings into the library."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject

from pyrpoc.app.model.library import LibraryModel
from pyrpoc.app.runtime.library import LIBRARY_LIMIT_BYTES, DataLibrary
from pyrpoc.app.runtime.recording_format import META_SUFFIX
from pyrpoc.app.runtime.runs import Runs
from pyrpoc.structs.data import Origin

FIXTURE = Path(__file__).parent / "fixtures" / "recordings" / "v1" / f"simulation{META_SUFFIX}"


def model_with(store: DataLibrary, parent: QObject) -> tuple[LibraryModel, list[str]]:
    model = LibraryModel(store, Runs(store, parent), parent)
    failures: list[str] = []
    model.load_failed.connect(failures.append)
    return model, failures


def test_load_adds_each_output(qt_parent: QObject) -> None:
    model, failures = model_with(DataLibrary(LIBRARY_LIMIT_BYTES), qt_parent)
    model.load(FIXTURE)

    (dataset,) = model.all()
    assert not failures
    assert dataset.origin is Origin.LOADED
    assert dataset.name == "simulation" and len(dataset) == 3


def test_load_is_refused_when_full(qt_parent: QObject) -> None:
    store = DataLibrary(100)
    model, failures = model_with(store, qt_parent)
    model.load(FIXTURE)
    model.load(FIXTURE)

    assert len(model.all()) == 1
    assert "auto-purge" in failures[0]


def test_a_broken_recording_is_reported(qt_parent: QObject, tmp_path: Path) -> None:
    broken = tmp_path / f"broken{META_SUFFIX}"
    broken.write_text("{not json", encoding="utf-8")
    model, failures = model_with(DataLibrary(LIBRARY_LIMIT_BYTES), qt_parent)
    model.load(broken)

    assert not model.all()
    assert str(broken) in failures[0]
