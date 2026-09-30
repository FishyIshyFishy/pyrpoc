"""The library's size limit: refusing new data, and auto-purge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QObject

from pyrpoc.acquisition.recording import Series
from pyrpoc.data_library.model import LibraryModel
from pyrpoc.data_library.store import LibraryFull, LibraryStore
from pyrpoc.plugins.programs.simulation import Simulation
from pyrpoc.structs.data import Mask2D
from pyrpoc.structs.dataset import Dataset, Origin, Provenance, utc_now

from .helpers import Recorder, small_simulation_blocks

# Smaller than one small simulation recording (3 frames of 2x8x8 float32).
TINY_LIMIT = 1000


def mask(name: str) -> Dataset:
    dataset = Dataset(
        output="mask",
        spec=Mask2D,
        provenance=Provenance(program_key="mask_editor", started_at=utc_now(), name=name),
        origin=Origin.AUTHORED,
    )
    dataset.append(np.ones((40, 40), dtype=np.uint8))
    return dataset


def test_new_recording_is_refused_over_the_limit(tmp_path: Path) -> None:
    recorder = Recorder(TINY_LIMIT)
    recorder.record(Simulation(), "simulation", small_simulation_blocks(), tmp_path, "a")
    assert recorder.library.over_limit

    with pytest.raises(LibraryFull, match="auto-purge"):
        recorder.record(Simulation(), "simulation", small_simulation_blocks(), tmp_path, "b")

    recorder.library.auto_purge = True
    recorder.record(Simulation(), "simulation", small_simulation_blocks(), tmp_path, "b")


def test_a_running_series_is_not_refused(tmp_path: Path) -> None:
    recorder, series = Recorder(TINY_LIMIT), Series()
    for _ in range(3):
        run = recorder.start(
            Simulation(), "simulation", small_simulation_blocks(), tmp_path, "s", series=series
        )
        recorder.wait(run)
    recorder.executor.end_series(series)

    (dataset,) = recorder.library.all()
    assert len(dataset) == 9


def test_purgeable_skips_live_and_drawn_entries(tmp_path: Path) -> None:
    recorder, series = Recorder(TINY_LIMIT), Series()
    # Lets starts through; nothing purges without the Qt model.
    recorder.library.auto_purge = True
    recorder.library.add(mask("drawn"))
    recorder.record(Simulation(), "simulation", small_simulation_blocks(), tmp_path, "done")
    run = recorder.start(
        Simulation(), "simulation", small_simulation_blocks(), tmp_path, "live", series=series
    )
    recorder.wait(run)

    assert [dataset.name for dataset in recorder.library.purgeable()] == ["done"]
    recorder.executor.end_series(series)


def acquired(name: str, size: int) -> Dataset:
    dataset = Dataset(
        output="spectrum",
        spec=Mask2D,
        provenance=Provenance(program_key="test", started_at=utc_now(), name=name),
        origin=Origin.ACQUIRED,
    )
    dataset.append(np.ones((size, 1), dtype=np.uint8))
    dataset.finalize(None)
    return dataset


def test_purge_closes_oldest_first_until_under_the_limit(qt_parent: QObject) -> None:
    store = LibraryStore(250)
    model = LibraryModel(store, qt_parent)
    for name in ("oldest", "middle", "newest"):
        store.add(acquired(name, 100))
    store.add(mask("drawn"))

    model.purge()
    assert [dataset.name for dataset in store.all()] == ["oldest", "middle", "newest", "drawn"]

    model.set_auto_purge(True)
    model.purge()
    # The mask alone is over the limit and cannot be purged, so everything
    # else goes, and purging stops there.
    assert [dataset.name for dataset in store.all()] == ["drawn"]
