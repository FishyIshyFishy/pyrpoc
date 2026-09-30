"""Saved recordings load, now and in every later version.

``fixtures/recordings/v<N>/`` hold recordings written by the version that
introduced format N. They are committed once and never regenerated: if a
change makes one of them fail to load, the change is what is wrong.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pyrpoc.data_library.format import META_SUFFIX, RecordingError, load_recording, write_notes
from pyrpoc.plugins.programs.simulation import Simulation
from pyrpoc.structs.data_library.data import Data, Image2D, Spectrum1D
from pyrpoc.structs.data_library.dataset import Origin
from pyrpoc.structs.plugins.params import BlockStore

from .helpers import Recorder, SpectrumProgram, small_simulation_blocks

FIXTURES = Path(__file__).parent / "fixtures" / "recordings"


@dataclass(frozen=True)
class Expected:
    output: str
    kind: type[Data]
    program_key: str
    channels: list[str]
    shape: tuple[int, ...]
    frames: int
    total: float


# What each committed fixture holds. Add a row per fixture; never edit one.
EXPECTED = {
    "v1/simulation": Expected(
        "intensity", Image2D, "simulation", ["sim0", "sim1"], (2, 8, 8), 3, 31.683820062004088
    ),
    "v1/spectrum": Expected("spectrum", Spectrum1D, "spectrum_fixture", ["ccd"], (1, 8), 3, 168.0),
}


def fixture_meta_paths() -> list[Path]:
    return sorted(FIXTURES.glob(f"v*/*{META_SUFFIX}"))


def fixture_key(meta_path: Path) -> str:
    return f"{meta_path.parent.name}/{meta_path.name.removesuffix(META_SUFFIX)}"


def test_every_fixture_has_expectations() -> None:
    assert {fixture_key(path) for path in fixture_meta_paths()} == set(EXPECTED)


@pytest.mark.parametrize("meta_path", fixture_meta_paths(), ids=fixture_key)
def test_fixture_loads(meta_path: Path) -> None:
    expected = EXPECTED[fixture_key(meta_path)]
    (dataset,) = load_recording(meta_path)
    frames = dataset.frames()

    assert dataset.output == expected.output
    assert dataset.spec is expected.kind
    assert dataset.provenance.program_key == expected.program_key
    assert dataset.channel_labels == expected.channels
    assert dataset.origin is Origin.LOADED and dataset.finished
    assert dataset.meta_path == meta_path
    assert len(frames) == expected.frames
    assert all(frame.shape == expected.shape for frame in frames)
    assert float(np.sum(np.stack(frames), dtype=np.float64)) == pytest.approx(expected.total)


def test_image_round_trip(tmp_path: Path) -> None:
    (saved,) = Recorder().record(
        Simulation(), "simulation", small_simulation_blocks(), tmp_path, "sample"
    )
    (loaded,) = load_recording(tmp_path / f"sample{META_SUFFIX}")

    assert loaded.channel_labels == saved.channel_labels
    assert loaded.provenance.parameters == saved.provenance.parameters
    for original, reread in zip(saved.frames(), loaded.frames(), strict=True):
        np.testing.assert_array_equal(original, reread)


def test_npz_round_trip_keeps_metadata(tmp_path: Path) -> None:
    (saved,) = Recorder().record(SpectrumProgram(), "spectrum", BlockStore(), tmp_path, "spec")
    (loaded,) = load_recording(tmp_path / f"spec{META_SUFFIX}")

    assert loaded.metadata == {"units": "counts"}
    for original, reread in zip(saved.frames(), loaded.frames(), strict=True):
        np.testing.assert_array_equal(original, reread)


def test_same_name_gets_numbered_stems(tmp_path: Path) -> None:
    recorder = Recorder()
    for _ in range(3):
        recorder.record(SpectrumProgram(), "spectrum", BlockStore(), tmp_path, "spec")

    names = sorted(path.name for path in tmp_path.glob(f"*{META_SUFFIX}"))
    assert names == [f"spec_2{META_SUFFIX}", f"spec_3{META_SUFFIX}", f"spec{META_SUFFIX}"]
    assert (tmp_path / "spec_2_spectrum.npz").is_file()


def copy_fixture(name: str, destination: Path) -> Path:
    for path in (FIXTURES / "v1").glob(f"{name}_*"):
        shutil.copy(path, destination / path.name)
    return destination / f"{name}{META_SUFFIX}"


def test_write_notes_keeps_unknown_fields(tmp_path: Path) -> None:
    meta_path = copy_fixture("spectrum", tmp_path)
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["added_by_a_later_version"] = {"kept": True}
    meta_path.write_text(json.dumps(raw), encoding="utf-8")

    write_notes(meta_path, "spectrum", "focus drifted after frame 2")

    (loaded,) = load_recording(meta_path)
    assert loaded.notes == "focus drifted after frame 2"
    assert json.loads(meta_path.read_text(encoding="utf-8"))["added_by_a_later_version"]


def test_unknown_format_version_is_refused(tmp_path: Path) -> None:
    meta_path = copy_fixture("spectrum", tmp_path)
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["format_version"] = 999
    meta_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RecordingError, match="format 999"):
        load_recording(meta_path)


def test_missing_data_file_is_named(tmp_path: Path) -> None:
    meta_path = copy_fixture("simulation", tmp_path)
    (tmp_path / "simulation_sim1.tiff").unlink()

    with pytest.raises(RecordingError, match="simulation_sim1.tiff"):
        load_recording(meta_path)
