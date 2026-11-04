"""Phase 1: the save folder is the Experiment."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyrpoc.core.program_input import ParameterError
from pyrpoc.core.saving import SaveTarget

from .helpers import FrameProgram, Harness, SpectrumProgram, saving_into


def test_frame_and_spectrum_land_in_the_experiment_folder(tmp_path: Path) -> None:
    harness = Harness()
    frame = harness.run(FrameProgram(), "test_frame", saving_into(tmp_path, "frame"))
    spectrum = harness.run(SpectrumProgram(), "test_spectrum", saving_into(tmp_path, "spectrum"))

    assert harness.failures == []
    assert sorted(path.name for path in tmp_path.glob("*_meta.json")) == [
        "frame_meta.json",
        "spectrum_meta.json",
    ]
    for run in (frame, spectrum):
        for output in run.outputs.values():
            assert output.meta_path is not None
            assert output.meta_path.parent == tmp_path


def test_no_folder_means_no_experiment() -> None:
    assert SaveTarget(name="frame", directory="", enabled=True).folder is None


def test_saving_with_no_folder_is_refused_and_says_why() -> None:
    harness = Harness()
    with pytest.raises(ParameterError, match="folder to save into"):
        harness.run(FrameProgram(), "test_frame", saving_into(None, "frame"))
    assert harness.library.all() == []


def test_saving_off_keeps_outputs_unsaved(tmp_path: Path) -> None:
    harness = Harness()
    save = SaveTarget(name="frame", directory=str(tmp_path), enabled=False)
    run = harness.run(FrameProgram(), "test_frame", save)
    assert [output.meta_path for output in run.outputs.values()] == [None]
    assert list(tmp_path.iterdir()) == []
