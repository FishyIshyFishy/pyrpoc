"""Phase 1: the Acquisition panel's blockers name the missing Experiment folder."""

from __future__ import annotations

from pathlib import Path

from .qt import application


def test_saving_with_no_folder_is_a_blocker(tmp_path: Path) -> None:
    app = application()
    runs = app.runs
    runs.set_save(name="frame", directory="", enabled=True)
    assert "an Experiment folder to save into" in runs.blockers("simulation")
    runs.set_save(directory=str(tmp_path))
    assert "an Experiment folder to save into" not in runs.blockers("simulation")
    runs.set_save(enabled=False, directory="")
    assert runs.blockers("simulation") == []
