"""The Sites panel, built off-screen with the app's own collections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loop_events import SQUARE, two_sites
from PyQt6.QtWidgets import QApplication

from pyrpoc.core.recording import EarlierRead, Recording
from pyrpoc.engine.acquisition.program_inputs import ProgramInputsStore
from pyrpoc.engine.acquisition.requests import RequestsStore
from pyrpoc.engine.data_library.store import LibraryStore
from pyrpoc.plugins.panels.autonomy.panel import AutonomyPanel
from pyrpoc.plugins.program_io.inputs import UNITS
from pyrpoc.plugins.program_io.outputs import (
    FocusPlane,
    LoopLogOutput,
    Placement,
    SurveyFinding,
    SurveyOutput,
    Unit,
)


def opened(name: str, location: Path, reads: dict[str, EarlierRead]) -> Recording:
    return Recording(
        program_key=name,
        name=name,
        started_at="2026-01-01T00:00:00+00:00",
        parts=[],
        reads=reads,
        location=location,
        note="",
        finished=False,
    )


def panel_on_survey(tmp_path: Path) -> tuple[AutonomyPanel, LibraryStore, ProgramInputsStore]:
    library, inputs = LibraryStore(1 << 30), ProgramInputsStore()
    survey = opened("survey", tmp_path / "survey_meta.json", {})
    tiles = SurveyOutput(name="tiles", recording=survey, root=None)
    found = SurveyFinding(FocusPlane(0.0, 0.0, 0.0, 0.0, 0), None, ())
    tiles.finish(found, np.zeros((1, 8, 8), dtype=np.float32), Placement(0.0, 0.0, 1.0))
    survey.outputs["tiles"] = tiles
    library.add(survey)
    return AutonomyPanel(library, inputs, RequestsStore()), library, inputs


def add_loop(library: LibraryStore, tmp_path: Path) -> LoopLogOutput:
    """A loop that read the survey, running, with two sites."""
    read = {"survey": EarlierRead("survey_meta.json", "tiles")}
    loop = opened("loop", tmp_path / "loop_meta.json", read)
    log = LoopLogOutput(name="log", recording=loop, root=None)
    log.log_all(two_sites())
    loop.outputs["log"] = log
    library.add(loop)
    return log


def test_drawn_sites_are_what_a_loop_starts_on(qt_app: QApplication, tmp_path: Path) -> None:
    panel, _library, inputs = panel_on_survey(tmp_path)
    panel.on_drawn(SQUARE)
    panel.on_drawn(SQUARE)
    assert len(inputs.get(UNITS.key)) == 2
    assert [panel.table.item(row, 0).text() for row in range(2)] == ["1", "2"]
    assert panel.status.text().startswith("2 sites drawn. Next: start Longitudinal Monitor")
    panel.table.selectRow(0)
    panel.delete_chosen()
    assert len(inputs.get(UNITS.key)) == 1


def test_a_running_loop_on_the_survey_is_followed(qt_app: QApplication, tmp_path: Path) -> None:
    panel, library, inputs = panel_on_survey(tmp_path)
    inputs.set(UNITS.key, (Unit.new(SQUARE, ""),))
    add_loop(library, tmp_path)
    panel.follow_loop()
    assert panel.running
    assert [panel.table.item(row, 2).text() for row in range(2)] == ["Waiting", "Waiting"]
    assert panel.loop_watch.heading.text() == "Loop running, under a minute so far"
    assert panel.loop_watch.activity.item(0).text().endswith("site 2 added")


def test_a_change_with_no_loop_to_take_it_says_so(qt_app: QApplication, tmp_path: Path) -> None:
    panel, library, _inputs = panel_on_survey(tmp_path)
    add_loop(library, tmp_path)
    panel.follow_loop()
    panel.table.selectRow(1)
    panel.toggle_pause()
    assert panel.status.text() == "No loop is running, so nothing was changed."


def test_a_stopped_loop_is_shown_read_only(qt_app: QApplication, tmp_path: Path) -> None:
    panel, library, _inputs = panel_on_survey(tmp_path)
    log = add_loop(library, tmp_path)
    log.recording.finished = True
    panel.follow_loop()
    assert not panel.draw_button.isEnabled()
    assert panel.loop_watch.heading.text().startswith("Loop stopped early")
    assert panel.table.item(0, 3).text() == ""
