"""The Sites panel's status line says where the user is and what to do next."""

from __future__ import annotations

from dataclasses import replace

from loop_events import two_sites

from pyrpoc.plugins.panels.autonomy.guidance import (
    LoopStage,
    PanelState,
    duration,
    guidance,
    loop_stage,
)
from pyrpoc.plugins.program_io.outputs import LoopEvent, LoopFinished, LoopView, RoutineEnded
from pyrpoc.plugins.program_io.outputs.loop_view import LoopFold

START = PanelState(
    has_survey=True, has_map=True, drawing=False, sites=0, stage=LoopStage.NONE, minutes=0.0
)


def view_of(events: list[LoopEvent]) -> LoopView:
    fold = LoopFold()
    for event in events:
        fold.take(event)
    return fold.view()


def test_each_state_says_what_to_do_next() -> None:
    assert guidance(replace(START, has_survey=False)).startswith("Run a Survey first")
    assert "when the survey finishes" in guidance(replace(START, has_map=False))
    assert guidance(START).startswith("Click Draw site")
    assert "Click each corner" in guidance(replace(START, drawing=True))
    assert guidance(replace(START, sites=1)).startswith("1 site drawn. Next: start")
    assert guidance(replace(START, sites=3, stage=LoopStage.RUNNING)).startswith("3 sites in")
    stopped = guidance(replace(START, sites=3, stage=LoopStage.STOPPED_EARLY, minutes=12))
    assert stopped.startswith("The loop stopped early, 12 min in.")


def test_a_loop_that_ended_with_work_left_was_stopped_early() -> None:
    assert loop_stage(None, False) is LoopStage.NONE
    view = view_of(two_sites())
    assert loop_stage(view, True) is LoopStage.RUNNING
    assert loop_stage(view, False) is LoopStage.STOPPED_EARLY
    ended = [RoutineEnded(200.0, 1, "monitor"), RoutineEnded(200.0, 2, "monitor")]
    assert loop_stage(view_of([*two_sites(), *ended, LoopFinished(200.0)]), False) is (
        LoopStage.FINISHED
    )


def test_durations_read_as_words() -> None:
    assert duration(0.4) == "under a minute"
    assert duration(12.2) == "12 min"
    assert duration(150.0) == "2.5 h"
