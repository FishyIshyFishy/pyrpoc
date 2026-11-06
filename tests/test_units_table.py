"""The Sites panel's table says when each site is next visited."""

from __future__ import annotations

from loop_events import SQUARE

from pyrpoc.plugins.panels.autonomy.shown import ShownUnit
from pyrpoc.plugins.panels.autonomy.units_table import next_text
from pyrpoc.plugins.program_io.outputs import UnitState


def site(state: UnitState, due_s: float) -> ShownUnit:
    return ShownUnit("key", 1, SQUARE, "", state, due_s)


def test_next_visit_counts_down_while_the_loop_runs() -> None:
    assert next_text(site(UnitState.WAITING, 70.0), 10.0) == "in 60 s"
    assert next_text(site(UnitState.WAITING, 5.0), 10.0) == "now"


def test_no_next_visit_for_a_paused_site_or_a_stopped_loop() -> None:
    assert next_text(site(UnitState.PAUSED, 5.0), 10.0) == ""
    assert next_text(site(UnitState.WAITING, 70.0), None) == ""
