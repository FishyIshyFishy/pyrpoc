"""The Sites panel's Activity list keeps only what a person needs."""

from __future__ import annotations

from loop_events import scanned, two_sites

from pyrpoc.plugins.panels.autonomy.activity import Activity
from pyrpoc.plugins.program_io.outputs import CommandRefused, UnitPaused


def texts(activity: Activity) -> list[str]:
    """The lines without their times, oldest first."""
    return [line.split("   ", 1)[1] for line in reversed(activity.lines)]


def test_routine_steps_and_measurements_are_left_out() -> None:
    activity = Activity()
    activity.take(two_sites())
    assert texts(activity) == ["loop started", "site 1 added", "site 2 added"]
    assert activity.taken == len(two_sites())


def test_tracking_is_reported_only_when_it_changes() -> None:
    activity = Activity()
    activity.take(two_sites())
    activity.take([scanned(1, 200.0, False), scanned(1, 260.0, False), scanned(1, 320.0, True)])
    assert texts(activity)[-2:] == ["site 1 lost track", "site 1 found again"]


def test_times_are_minutes_since_the_loop_started() -> None:
    activity = Activity()
    activity.take([*two_sites(), UnitPaused(400.0, 2)])
    assert activity.lines[0] == "  5.0 min   site 2 paused"


def test_a_refusal_says_why() -> None:
    activity = Activity()
    activity.take([*two_sites(), CommandRefused(130.0, "pause", "key-9", "no such unit")])
    assert texts(activity)[-1] == "could not pause a site: no such unit"
