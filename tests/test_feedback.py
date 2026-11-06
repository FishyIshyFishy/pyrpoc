"""A change sent to a running loop is announced, then confirmed or refused."""

from __future__ import annotations

from loop_events import SQUARE

from pyrpoc.plugins.panels.autonomy.feedback import Feedback
from pyrpoc.plugins.program_io.outputs import (
    CommandRefused,
    Unit,
    UnitJoined,
    UnitPaused,
)
from pyrpoc.plugins.program_io.requests import AddUnit, PauseUnit


def test_pausing_is_announced_then_confirmed() -> None:
    feedback = Feedback()
    assert feedback.ask(PauseUnit("key-2"), 2) == "Pausing site 2…"
    assert feedback.take([UnitPaused(5.0, 1)]) is None
    assert feedback.take([UnitPaused(6.0, 2)]) == "Site 2 paused."
    assert feedback.take([UnitPaused(7.0, 2)]) is None


def test_a_new_site_is_confirmed_with_the_number_the_loop_gave_it() -> None:
    feedback = Feedback()
    unit = Unit("key-new", SQUARE, "")
    assert feedback.ask(AddUnit(unit), None) == "Adding the new site to the loop…"
    assert feedback.take([UnitJoined(5.0, 7, "key-other", "", SQUARE)]) is None
    assert feedback.take([UnitJoined(6.0, 8, "key-new", "", SQUARE)]) == (
        "Site 8 added to the loop."
    )


def test_a_refusal_gives_the_loop_reason() -> None:
    feedback = Feedback()
    feedback.ask(PauseUnit("key-2"), 2)
    refused = CommandRefused(5.0, "pause", "key-2", "it has left")
    assert feedback.take([refused]) == "The loop could not pause site 2: it has left."


def test_nothing_is_said_with_nothing_sent() -> None:
    assert Feedback().take([UnitPaused(6.0, 2)]) is None
