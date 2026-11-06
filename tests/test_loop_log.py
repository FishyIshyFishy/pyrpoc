"""A loop log's summary is kept as it logs, and equals replaying every event."""

from __future__ import annotations

from loop_events import scanned, two_sites

from pyrpoc.core.recording import Recording
from pyrpoc.plugins.program_io.outputs import LoopLogOutput
from pyrpoc.plugins.program_io.outputs.loop_view import LoopFold


def recording() -> Recording:
    return Recording(
        program_key="loop",
        name="loop",
        started_at="2026-01-01T00:00:00+00:00",
        parts=[],
        reads={},
        location=None,
        note="",
        finished=False,
    )


def test_view_equals_a_full_replay() -> None:
    log = LoopLogOutput(name="log", recording=recording(), root=None)
    events = [*two_sites(), scanned(1, 200.0, False)]
    log.log_all(events[:4])
    for event in events[4:]:
        log.log(event)
    replay = LoopFold()
    for event in events:
        replay.take(event)
    assert log.view() == replay.view()


def test_events_after_gives_only_the_new_ones() -> None:
    log = LoopLogOutput(name="log", recording=recording(), root=None)
    log.log_all(two_sites())
    assert log.events_after(len(two_sites())) == []
    log.log(scanned(1, 200.0, False))
    assert log.events_after(len(two_sites())) == [scanned(1, 200.0, False)]
