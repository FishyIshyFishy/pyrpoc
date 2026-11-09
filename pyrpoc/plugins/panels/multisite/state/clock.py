"""``LoopClock``: the instrument's clock as the panel can tell it from the events it sees
arrive."""

from __future__ import annotations

# The rate is learnt only from arrivals at least this many wall seconds apart.
MIN_SPAN_S = 3.0


class LoopClock:
    """Instrument seconds from wall seconds, anchored on the newest event seen arriving and
    running at the rate events have arrived at."""

    def __init__(self) -> None:
        # (wall, instrument) seconds: the first and newest events seen arriving.
        self.first: tuple[float, float] | None = None
        self.newest: tuple[float, float] | None = None
        self.backlog = True

    def seen(self, instrument_s: float, wall_s: float) -> None:
        """Events up to ``instrument_s`` had arrived by ``wall_s``."""
        self.newest = (wall_s, instrument_s)
        # The first events seen were logged before the panel looked, perhaps long before, so
        # they say nothing of the rate.
        if self.backlog:
            self.backlog = False
        elif self.first is None:
            self.first = self.newest

    def rate(self) -> float:
        """Instrument seconds per wall second: 1 for a real instrument, more for a simulated
        dish."""
        if self.first is None or self.newest is None:
            return 1.0
        wall = self.newest[0] - self.first[0]
        if wall < MIN_SPAN_S:
            return 1.0
        # No instrument's clock runs slower than the wall's; a slower estimate is arrival lag.
        return max(1.0, (self.newest[1] - self.first[1]) / wall)

    def now(self, wall_s: float) -> float:
        if self.newest is None:
            return 0.0
        wall, instrument = self.newest
        return instrument + (wall_s - wall) * self.rate()
