"""``DishClock``: the simulated dish's own time, which runs faster than wall time."""

from __future__ import annotations

import time


class DishClock:
    """Seconds of dish time since the dish was made, running ``speed`` times wall time."""

    def __init__(self, speed: float):
        self.speed = speed
        self.started = time.monotonic()

    def now(self) -> float:
        return (time.monotonic() - self.started) * self.speed

    def wall_seconds(self, dish_seconds: float) -> float:
        """How long ``dish_seconds`` of dish time takes on the wall clock."""
        return dish_seconds / self.speed
