"""``StageMotion``: where the simulated stage is, travelling at a set speed in dish time."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Literal

from .clock import DishClock

Axis = Literal["xy", "z"]

XY_SPEED_UM_S = 2000.0
Z_SPEED_UM_S = 200.0


@dataclass
class Leg:
    """One axis group's move: from where, to where, starting when, how fast."""

    start: tuple[float, ...]
    target: tuple[float, ...]
    started_s: float
    speed: float

    def duration(self) -> float:
        return math.dist(self.start, self.target) / self.speed

    def position(self, now_s: float) -> tuple[float, ...]:
        duration = self.duration()
        fraction = 1.0 if duration == 0 else min(1.0, (now_s - self.started_s) / duration)
        return tuple(a + (b - a) * fraction for a, b in zip(self.start, self.target, strict=True))

    def done(self, now_s: float) -> bool:
        return now_s - self.started_s >= self.duration()


class StageMotion:
    """The stage's xy and z, each moving toward its last target at its own speed."""

    def __init__(self, clock: DishClock):
        self.clock = clock
        self.lock = threading.Lock()
        now = clock.now()
        self.xy = Leg((0.0, 0.0), (0.0, 0.0), now, XY_SPEED_UM_S)
        self.z = Leg((0.0,), (0.0,), now, Z_SPEED_UM_S)

    def position(self) -> tuple[float, float, float]:
        now = self.clock.now()
        with self.lock:
            x, y = self.xy.position(now)
            (z,) = self.z.position(now)
        return x, y, z

    def move_xy_to(self, x_um: float, y_um: float) -> None:
        now = self.clock.now()
        with self.lock:
            self.xy = Leg(self.xy.position(now), (x_um, y_um), now, XY_SPEED_UM_S)

    def move_z_to(self, z_um: float) -> None:
        now = self.clock.now()
        with self.lock:
            self.z = Leg(self.z.position(now), (z_um,), now, Z_SPEED_UM_S)

    def stop(self) -> None:
        """Stop both axes where they are."""
        now = self.clock.now()
        with self.lock:
            self.xy = Leg(self.xy.position(now), self.xy.position(now), now, XY_SPEED_UM_S)
            self.z = Leg(self.z.position(now), self.z.position(now), now, Z_SPEED_UM_S)

    def busy(self, axis: Axis) -> bool:
        now = self.clock.now()
        with self.lock:
            leg = self.xy if axis == "xy" else self.z
            return not leg.done(now)

    def remaining_s(self, axis: Axis) -> float:
        """Dish seconds until ``axis`` arrives."""
        now = self.clock.now()
        with self.lock:
            leg = self.xy if axis == "xy" else self.z
            return max(0.0, leg.duration() - (now - leg.started_s))
