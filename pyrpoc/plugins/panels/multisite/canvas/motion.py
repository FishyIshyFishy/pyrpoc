"""``Travel``: a glide from one point to another that eases off and settles, like the stage."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF


@dataclass
class Travel:
    start: QPointF
    end: QPointF
    began_s: float
    takes_s: float

    def at(self, now_s: float) -> QPointF:
        t = min(1.0, (now_s - self.began_s) / self.takes_s)
        eased = t * t * (3 - 2 * t)
        return self.start + (self.end - self.start) * eased

    def done(self, now_s: float) -> bool:
        return now_s - self.began_s >= self.takes_s


def distance(a: QPointF, b: QPointF) -> float:
    return math.hypot(b.x() - a.x(), b.y() - a.y())
