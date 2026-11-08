"""``FakeDish``: a made-up dish of drifting, flickering cells in two channels, imaged whole for the
survey and one field at a time for the loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.outputs import Outline

WIDTH_UM, HEIGHT_UM = 3200.0, 2200.0
OVERVIEW_UM_PER_PX = 2.0
FIELD_UM = 160.0
FIELD_PX = 320
CELLS = 340
# How fast cells wander, in µm a second: fast enough to see the loop follow them.
DRIFT_UM_S = 0.6
CHANNELS = ("SRS", "TPEF")


@dataclass(frozen=True)
class Cells:
    """Each cell's start position, radius and brightness per channel, and its drift."""

    xy: np.ndarray
    radius: np.ndarray
    bright: np.ndarray
    drift: np.ndarray
    flicker: np.ndarray


def made_cells(rng: np.random.Generator) -> Cells:
    xy = rng.uniform((60, 60), (WIDTH_UM - 60, HEIGHT_UM - 60), size=(CELLS, 2))
    angle = rng.uniform(0, 2 * np.pi, CELLS)
    speed = rng.uniform(0.2, 1.0, CELLS) * DRIFT_UM_S
    drift = np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]
    return Cells(
        xy,
        rng.uniform(6, 15, CELLS),
        rng.uniform(0.4, 1.0, (CELLS, len(CHANNELS))),
        drift,
        rng.uniform(0.02, 0.08, CELLS),
    )


class FakeDish:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.cells = made_cells(self.rng)

    def at(self, cell: int, t_s: float) -> tuple[float, float]:
        """Where ``cell`` is ``t_s`` seconds in."""
        x, y = self.cells.xy[cell] + self.cells.drift[cell] * t_s
        return float(x), float(y)

    def nearest(self, x_um: float, y_um: float) -> int:
        offsets = self.cells.xy - np.asarray((x_um, y_um))
        return int(np.argmin(np.hypot(offsets[:, 0], offsets[:, 1])))

    def outline_of(self, cell: int) -> Outline:
        """A ring round ``cell`` where it started."""
        x, y = self.at(cell, 0.0)
        r = float(self.cells.radius[cell]) * 2.2
        angles = np.linspace(0, 2 * np.pi, 18, endpoint=False)
        return tuple((x + r * float(np.cos(a)), y + r * float(np.sin(a))) for a in angles)

    def overview(self) -> np.ndarray:
        """The whole dish at the start, ``(C, H, W)``."""
        return self.image(0.0, 0.0, WIDTH_UM, HEIGHT_UM, OVERVIEW_UM_PER_PX, 0.0)

    def field(self, x_um: float, y_um: float, t_s: float) -> np.ndarray:
        """One field centred on (``x_um``, ``y_um``), ``t_s`` seconds in."""
        um_per_px = FIELD_UM / FIELD_PX
        left, top = x_um - FIELD_UM / 2, y_um - FIELD_UM / 2
        return self.image(left, top, FIELD_UM, FIELD_UM, um_per_px, t_s)

    def image(
        self, left: float, top: float, width: float, height: float, um_per_px: float, t_s: float
    ) -> np.ndarray:
        cols, rows = int(round(width / um_per_px)), int(round(height / um_per_px))
        planes = self.rng.normal(0.06, 0.02, (len(CHANNELS), rows, cols)).astype(np.float32)
        cells = self.cells
        centres = cells.xy + cells.drift * t_s
        glow = 1.0 + 0.25 * np.sin(t_s * cells.flicker * 2 * np.pi)
        for index in range(CELLS):
            self.add_cell(
                planes,
                centres[index],
                cells.radius[index],
                cells.bright[index] * glow[index],
                left,
                top,
                um_per_px,
            )
        return planes

    @staticmethod
    def add_cell(
        planes: np.ndarray,
        centre: np.ndarray,
        radius: float,
        bright: np.ndarray,
        left: float,
        top: float,
        um_per_px: float,
    ) -> None:
        """Add one cell's glow to the pixels it reaches."""
        reach = 3 * radius
        _, rows, cols = planes.shape
        c0 = max(0, int((centre[0] - reach - left) / um_per_px))
        c1 = min(cols, int((centre[0] + reach - left) / um_per_px) + 1)
        r0 = max(0, int((centre[1] - reach - top) / um_per_px))
        r1 = min(rows, int((centre[1] + reach - top) / um_per_px) + 1)
        if c0 >= c1 or r0 >= r1:
            return
        xs = left + (np.arange(c0, c1) + 0.5) * um_per_px - centre[0]
        ys = top + (np.arange(r0, r1) + 0.5) * um_per_px - centre[1]
        blob = np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) / (2 * (radius / 1.6) ** 2))
        for channel, level in enumerate(bright):
            planes[channel, r0:r1, c0:c1] += float(level) * blob
