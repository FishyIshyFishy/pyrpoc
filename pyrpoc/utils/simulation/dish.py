"""``Dish``: the simulated sample, with its clock, its cells, its stage and what each scan
of it sees."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .cells import Cell, cells_near
from .clock import DishClock
from .optics import TREATMENT_GAIN, FocalSurface, bleached, blur_sigma_um, blurred, detected
from .stage import StageMotion

CHANNELS = 2
# Softness of a cell's edge, as a fraction of its radius.
EDGE = 0.08
NUCLEUS_SCALE = 0.45
NUCLEUS_OFFSET = 0.15


@dataclass(frozen=True)
class DishSettings:
    """Everything that makes one dish: its seed, how fast cells move and its clock runs,
    its focal surface, how its stage drifts, and the galvo's true scale."""

    seed: int
    cell_speed_um_min: float
    clock_speed: float
    focus: FocalSurface
    stage_drift_um_min: tuple[float, float]
    um_per_volt: tuple[float, float]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def cell_planes(
    cell: Cell, centre: tuple[float, float], xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The cell's body and nucleus at the grid ``xs`` by ``ys``, in µm."""
    dx = xs[np.newaxis, :] - centre[0]
    dy = ys[:, np.newaxis] - centre[1]
    cos, sin = np.cos(cell.angle), np.sin(cell.angle)
    u = (dx * cos + dy * sin) / cell.radius_um
    v = (-dx * sin + dy * cos) / (cell.radius_um * cell.aspect)
    body = sigmoid((1.0 - np.hypot(u, v)) / EDGE)
    inner = np.hypot((u - NUCLEUS_OFFSET) / NUCLEUS_SCALE, v / NUCLEUS_SCALE)
    nucleus = sigmoid((1.0 - inner) / (EDGE * 1.5))
    return body, nucleus


def pixel_span(axis_um: np.ndarray, low: float, high: float) -> slice:
    """The indices of the evenly spaced, increasing ``axis_um`` between ``low`` and ``high``."""
    start = int(np.searchsorted(axis_um, low, side="left"))
    stop = int(np.searchsorted(axis_um, high, side="right"))
    return slice(start, stop)


class Dish:
    """One simulated dish. Its cells move, its focus drifts and its stage travels in its own
    time, and every scan of a cell adds to that cell's light dose."""

    def __init__(self, settings: DishSettings):
        self.settings = settings
        self.clock = DishClock(settings.clock_speed)
        self.stage = StageMotion(self.clock)
        self.doses: dict[int, float] = {}
        self.lock = threading.Lock()
        self.scans = 0

    def minutes(self) -> float:
        return self.clock.now() / 60.0

    def drift_um(self, minutes: float) -> tuple[float, float]:
        """How far the stage has crept from where it says it is."""
        rate_x, rate_y = self.settings.stage_drift_um_min
        return rate_x * minutes, rate_y * minutes

    def focus_z(self, x_um: float, y_um: float) -> float:
        """Where focus is at stage position (``x_um``, ``y_um``) now."""
        minutes = self.minutes()
        dx, dy = self.drift_um(minutes)
        return self.settings.focus.z_at(x_um + dx, y_um + dy, minutes)

    def cell_positions(
        self, box: tuple[float, float, float, float]
    ) -> dict[int, tuple[float, float]]:
        """Where each cell near ``box`` (dish µm) is now, by id, for checking a tracker."""
        minutes = self.minutes()
        seeded = cells_near(box, self.settings.seed, self.settings.cell_speed_um_min)
        return {cell.id: cell.position(minutes) for cell in seeded}

    def scan(
        self, fast_volts: np.ndarray, slow_volts: np.ndarray, dwell_us: float, treating: bool
    ) -> np.ndarray:
        """A ``(C, H, W)`` frame of the galvo held at each (``fast_volts``, ``slow_volts``) for
        ``dwell_us``, at the stage's current position, with the treatment light on or off."""
        minutes = self.minutes()
        stage_x, stage_y, stage_z = self.stage.position()
        dx, dy = self.drift_um(minutes)
        xs = stage_x + dx + np.asarray(fast_volts, dtype=np.float64) * self.settings.um_per_volt[0]
        ys = stage_y + dy + np.asarray(slow_volts, dtype=np.float64) * self.settings.um_per_volt[1]
        gain = TREATMENT_GAIN if treating else 1.0
        expected = self.render(xs, ys, minutes, dwell_us * gain)
        pixel_um = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 1.0
        centre_x, centre_y = float(xs.mean()), float(ys.mean())
        defocus = stage_z - self.settings.focus.z_at(centre_x, centre_y, minutes)
        sigma_px = blur_sigma_um(defocus) / pixel_um
        sharp = np.stack([blurred(plane, sigma_px) for plane in expected])
        with self.lock:
            self.scans += 1
            generator = np.random.default_rng([self.settings.seed, self.scans])
        return detected(sharp, dwell_us, generator)

    def render(self, xs: np.ndarray, ys: np.ndarray, minutes: float, dwell_us: float) -> np.ndarray:
        """The in-focus brightness over the grid, and each cell's dose from this scan."""
        planes = np.zeros((CHANNELS, len(ys), len(xs)), dtype=np.float32)
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        # Light per µm² from this scan, so a dense scan bleaches more than a sparse one.
        pixel_area = abs(float(xs[-1] - xs[0]) * float(ys[-1] - ys[0])) / max(1, len(xs) * len(ys))
        dose = dwell_us / max(pixel_area, 1e-9)
        for cell in cells_near(box, self.settings.seed, self.settings.cell_speed_um_min):
            self.add_cell(planes, cell, cell.position(minutes), xs, ys, dose)
        return planes

    def add_cell(
        self,
        planes: np.ndarray,
        cell: Cell,
        centre: tuple[float, float],
        xs: np.ndarray,
        ys: np.ndarray,
        dose: float,
    ) -> None:
        reach = cell.radius_um * 1.3
        columns = pixel_span(xs, centre[0] - reach, centre[0] + reach)
        rows = pixel_span(ys, centre[1] - reach, centre[1] + reach)
        if columns.start >= columns.stop or rows.start >= rows.stop:
            return
        with self.lock:
            left = bleached(self.doses.get(cell.id, 0.0))
            self.doses[cell.id] = self.doses.get(cell.id, 0.0) + dose
        body, nucleus = cell_planes(cell, centre, xs[columns], ys[rows])
        planes[0, rows, columns] += cell.body * left * body
        planes[1, rows, columns] += cell.nucleus * left * nucleus
