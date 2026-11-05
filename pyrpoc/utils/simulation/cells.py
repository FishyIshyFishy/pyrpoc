"""Seeded cells on an endless plane: where each is at a given dish time, and how it looks."""

# Cells are seeded per lattice square from its own index, so any region can be asked for
# without holding the whole plane, and a cell is the same however often it is sampled.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# The lattice the cells are seeded on, and the coarser one the empty regions are.
LATTICE_UM = 40.0
REGION_UM = 300.0
# Fraction of the region field below which a square holds no cells.
EMPTY_BELOW = 0.4
CELLS_PER_SQUARE = 2.0
RADIUS_UM = (4.0, 8.0)
# Each axis's motion is a sum of this many slow sinusoids, with periods in this range.
WAVES = 3
PERIOD_MIN = (60.0, 360.0)


def seeded(*parts: int) -> np.random.Generator:
    return np.random.default_rng([part % (2**32) for part in parts])


@dataclass(frozen=True)
class Cell:
    """One cell: an identity, its home position, its shape, its brightness in each channel
    (body, then nucleus) and its motion."""

    id: int
    home_um: tuple[float, float]
    radius_um: float
    aspect: float
    angle: float
    body: float
    nucleus: float
    # Per axis, ``WAVES`` (displacement amplitude, angular frequency per minute, phase).
    waves: np.ndarray

    def position(self, minutes: float) -> tuple[float, float]:
        """Where the cell is ``minutes`` into the dish's life."""
        amplitude, omega, phase = self.waves[..., 0], self.waves[..., 1], self.waves[..., 2]
        offset = np.sum(amplitude * (np.sin(omega * minutes + phase) - np.sin(phase)), axis=1)
        return self.home_um[0] + float(offset[0]), self.home_um[1] + float(offset[1])


def region_level(x_um: float, y_um: float, seed: int) -> float:
    """A smooth field in [0, 1): bilinear value noise on the region lattice."""
    gx, gy = x_um / REGION_UM, y_um / REGION_UM
    ix, iy = int(np.floor(gx)), int(np.floor(gy))
    fx, fy = gx - ix, gy - iy

    def corner(cx: int, cy: int) -> float:
        return float(seeded(seed, cx, cy, 0xE17).random())

    top = corner(ix, iy) * (1 - fx) + corner(ix + 1, iy) * fx
    bottom = corner(ix, iy + 1) * (1 - fx) + corner(ix + 1, iy + 1) * fx
    return top * (1 - fy) + bottom * fy


def wave_table(generator: np.random.Generator, mean_speed_um_min: float) -> np.ndarray:
    """``(2, WAVES, 3)`` sinusoids whose summed velocity has mean speed ``mean_speed_um_min``."""
    # The summed velocity is close to Gaussian per axis, so its 2D speed is Rayleigh, whose
    # mean is sigma * sqrt(pi / 2); each wave carries an equal share of sigma squared.
    sigma = mean_speed_um_min / np.sqrt(np.pi / 2.0)
    velocity = sigma * np.sqrt(2.0 / WAVES)
    omega = 2.0 * np.pi / generator.uniform(*PERIOD_MIN, (2, WAVES))
    phase = generator.uniform(0.0, 2.0 * np.pi, (2, WAVES))
    return np.stack([velocity / omega, omega, phase], axis=-1)


def reach_um(mean_speed_um_min: float) -> float:
    """The furthest any cell can wander from home, plus its own size."""
    sigma = mean_speed_um_min / np.sqrt(np.pi / 2.0)
    slowest = 2.0 * np.pi / PERIOD_MIN[1]
    return 2.0 * WAVES * sigma * np.sqrt(2.0 / WAVES) / slowest + RADIUS_UM[1]


# Cells never change once seeded, and scans ask for the same squares over and over.
@lru_cache(maxsize=4096)
def square_cells(ix: int, iy: int, seed: int, mean_speed_um_min: float) -> tuple[Cell, ...]:
    """The cells seeded in lattice square (``ix``, ``iy``)."""
    centre = ((ix + 0.5) * LATTICE_UM, (iy + 0.5) * LATTICE_UM)
    if region_level(*centre, seed) < EMPTY_BELOW:
        return ()
    generator = seeded(seed, ix, iy)
    cells = []
    for index in range(generator.poisson(CELLS_PER_SQUARE)):
        cells.append(
            Cell(
                id=(ix * 73856093) ^ (iy * 19349663) ^ index,
                home_um=(
                    (ix + generator.random()) * LATTICE_UM,
                    (iy + generator.random()) * LATTICE_UM,
                ),
                radius_um=float(generator.uniform(*RADIUS_UM)),
                aspect=float(generator.uniform(0.65, 1.0)),
                angle=float(generator.uniform(0.0, np.pi)),
                body=float(generator.uniform(0.35, 1.0)),
                nucleus=float(generator.uniform(0.5, 1.0)),
                waves=wave_table(generator, mean_speed_um_min),
            )
        )
    return tuple(cells)


def cells_near(
    box: tuple[float, float, float, float], seed: int, mean_speed_um_min: float
) -> Iterator[Cell]:
    """Every cell that could be inside ``box`` (left, top, right, bottom, in µm)."""
    margin = reach_um(mean_speed_um_min)
    left, top, right, bottom = box
    for iy in range(
        int(np.floor((top - margin) / LATTICE_UM)), int((bottom + margin) // LATTICE_UM) + 1
    ):
        for ix in range(
            int(np.floor((left - margin) / LATTICE_UM)), int((right + margin) // LATTICE_UM) + 1
        ):
            yield from square_cells(ix, iy, seed, mean_speed_um_min)
