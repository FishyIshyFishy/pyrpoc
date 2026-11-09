"""How much of a plane is signal: its background level and noise, by robust statistics."""

# The background is found where the plane is darkest, not across the whole of it. A median and
# its absolute deviation over the whole plane ignore up to half of it being signal, but a dense
# culture covers more than half a field, and then they measure the cells: the noise comes out
# many times too wide and little stands clear of it. The darkest blocks give a first level and
# noise, which are refined over every pixel not clearly above that level (sigma clipping from
# above only, so that no signal is counted and no background is lost), as sky backgrounds are in
# crowded astronomical fields (Bertin and Arnouts 1996).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Scales a median absolute deviation to a Gaussian standard deviation.
MAD_TO_SIGMA = 1.4826
# The fraction of pixels taken as a plane's bright end, so one hot pixel cannot carry it.
BRIGHT_QUANTILE = 0.995
# The blocks the darkest share of which start the background, and that share.
BLOCK_PX = 16
DARK_SHARE = 0.25
# Pixels further above the level than this many noise widths are not background.
CLIP_DEVIATIONS = 3.0
CLIP_ROUNDS = 5


@dataclass(frozen=True)
class Floor:
    """A plane's background level and the spread of its noise."""

    level: float
    sigma: float

    def score(self, value: float) -> float:
        """How many noise widths ``value`` stands above the background."""
        return (value - self.level) / self.sigma


def median_floor(values: np.ndarray) -> Floor:
    """The median of ``values`` and their median absolute deviation, as a Gaussian's width."""
    level = float(np.median(values))
    spread = float(np.median(np.abs(values - level))) * MAD_TO_SIGMA
    return Floor(level, max(spread, 1e-9))


def darkest_blocks(plane: np.ndarray) -> np.ndarray:
    """The pixels of the darkest ``DARK_SHARE`` of the plane's square blocks."""
    side = min(BLOCK_PX, *plane.shape)
    rows, cols = plane.shape[0] // side, plane.shape[1] // side
    cropped = plane[: rows * side, : cols * side]
    blocks = cropped.reshape(rows, side, cols, side).swapaxes(1, 2).reshape(rows * cols, -1)
    darkest = np.argsort(np.median(blocks, axis=1))[: max(1, int(DARK_SHARE * len(blocks)))]
    return blocks[darkest].ravel()


def floor_of(plane: np.ndarray) -> Floor:
    """The background level and noise of a plane, however much of it is signal, so long as
    some of it is background."""
    floor = median_floor(darkest_blocks(plane))
    for _ in range(CLIP_ROUNDS):
        floor = median_floor(plane[plane <= floor.level + CLIP_DEVIATIONS * floor.sigma])
    return floor


def signal_score(plane: np.ndarray) -> float:
    """How far a plane's bright end stands above its noise, in noise widths."""
    return floor_of(plane).score(float(np.quantile(plane, BRIGHT_QUANTILE)))


def strongest_channel(frame: np.ndarray) -> int:
    """The channel of a ``(C, H, W)`` frame with the most signal over its noise."""
    return int(np.argmax([signal_score(plane) for plane in frame]))
