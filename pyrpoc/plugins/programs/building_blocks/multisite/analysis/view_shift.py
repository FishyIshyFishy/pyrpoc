"""How far the whole view moved between two frames of the same place: the normalised
cross-correlation of the frames over their overlap at every offset, peaked to a fraction of a
pixel near where the move was predicted to put it."""

# When the stage carries the sample, everything in view moves together, so every pixel with
# structure is evidence of the one shift. Correlating whole frames uses all of it, where following
# one feature uses a corner of it and needs that corner to be a bright, isolated thing; a dense
# field (a sheet of beads, a confluent dish) has no such feature but registers best of all.
#
# The frames are band-passed first. The slow part of a frame (uneven illumination, vignetting
# toward the scan's edges) is fixed to the scanner rather than the sample, so it does not move with
# the stage and would pull the peak toward no shift at all; the fastest part is shot noise, which
# matches nothing. What is left is the sample's own structure.

from __future__ import annotations

import cv2
import numpy as np

from .stitching import best_near, overlap_scores

# Structure broader than this share of the field is taken to be the scanner's, not the sample's.
SHADING_FRACTION = 1.0 / 16.0
NOISE_SIGMA_PX = 1.0
# Below this normalised correlation the two frames are not taken to show the same place.
MATCH_FLOOR = 0.4
# A frame that matches itself this well somewhere other than in place could be mistaken for
# itself shifted there.
AMBIGUOUS_SCORE = 0.5


def structure(plane: np.ndarray) -> np.ndarray:
    """The plane band-passed to the sample's own structure: shot noise and the scanner's
    shading taken out."""
    plane = plane.astype(np.float32)
    shading_sigma = SHADING_FRACTION * max(plane.shape)
    smooth = cv2.GaussianBlur(plane, (0, 0), NOISE_SIGMA_PX)
    return smooth - cv2.GaussianBlur(plane, (0, 0), shading_sigma)


def ambiguity_px(plane: np.ndarray, within: float) -> float:
    """How near to no shift, out to ``within`` pixels, ``structure`` plane matches itself at
    ``AMBIGUOUS_SCORE`` again; ``inf`` if nowhere."""
    # A regular sample (a lattice of beads) matches itself one spacing over nearly as well as in
    # place, so a shift searched for any further than half that from its prediction could be read
    # a spacing off. The plane's own correlation says how far that is.
    scores = np.fft.fftshift(overlap_scores(plane, plane))
    centre_y, centre_x = scores.shape[0] // 2, scores.shape[1] // 2
    high = (scores >= AMBIGUOUS_SCORE).astype(np.uint8)
    # The peak at no shift is as broad as the features are; everything joined to it is that peak.
    _, labels = cv2.connectedComponents(high, connectivity=8)
    others = (high > 0) & (labels != labels[centre_y, centre_x])
    dy, dx = np.nonzero(others)
    distance = np.hypot(dy - centre_y, dx - centre_x)
    distance = distance[distance <= within]
    return float(distance.min()) if distance.size else float("inf")


def view_shift(
    before: np.ndarray, after: np.ndarray, predicted: tuple[float, float], radius: float
) -> tuple[tuple[float, float], float] | None:
    """The (x, y) pixels the content of ``structure`` plane ``before`` moved by to reach
    ``after``, searched for within ``radius`` of ``predicted``, and the correlation there;
    None when the frames do not match anywhere in that range."""
    # ``overlap_scores`` gives where ``after``'s corner sits in ``before``'s pixels, which is
    # the content's shift with its sign turned.
    scores = overlap_scores(before, after)
    (dy, dx), score = best_near(scores, (-predicted[1], -predicted[0]), radius)
    if not np.isfinite(score) or score < MATCH_FLOOR:
        return None
    return (-dx, -dy), score
