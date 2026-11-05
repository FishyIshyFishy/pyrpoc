"""What a scan of the dish sees: the focal surface, defocus, the galvo's scale, the
detector's photon counts and bleaching."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The blur of a perfectly focused spot, and how fast it grows away from focus.
FOCUSED_SIGMA_UM = 0.35
SIGMA_PER_UM_DEFOCUS = 0.45
# Photons collected per microsecond from a fully bright pixel, and from the background.
PHOTONS_PER_US = 20.0
BACKGROUND = 0.03
# The light dose, in µs of dwell per µm², that bleaches a cell to 1/e.
BLEACH_DOSE = 2000.0


@dataclass(frozen=True)
class FocalSurface:
    """The z at which the dish is in focus: a tilted plane that drifts with time."""

    z0_um: float
    tilt_x: float
    tilt_y: float
    drift_um_per_min: float

    def z_at(self, x_um: float, y_um: float, minutes: float) -> float:
        return (
            self.z0_um + self.tilt_x * x_um + self.tilt_y * y_um + self.drift_um_per_min * minutes
        )


def blur_sigma_um(defocus_um: float) -> float:
    """The width of the blur ``defocus_um`` away from focus."""
    return FOCUSED_SIGMA_UM + SIGMA_PER_UM_DEFOCUS * abs(defocus_um)


def blurred(plane: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px < 0.3:
        return plane
    return cv2.GaussianBlur(plane, (0, 0), sigmaX=sigma_px, borderType=cv2.BORDER_REFLECT)


def detected(expected: np.ndarray, dwell_us: float, generator: np.random.Generator) -> np.ndarray:
    """``expected`` brightness as the detector reports it: photons counted over the dwell,
    with shot noise, scaled back so the mean does not depend on dwell."""
    # As the rig's frame is the mean of each pixel's samples, a longer dwell lowers the noise
    # but not the level.
    photons_per_pixel = PHOTONS_PER_US * dwell_us
    counts = generator.poisson((expected + BACKGROUND) * photons_per_pixel)
    return (counts / photons_per_pixel).astype(np.float32)


def bleached(dose: float) -> float:
    """The brightness left after ``dose``."""
    return float(np.exp(-dose / BLEACH_DOSE))
