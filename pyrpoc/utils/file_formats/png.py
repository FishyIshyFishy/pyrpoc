"""8-bit greyscale PNGs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_png(path: Path, image: np.ndarray) -> None:
    """A 2-D ``uint8`` array as a greyscale PNG at ``path``."""
    # Encoded in memory and written by Python: OpenCV's own file calls fail on non-ASCII paths.
    encoded, data = cv2.imencode(".png", np.ascontiguousarray(image, dtype=np.uint8))
    if not encoded:
        raise OSError(f"could not encode a PNG for '{path}'")
    path.write_bytes(data.tobytes())


def read_png(path: Path) -> np.ndarray:
    """The image at ``path`` as a 2-D ``uint8`` array, any colour flattened to grey."""
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"'{path.name}' is not an image that can be read")
    return image
