"""Mask presets: a file path plus the digital line it drives.

A mask is authored, not acquired — you draw it once, save it, and load it into
runs for months. So it is a plain file referenced by a parameter, not an entry
in the dataset library.

This is the one place ``core/`` touches the filesystem. Mask files are an input
format, not dataset output, so putting the loader in ``data/io.py`` would muddy
that module; and ``programs/hardware/`` may not import ``data/`` at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import ParameterError


@dataclass(frozen=True)
class MaskBinding:
    """One mask file wired to one digital output line."""

    path: Path
    port: int = 0
    line: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(str(self.path)))
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(self, "line", int(self.line))

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this binding drives."""
        return f"{device_name}/port{self.port}/line{self.line}"

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "port": self.port, "line": self.line}

    @classmethod
    def from_dict(cls, raw: Any) -> "MaskBinding":
        if isinstance(raw, MaskBinding):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a mask binding must be an object with path/port/line")
        return cls(
            path=Path(str(raw.get("path", ""))),
            port=int(raw.get("port", 0)),
            line=int(raw.get("line", 0)),
        )


def load_mask(path: Path | str) -> np.ndarray:
    """Read a mask image as a 2-D uint8 array."""
    resolved = Path(str(path)).expanduser()
    image = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read a mask from '{resolved}'")
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={array.shape}")
    return array.astype(np.uint8, copy=True)


def save_mask(path: Path | str, mask: np.ndarray) -> Path:
    """Write a 2-D mask to disk. Returns the path written."""
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={array.shape}")
    resolved = Path(str(path)).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(resolved), array):
        raise OSError(f"failed to write a mask to '{resolved}'")
    return resolved
