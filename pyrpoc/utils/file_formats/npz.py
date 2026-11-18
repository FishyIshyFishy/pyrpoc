"""Named arrays in one compressed ``.npz`` file."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """``arrays`` as one compressed ``.npz`` at ``path``, each under its name."""
    # Not pickled, so reading a file back never runs code from it.
    np.savez_compressed(str(path), allow_pickle=False, **arrays)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """Every array in the ``.npz`` at ``path``, by name."""
    with np.load(str(path)) as archive:
        return {name: archive[name] for name in archive.files}
