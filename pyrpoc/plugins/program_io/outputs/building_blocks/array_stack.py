"""``ArrayStack``: a program output that is a growing stack of arrays with named axes, the
same channels throughout."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import Recording


class ArrayStack(ProgramOutput):
    """Arrays added one at a time, each with ``axes``. A first axis of ``"channel"`` is
    labelled by ``channels``, fixed by the first array added."""

    axes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.arrays: list[np.ndarray] = []
        self.channels: tuple[str, ...] = ()

    def add(self, array: np.ndarray, channels: Sequence[str]) -> None:
        """Add one array, write it if saving, then tell subscribers."""
        frozen = self.checked(array)
        labels = self.labels_for(frozen, channels)
        with self.lock:
            if self.arrays and labels != self.channels:
                raise ValueError(f"{self.name} has channels {self.channels}, not {labels}")
            self.channels = labels
            self.arrays.append(frozen)
            if self.root is not None:
                self.write(self.root, frozen)
        self.changed()

    def write(self, root: Path, array: np.ndarray) -> None:
        """Save one array beside ``root`` as it is added; a type that writes only at close
        does nothing."""
        del root, array

    def checked(self, array: np.ndarray) -> np.ndarray:
        """``array`` as float32 with the declared axes, read-only so readers may hold it."""
        frozen = np.array(array, dtype=np.float32)
        if frozen.ndim != len(self.axes):
            raise ValueError(f"{self.name} expects axes {self.axes}, got shape {frozen.shape}")
        if any(size <= 0 for size in frozen.shape):
            raise ValueError(f"{self.name} received an empty axis: shape {frozen.shape}")
        frozen.setflags(write=False)
        return frozen

    def labels_for(self, array: np.ndarray, channels: Sequence[str]) -> tuple[str, ...]:
        if self.axes[0] != "channel":
            if channels:
                raise ValueError(f"{self.name} has no channel axis to label")
            return ()
        count = array.shape[0]
        if not channels:
            return tuple(f"channel_{index}" for index in range(count))
        if len(channels) != count:
            raise ValueError(f"{self.name} has {count} channels but {len(channels)} labels")
        return tuple(channels)

    def __len__(self) -> int:
        with self.lock:
            return len(self.arrays)

    def __getitem__(self, index: int) -> np.ndarray:
        with self.lock:
            return self.arrays[index]

    @property
    def nbytes(self) -> int:
        with self.lock:
            return sum(array.nbytes for array in self.arrays)

    def fill(self, arrays: Sequence[np.ndarray], channels: Sequence[str]) -> None:
        """Take arrays read back from files, without writing or telling anyone."""
        for array in arrays:
            frozen = self.checked(array)
            self.channels = self.labels_for(frozen, channels)
            self.arrays.append(frozen)
