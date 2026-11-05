"""``Check``: a condition a program sets on starting, judged before its recording opens, and the
``CheckContext`` it is judged in."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class CheckContext(ABC):
    """What a check can see of a run about to start."""

    @property
    @abstractmethod
    def folder(self) -> Path | None:
        """The save folder, which the run is saved in and reads from; None when none is set."""

    @abstractmethod
    def recordings_in_folder(self) -> list[str]:
        """The name of every recording saved in the folder, whether or not it can be read."""

    @abstractmethod
    def reading(self) -> dict[str, str]:
        """For each name the program reads, the recording it would read from."""


@dataclass(frozen=True)
class Check:
    """One condition for starting. Declared once on a program, judged before every start."""

    def blockers(self, ctx: CheckContext) -> list[str]:
        """What stops the run starting, each said as what it needs; empty when it may start."""
        raise NotImplementedError
