"""``InputSection``: a labelled list of program inputs, which is how a panel
groups a program's inputs on screen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .program_inputs import ProgramInput


@dataclass(frozen=True)
class InputSection:
    """A heading and the program inputs under it."""

    label: str
    inputs: tuple[ProgramInput[Any], ...]
