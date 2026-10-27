"""Decay fit: which exponential model a decay trace is fitted with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block, choice_field

SINGLE = "Single exponential"
DOUBLE = "Two exponentials"
DOUBLE_FIXED = "Two exponentials, control rate fixed"
MODELS = (SINGLE, DOUBLE, DOUBLE_FIXED)


@block
@dataclass
class DecayFitGroup(Group):
    label: ClassVar[str] = "Decay Fit"

    model: str = choice_field(
        "Model",
        SINGLE,
        choices=MODELS,
        tooltip="Single: one rate, fitted on treated ÷ control. "
        "Two: a fast and a slow rate on treated ÷ control; the slow one is reported. "
        "Control rate fixed: the control's own rate is one of two components, fitted "
        "on the uncorrected treated trace; the other rate is reported. Needs control cells",
    )
