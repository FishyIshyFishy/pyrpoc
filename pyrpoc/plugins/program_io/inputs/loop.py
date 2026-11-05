"""How a loop runs: how often a monitored unit is scanned and for how long, the treatment at
the start, and how often focus is checked."""

from __future__ import annotations

from pyrpoc.core.program_input import declare
from pyrpoc.utils.value_rules import TextField, bool_input, float_input, int_input

PERIOD_S = float_input(
    "loop_period_s",
    "Period (s)",
    60.0,
    minimum=1.0,
    step=10.0,
    decimals=1,
    tooltip="How long after one scan of a unit its next is due",
)
DURATION_MIN = float_input(
    "loop_duration_min",
    "Duration (min)",
    30.0,
    minimum=0.1,
    step=5.0,
    decimals=1,
    tooltip="How long each unit is monitored for",
)
FOCUS_CHECK_S = float_input(
    "loop_focus_check_s",
    "Focus Check (s)",
    300.0,
    minimum=10.0,
    step=60.0,
    decimals=0,
    tooltip="How often focus is found again, to follow it as it drifts",
)
TREAT = bool_input(
    "loop_treat",
    "Treat at Start",
    False,
    tooltip="Before monitoring, scan each unit of the treated group once with the line driven",
)
TREATED_GROUP = declare(
    "loop_treated_group",
    TextField("Treated Group", "The group whose units are treated at the start"),
    "treated",
)
TREATMENT_PORT = int_input(
    "loop_treatment_port", "Treatment Port", 0, minimum=0, tooltip="The DAQ port of the line"
)
TREATMENT_LINE = int_input(
    "loop_treatment_line",
    "Treatment Line",
    0,
    minimum=0,
    tooltip="The digital line driven high over a unit while it is treated",
)
