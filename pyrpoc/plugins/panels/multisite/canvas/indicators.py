"""``Indicator``: the ways the dish can show what the loop is doing, each switched on or off."""

from __future__ import annotations

from enum import Flag, auto


class Indicator(Flag):
    NONE = 0
    # A field-sized box that travels to the unit and shows the phase.
    RETICLE = auto()
    # Each unit's outline drawn by its state, and the active one by the phase.
    OUTLINE_STATE = auto()
    # The next routines to run, marked on their units with a countdown.
    NEXT_UP = auto()
    # A line of text in the corner.
    STATUS_LINE = auto()
    ALL = RETICLE | OUTLINE_STATE | NEXT_UP | STATUS_LINE


NAMES = {
    Indicator.RETICLE: "Field-of-view reticle",
    Indicator.OUTLINE_STATE: "Unit outline state",
    Indicator.NEXT_UP: "Next-up preview",
    Indicator.STATUS_LINE: "Corner status line",
}
