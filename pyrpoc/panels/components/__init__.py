"""Qt building blocks shared by more than one panel.

Cards, the range slider, the read-only list table, the channel colour
palette -- generic widgets and helpers, not application logic. Nothing here
may import a panel, a program, or run/; a panel imports what it needs from
here, never the other way around.
"""

from .cards import BaseCardWidget, RemovableCardWidget
from .colors import color_for_index
from .range_slider import RangeSlider
from .table import ListTable

__all__ = [
    "BaseCardWidget",
    "RemovableCardWidget",
    "color_for_index",
    "RangeSlider",
    "ListTable",
]
