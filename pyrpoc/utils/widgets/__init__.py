"""Generic Qt widgets that know no feature: the panel bases, tables, cards, the
parameter form, icons, colours and histogram levels."""

from __future__ import annotations

from .cards import BaseCardWidget, RemovableCardWidget
from .colors import color_for_index
from .table import ListTable

__all__ = ["BaseCardWidget", "RemovableCardWidget", "color_for_index", "ListTable"]
