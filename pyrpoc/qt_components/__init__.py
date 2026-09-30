"""Generic Qt widgets: tables, cards, the parameter form, icons, colours and
histogram levels. They know no feature, so any panel can use them; they import
only Qt and structs."""

from __future__ import annotations

from .cards import BaseCardWidget, RemovableCardWidget
from .colors import color_for_index
from .table import ListTable

__all__ = ["BaseCardWidget", "RemovableCardWidget", "color_for_index", "ListTable"]
