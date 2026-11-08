"""Every key the panel answers to, in one place."""

from __future__ import annotations

from PyQt6.QtCore import Qt

# Hold to peek at the overlay, tap to keep it up.
OVERLAY = Qt.Key.Key_Tab
CLOSE = Qt.Key.Key_Escape
# Held while dragging on the dish, draws a unit.
DRAW = Qt.KeyboardModifier.ShiftModifier
FIT = Qt.Key.Key_F

UP = Qt.Key.Key_Up
DOWN = Qt.Key.Key_Down
# Held with UP or DOWN, moves the chosen routine instead of the choice.
REORDER = Qt.KeyboardModifier.ShiftModifier
PAUSE = Qt.Key.Key_P
RUN_NOW = Qt.Key.Key_R
REMOVE = (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
LOCATE = (Qt.Key.Key_Return, Qt.Key.Key_Enter)

# The names shown in hints, matching the keys above.
OVERLAY_NAME = "Tab"
DRAW_NAME = "Shift+drag"
FIT_NAME = "F"
