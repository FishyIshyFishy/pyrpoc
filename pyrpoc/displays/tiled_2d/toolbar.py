"""Analysis tool selector shown above the tiled display."""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QToolButton, QWidget


class AnalysisTool(Enum):
    PAN = "pan"
    INSPECT = "inspect"
    LINE = "line"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"


def tool_from_value(value: str) -> AnalysisTool:
    for tool in AnalysisTool:
        if tool.value == value:
            return tool
    return AnalysisTool.PAN


tool_buttons = [
    (AnalysisTool.PAN, "Pan/Zoom", "Drag to pan, scroll to zoom (no analysis overlay)"),
    (AnalysisTool.INSPECT, "Inspect", "Read the cursor position and every channel's value"),
    (AnalysisTool.LINE, "Line", "Draw a line to plot an intensity profile across channels"),
    (AnalysisTool.RECTANGLE, "Rectangle", "Measure statistics inside a rectangle"),
    (AnalysisTool.ELLIPSE, "Oval", "Measure statistics inside an oval"),
]


class AnalysisToolbar(QWidget):
    tool_changed = pyqtSignal(object)
    clear_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[AnalysisTool, QToolButton] = {}

        for tool, label, tip in tool_buttons:
            button = QToolButton(self)
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, t=tool: self.tool_changed.emit(t))
            self._group.addButton(button)
            self._buttons[tool] = button
            layout.addWidget(button)

        self._buttons[AnalysisTool.PAN].setChecked(True)

        clear_button = QPushButton("Clear", self)
        clear_button.setToolTip("Remove the analysis overlay and return to Pan/Zoom")
        clear_button.clicked.connect(self.clear_requested.emit)
        layout.addSpacing(12)
        layout.addWidget(clear_button)
        layout.addStretch(1)

    def set_active_tool(self, tool: AnalysisTool) -> None:
        button = self._buttons.get(tool)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)
