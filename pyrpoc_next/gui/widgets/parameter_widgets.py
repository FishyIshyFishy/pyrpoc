"""Render structs.Parameter declarations to Qt widgets.

The Qt half of the declarative parameter system: structs describes a parameter, this
turns it into an editor, reads it back, and reports changes. Dispatch is by type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from pyrpoc_next.structs.parameters import (
    ChannelSelectionParameter,
    CheckboxParameter,
    ChoiceParameter,
    NumberParameter,
    Parameter,
    PathParameter,
    TextParameter,
)


class ChannelSelector(QWidget):
    """A row of toggle buttons picking active channel indices."""

    def __init__(self, count: int):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.buttons: list[QToolButton] = []
        for index in range(count):
            button = QToolButton()
            button.setText(f"AI{index}")
            button.setCheckable(True)
            layout.addWidget(button)
            self.buttons.append(button)

    def selected(self) -> list[int]:
        return [index for index, button in enumerate(self.buttons) if button.isChecked()]

    def set_selected(self, indices: list[int]) -> None:
        chosen = set(indices)
        for index, button in enumerate(self.buttons):
            button.setChecked(index in chosen)


class PathEditor(QWidget):
    """A line edit with a Browse button for a file path."""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line = QLineEdit()
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.pick)
        layout.addWidget(self.line)
        layout.addWidget(browse)

    def pick(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Choose file")
        if path:
            self.line.setText(path)


def create_widget(parameter: Parameter) -> QWidget:
    """Build the editor widget for a parameter declaration."""
    if isinstance(parameter, PathParameter):
        return PathEditor()
    if isinstance(parameter, TextParameter):
        return QLineEdit()
    if isinstance(parameter, NumberParameter):
        if parameter.number_type is int:
            spin = QSpinBox()
            low = int(parameter.minimum) if parameter.minimum is not None else -2_000_000_000
            high = int(parameter.maximum) if parameter.maximum is not None else 2_000_000_000
            spin.setRange(low, high)
            spin.setSingleStep(max(1, int(parameter.step)))
            return spin
        spin = QDoubleSpinBox()
        spin.setRange(parameter.minimum if parameter.minimum is not None else -1e12,
                      parameter.maximum if parameter.maximum is not None else 1e12)
        spin.setSingleStep(parameter.step)
        spin.setDecimals(6)
        return spin
    if isinstance(parameter, CheckboxParameter):
        return QCheckBox()
    if isinstance(parameter, ChoiceParameter):
        combo = QComboBox()
        combo.addItems(parameter.choices)
        return combo
    if isinstance(parameter, ChannelSelectionParameter):
        return ChannelSelector(parameter.channel_count)
    raise TypeError(f"no widget for {type(parameter).__name__}")


def read_value(parameter: Parameter, widget: QWidget) -> Any:
    """Read the current value out of a parameter's widget."""
    if isinstance(widget, PathEditor):
        return widget.line.text()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, ChannelSelector):
        return widget.selected()
    raise TypeError(f"cannot read {type(widget).__name__}")


def set_value(parameter: Parameter, widget: QWidget, value: Any) -> None:
    """Write a value into a parameter's widget."""
    if isinstance(widget, PathEditor):
        widget.line.setText(str(value))
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value))
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(value)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QComboBox):
        widget.setCurrentText(str(value))
    elif isinstance(widget, ChannelSelector):
        widget.set_selected(list(value))


def connect_changed(widget: QWidget, callback: Callable[[], None]) -> None:
    """Fire callback whenever the widget's value changes."""
    if isinstance(widget, PathEditor):
        widget.line.textChanged.connect(lambda *_: callback())
    elif isinstance(widget, QLineEdit):
        widget.textChanged.connect(lambda *_: callback())
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.valueChanged.connect(lambda *_: callback())
    elif isinstance(widget, QCheckBox):
        widget.toggled.connect(lambda *_: callback())
    elif isinstance(widget, QComboBox):
        widget.currentTextChanged.connect(lambda *_: callback())
    elif isinstance(widget, ChannelSelector):
        for button in widget.buttons:
            button.toggled.connect(lambda *_: callback())
