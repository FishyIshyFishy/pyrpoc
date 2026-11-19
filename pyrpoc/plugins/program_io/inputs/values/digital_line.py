"""``DigitalLine``: one DAQ port and line that is switched on or off, and its rules and editor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from pyrpoc.core.program_input import Field, ParameterError
from pyrpoc.utils.widgets.editors import Editor, FieldValues, editor_for
from pyrpoc.utils.widgets.port_line_edit import PortLineEdit


@dataclass(frozen=True)
class DigitalLine:
    """A port and line, and whether it is on."""

    port: int
    line: int
    on: bool

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this line is."""
        return f"{device_name}/port{self.port}/line{self.line}"

    def describe(self) -> str:
        return f"p{self.port}.{self.line} {'on' if self.on else 'off'}"

    def to_dict(self) -> dict[str, Any]:
        return {"port": self.port, "line": self.line, "on": self.on}

    @classmethod
    def from_value(cls, raw: Any, label: str) -> DigitalLine:
        """A line from the form (already a ``DigitalLine``) or from JSON (a dict)."""
        if isinstance(raw, DigitalLine):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError(f"{label}: expected an object with a port, line and on")
        try:
            return cls(int(raw["port"]), int(raw["line"]), bool(raw["on"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ParameterError(f"{label}: needs an integer port and line, and on: {exc}") from exc


@dataclass(frozen=True)
class DigitalLineField(Field):
    """A port and line with a checkbox, whose meaning the label and tooltip give."""

    def coerce(self, value: Any) -> DigitalLine:
        return DigitalLine.from_value(value, self.label)

    def encode(self, value: DigitalLine) -> Any:
        return value.to_dict()


class DigitalLineRow(QWidget):
    """A checkbox beside a ``p<port>.<line>`` box."""

    def __init__(self, tooltip: str, parent: QWidget):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.on = QCheckBox(self)
        self.on.setToolTip(tooltip)
        row.addWidget(self.on)
        self.port_line = PortLineEdit(0, 0, self)
        row.addWidget(self.port_line)
        row.addStretch(1)

    def value(self) -> DigitalLine:
        return DigitalLine(self.port_line.port, self.port_line.line, self.on.isChecked())

    def set_value(self, value: DigitalLine) -> None:
        self.on.setChecked(value.on)
        self.port_line.set_parts(value.port, value.line)


@editor_for(DigitalLineField)
def digital_line_editor(spec: DigitalLineField, parent: QWidget, values: FieldValues) -> Editor:
    del values
    row = DigitalLineRow(spec.tooltip, parent)

    def connect(callback: Callable[[], None]) -> object:
        row.on.toggled.connect(lambda *_: callback())
        return row.port_line.changed.connect(callback)

    return Editor(
        row,
        get=row.value,
        set=row.set_value,
        connect=connect,
        summary=lambda: row.value().describe(),
        spec=spec,
    )
