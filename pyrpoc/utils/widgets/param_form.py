"""``ParamForm``: a form with one row per value, writing every edit straight
into where the values are kept."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.device import Device
from pyrpoc.core.device_config import DeviceConfig, config_fields
from pyrpoc.core.program_input import Field, ParameterError, ProgramInput

from .cards import BaseCardWidget
from .editors import Editor, build_editor
from .flow_grid import FlowGrid

# The narrowest a field and its label are given before the grid drops to fewer columns.
CELL_PX = 110


@dataclass(frozen=True)
class FormSection:
    """A heading and the ``(key, rules)`` rows under it."""

    label: str
    rows: tuple[tuple[str, Field], ...]


def input_form_section(label: str, inputs: Sequence[ProgramInput[Any]]) -> FormSection:
    """A form section showing ``inputs`` under ``label``."""
    return FormSection(label, tuple((each.key, each.rule) for each in inputs))


def config_form_section(cls: type[DeviceConfig]) -> FormSection:
    """A form section showing every field of a device configuration, keyed by field name."""
    return FormSection("", tuple(config_fields(cls)))


class FormValues(Protocol):
    """Where a form reads and writes its values, by key."""

    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: Any) -> None:
        """Store ``value``, raising ``ParameterError`` if its rules refuse it."""
        ...


class ConfigValues:
    """One device's configuration, as a form reads and writes it."""

    def __init__(self, device: Device):
        self.device = device

    def get(self, key: str) -> Any:
        return getattr(self.device.config, key)

    def set(self, key: str, value: Any) -> None:
        spec = dict(config_fields(self.device.config))[key]
        setattr(self.device.config, key, spec.coerce(value))


class ParamForm(QWidget):
    """One row per value, grouped into sections, optionally as collapsible cards.
    Every edit is written straight into ``values``."""

    changed = pyqtSignal()
    invalid = pyqtSignal(str)

    def __init__(
        self,
        form_sections: Sequence[FormSection],
        values: FormValues,
        parent: QWidget,
        *,
        cards: bool,
    ):
        super().__init__(parent)
        self.values = values
        self.fields: dict[str, Editor] = {}
        self._cards: list[tuple[BaseCardWidget, list[str]]] = []
        self._loading = False
        self._writing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        for section in form_sections:
            root.addWidget(self.build_section(section, cards=cards))
        root.addStretch(1)
        self.reload()

    def build_section(self, section: FormSection, *, cards: bool) -> QWidget:
        body = QWidget(self)
        if cards:
            grid = FlowGrid(body, min_cell=CELL_PX, spacing=10)
            grid.setContentsMargins(4, 4, 4, 4)
        else:
            form = QFormLayout(body)
            form.setContentsMargins(4, 4, 4, 4)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        keys: list[str] = []
        for key, spec in section.rows:
            field = build_editor(spec, body, self.values)
            field.connect(lambda key=key: self.on_field_changed(key))
            self.fields[key] = field
            keys.append(key)
            if cards:
                grid.add(self.build_cell(spec.label, field.widget, body), wide=field.wide)
            else:
                form.addRow(spec.label, field.widget)
        if not cards:
            return body
        card = BaseCardWidget(section.label, self)
        card.set_body_widget(body)
        self._cards.append((card, keys))
        return card

    def build_cell(self, label: str, widget: QWidget, parent: QWidget) -> QWidget:
        """A field under its label, so a cell is as wide as its field needs."""
        cell = QWidget(parent)
        column = QVBoxLayout(cell)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        title = QLabel(label, cell)
        # Wraps rather than widening the cell, so a long name never forces a column wider.
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        column.addWidget(title)
        column.addWidget(widget)
        widget.setParent(cell)
        return cell

    def reload(self) -> None:
        """Show every value ``values`` holds now."""
        for key in self.fields:
            self.show_value(key)
        self.refresh_summaries()

    def show_value(self, key: str) -> None:
        """Show the value ``values`` holds for ``key``, unless this form is the
        one writing it."""
        # A custom editor may rebuild itself on ``set``, which must not happen
        # under the user's hands while they edit it.
        if self._writing or key not in self.fields:
            return
        self._loading = True
        try:
            self.fields[key].set(self.values.get(key))
        finally:
            self._loading = False
        self.refresh_summaries()

    def on_field_changed(self, key: str) -> None:
        """Write the edited value; a value its rules refuse is reported, not raised."""
        if self._loading:
            return
        self._writing = True
        try:
            self.values.set(key, self.fields[key].get())
        except ParameterError as exc:
            self.invalid.emit(str(exc))
            return
        finally:
            self._writing = False
        self.refresh_summaries()
        self.changed.emit()

    def refresh_summaries(self) -> None:
        for card, keys in self._cards:
            card.set_summary([(self.fields[k].spec.label, self.fields[k].summary()) for k in keys])
