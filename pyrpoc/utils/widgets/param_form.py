"""``ParamForm``: a form with one row per value, writing every edit straight
into where the values are kept."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core import program_input as P
from pyrpoc.core.device import Device
from pyrpoc.core.device_config import DeviceConfig, config_fields
from pyrpoc.core.program_input import InputSection, ParameterError
from pyrpoc.utils import value_rules as R

from .cards import BaseCardWidget
from .flow_grid import FlowGrid

# The narrowest a field and its label are given before the grid drops to fewer columns.
CELL_PX = 110
# Enough for a few digits and the arrow buttons.
SPIN_MIN_PX = 64

CHANNEL_BUTTON_CSS = (
    "QToolButton {"
    "padding: 4px 8px;"
    "border: 1px solid palette(mid);"
    "border-radius: 10px;"
    "background: palette(base);"
    "}"
    "QToolButton:checked {"
    "background: palette(highlight);"
    "color: palette(highlighted-text);"
    "border: 1px solid palette(highlight);"
    "font-weight: 700;"
    "}"
)


def build_int(spec: R.IntField, parent: QWidget) -> P.Editor:
    widget = QSpinBox(parent)
    widget.setMinimumWidth(SPIN_MIN_PX)
    widget.setMinimum(spec.minimum if spec.minimum is not None else -1_000_000)
    widget.setMaximum(spec.maximum if spec.maximum is not None else 1_000_000)
    widget.setSingleStep(spec.step)
    return P.Editor(
        widget,
        get=widget.value,
        set=lambda value: widget.setValue(int(value)),
        connect=lambda cb: widget.valueChanged.connect(lambda *_: cb()),
        summary=lambda: str(widget.value()),
        spec=spec,
    )


class TrimmedSpinBox(QDoubleSpinBox):
    """A spin box that drops trailing zeros: 2 rather than 2.000000."""

    def textFromValue(self, v: float) -> str:
        text = f"{v:.{self.decimals()}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text


def build_float(spec: R.FloatField, parent: QWidget) -> P.Editor:
    widget = TrimmedSpinBox(parent)
    widget.setMinimumWidth(SPIN_MIN_PX)
    widget.setDecimals(spec.decimals)
    widget.setMinimum(spec.minimum if spec.minimum is not None else -1e12)
    widget.setMaximum(spec.maximum if spec.maximum is not None else 1e12)
    widget.setSingleStep(spec.step)
    return P.Editor(
        widget,
        get=widget.value,
        set=lambda value: widget.setValue(float(value)),
        connect=lambda cb: widget.valueChanged.connect(lambda *_: cb()),
        summary=lambda: f"{widget.value():g}",
        spec=spec,
    )


def build_text(spec: R.TextField, parent: QWidget) -> P.Editor:
    widget = QLineEdit(parent)
    return P.Editor(
        widget,
        get=widget.text,
        set=lambda value: widget.setText(str(value)),
        connect=lambda cb: widget.textChanged.connect(lambda *_: cb()),
        summary=lambda: widget.text() or "-",
        spec=spec,
    )


def build_path(spec: R.PathField, parent: QWidget) -> P.Editor:
    root = QWidget(parent)
    layout = QHBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(root)
    edit.setPlaceholderText("Path")
    browse = QPushButton("Browse...", root)
    layout.addWidget(edit, 1)
    layout.addWidget(browse)

    def pick() -> None:
        current = edit.text().strip()
        start = str(Path(current).expanduser()) if current else str(Path.cwd())
        selected, _ = QFileDialog.getSaveFileName(
            root, "Select output path", start, spec.dialog_filter
        )
        if selected:
            edit.setText(selected)

    browse.clicked.connect(pick)
    return P.Editor(
        root,
        get=edit.text,
        set=lambda value: edit.setText(str(value)),
        connect=lambda cb: edit.textChanged.connect(lambda *_: cb()),
        summary=lambda: Path(edit.text()).name if edit.text().strip() else "-",
        spec=spec,
        wide=True,
    )


def build_bool(spec: R.BoolField, parent: QWidget) -> P.Editor:
    widget = QCheckBox(parent)
    return P.Editor(
        widget,
        get=widget.isChecked,
        set=lambda value: widget.setChecked(bool(value)),
        connect=lambda cb: widget.toggled.connect(lambda *_: cb()),
        summary=lambda: "on" if widget.isChecked() else "off",
        spec=spec,
    )


def build_choice(spec: R.ChoiceField, parent: QWidget) -> P.Editor:
    widget = QComboBox(parent)
    widget.addItems(list(spec.choices))
    return P.Editor(
        widget,
        get=widget.currentText,
        set=lambda value: widget.setCurrentText(str(value)),
        connect=lambda cb: widget.currentTextChanged.connect(lambda *_: cb()),
        summary=widget.currentText,
        spec=spec,
    )


def build_channels(spec: R.ChannelsField, parent: QWidget) -> P.Editor:
    root = QWidget(parent)
    layout = QHBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

    buttons: list[QToolButton] = []
    for index in range(spec.num_channels):
        button = QToolButton(root)
        button.setCheckable(True)
        button.setText(f"AI{index}")
        button.setToolTip(f"Toggle AI{index}")
        button.setStyleSheet(CHANNEL_BUTTON_CSS)
        layout.addWidget(button)
        buttons.append(button)

    def get() -> tuple[int, ...]:
        return tuple(i for i, button in enumerate(buttons) if button.isChecked())

    def set_value(value: tuple[int, ...]) -> None:
        for index, button in enumerate(buttons):
            button.blockSignals(True)
            button.setChecked(index in value)
            button.blockSignals(False)

    def connect(cb: Callable[[], None]) -> None:
        for button in buttons:
            button.toggled.connect(lambda *_: cb())

    return P.Editor(
        root,
        get=get,
        set=set_value,
        connect=connect,
        summary=lambda: ", ".join(f"AI{i}" for i in get()) or "none",
        spec=spec,
        wide=True,
    )


BUILDERS: dict[type, Callable[[Any, QWidget], P.Editor]] = {
    R.IntField: build_int,
    R.FloatField: build_float,
    R.TextField: build_text,
    R.PathField: build_path,
    R.BoolField: build_bool,
    R.ChoiceField: build_choice,
    R.ChannelsField: build_channels,
}


def build_field(spec: P.Field, parent: QWidget, values: P.FieldValues) -> P.Editor:
    """The field's own editor if it has one, otherwise the form's."""
    field = spec.editor(parent, values) or BUILDERS[type(spec)](spec, parent)
    if spec.tooltip:
        field.widget.setToolTip(spec.tooltip)
    return field


@dataclass(frozen=True)
class FormSection:
    """A heading and the ``(key, rules)`` rows under it."""

    label: str
    rows: tuple[tuple[str, P.Field], ...]


def input_form_section(section: InputSection) -> FormSection:
    """A form section showing the program inputs of ``section``."""
    return FormSection(section.label, tuple((each.key, each.rule) for each in section.inputs))


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
        self.fields: dict[str, P.Editor] = {}
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
            if not spec.shown:
                continue
            field = build_field(spec, body, self.values)
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
