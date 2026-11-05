"""``Editor``: one field's widget, the editors of the common value rules, and the lookup that
finds the editor for any ``Field``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from pyrpoc.core.program_input import Field
from pyrpoc.utils import value_rules as R

FieldT = TypeVar("FieldT", bound=Field)


class FieldValues(Protocol):
    """The values a form shows, by key."""

    def get(self, key: str) -> Any: ...


@dataclass
class Editor:
    """One field's widget, and how a form reads, writes and watches it."""

    widget: QWidget
    get: Callable[[], Any]
    set: Callable[[Any], None]
    # Returns object: the result is ignored, and Qt's connect() returns a handle.
    connect: Callable[[Callable[[], None]], object]
    summary: Callable[[], str]
    spec: Field
    # Whether a form gives it a whole row rather than one cell of a shared row.
    wide: bool = False


# Builds a field's editor. ``values`` is the form's, for an editor that shows other fields'
# values beside its own.
EditorBuilder = Callable[[Any, QWidget, FieldValues], Editor]

# The editor of each type of ``Field``, found for a field by its type or the nearest it derives
# from.
editor_builders: dict[type[Field], EditorBuilder] = {}


def editor_for(
    field_type: type[FieldT],
) -> Callable[
    [Callable[[FieldT, QWidget, FieldValues], Editor]],
    Callable[[FieldT, QWidget, FieldValues], Editor],
]:
    """Register the decorated function as the editor of ``field_type``."""

    def register(
        build: Callable[[FieldT, QWidget, FieldValues], Editor],
    ) -> Callable[[FieldT, QWidget, FieldValues], Editor]:
        if field_type in editor_builders:
            raise ValueError(f"{field_type.__name__} already has an editor")
        editor_builders[field_type] = build
        return build

    return register


def build_editor(spec: Field, parent: QWidget, values: FieldValues) -> Editor:
    """The editor of ``spec``'s type, with its tooltip."""
    build = next(
        (editor_builders[cls] for cls in type(spec).__mro__ if cls in editor_builders), None
    )
    if build is None:
        raise TypeError(f"no editor is registered for {type(spec).__name__}")
    editor = build(spec, parent, values)
    if spec.tooltip:
        editor.widget.setToolTip(spec.tooltip)
    return editor


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


@editor_for(R.IntField)
def build_int(spec: R.IntField, parent: QWidget, values: FieldValues) -> Editor:
    del values
    widget = QSpinBox(parent)
    widget.setMinimumWidth(SPIN_MIN_PX)
    widget.setMinimum(spec.minimum if spec.minimum is not None else -1_000_000)
    widget.setMaximum(spec.maximum if spec.maximum is not None else 1_000_000)
    widget.setSingleStep(spec.step)
    return Editor(
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


@editor_for(R.FloatField)
def build_float(spec: R.FloatField, parent: QWidget, values: FieldValues) -> Editor:
    del values
    widget = TrimmedSpinBox(parent)
    widget.setMinimumWidth(SPIN_MIN_PX)
    widget.setDecimals(spec.decimals)
    widget.setMinimum(spec.minimum if spec.minimum is not None else -1e12)
    widget.setMaximum(spec.maximum if spec.maximum is not None else 1e12)
    widget.setSingleStep(spec.step)
    return Editor(
        widget,
        get=widget.value,
        set=lambda value: widget.setValue(float(value)),
        connect=lambda cb: widget.valueChanged.connect(lambda *_: cb()),
        summary=lambda: f"{widget.value():g}",
        spec=spec,
    )


@editor_for(R.TextField)
def build_text(spec: R.TextField, parent: QWidget, values: FieldValues) -> Editor:
    del values
    widget = QLineEdit(parent)
    return Editor(
        widget,
        get=widget.text,
        set=lambda value: widget.setText(str(value)),
        connect=lambda cb: widget.textChanged.connect(lambda *_: cb()),
        summary=lambda: widget.text() or "-",
        spec=spec,
    )


@editor_for(R.PathField)
def build_path(spec: R.PathField, parent: QWidget, values: FieldValues) -> Editor:
    del values
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
    return Editor(
        root,
        get=edit.text,
        set=lambda value: edit.setText(str(value)),
        connect=lambda cb: edit.textChanged.connect(lambda *_: cb()),
        summary=lambda: Path(edit.text()).name if edit.text().strip() else "-",
        spec=spec,
        wide=True,
    )


@editor_for(R.BoolField)
def build_bool(spec: R.BoolField, parent: QWidget, values: FieldValues) -> Editor:
    del values
    widget = QCheckBox(parent)
    return Editor(
        widget,
        get=widget.isChecked,
        set=lambda value: widget.setChecked(bool(value)),
        connect=lambda cb: widget.toggled.connect(lambda *_: cb()),
        summary=lambda: "on" if widget.isChecked() else "off",
        spec=spec,
    )


@editor_for(R.ChoiceField)
def build_choice(spec: R.ChoiceField, parent: QWidget, values: FieldValues) -> Editor:
    del values
    widget = QComboBox(parent)
    widget.addItems(list(spec.choices))
    return Editor(
        widget,
        get=widget.currentText,
        set=lambda value: widget.setCurrentText(str(value)),
        connect=lambda cb: widget.currentTextChanged.connect(lambda *_: cb()),
        summary=widget.currentText,
        spec=spec,
    )


@editor_for(R.ChannelsField)
def build_channels(spec: R.ChannelsField, parent: QWidget, values: FieldValues) -> Editor:
    del values
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

    return Editor(
        root,
        get=get,
        set=set_value,
        connect=connect,
        summary=lambda: ", ".join(f"AI{i}" for i in get()) or "none",
        spec=spec,
        wide=True,
    )
