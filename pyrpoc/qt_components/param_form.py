"""A form generated from parameter blocks, writing back into them.

The blocks are authoritative: every widget change writes straight into its
block, so nothing scrapes the form at play time. A field type declared outside
``structs`` brings its own widget through ``Field.editor``, so the form never
learns what it is editing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.structs import params as P
from pyrpoc.structs.params import ParameterError

from .cards import BaseCardWidget

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


def build_int(spec: P.IntField, parent: QWidget) -> P.Editor:
    widget = QSpinBox(parent)
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


def build_float(spec: P.FloatField, parent: QWidget) -> P.Editor:
    widget = QDoubleSpinBox(parent)
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


def build_text(spec: P.TextField, parent: QWidget) -> P.Editor:
    widget = QLineEdit(parent)
    return P.Editor(
        widget,
        get=widget.text,
        set=lambda value: widget.setText(str(value)),
        connect=lambda cb: widget.textChanged.connect(lambda *_: cb()),
        summary=lambda: widget.text() or "-",
        spec=spec,
    )


def build_path(spec: P.PathField, parent: QWidget) -> P.Editor:
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
    )


def build_bool(spec: P.BoolField, parent: QWidget) -> P.Editor:
    widget = QCheckBox(parent)
    return P.Editor(
        widget,
        get=widget.isChecked,
        set=lambda value: widget.setChecked(bool(value)),
        connect=lambda cb: widget.toggled.connect(lambda *_: cb()),
        summary=lambda: "on" if widget.isChecked() else "off",
        spec=spec,
    )


def build_choice(spec: P.ChoiceField, parent: QWidget) -> P.Editor:
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


def build_channels(spec: P.ChannelsField, parent: QWidget) -> P.Editor:
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
    )


BUILDERS: dict[type, Callable[[Any, QWidget], P.Editor]] = {
    P.IntField: build_int,
    P.FloatField: build_float,
    P.TextField: build_text,
    P.PathField: build_path,
    P.BoolField: build_bool,
    P.ChoiceField: build_choice,
    P.ChannelsField: build_channels,
}


def build_field(spec: P.Field, parent: QWidget, context: P.FieldContext) -> P.Editor:
    """The field's own editor if it has one, otherwise the form's."""
    field = spec.editor(parent, context) or BUILDERS[type(spec)](spec, parent)
    if spec.tooltip:
        field.widget.setToolTip(spec.tooltip)
    return field


class ParamForm(QWidget):
    """One section per parameter block, optionally as collapsible cards.

    Blocks are shared across programs, so switching programs hands nothing
    over: the instance this form edits is the one every other declaring
    program holds.
    """

    changed = pyqtSignal()
    invalid = pyqtSignal(str)

    def __init__(
        self,
        blocks: Sequence[P.Group],
        parent: QWidget,
        *,
        cards: bool,
        context: P.FieldContext,
    ):
        """``context`` is what the owner can offer fields that build their own
        editor, such as the open data for a field whose value names it."""
        super().__init__(parent)
        self.blocks: list[P.Group] = list(blocks)
        self.index: dict[str, P.Group] = P.index(self.blocks)
        self.fields: dict[str, P.Editor] = {}
        self._cards: list[tuple[BaseCardWidget, list[str]]] = []
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        for section in P.sections(self.blocks):
            root.addWidget(self.build_section(section, cards=cards, context=context))
        root.addStretch(1)
        self.reload()

    def build_section(self, section: P.Section, *, cards: bool, context: P.FieldContext) -> QWidget:
        body = QWidget(self)
        form = QFormLayout(body)
        form.setContentsMargins(4, 4, 4, 4)
        paths: list[str] = []
        for path, spec in section.entries:
            field = build_field(spec, body, context)
            field.connect(self.on_field_changed)
            self.fields[path] = field
            paths.append(path)
            form.addRow(spec.label, field.widget)
        if not cards:
            return body
        card = BaseCardWidget(section.label, self)
        card.set_body_widget(body)
        self._cards.append((card, paths))
        return card

    def reload(self) -> None:
        """Blocks -> form. Call after something else has written the blocks."""
        self._loading = True
        try:
            for path, field in self.fields.items():
                field.set(P.get_path(self.index, path))
        finally:
            self._loading = False
        self.refresh_summaries()

    def read_into_blocks(self) -> None:
        """Form -> blocks. Coerces everything before writing anything, so a
        value that fails its bounds leaves every block untouched."""
        coerced = {path: field.spec.coerce(field.get()) for path, field in self.fields.items()}
        for path, value in coerced.items():
            P.set_path(self.index, path, value)

    def on_field_changed(self) -> None:
        """A bad value is reported, not raised: this is user input, and the
        widgets normally clamp but that is not a guarantee."""
        if self._loading:
            return
        try:
            self.read_into_blocks()
        except ParameterError as exc:
            self.invalid.emit(str(exc))
            return
        self.refresh_summaries()
        self.changed.emit()

    def refresh_summaries(self) -> None:
        for card, paths in self._cards:
            parts = [f"{self.fields[p].spec.label}: {self.fields[p].summary()}" for p in paths]
            card.set_description("  |  ".join(parts))
