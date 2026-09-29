"""A form generated from parameter blocks, writing back into them.

The blocks are authoritative. Every widget change writes straight back into the
block instance, so nothing has to scrape the form at play time and anything
other than the form can parameterise a run.

One generator serves the acquisition form and the device panels both, so adding
a field to a device config adds its row with no panel edit.

Imports only ``structs``. The generic fields are drawn here; a field type
declared elsewhere -- a mask binding, a galvo point -- brings its own widget
through ``Field.editor``, so the form never learns what it is editing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

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

from pyrpoc.src.structs import params as P
from pyrpoc.src.structs.params import ParameterError

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


#: One field's widget. The record lives in ``structs`` so a field declared
#: elsewhere can build its own through ``Field.editor``.
FieldWidget = P.Editor


# --------------------------------------------------------------------------- #
# One widget per field type                                                    #
# --------------------------------------------------------------------------- #


def build_int(spec: P.IntField, parent) -> FieldWidget:
    widget = QSpinBox(parent)
    widget.setMinimum(int(spec.minimum) if spec.minimum is not None else -1_000_000)
    widget.setMaximum(int(spec.maximum) if spec.maximum is not None else 1_000_000)
    widget.setSingleStep(int(spec.step or 1))
    return FieldWidget(
        widget,
        get=widget.value,
        set=lambda value: widget.setValue(int(value if value is not None else 0)),
        connect=lambda cb: widget.valueChanged.connect(lambda *_: cb()),
        summary=lambda: str(widget.value()),
    )


def build_float(spec: P.FloatField, parent) -> FieldWidget:
    widget = QDoubleSpinBox(parent)
    widget.setDecimals(spec.decimals)
    widget.setMinimum(float(spec.minimum) if spec.minimum is not None else -1e12)
    widget.setMaximum(float(spec.maximum) if spec.maximum is not None else 1e12)
    widget.setSingleStep(float(spec.step or 0.1))
    return FieldWidget(
        widget,
        get=widget.value,
        set=lambda value: widget.setValue(float(value if value is not None else 0.0)),
        connect=lambda cb: widget.valueChanged.connect(lambda *_: cb()),
        summary=lambda: f"{widget.value():g}",
    )


def build_text(spec: P.TextField, parent) -> FieldWidget:
    widget = QLineEdit(parent)
    return FieldWidget(
        widget,
        get=widget.text,
        set=lambda value: widget.setText("" if value is None else str(value)),
        connect=lambda cb: widget.textChanged.connect(lambda *_: cb()),
        summary=lambda: widget.text() or "-",
    )


def build_path(spec: P.PathField, parent) -> FieldWidget:
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
        selected, _ = QFileDialog.getSaveFileName(root, "Select output path", start, spec.dialog_filter)
        if selected:
            edit.setText(selected)

    browse.clicked.connect(pick)
    return FieldWidget(
        root,
        get=edit.text,
        set=lambda value: edit.setText("" if value is None else str(value)),
        connect=lambda cb: edit.textChanged.connect(lambda *_: cb()),
        summary=lambda: Path(edit.text()).name if edit.text().strip() else "-",
    )


def build_bool(spec: P.BoolField, parent) -> FieldWidget:
    widget = QCheckBox(parent)
    return FieldWidget(
        widget,
        get=widget.isChecked,
        set=lambda value: widget.setChecked(bool(value)),
        connect=lambda cb: widget.toggled.connect(lambda *_: cb()),
        summary=lambda: "on" if widget.isChecked() else "off",
    )


def build_choice(spec: P.ChoiceField, parent) -> FieldWidget:
    widget = QComboBox(parent)
    widget.addItems(list(spec.choices))
    return FieldWidget(
        widget,
        get=widget.currentText,
        set=lambda value: widget.setCurrentText(str(value)),
        connect=lambda cb: widget.currentTextChanged.connect(lambda *_: cb()),
        summary=widget.currentText,
    )


def build_channels(spec: P.ChannelsField, parent) -> FieldWidget:
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

    def set_value(value) -> None:
        active = set(value or ())
        for index, button in enumerate(buttons):
            button.blockSignals(True)
            button.setChecked(index in active)
            button.blockSignals(False)

    def connect(cb) -> None:
        for button in buttons:
            button.toggled.connect(lambda *_: cb())

    return FieldWidget(
        root,
        get=get,
        set=set_value,
        connect=connect,
        summary=lambda: ", ".join(f"AI{i}" for i in get()) or "none",
    )


BUILDERS: dict[type, Callable[[Any, QWidget], FieldWidget]] = {
    P.IntField: build_int,
    P.FloatField: build_float,
    P.TextField: build_text,
    P.PathField: build_path,
    P.BoolField: build_bool,
    P.ChoiceField: build_choice,
    P.ChannelsField: build_channels,
}


def build_field(spec: P.Field, parent: QWidget, context: P.FieldContext) -> FieldWidget:
    """The field's own editor if it has one, otherwise the form's."""
    field = spec.editor(parent, context)
    if field is None:
        builder = BUILDERS.get(type(spec))
        if builder is None:
            raise TypeError(f"no widget for {type(spec).__name__}")
        field = builder(spec, parent)
    field.spec = spec
    if spec.tooltip:
        field.widget.setToolTip(spec.tooltip)
    return field


# --------------------------------------------------------------------------- #
# The form                                                                     #
# --------------------------------------------------------------------------- #


class ParamForm(QWidget):
    """One collapsible card per parameter block, generated from the blocks.

    The blocks are authoritative and shared: every widget change writes
    straight back into the instance it came from, and that instance is the one
    every other modality declaring the same block is also holding. So switching
    programs does not need the form to hand anything over, and nothing has to
    scrape widgets at play time.

    Takes a sequence, so a device configuration is simply a one-block form.
    """

    changed = pyqtSignal()
    invalid = pyqtSignal(str)

    def __init__(
        self,
        blocks: Sequence[P.Group] | P.Group,
        parent: QWidget | None = None,
        *,
        cards: bool = True,
        context: P.FieldContext | None = None,
    ):
        """``context`` is what the owner can offer fields that build their own
        editor -- the open data, for one whose value names it. Optional because
        a device configuration is the same form with none of those in it.
        """
        super().__init__(parent)
        if isinstance(blocks, P.Group):
            blocks = [blocks]
        self.blocks: list[P.Group] = list(blocks)
        self.index: dict[str, P.Group] = P.index(self.blocks)
        self.last_error: str | None = None
        self.fields: dict[str, FieldWidget] = {}
        self._cards: list[tuple[BaseCardWidget, list[str]]] = []
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        for section in P.sections(self.blocks):
            body = QWidget(self)
            form = QFormLayout(body)
            form.setContentsMargins(4, 4, 4, 4)

            paths: list[str] = []
            for path, spec in section.entries:
                field = build_field(spec, body, context or P.FieldContext())
                field.connect(self.on_field_changed)
                self.fields[path] = field
                paths.append(path)
                form.addRow(spec.label, field.widget)

            if cards:
                card = BaseCardWidget(None, section.label, self)
                card.set_toggle_visible(False)
                card.expand_requested.connect(
                    lambda _obj, c=card: c.set_expanded(not c.is_expanded())
                )
                card.set_body_widget(body)
                self._cards.append((card, paths))
                root.addWidget(card)
            else:
                root.addWidget(body)

        root.addStretch(1)
        self.reload()

    # -- blocks <-> form ---------------------------------------------------- #

    def reload(self) -> None:
        """Blocks -> form. Call after something else has written the blocks."""
        self._loading = True
        try:
            for path, field in self.fields.items():
                field.set(P.get_path(self.index, path))
        finally:
            self._loading = False
        self.refresh_summaries()

    def read_into(self, target: Any | None = None) -> Any:
        """Form -> blocks, coercing each value through its field spec.

        Coerces everything before writing anything, so a value that fails its
        bounds leaves every block untouched rather than half-updated.
        """
        lookup = self.index if target is None else P.index(
            [target] if isinstance(target, P.Group) else list(target)
        )
        coerced = {
            path: (field.spec or P.spec_at(lookup, path)).coerce(field.get())
            for path, field in self.fields.items()
        }
        for path, value in coerced.items():
            P.set_path(lookup, path, value)
        return self.blocks if target is None else target

    def on_field_changed(self) -> None:
        """Runs inside a Qt slot, so nothing may escape from here.

        An exception raised in a PyQt slot aborts the process rather than
        unwinding, so a value that fails its bounds must be reported, not
        raised. The widgets normally clamp, but "normally" is not a guarantee
        worth crashing on.
        """
        if self._loading:
            return
        try:
            self.read_into()
        except ParameterError as exc:
            self.last_error = str(exc)
            self.invalid.emit(self.last_error)
            return
        self.last_error = None
        self.refresh_summaries()
        self.changed.emit()

    def refresh_summaries(self) -> None:
        for card, paths in self._cards:
            parts = []
            for path in paths:
                field = self.fields[path]
                spec = field.spec or P.spec_at(self.index, path)
                parts.append(f"{spec.label}: {field.summary()}")
            card.set_description("  |  ".join(parts))
