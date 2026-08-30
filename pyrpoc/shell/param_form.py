"""A form generated from a parameter model, writing back into it.

The Qt half of v3.0's ``backend_utils/parameter_utils.py``. What that module did
in one class per field -- definition, coercion, widget, get, set, connect -- is
split: ``core/params.py`` holds the definition and coercion, this holds the
widget.

The model is authoritative. Every widget change writes straight back into it, so
nothing has to scrape the form at play time and anything other than the form can
parameterise a run. That is settled statement 3.

One generator serves both the acquisition form and the device panels, so adding
a field to a device config adds its row with no panel edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core import params as P
from pyrpoc.core.errors import ParameterError
from pyrpoc.core.modulation import MaskBinding

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


@dataclass
class FieldWidget:
    widget: QWidget
    get: Callable[[], Any]
    set: Callable[[Any], None]
    connect: Callable[[Callable[[], None]], None]
    summary: Callable[[], str]


# --------------------------------------------------------------------------- #
# The mask table -- the replacement for the whole optocontrol subsystem        #
# --------------------------------------------------------------------------- #


class MaskTable(QWidget):
    """Mask file, port, line -- one row per binding.

    Replaces BaseOptoControl, BaseOptoControlWidget, the optocontrol registry
    and manager panel, prepare_for_acquisition, get_context, MaskContext,
    extract_mask_contexts, allowed_optocontrols and the optocontrols list in
    AppState. Masks stop being globally toggleable objects and become run
    parameters, so a run's saved metadata records exactly which masks were
    applied on which lines -- which v3.0 did not, because the enabled flag lived
    outside the modality.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Mask file", "Port", "Line"])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(90)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())
        root.addWidget(self.table)

        buttons = QHBoxLayout()
        self.add_btn = QPushButton("Add mask", self)
        self.remove_btn = QPushButton("Remove", self)
        self.add_btn.clicked.connect(self.on_add_clicked)
        self.remove_btn.clicked.connect(self.on_remove_clicked)
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.remove_btn)
        buttons.addStretch(1)
        root.addLayout(buttons)

    # -- rows -------------------------------------------------------------- #

    def add_row(self, binding: MaskBinding) -> None:
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(binding.path)))

        for column, value in ((1, binding.port), (2, binding.line)):
            spin = QSpinBox(self.table)
            spin.setRange(0, 1024)
            spin.setValue(int(value))
            spin.valueChanged.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(row, column, spin)
        self.table.blockSignals(False)

    def on_add_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a mask file", "", "Images (*.png *.tif *.tiff *.bmp);;All Files (*)"
        )
        if not path:
            return
        self.add_row(MaskBinding(Path(path)))
        self.changed.emit()

    def on_remove_clicked(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)
        self.changed.emit()

    # -- value ------------------------------------------------------------- #

    def value(self) -> tuple[MaskBinding, ...]:
        out: list[MaskBinding] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            text = (item.text() if item is not None else "").strip()
            if not text:
                continue
            port = self.table.cellWidget(row, 1)
            line = self.table.cellWidget(row, 2)
            out.append(
                MaskBinding(
                    Path(text),
                    port.value() if isinstance(port, QSpinBox) else 0,
                    line.value() if isinstance(line, QSpinBox) else 0,
                )
            )
        return tuple(out)

    def set_value(self, bindings) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        for binding in bindings or ():
            self.add_row(binding)

    def summary(self) -> str:
        count = self.table.rowCount()
        if count == 0:
            return "none"
        return f"{count} mask" + ("" if count == 1 else "s")


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


def build_masks(spec: P.MasksField, parent) -> FieldWidget:
    table = MaskTable(parent)
    return FieldWidget(
        table,
        get=table.value,
        set=table.set_value,
        connect=lambda cb: table.changed.connect(lambda *_: cb()),
        summary=table.summary,
    )


BUILDERS: dict[type, Callable[[Any, QWidget], FieldWidget]] = {
    P.IntField: build_int,
    P.FloatField: build_float,
    P.TextField: build_text,
    P.PathField: build_path,
    P.BoolField: build_bool,
    P.ChoiceField: build_choice,
    P.ChannelsField: build_channels,
    P.MasksField: build_masks,
}


def build_field(spec: P.Field, parent: QWidget) -> FieldWidget:
    builder = BUILDERS.get(type(spec))
    if builder is None:
        raise TypeError(f"no widget for {type(spec).__name__}")
    field = builder(spec, parent)
    if spec.tooltip:
        field.widget.setToolTip(spec.tooltip)
    return field


# --------------------------------------------------------------------------- #
# The form                                                                     #
# --------------------------------------------------------------------------- #


class ParamForm(QWidget):
    """Sections as collapsible cards, generated from a parameter model."""

    changed = pyqtSignal()
    invalid = pyqtSignal(str)

    def __init__(self, model: Any, parent: QWidget | None = None, *, cards: bool = True):
        super().__init__(parent)
        self.model = model
        self.last_error: str | None = None
        self.fields: dict[str, FieldWidget] = {}
        self._cards: list[tuple[BaseCardWidget, list[str]]] = []
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        for section in P.sections(model):
            body = QWidget(self)
            form = QFormLayout(body)
            form.setContentsMargins(4, 4, 4, 4)

            paths: list[str] = []
            for path, spec in section.entries:
                field = build_field(spec, body)
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
        self.write_from(model)

    # -- model <-> form ----------------------------------------------------- #

    def write_from(self, model: Any) -> None:
        """Model -> form."""
        self.model = model
        self._loading = True
        try:
            for path, field in self.fields.items():
                field.set(P.get_path(model, path))
        finally:
            self._loading = False
        self.refresh_summaries()

    def read_into(self, model: Any | None = None) -> Any:
        """Form -> model, coercing each value through its field spec."""
        target = model if model is not None else self.model
        for path, field in self.fields.items():
            spec = P.spec_at(target, path)
            P.set_path(target, path, spec.coerce(field.get()))
        return target

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
                spec = P.spec_at(self.model, path)
                parts.append(f"{spec.label}: {self.fields[path].summary()}")
            card.set_description("  |  ".join(parts))
