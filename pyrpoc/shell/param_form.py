"""A form generated from parameter blocks, writing back into them.

The Qt half of v3.0's ``backend_utils/parameter_utils.py``. What that module did
in one class per field -- definition, coercion, widget, get, set, connect -- is
split: ``core/params.py`` holds the definition and coercion, this holds the
widget.

The blocks are authoritative. Every widget change writes straight back into the
block instance, so nothing has to scrape the form at play time and anything
other than the form can parameterise a run. Because a block instance is shared
by every modality that declares it, that write is also how the value reaches
the other modalities.

One generator serves the acquisition form and the device panels both, so adding
a field to a device config adds its row with no panel edit.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    QHeaderView,
    QLabel,
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
from pyrpoc.programs.components import Mask, MasksField, Point, PointField

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
    #: The spec this widget was built from. Carried rather than looked up:
    #: resolving it per path per keystroke was quadratic in field count.
    spec: P.Field | None = None

    #: Subscribe to "the user asked to pick this value off a display". Set only
    #: by widgets that can be filled by something outside the form; every other
    #: builder leaves it None and the form skips it.
    arm: Callable[[Callable[[bool], None]], None] | None = None

    #: Show or clear the armed state. The reverse of ``arm``, and the reason
    #: there are two hooks rather than one: the application disarms after a
    #: pick, so the widget has to be told, not just asked.
    set_armed: Callable[[bool], None] | None = None


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

    def add_row(self, binding: Mask) -> None:
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
        self.add_row(Mask(Path(path)))
        self.changed.emit()

    def on_remove_clicked(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)
        self.changed.emit()

    # -- value ------------------------------------------------------------- #

    def value(self) -> tuple[Mask, ...]:
        out: list[Mask] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            text = (item.text() if item is not None else "").strip()
            if not text:
                continue
            port = self.table.cellWidget(row, 1)
            line = self.table.cellWidget(row, 2)
            out.append(
                Mask(
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
# The point picker -- two volts and a way to pick them off a display          #
# --------------------------------------------------------------------------- #


#: Shown under the spin boxes when no point has been set yet.
NO_ORIGIN = "—"


class PointPicker(QWidget):
    """Galvo volts, typed or picked off an image.

    The button is a widget affordance, not a parameter: which is the whole
    reason arming is not a field. ``Browse...`` on a path field is the same
    idea -- press it, something outside the form temporarily takes over to fill
    one value, it ends. Nothing about the button is validated, encoded or
    persisted, so the application can never come back from a relaunch armed at
    hardware.

    It is checkable because arming outlives the press: the click that fills it
    happens somewhere else entirely. ``set_armed`` exists for that reason -- the
    application decides when arming ends, and says so.
    """

    changed = pyqtSignal()
    arm_requested = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._point = Point()
        #: True while ``set_value`` is driving the spin boxes, so a programmatic
        #: write keeps the provenance a hand edit is supposed to clear.
        self._programmatic = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        volts = QHBoxLayout()
        volts.setContentsMargins(0, 0, 0, 0)
        self.fast_spin = self.build_spin("Fast axis (X) volts")
        self.slow_spin = self.build_spin("Slow axis (Y) volts")
        volts.addWidget(QLabel("X", self))
        volts.addWidget(self.fast_spin, 1)
        volts.addWidget(QLabel("Y", self))
        volts.addWidget(self.slow_spin, 1)
        root.addLayout(volts)

        self.pick_btn = QPushButton("\u2295 Acquire at point\u2026", self)
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip(
            "Arm, then click a point on an image to park the galvos there and "
            "acquire. Clicking moves hardware."
        )
        root.addWidget(self.pick_btn)

        self.origin_label = QLabel(f"from: {NO_ORIGIN}", self)
        self.origin_label.setEnabled(False)
        root.addWidget(self.origin_label)

        self.fast_spin.valueChanged.connect(self.on_spin_changed)
        self.slow_spin.valueChanged.connect(self.on_spin_changed)
        self.pick_btn.toggled.connect(self.arm_requested.emit)

    def build_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.01)
        spin.setSuffix(" V")
        spin.setToolTip(tooltip)
        return spin

    # -- value -------------------------------------------------------------- #

    def value(self) -> Point:
        return Point(
            self.fast_spin.value(),
            self.slow_spin.value(),
            self._point.source_id,
            self._point.source_label,
            self._point.pixel_x,
            self._point.pixel_y,
        )

    def set_value(self, value: Any) -> None:
        """Blocks -> widget. Keeps the provenance the incoming point carries."""
        point = Point.from_dict(value)
        self._point = point
        self._programmatic = True
        try:
            self.fast_spin.setValue(point.fast_v)
            self.slow_spin.setValue(point.slow_v)
        finally:
            self._programmatic = False
        origin = point.describe() if point.picked else NO_ORIGIN
        self.origin_label.setText(f"from: {origin}")

    def on_spin_changed(self) -> None:
        """A hand edit is no longer the pixel it was picked from, so say so.

        ``changed`` is emitted either way. A programmatic set only ever happens
        inside ``ParamForm.reload``, whose ``_loading`` guard swallows it, so
        emitting unconditionally costs nothing and means a typed value can never
        be the one case that fails to reach the block.
        """
        if not self._programmatic:
            self._point = Point(self.fast_spin.value(), self.slow_spin.value())
            self.origin_label.setText("from: typed")
        self.changed.emit()

    def summary(self) -> str:
        return f"{self.fast_spin.value():.3f} / {self.slow_spin.value():.3f} V"

    # -- arming ------------------------------------------------------------- #

    def set_armed(self, active: bool) -> None:
        """Reflect the application's arming state without asking for a change."""
        if self.pick_btn.isChecked() == bool(active):
            return
        self.pick_btn.blockSignals(True)
        self.pick_btn.setChecked(bool(active))
        self.pick_btn.blockSignals(False)


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


def build_masks(spec: MasksField, parent) -> FieldWidget:
    table = MaskTable(parent)
    return FieldWidget(
        table,
        get=table.value,
        set=table.set_value,
        connect=lambda cb: table.changed.connect(lambda *_: cb()),
        summary=table.summary,
    )


def build_point(spec: PointField, parent) -> FieldWidget:
    del spec
    picker = PointPicker(parent)

    # Statement bodies rather than lambdas: ``connect`` returns a Connection,
    # and the hooks are declared as returning None.
    def connect(cb: Callable[[], None]) -> None:
        picker.changed.connect(lambda *_: cb())

    def arm(cb: Callable[[bool], None]) -> None:
        picker.arm_requested.connect(cb)

    return FieldWidget(
        picker,
        get=picker.value,
        set=picker.set_value,
        connect=connect,
        summary=picker.summary,
        arm=arm,
        set_armed=picker.set_armed,
    )


BUILDERS: dict[type, Callable[[Any, QWidget], FieldWidget]] = {
    P.IntField: build_int,
    P.FloatField: build_float,
    P.TextField: build_text,
    P.PathField: build_path,
    P.BoolField: build_bool,
    P.ChoiceField: build_choice,
    P.ChannelsField: build_channels,
    MasksField: build_masks,
    PointField: build_point,
}


def build_field(spec: P.Field, parent: QWidget) -> FieldWidget:
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
    #: A field asked to be filled from outside the form. Bubbled rather than
    #: handled, exactly as ``changed`` is: the form does not know what a pick
    #: is, only that a widget offered one.
    pick_armed = pyqtSignal(bool)

    def __init__(
        self,
        blocks: Sequence[P.Group] | P.Group,
        parent: QWidget | None = None,
        *,
        cards: bool = True,
    ):
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
                field = build_field(spec, body)
                field.connect(self.on_field_changed)
                if field.arm is not None:
                    field.arm(self.pick_armed.emit)
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

    def show_pick_armed(self, active: bool) -> None:
        """Push the armed state down to whichever field can be picked.

        Fans out over every field because the form has no reason to know which
        one it is; builders that left ``set_armed`` unset are skipped.
        """
        for field in self.fields.values():
            if field.set_armed is not None:
                field.set_armed(active)

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
