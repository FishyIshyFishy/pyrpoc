"""Widgets for the field types declared in ``param_groups``.

The form draws the generic fields itself and asks any other field for its own
editor through ``Field.editor``; these are the answers for masks and points.
They live beside the fields they edit so the form needs nothing but ``structs``,
and ``param_groups`` imports this module only when an editor is asked for, so a
program can still be imported with no Qt in sight.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.src.structs.data import Mask2D
from pyrpoc.src.structs.params import Editor, FieldContext

from .param_groups import Mask, Point

# --------------------------------------------------------------------------- #
# The mask table -- the replacement for the whole optocontrol subsystem        #
# --------------------------------------------------------------------------- #


class MaskSourceCombo(QComboBox):
    """Picks a ``Mask2D`` entry out of the open data.

    Repopulated in ``showPopup`` rather than from a library subscription, and
    that is deliberate. The form is rebuilt on every modality change, and the
    library notifies its subscribers from inside ``Executor.start``; a stale
    callback into a deleted row would raise there, taking out the run that was
    starting. A list built when it is opened cannot go stale.
    """

    def __init__(self, table: MaskTable, parent: QWidget | None = None):
        super().__init__(parent)
        self._table = table
        self.setToolTip("A mask drawn in the Mask Editor. Add one there to see it here.")

    def showPopup(self) -> None:
        chosen = self.currentData()
        label = self.currentText()
        self.blockSignals(True)
        self.clear()
        found = False
        for dataset in self._table.sources():
            self.addItem(dataset.label, dataset.id)
            if dataset.id == chosen:
                self.setCurrentIndex(self.count() - 1)
                found = True
        if isinstance(chosen, str) and chosen and not found:
            # The entry was closed in the data panel. Keep naming it rather than
            # silently rebinding the row to an unrelated mask.
            self.addItem(f"{label} (closed)", chosen)
            self.setCurrentIndex(self.count() - 1)
        self.blockSignals(False)
        super().showPopup()


class MaskTable(QWidget):
    """A mask, and the port and line it drives -- one row per binding.

    Replaces BaseOptoControl, BaseOptoControlWidget, the optocontrol registry
    and manager panel, prepare_for_acquisition, get_context, MaskContext,
    extract_mask_contexts, allowed_optocontrols and the optocontrols list in
    AppState. Masks stop being globally toggleable objects and become run
    parameters, so a run's saved metadata records exactly which masks were
    applied on which lines -- which v3.0 did not, because the enabled flag lived
    outside the modality.

    A row names a dataset rather than a file. The mask editor files what it draws
    in the library and this asks the library what it holds, so neither knows the
    other exists and drawing a mask no longer means saving a PNG and browsing
    back to it.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # The open data, from the form's ``FieldContext``. Duck-typed: the
        # library is the app's, and programs/ does not import app/.
        self._library: Any = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Mask", "Port", "Line"])
        # A table builds both headers in its constructor; PyQt types them as
        # optional regardless.
        rows = self.table.verticalHeader()
        header = self.table.horizontalHeader()
        if rows is None or header is None:
            raise RuntimeError("QTableWidget was built without headers")
        rows.setVisible(False)
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

        self.hint = QLabel("Draw a mask in the Mask Editor to bind one here.", self)
        self.hint.setStyleSheet("color: palette(mid);")
        root.addWidget(self.hint)

    # -- the open data ----------------------------------------------------- #

    def attach_library(self, library: Any) -> None:
        self._library = library

    def is_open(self, dataset_id: str) -> bool:
        return self._library is not None and self._library.by_id(dataset_id) is not None

    def sources(self) -> list:
        """The mask entries currently open, newest first."""
        if self._library is None:
            return []
        return self._library.matching(Mask2D)

    # -- rows -------------------------------------------------------------- #

    def add_row(self, binding: Mask) -> None:
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)

        combo = MaskSourceCombo(self, self.table)
        if binding.source_id:
            combo.addItem(binding.source_label or binding.source_id, binding.source_id)
        else:
            for dataset in self.sources():
                combo.addItem(dataset.label, dataset.id)
        combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.table.setCellWidget(row, 0, combo)

        for column, value in ((1, binding.port), (2, binding.line)):
            spin = QSpinBox(self.table)
            spin.setRange(0, 1024)
            spin.setValue(int(value))
            spin.valueChanged.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(row, column, spin)
        self.table.blockSignals(False)
        self.refresh_hint()

    def used_lines(self) -> set[int]:
        lines: set[int] = set()
        for row in range(self.table.rowCount()):
            spin = self.table.cellWidget(row, 2)
            if isinstance(spin, QSpinBox):
                lines.add(int(spin.value()))
        return lines

    def next_free_line(self) -> int:
        """The lowest line no row is already driving.

        Defaulting every row to line 0 would leave two masks fighting over one
        output with nothing on screen saying so.
        """
        used = self.used_lines()
        line = 0
        while line in used:
            line += 1
        return line

    def used_sources(self) -> set[str]:
        taken: set[str] = set()
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            chosen = combo.currentData() if isinstance(combo, QComboBox) else None
            if isinstance(chosen, str) and chosen:
                taken.add(chosen)
        return taken

    def next_source(self):
        """The newest mask no row has taken yet, or the newest if all are taken.

        The same courtesy as ``next_free_line`` and needed for the same reason:
        a second row defaulting to the mask the first row already has looks like
        two bindings and behaves like one.
        """
        taken = self.used_sources()
        available = self.sources()
        for dataset in available:
            if dataset.id not in taken:
                return dataset
        return available[0] if available else None

    def on_add_clicked(self) -> None:
        newest = self.next_source()
        self.add_row(
            Mask(
                source_id=newest.id if newest is not None else "",
                source_label=newest.label if newest is not None else "",
                port=0,
                line=self.next_free_line(),
            )
        )
        self.changed.emit()

    def on_remove_clicked(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)
        self.refresh_hint()
        self.changed.emit()

    def refresh_hint(self) -> None:
        self.hint.setVisible(self.table.rowCount() == 0)

    # -- value ------------------------------------------------------------- #

    def value(self) -> tuple[Mask, ...]:
        """Every row that names a mask, as a reference.

        A row whose entry has been closed is kept: the binding is the user's,
        and dropping it silently would run without a mask they asked for. The
        run resolves the reference and refuses to start on one that is gone.
        """
        out: list[Mask] = []
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            dataset_id = combo.currentData() if isinstance(combo, QComboBox) else None
            if not isinstance(dataset_id, str) or not dataset_id:
                continue
            label = combo.currentText() if isinstance(combo, QComboBox) else ""
            dataset = self._library.by_id(dataset_id) if self._library is not None else None
            if dataset is not None:
                label = dataset.label
            elif label.endswith(" (closed)"):
                label = label[: -len(" (closed)")]
            port = self.table.cellWidget(row, 1)
            line = self.table.cellWidget(row, 2)
            out.append(
                Mask(
                    source_id=dataset_id,
                    source_label=label,
                    port=port.value() if isinstance(port, QSpinBox) else 0,
                    line=line.value() if isinstance(line, QSpinBox) else 0,
                )
            )
        return tuple(out)

    def set_value(self, bindings) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        for binding in bindings or ():
            self.add_row(binding)
        self.refresh_hint()

    def summary(self) -> str:
        """How many masks will be applied, and how many rows will not be.

        A row whose entry was closed in the data panel will stop the run.
        Counting only the rows would hide that, so the two numbers are reported
        separately.
        """
        rows = self.table.rowCount()
        if rows == 0:
            return "none"
        live = sum(1 for mask in self.value() if self.is_open(mask.source_id))
        text = f"{live} mask" + ("" if live == 1 else "s")
        closed = rows - live
        return text + (f", {closed} closed" if closed else "")


# --------------------------------------------------------------------------- #
# The point editor -- two volts and where they came from                       #
# --------------------------------------------------------------------------- #


# Shown under the spin boxes when no point has been set yet.
NO_ORIGIN = "—"


class PointPicker(QWidget):
    """Galvo volts, typed or picked off an image, and where they came from.

    A plain editor. Picking a point off an image is an entry point of the
    program that wants one -- an ``ArmAndRun`` runner -- not something this
    widget does; it shows the result when the form reloads.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._point = Point()
        # True while ``set_value`` is driving the spin boxes, so a programmatic
        # write keeps the provenance a hand edit is supposed to clear.
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

        self.origin_label = QLabel(f"from: {NO_ORIGIN}", self)
        self.origin_label.setEnabled(False)
        root.addWidget(self.origin_label)

        self.fast_spin.valueChanged.connect(self.on_spin_changed)
        self.slow_spin.valueChanged.connect(self.on_spin_changed)

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


# --------------------------------------------------------------------------- #
# Editors, as ``Field.editor`` hands them to the form                         #
# --------------------------------------------------------------------------- #


def mask_editor(parent: Any, context: FieldContext) -> Editor:
    table = MaskTable(parent)
    table.attach_library(context.library)
    return Editor(
        table,
        get=table.value,
        set=table.set_value,
        connect=lambda cb: table.changed.connect(lambda *_: cb()),
        summary=table.summary,
    )


def point_editor(parent: Any, context: FieldContext) -> Editor:
    del context
    picker = PointPicker(parent)

    # Statement bodies rather than lambdas: ``connect`` returns a Connection,
    # and the hooks are declared as returning None.
    def connect(cb: Callable[[], None]) -> None:
        picker.changed.connect(lambda *_: cb())

    return Editor(
        picker,
        get=picker.value,
        set=picker.set_value,
        connect=connect,
        summary=picker.summary,
    )
