"""Widgets for the field types declared in ``param_groups``, handed to the form
through ``Field.editor``. The one Qt module in programs/, imported only when an
editor is asked for, so programs import without Qt."""

from __future__ import annotations

from dataclasses import dataclass

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

from pyrpoc.structs.data import Dataset, Library, Mask2D
from pyrpoc.structs.params import Editor, FieldContext

from .param_groups import Mask, MasksField, Point, PointField

CLOSED_SUFFIX = " (closed)"


class MaskSourceCombo(QComboBox):
    """Picks a ``Mask2D`` entry out of the open data.

    Filled when opened rather than from a library subscription: the form is
    rebuilt on every program change, and a stale callback into a deleted row
    would raise inside the library's notify, which runs at run start.
    """

    def __init__(self, table: MaskTable, parent: QWidget):
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
        if chosen and not found:
            # Closed in the data panel: keep naming it rather than silently
            # rebinding the row to an unrelated mask.
            self.addItem(f"{label}{CLOSED_SUFFIX}", chosen)
            self.setCurrentIndex(self.count() - 1)
        self.blockSignals(False)
        super().showPopup()


@dataclass
class MaskRow:
    source: MaskSourceCombo
    port: QSpinBox
    line: QSpinBox


class MaskTable(QWidget):
    """A mask and the port and line it drives, one row per binding. A row names
    a library entry, so the run's metadata records exactly which masks drove
    which lines."""

    changed = pyqtSignal()

    def __init__(self, library: Library, parent: QWidget):
        super().__init__(parent)
        self.library = library
        self.rows: list[MaskRow] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Mask", "Port", "Line"])
        rows = self.table.verticalHeader()
        header = self.table.horizontalHeader()
        # Narrows for the type checker: a table builds both headers itself.
        if rows is None or header is None:
            raise RuntimeError("QTableWidget was built without headers")
        rows.setVisible(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(90)
        root.addWidget(self.table)
        root.addLayout(self.build_buttons())
        self.hint = QLabel("Draw a mask in the Mask Editor to bind one here.", self)
        self.hint.setStyleSheet("color: palette(mid);")
        root.addWidget(self.hint)

    def build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        add_btn = QPushButton("Add mask", self)
        remove_btn = QPushButton("Remove", self)
        add_btn.clicked.connect(self.on_add_clicked)
        remove_btn.clicked.connect(self.on_remove_clicked)
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        row.addStretch(1)
        return row

    def sources(self) -> list[Dataset]:
        """The mask entries currently open, newest first."""
        return self.library.matching(Mask2D)

    def add_row(self, binding: Mask) -> None:
        index = self.table.rowCount()
        self.table.insertRow(index)
        source = MaskSourceCombo(self, self.table)
        if binding.source_id:
            source.addItem(binding.source_label or binding.source_id, binding.source_id)
        else:
            for dataset in self.sources():
                source.addItem(dataset.label, dataset.id)
        source.currentIndexChanged.connect(self.changed)
        self.table.setCellWidget(index, 0, source)

        spins = []
        for column, value in ((1, binding.port), (2, binding.line)):
            spin = QSpinBox(self.table)
            spin.setRange(0, 1024)
            spin.setValue(value)
            spin.valueChanged.connect(self.changed)
            self.table.setCellWidget(index, column, spin)
            spins.append(spin)
        self.rows.append(MaskRow(source, *spins))
        self.refresh_hint()

    def next_free_line(self) -> int:
        """The lowest line no row drives, so two rows never silently share one."""
        used = {row.line.value() for row in self.rows}
        line = 0
        while line in used:
            line += 1
        return line

    def next_source(self) -> Dataset | None:
        """The newest mask no row has taken, or the newest if all are taken, so a
        second row does not default to the first row's mask."""
        taken = {row.source.currentData() for row in self.rows}
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
        index = self.table.currentRow()
        if index < 0:
            return
        self.table.removeRow(index)
        del self.rows[index]
        self.refresh_hint()
        self.changed.emit()

    def refresh_hint(self) -> None:
        self.hint.setVisible(not self.rows)

    def value(self) -> tuple[Mask, ...]:
        """Every row that names a mask, as a reference. A row whose entry was
        closed is kept: the run resolves it and refuses to start without it."""
        out: list[Mask] = []
        for row in self.rows:
            dataset_id = row.source.currentData()
            if not dataset_id:
                continue
            dataset = self.library.by_id(dataset_id)
            label = dataset.label if dataset is not None else row.source.currentText()
            out.append(
                Mask(
                    source_id=dataset_id,
                    source_label=label.removesuffix(CLOSED_SUFFIX),
                    port=row.port.value(),
                    line=row.line.value(),
                )
            )
        return tuple(out)

    def set_value(self, bindings: tuple[Mask, ...]) -> None:
        self.table.setRowCount(0)
        self.rows = []
        for binding in bindings:
            self.add_row(binding)
        self.refresh_hint()

    def summary(self) -> str:
        """How many masks will apply, and separately how many rows name a
        closed entry, since those will stop the run."""
        if not self.rows:
            return "none"
        live = sum(1 for mask in self.value() if self.library.by_id(mask.source_id))
        closed = len(self.rows) - live
        text = f"{live} mask" + ("" if live == 1 else "s")
        return text + (f", {closed} closed" if closed else "")


# Shown under the spin boxes when no point has been set yet.
NO_ORIGIN = "—"


class PointPicker(QWidget):
    """Galvo volts, typed or picked off an image, and where they came from.
    Picking is a runner's job; this shows the result when the form reloads."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._point = Point()
        # True while ``set_value`` drives the spin boxes, so a programmatic
        # write keeps the provenance a hand edit clears.
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

    def value(self) -> Point:
        return Point(
            self.fast_spin.value(),
            self.slow_spin.value(),
            self._point.source_id,
            self._point.source_label,
            self._point.pixel_x,
            self._point.pixel_y,
        )

    def set_value(self, point: Point) -> None:
        """Blocks -> widget, keeping the provenance the point carries."""
        self._point = point
        self._programmatic = True
        try:
            self.fast_spin.setValue(point.fast_v)
            self.slow_spin.setValue(point.slow_v)
        finally:
            self._programmatic = False
        self.origin_label.setText(f"from: {point.describe() if point.picked else NO_ORIGIN}")

    def on_spin_changed(self) -> None:
        """A hand edit is no longer the pixel it was picked from, so say so."""
        if not self._programmatic:
            self._point = Point(self.fast_spin.value(), self.slow_spin.value())
            self.origin_label.setText("from: typed")
        self.changed.emit()

    def summary(self) -> str:
        return f"{self.fast_spin.value():.3f} / {self.slow_spin.value():.3f} V"


def mask_editor(spec: MasksField, parent: QWidget, context: FieldContext) -> Editor:
    # Narrows for the type checker: every form holding masks offers the library.
    if context.library is None:
        raise ValueError("a masks field needs the open data")
    table = MaskTable(context.library, parent)
    return Editor(
        table,
        get=table.value,
        set=table.set_value,
        connect=lambda cb: table.changed.connect(cb),
        summary=table.summary,
        spec=spec,
    )


def point_editor(spec: PointField, parent: QWidget) -> Editor:
    picker = PointPicker(parent)
    return Editor(
        picker,
        get=picker.value,
        set=picker.set_value,
        connect=lambda cb: picker.changed.connect(cb),
        summary=picker.summary,
        spec=spec,
    )
