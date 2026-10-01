"""Modulation: masks driving digital lines during a scan. A mask is bound by
reference to a library entry, so it has its own field type and table widget."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from typing import Any, ClassVar

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.structs.data_library.data import Mask2D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.plugins.params import (
    Editor,
    Field,
    FieldContext,
    Group,
    ParameterError,
    block,
    spec_field,
)


@dataclass(frozen=True)
class Mask:
    """One authored mask wired to one digital output line, by reference.

    The stored value names a library entry; ``source_label`` sits beside the id
    because an id means nothing in metadata read months later. ``array`` is
    filled only in a run's copy, by ``MasksField.resolve``: a program has no
    library, so the pixels arrive with its parameters while the shared block
    and session keep only the reference. ``compare=False`` because comparing
    arrays in a frozen dataclass's ``__eq__`` raises.
    """

    source_id: str = ""
    source_label: str = ""
    array: np.ndarray | None = dc_field(default=None, compare=False)
    port: int = 0
    line: int = 0

    def describe(self) -> str:
        """Which library entry this is, for a row that has lost it."""
        return self.source_label or self.source_id or "no mask"

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this mask drives."""
        return f"{device_name}/port{self.port}/line{self.line}"

    def on_grid(self, height: int, width: int) -> np.ndarray:
        """The masked pixels as booleans, resized nearest-neighbour to ``(height, width)``."""
        # Narrows for the type checker: a run's masks are resolved at start.
        if self.array is None:
            raise ValueError(f"mask '{self.describe()}' was not resolved")
        lit = self.array > 0
        if lit.shape == (height, width):
            return lit
        source_h, source_w = lit.shape
        rows = np.minimum((np.arange(height, dtype=np.int64) * source_h) // height, source_h - 1)
        cols = np.minimum((np.arange(width, dtype=np.int64) * source_w) // width, source_w - 1)
        return lit[np.ix_(rows, cols)]

    def to_dict(self) -> dict[str, Any]:
        """Provenance only: the array is data, and this is a parameter."""
        return {
            "source_id": self.source_id,
            "source_label": self.source_label,
            "port": self.port,
            "line": self.line,
        }

    @classmethod
    def from_value(cls, raw: Any) -> Mask:
        """A mask from the form (already a ``Mask``) or from JSON (a dict)."""
        if isinstance(raw, Mask):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a mask must be an object with source_id/port/line")
        try:
            return cls(
                source_id=str(raw.get("source_id", "")),
                source_label=str(raw.get("source_label", "")),
                port=int(raw.get("port", 0)),
                line=int(raw.get("line", 0)),
            )
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"a mask's port and line must be integers: {exc}") from exc


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


@dataclass(frozen=True)
class MasksField(Field):
    """The Modulation table: library entry, port, line, one row per mask."""

    def coerce(self, value: Any) -> tuple[Mask, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of masks")
        return tuple(Mask.from_value(row) for row in value)

    def encode(self, value: tuple[Mask, ...]) -> Any:
        return [mask.to_dict() for mask in value]

    def resolve(self, value: tuple[Mask, ...], library: Library) -> tuple[Mask, ...]:
        """Each binding with its pixels. One whose entry is not open refuses the
        run: acquiring without a mask the user bound is worse than not acquiring."""
        out: list[Mask] = []
        for mask in value:
            dataset = library.by_id(mask.source_id)
            array = dataset.latest() if dataset is not None else None
            if array is None:
                raise ParameterError(f"mask '{mask.describe()}' is not open")
            out.append(replace(mask, array=array))
        return tuple(out)

    def editor(self, parent: Any, context: FieldContext) -> Editor:
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
            spec=self,
        )


@block
@dataclass
class ModulationGroup(Group):
    label: ClassVar[str] = "Modulation"

    masks: tuple[Mask, ...] = spec_field(
        None,
        MasksField(
            "Masks",
            "Masks driving digital output lines during the scan. "
            "Draw one in the Mask Editor to add it here",
        ),
        factory=tuple,
    )
