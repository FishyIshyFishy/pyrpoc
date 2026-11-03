"""Modulation: masks that drive digital lines during a scan, one row per mask
with the port and line it drives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.mask import Mask
from pyrpoc.core.program_inputs import Editor, Field, ParameterError, declare

from .mask_cell import MaskCell


@dataclass(frozen=True, eq=False)
class MaskBinding:
    """One mask wired to one digital output line. A row with no mask drives nothing."""

    mask: Mask | None
    port: int
    line: int

    @property
    def line_name(self) -> str:
        """The port and line it drives, as a mask is placed by them."""
        return f"port{self.port}/line{self.line}"

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this mask drives."""
        return f"{device_name}/{self.line_name}"

    def on_grid(self, height: int, width: int) -> np.ndarray:
        """The lit pixels on a ``(height, width)`` grid; none for a row with no mask."""
        if self.mask is None:
            return np.zeros((height, width), dtype=bool)
        return self.mask.on_grid(height, width)

    def to_dict(self) -> dict[str, Any]:
        mask = None if self.mask is None else self.mask.to_dict()
        return {"mask": mask, "port": self.port, "line": self.line}

    @classmethod
    def from_value(cls, raw: Any) -> MaskBinding:
        """A binding from the form (already a ``MaskBinding``) or from JSON (a dict)."""
        if isinstance(raw, MaskBinding):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a mask row must be an object with a mask, port and line")
        try:
            port, line = int(raw.get("port", 0)), int(raw.get("line", 0))
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"a mask's port and line must be integers: {exc}") from exc
        # A row saved before masks were held inline names a library entry, which
        # did not outlive its session: it comes back without a mask.
        mask = raw.get("mask")
        return cls(None if mask is None else Mask.from_dict(mask), port, line)


@dataclass
class MaskRow:
    cell: MaskCell
    port: QSpinBox
    line: QSpinBox


class MaskTable(QWidget):
    """A mask and the port and line it drives, one row per binding."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
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
        self.hint = QLabel("Draw a mask in a Mask Editor and choose a line there.", self)
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

    def add_row(self, binding: MaskBinding) -> None:
        index = self.table.rowCount()
        self.table.insertRow(index)
        cell = MaskCell("no mask", self.table)
        cell.set_value(binding.mask)
        cell.changed.connect(self.changed)
        self.table.setCellWidget(index, 0, cell)

        spins = []
        for column, value in ((1, binding.port), (2, binding.line)):
            spin = QSpinBox(self.table)
            spin.setRange(0, 1024)
            spin.setValue(value)
            spin.valueChanged.connect(self.changed)
            self.table.setCellWidget(index, column, spin)
            spins.append(spin)
        self.rows.append(MaskRow(cell, *spins))
        self.table.resizeRowsToContents()
        self.refresh_hint()

    def on_add_clicked(self) -> None:
        self.add_row(MaskBinding(None, 0, next_free_line(self.value())))
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

    def value(self) -> tuple[MaskBinding, ...]:
        return tuple(
            MaskBinding(row.cell.value(), row.port.value(), row.line.value()) for row in self.rows
        )

    def set_value(self, bindings: tuple[MaskBinding, ...]) -> None:
        self.table.setRowCount(0)
        self.rows = []
        for binding in bindings:
            self.add_row(binding)
        self.refresh_hint()

    def summary(self) -> str:
        """How many masks will drive a line, and how many rows have none."""
        if not self.rows:
            return "none"
        live = sum(1 for row in self.rows if row.cell.value() is not None)
        empty = len(self.rows) - live
        text = f"{live} mask" + ("" if live == 1 else "s")
        return text + (f", {empty} empty" if empty else "")


@dataclass(frozen=True)
class MasksField(Field):
    """The Modulation table: a mask, its port and its line, one row per mask. A mask is
    placed by the line it drives: an existing row's, or the next free line's."""

    accepts = Mask

    def coerce(self, value: Any) -> tuple[MaskBinding, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of masks")
        return tuple(MaskBinding.from_value(row) for row in value)

    def encode(self, value: tuple[MaskBinding, ...]) -> Any:
        return [binding.to_dict() for binding in value]

    def slots(self, value: tuple[MaskBinding, ...]) -> tuple[str, ...]:
        free = MaskBinding(None, 0, next_free_line(value))
        return (*(binding.line_name for binding in value), free.line_name)

    def place(
        self, value: tuple[MaskBinding, ...], slot: str, item: Mask | None
    ) -> tuple[MaskBinding, ...]:
        """``value`` with the row driving line ``slot`` holding ``item``, adding the row if
        no row drives it yet."""
        rows = [replace(b, mask=item) if b.line_name == slot else b for b in value]
        if all(binding.line_name != slot for binding in value):
            port, line = slot.removeprefix("port").split("/line")
            rows.append(MaskBinding(item, int(port), int(line)))
        return tuple(rows)

    def editor(self, parent: Any) -> Editor:
        table = MaskTable(parent)
        return Editor(
            table,
            get=table.value,
            set=table.set_value,
            connect=lambda cb: table.changed.connect(cb),
            summary=table.summary,
            spec=self,
        )


MASKS = declare(
    "modulation.masks",
    MasksField(
        "Modulation masks",
        "Masks driving digital output lines during the scan. "
        "Draw one in a Mask Editor and choose its line there",
    ),
    (),
)

MODULATION_SECTION = InputSection("Modulation", (MASKS,))


def next_free_line(bindings: tuple[MaskBinding, ...]) -> int:
    """The lowest line on port 0 no row drives."""
    # So two rows never silently share one.
    used = {binding.line for binding in bindings if binding.port == 0}
    line = 0
    while line in used:
        line += 1
    return line
