"""Modulation: each drawn mask drives one digital line during a scan, on the port and line
set beside it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QLabel, QSpinBox, QWidget

from pyrpoc.core.mask import MASKS, Mask
from pyrpoc.core.program_inputs import (
    Editor,
    Field,
    FieldValues,
    ParameterError,
    declare,
)

# Wide enough for four digits, so a port or line box stays small beside the mask's name.
SPIN_WIDTH = 56


@dataclass(frozen=True)
class MaskLine:
    """The port and line one drawn mask drives, by the mask's id."""

    mask_id: str
    port: int
    line: int

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this line is."""
        return f"{device_name}/port{self.port}/line{self.line}"

    def to_dict(self) -> dict[str, Any]:
        return {"mask_id": self.mask_id, "port": self.port, "line": self.line}

    @classmethod
    def from_value(cls, raw: Any) -> MaskLine:
        """A line from the form (already a ``MaskLine``) or from JSON (a dict)."""
        if isinstance(raw, MaskLine):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a mask's line must be an object with a mask id, port and line")
        try:
            return cls(str(raw["mask_id"]), int(raw["port"]), int(raw["line"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ParameterError(f"a mask's port and line must be integers: {exc}") from exc


def lines_for(masks: tuple[Mask, ...], lines: tuple[MaskLine, ...]) -> tuple[MaskLine, ...]:
    """The line each mask drives, in the masks' order: the one set for it, or else the lowest
    line on port 0 no other mask drives."""
    # The form shows exactly this, so a mask drives the line the user sees beside it.
    present = {mask.id for mask in masks}
    chosen = {line.mask_id: line for line in lines if line.mask_id in present}
    used = {(line.port, line.line) for line in chosen.values()}
    out = []
    for mask in masks:
        if mask.id not in chosen:
            free = next(line for line in range(len(used) + 1) if (0, line) not in used)
            used.add((0, free))
            chosen[mask.id] = MaskLine(mask.id, 0, free)
        out.append(chosen[mask.id])
    return tuple(out)


class MaskLinesTable(QWidget):
    """One row per drawn mask: its name, then the port and line it drives. The rows follow
    the Mask Editor's masks; only the port and line are edited here."""

    changed = pyqtSignal()

    def __init__(self, values: FieldValues, parent: QWidget):
        super().__init__(parent)
        self.values = values
        self.spins: list[tuple[str, QSpinBox, QSpinBox]] = []
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(4)
        self.grid.setColumnStretch(0, 1)

    def masks(self) -> tuple[Mask, ...]:
        return self.values.get(MASKS.key)

    def value(self) -> tuple[MaskLine, ...]:
        return tuple(
            MaskLine(mask_id, port.value(), line.value()) for mask_id, port, line in self.spins
        )

    def set_value(self, lines: tuple[MaskLine, ...]) -> None:
        """Show a row for each drawn mask, with the line it drives."""
        while (item := self.grid.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self.spins = []
        masks = self.masks()
        if not masks:
            self.grid.addWidget(muted("No masks to assign.", self))
            return
        for column, text in ((1, "Port"), (2, "Line")):
            self.grid.addWidget(muted(text, self), 0, column, Qt.AlignmentFlag.AlignHCenter)
        for row, (mask, mask_line) in enumerate(
            zip(masks, lines_for(masks, lines), strict=True), start=1
        ):
            name = QLabel(mask.label or "mask", self)
            name.setToolTip(mask.describe())
            port, line = self.spin(mask_line.port), self.spin(mask_line.line)
            self.grid.addWidget(name, row, 0)
            self.grid.addWidget(port, row, 1)
            self.grid.addWidget(line, row, 2)
            self.spins.append((mask.id, port, line))

    def spin(self, value: int) -> QSpinBox:
        """A compact number box: typed, scrolled or stepped with the arrow keys."""
        spin = QSpinBox(self)
        spin.setRange(0, 1024)
        spin.setValue(value)
        spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spin.setFixedWidth(SPIN_WIDTH)
        spin.valueChanged.connect(self.changed)
        return spin

    def summary(self) -> str:
        masks = self.masks()
        parts = [
            f"{mask.label or 'mask'} → port{line.port}/line{line.line}"
            for mask, line in zip(masks, lines_for(masks, self.value()), strict=True)
        ]
        return ", ".join(parts) or "none"


def muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setEnabled(False)
    return label


@dataclass(frozen=True)
class MaskLinesField(Field):
    """The port and line each drawn mask drives, by mask id. A mask with none set drives the
    next free line on port 0."""

    def coerce(self, value: Any) -> tuple[MaskLine, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of mask lines")
        return tuple(MaskLine.from_value(each) for each in value)

    def encode(self, value: tuple[MaskLine, ...]) -> Any:
        return [line.to_dict() for line in value]

    def editor(self, parent: Any, values: FieldValues) -> Editor:
        table = MaskLinesTable(values, parent)
        return Editor(
            table,
            get=table.value,
            set=table.set_value,
            connect=lambda cb: table.changed.connect(cb),
            summary=table.summary,
            spec=self,
        )


MASK_LINES = declare(
    "mask_lines",
    MaskLinesField("Mask Lines", "The digital output line each drawn mask drives during the scan"),
    (),
)
