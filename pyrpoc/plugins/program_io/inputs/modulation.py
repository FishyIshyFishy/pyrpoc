"""Modulation: each drawn mask drives one digital line during a scan, on the port and line
set beside it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.program_input import Editor, Field, FieldValues, ParameterError, declare
from pyrpoc.plugins.program_io.inputs.mask import Mask
from pyrpoc.plugins.program_io.inputs.masks import MASKS, thumbnail
from pyrpoc.utils.widgets.empty_state import EmptyState
from pyrpoc.utils.widgets.flow_grid import FlowGrid

# Wide enough for the prefix and two digits; a wider row gives the boxes more.
SPIN_WIDTH = 60
THUMBNAIL_PX = 44
# The narrowest a mask's row is given before the rows drop to fewer columns.
ROW_PX = 230


@dataclass(frozen=True)
class MaskLine:
    """The port and line one drawn mask drives, by the mask's id."""

    mask_id: str
    port: int
    line: int
    # Off, the mask drives nothing, as if it were not drawn.
    enabled: bool

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this line is."""
        return f"{device_name}/port{self.port}/line{self.line}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "port": self.port,
            "line": self.line,
            "enabled": self.enabled,
        }

    @classmethod
    def from_value(cls, raw: Any) -> MaskLine:
        """A line from the form (already a ``MaskLine``) or from JSON (a dict)."""
        if isinstance(raw, MaskLine):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a mask's line must be an object with a mask id, port and line")
        try:
            # Recordings from before masks could be switched off have no flag; they were all on.
            return cls(
                str(raw["mask_id"]),
                int(raw["port"]),
                int(raw["line"]),
                bool(raw.get("enabled", True)),
            )
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
            chosen[mask.id] = MaskLine(mask.id, 0, free, True)
        out.append(chosen[mask.id])
    return tuple(out)


class MaskRow(QFrame):
    """One mask's modulation: a checkbox to use it, its picture and name, and the port and line
    it drives."""

    changed = pyqtSignal()

    def __init__(self, mask: Mask, mask_line: MaskLine, parent: QWidget):
        super().__init__(parent)
        self.mask_id = mask.id
        self.setObjectName("maskRow")
        self.setStyleSheet("#maskRow { border: 1px solid palette(midlight); border-radius: 4px; }")
        self.setToolTip(mask.describe())
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(8)
        self.enabled = QCheckBox(self)
        self.enabled.setToolTip("Drive this line with the mask during a scan")
        self.enabled.setChecked(mask_line.enabled)
        row.addWidget(self.enabled)
        self.picture = QLabel(self)
        self.picture.setPixmap(thumbnail(mask, THUMBNAIL_PX))
        row.addWidget(self.picture)
        column = QVBoxLayout()
        column.setSpacing(4)
        self.name = QLabel(mask.label or "mask", self)
        column.addWidget(self.name)
        spins = QHBoxLayout()
        spins.setSpacing(4)
        self.port = self.spin("Port ", mask_line.port)
        self.line = self.spin("Line ", mask_line.line)
        spins.addWidget(self.port)
        spins.addWidget(self.line)
        column.addLayout(spins)
        row.addLayout(column, 1)
        self.apply_enabled(mask_line.enabled)
        self.enabled.toggled.connect(self.on_toggled)

    def spin(self, prefix: str, value: int) -> QSpinBox:
        """A compact number box: typed, scrolled or stepped with the arrow keys."""
        spin = QSpinBox(self)
        spin.setRange(0, 1024)
        spin.setValue(value)
        spin.setPrefix(prefix)
        spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spin.setMinimumWidth(SPIN_WIDTH)
        spin.valueChanged.connect(self.changed)
        return spin

    def apply_enabled(self, enabled: bool) -> None:
        for each in (self.picture, self.name, self.port, self.line):
            each.setEnabled(enabled)

    def on_toggled(self, checked: bool) -> None:
        self.apply_enabled(checked)
        self.changed.emit()

    def value(self) -> MaskLine:
        return MaskLine(
            self.mask_id, self.port.value(), self.line.value(), self.enabled.isChecked()
        )


class MaskLinesTable(QWidget):
    """One row per drawn mask, following the Mask Editor's masks; the picture is read-only
    and the rest is edited here."""

    changed = pyqtSignal()

    def __init__(self, values: FieldValues, parent: QWidget):
        super().__init__(parent)
        self.values = values
        self.rows: list[MaskRow] = []
        self.empty = EmptyState("No masks yet", "Draw one in a Mask Editor.", self)
        self.grid_host = QWidget(self)
        self.grid = FlowGrid(self.grid_host, min_cell=ROW_PX, spacing=6)
        self.grid.setContentsMargins(0, 0, 0, 0)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.addWidget(self.empty)
        outer.addWidget(self.grid_host)

    def masks(self) -> tuple[Mask, ...]:
        return self.values.get(MASKS.key)

    def value(self) -> tuple[MaskLine, ...]:
        return tuple(row.value() for row in self.rows)

    def set_value(self, lines: tuple[MaskLine, ...]) -> None:
        """Show a row for each drawn mask, with the line it drives."""
        while (item := self.grid.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        masks = self.masks()
        self.rows = [
            MaskRow(mask, mask_line, self.grid_host)
            for mask, mask_line in zip(masks, lines_for(masks, lines), strict=True)
        ]
        for row in self.rows:
            row.changed.connect(self.changed)
            self.grid.add(row, wide=False)
        self.empty.setVisible(not masks)
        self.grid_host.setVisible(bool(masks))
        self.grid.invalidate()

    def summary(self) -> str:
        masks = self.masks()
        parts = [
            f"{mask.label or 'mask'} → port{line.port}/line{line.line}"
            for mask, line in zip(masks, lines_for(masks, self.value()), strict=True)
            if line.enabled
        ]
        return ", ".join(parts) or "none"


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
            wide=True,
        )


MASK_LINES = declare(
    "mask_lines",
    MaskLinesField(
        "Masks", "Whether each drawn mask drives a digital output line during the scan, and which"
    ),
    (),
)
