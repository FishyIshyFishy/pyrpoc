"""``InputTargetCombo``: a dropdown naming where a panel puts the values it makes."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QWidget

from pyrpoc.core.program_inputs import (
    InputTarget,
    ProgramInputs,
    program_input_registry,
    targets_for,
)


class InputTargetCombo(QComboBox):
    """Every slot of every program input that accepts a ``kind``, after a first entry for
    none. It follows the inputs, since an input's slots can change with its value."""

    # The user chose a different target.
    chosen = pyqtSignal()

    def __init__(self, kind: type, inputs: ProgramInputs, parent: QWidget):
        super().__init__(parent)
        self.kind = kind
        self.inputs = inputs
        self.targets: list[InputTarget] = []
        self.target: InputTarget | None = None
        self.currentIndexChanged.connect(self.on_index_changed)
        inputs.subscribe(self.on_input_changed)
        self.refresh()

    def refresh(self) -> None:
        """List the targets there are now, keeping the chosen one if it is still there."""
        self.targets = targets_for(self.kind, self.inputs)
        if self.target not in self.targets:
            self.target = None
        self.blockSignals(True)
        self.clear()
        self.addItem("Nothing")
        for target in self.targets:
            self.addItem(target.label)
        self.setCurrentIndex(0 if self.target is None else self.targets.index(self.target) + 1)
        self.blockSignals(False)

    def on_input_changed(self, key: str) -> None:
        if program_input_registry[key].rule.accepts is self.kind:
            self.refresh()

    def on_index_changed(self, index: int) -> None:
        self.target = None if index <= 0 else self.targets[index - 1]
        self.chosen.emit()

    def detach(self) -> None:
        """Stop following the inputs."""
        self.inputs.unsubscribe(self.on_input_changed)

    def export_state(self) -> dict[str, str] | None:
        if self.target is None:
            return None
        return {"key": self.target.program_input.key, "slot": self.target.slot}

    def import_state(self, raw: Any) -> None:
        """Choose the target saved by ``export_state``, if it is still there."""
        if not isinstance(raw, dict):
            return
        for target in self.targets:
            if target.program_input.key == raw.get("key") and target.slot == raw.get("slot"):
                self.target = target
        self.refresh()
