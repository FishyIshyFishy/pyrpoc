"""Routine editor: a Ctrl+R dock that composes the routine's block structure.

Each block picks a modality and which modifiers it makes available. Parameter values
are set in the Acquisition tab, not here. Editing emits ``changed`` so the Acquisition
tab re-renders.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc_next.acquisition import modality_registry, modifier_registry
from pyrpoc_next.structs.routine import ModifierSlot, Routine, RoutineBlock


class BlockRow(QGroupBox):
    """One block: a modality and checkboxes for which modifiers it offers."""

    changed = pyqtSignal()

    def __init__(self):
        super().__init__("Block")
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.modality_combo = QComboBox()
        for key in modality_registry.available():
            self.modality_combo.addItem(modality_registry.manifest(key).display_name, key)
        self.remove_button = QPushButton("Remove")
        top.addWidget(QLabel("Modality"))
        top.addWidget(self.modality_combo, 1)
        top.addWidget(self.remove_button)
        layout.addLayout(top)

        layout.addWidget(QLabel("Available modifiers:"))
        self.modifier_area = QVBoxLayout()
        layout.addLayout(self.modifier_area)

        self.checks: dict = {}
        self.modality_combo.currentIndexChanged.connect(self.on_modality_changed)
        self.rebuild_modifiers()

    def modality_key(self):
        return self.modality_combo.currentData()

    def on_modality_changed(self) -> None:
        self.rebuild_modifiers()
        self.changed.emit()

    def rebuild_modifiers(self) -> None:
        while (item := self.modifier_area.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.checks = {}
        for key in modality_registry.manifest(self.modality_key()).realizable_modifiers:
            check = QCheckBox(modifier_registry.manifest(key).display_name)
            check.toggled.connect(lambda *_: self.changed.emit())
            self.modifier_area.addWidget(check)
            self.checks[key] = check

    def to_block(self) -> RoutineBlock:
        modifiers = [
            ModifierSlot(key=key, available=check.isChecked(), enabled=False)
            for key, check in self.checks.items()
        ]
        return RoutineBlock(modality=self.modality_key(), values=[], modifiers=modifiers)


class RoutineEditor(QWidget):
    """Compose the routine's blocks. Emits ``changed`` when the structure changes."""

    changed = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.state = controller.state
        root = QVBoxLayout(self)

        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.rows_container)
        root.addWidget(scroll, 1)

        add = QPushButton("Add Block")
        add.clicked.connect(self.add_block)
        root.addWidget(add)

        self.rows: list[BlockRow] = []
        self.add_block()  # seed one block so the Acquisition tab is populated

    def add_block(self) -> None:
        row = BlockRow()
        row.changed.connect(self.sync_and_emit)
        row.remove_button.clicked.connect(lambda: self.remove_block(row))
        self.rows.append(row)
        self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
        self.sync_and_emit()

    def remove_block(self, row: BlockRow) -> None:
        if row in self.rows:
            self.rows.remove(row)
        row.setParent(None)
        self.sync_and_emit()

    def sync_and_emit(self) -> None:
        blocks = [row.to_block() for row in self.rows]
        active = min(self.state.routine.active_index, max(0, len(blocks) - 1))
        self.state.routine = Routine(name="session", blocks=blocks, active_index=active)
        self.changed.emit()
