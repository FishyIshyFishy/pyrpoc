"""Acquisition tab: the settings menu where the routine's blocks are laid out.

Renders every block of the routine as a card (the pre-existing parameter-group cards
plus the block's available modifier cards). One block is marked active and is what
Play runs. This is where the routine's setup is 'dumped' as parameter blocks.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from pyrpoc_next.acquisition import MaskModifier, modality_registry, modifier_registry
from pyrpoc_next.gui.panels.param_cards import collect_values, group_cards
from pyrpoc_next.gui.widgets.cards import BaseCardWidget
from pyrpoc_next.structs.keys import ModifierKey


class AcquisitionPanel(QWidget):
    """Transport controls plus the routine's blocks rendered as nested cards."""

    def __init__(self, controller, bridge):
        super().__init__()
        self.controller = controller
        self.state = controller.state
        self.block_cards: list[dict] = []
        self.masks: dict = {}

        root = QVBoxLayout(self)
        style = self.style()
        controls = QHBoxLayout()
        self.play_button = QPushButton()
        self.continuous_button = QPushButton()
        self.stop_button = QPushButton()
        if style is not None:
            self.play_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.continuous_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
            self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.play_button.setToolTip("Play active block")
        self.continuous_button.setToolTip("Play continuously")
        self.stop_button.setToolTip("Stop")
        self.stop_button.setEnabled(False)
        self.play_button.clicked.connect(lambda: self.play(continuous=False))
        self.continuous_button.clicked.connect(lambda: self.play(continuous=True))
        self.stop_button.clicked.connect(self.controller.stop)
        controls.addWidget(self.play_button)
        controls.addWidget(self.continuous_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        root.addLayout(controls)

        self.status = QLabel("Status: idle")
        root.addWidget(self.status)

        self.blocks_container = QWidget()
        self.blocks_layout = QVBoxLayout(self.blocks_container)
        self.blocks_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.blocks_container)
        root.addWidget(scroll, 1)

        bridge.started.connect(self.on_started)
        bridge.stopped.connect(self.on_stopped)
        bridge.errored.connect(self.on_error)
        self.rebuild()

    def rebuild(self) -> None:
        """Re-render every block of the current routine as a card."""
        while (item := self.blocks_layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.block_cards = []

        for index, block in enumerate(self.state.routine.blocks):
            manifest = modality_registry.manifest(block.modality)
            card = BaseCardWidget(None, f"Block {index + 1}: {manifest.display_name}")
            card.toggle_checkbox.setToolTip("Active")
            card.set_toggle_checked(index == self.state.routine.active_index)
            card.expand_requested.connect(lambda _, c=card: c.set_expanded(not c.is_expanded()))
            card.toggle_changed.connect(lambda _obj, checked, i=index: self.on_active(i, checked))

            body = QWidget()
            body_layout = QVBoxLayout(body)
            groups = group_cards(manifest.parameter_groups, block.values)
            for group_card in groups:
                body_layout.addWidget(group_card)
            modifier_cards = [self.build_modifier_card(slot) for slot in block.modifiers if slot.available]
            for modifier_card in modifier_cards:
                body_layout.addWidget(modifier_card["card"])
            card.set_body_widget(body)
            card.set_expanded(True)

            self.blocks_layout.insertWidget(self.blocks_layout.count() - 1, card)
            self.block_cards.append({"index": index, "card": card, "groups": groups, "modifiers": modifier_cards})

    def build_modifier_card(self, slot) -> dict:
        manifest = modifier_registry.manifest(slot.key)
        card = BaseCardWidget(None, f"Modifier: {manifest.display_name}")
        card.toggle_checkbox.setToolTip("Enabled")
        card.set_toggle_checked(slot.enabled)
        card.expand_requested.connect(lambda _: card.set_expanded(not card.is_expanded()))

        body = QWidget()
        body_layout = QVBoxLayout(body)
        groups = group_cards(manifest.parameter_groups, slot.values)
        for group_card in groups:
            body_layout.addWidget(group_card)
        if slot.key is ModifierKey.mask:
            load = QPushButton("Load Mask...")
            load.clicked.connect(self.load_mask)
            body_layout.addWidget(load)
        card.set_body_widget(body)
        return {"card": card, "groups": groups, "key": slot.key}

    def on_active(self, index: int, checked: bool) -> None:
        if not checked:
            return
        for entry in self.block_cards:
            if entry["index"] != index:
                entry["card"].set_toggle_checked(False)
        self.state.routine.active_index = index

    def load_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load mask", "", "Images (*.png *.tif *.tiff *.bmp)")
        if not path:
            return
        import cv2

        self.masks[ModifierKey.mask] = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    def sync_routine(self) -> None:
        """Write the panel's current values back into the routine before a run."""
        for entry in self.block_cards:
            block = self.state.routine.blocks[entry["index"]]
            block.values = collect_values(entry["groups"])
            for modifier_card, slot in zip(entry["modifiers"], [s for s in block.modifiers if s.available]):
                slot.enabled = modifier_card["card"].toggle_checkbox.isChecked()
                slot.values = collect_values(modifier_card["groups"])

    def attach_mask(self, modifier, slot) -> None:
        if isinstance(modifier, MaskModifier):
            modifier.mask = self.masks.get(ModifierKey.mask)

    def play(self, continuous: bool) -> None:
        self.sync_routine()
        self.controller.prepare_modifier = self.attach_mask
        report = self.controller.play(continuous=continuous)
        if report.blocked:
            QMessageBox.warning(self, "Cannot start acquisition",
                                "\n".join(issue.message for issue in report.issues))

    def on_started(self) -> None:
        self.status.setText("Status: acquiring")
        self.play_button.setEnabled(False)
        self.continuous_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def on_stopped(self) -> None:
        self.status.setText("Status: idle")
        self.play_button.setEnabled(True)
        self.continuous_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def on_error(self, message: str) -> None:
        self.status.setText(f"Status: error - {message}")
        self.play_button.setEnabled(True)
        self.continuous_button.setEnabled(True)
        self.stop_button.setEnabled(False)
