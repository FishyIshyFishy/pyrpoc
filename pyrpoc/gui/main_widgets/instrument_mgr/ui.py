from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


@dataclass
class InstrumentManagerUI:
    type_combo: QComboBox
    add_btn: QPushButton
    instances_list: QListWidget
    connect_btn: QPushButton
    disconnect_btn: QPushButton
    remove_btn: QPushButton
    status_label: QLabel
    config_box: QGroupBox
    config_form: QFormLayout
    actions_box: QGroupBox
    actions_layout: QVBoxLayout


def build_instrument_manager_ui(owner: QWidget) -> InstrumentManagerUI:
    root = QVBoxLayout(owner)

    add_row = QHBoxLayout()
    add_row.addWidget(QLabel("Instrument:", owner))
    type_combo = QComboBox(owner)
    add_row.addWidget(type_combo, 1)
    add_btn = QPushButton("Add", owner)
    add_row.addWidget(add_btn)
    root.addLayout(add_row)

    root.addWidget(QLabel("Instances:", owner))
    instances_list = QListWidget(owner)
    root.addWidget(instances_list)

    action_row = QHBoxLayout()
    connect_btn = QPushButton("Connect", owner)
    disconnect_btn = QPushButton("Disconnect", owner)
    remove_btn = QPushButton("Remove", owner)
    action_row.addWidget(connect_btn)
    action_row.addWidget(disconnect_btn)
    action_row.addWidget(remove_btn)
    action_row.addStretch(1)
    root.addLayout(action_row)

    status_label = QLabel("Status: ready", owner)
    root.addWidget(status_label)

    config_box = QGroupBox("Connection Parameters", owner)
    config_form = QFormLayout(config_box)
    root.addWidget(config_box)

    actions_box = QGroupBox("Actions", owner)
    actions_layout = QVBoxLayout(actions_box)
    actions_layout.setContentsMargins(8, 8, 8, 8)
    actions_layout.setSpacing(8)
    actions_layout.addStretch(1)

    actions_scroll = QScrollArea(owner)
    actions_scroll.setWidgetResizable(True)
    actions_scroll.setWidget(actions_box)
    root.addWidget(actions_scroll, 1)

    return InstrumentManagerUI(
        type_combo=type_combo,
        add_btn=add_btn,
        instances_list=instances_list,
        connect_btn=connect_btn,
        disconnect_btn=disconnect_btn,
        remove_btn=remove_btn,
        status_label=status_label,
        config_box=config_box,
        config_form=config_form,
        actions_box=actions_box,
        actions_layout=actions_layout,
    )
