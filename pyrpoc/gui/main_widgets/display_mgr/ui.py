from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass
class DisplayManagerUI:
    display_combo: QComboBox
    name_input: QLineEdit
    add_btn: QPushButton
    instances_list: QListWidget
    attach_btn: QPushButton
    detach_btn: QPushButton
    remove_btn: QPushButton
    status_label: QLabel


def build_display_manager_ui(owner: QWidget) -> DisplayManagerUI:
    root = QVBoxLayout(owner)

    add_row = QHBoxLayout()
    add_row.addWidget(QLabel("Display:", owner))
    display_combo = QComboBox(owner)
    add_row.addWidget(display_combo, 1)
    add_btn = QPushButton("Add", owner)
    add_row.addWidget(add_btn, 0)
    root.addLayout(add_row)

    name_row = QHBoxLayout()
    name_row.addWidget(QLabel("Display Name:", owner))
    name_input = QLineEdit(owner)
    name_input.setPlaceholderText("Optional display name")
    name_row.addWidget(name_input, 1)
    root.addLayout(name_row)

    root.addWidget(QLabel("Active Displays:", owner))
    instances_list = QListWidget(owner)
    root.addWidget(instances_list, 1)

    action_row = QHBoxLayout()
    attach_btn = QPushButton("Attach", owner)
    detach_btn = QPushButton("Detach", owner)
    remove_btn = QPushButton("Remove", owner)
    action_row.addWidget(attach_btn)
    action_row.addWidget(detach_btn)
    action_row.addWidget(remove_btn)
    action_row.addStretch(1)
    root.addLayout(action_row)

    status_label = QLabel("Status: ready", owner)
    root.addWidget(status_label)

    # Remove per-widget display tabs; display widgets now live in main ADS docks.
    return DisplayManagerUI(
        display_combo=display_combo,
        name_input=name_input,
        add_btn=add_btn,
        instances_list=instances_list,
        attach_btn=attach_btn,
        detach_btn=detach_btn,
        remove_btn=remove_btn,
        status_label=status_label,
    )
