from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass
class DisplayManagerUI:
    display_combo: QComboBox
    add_btn: QPushButton
    instances_list: QListWidget
    attach_btn: QPushButton
    detach_btn: QPushButton
    remove_btn: QPushButton
    status_label: QLabel
    display_tabs: QTabWidget


def build_display_manager_ui(owner: QWidget) -> DisplayManagerUI:
    root = QVBoxLayout(owner)

    add_row = QHBoxLayout()
    add_row.addWidget(QLabel("Display:", owner))
    display_combo = QComboBox(owner)
    add_row.addWidget(display_combo, 1)
    add_btn = QPushButton("Add", owner)
    add_row.addWidget(add_btn)
    root.addLayout(add_row)

    root.addWidget(QLabel("Active Displays:", owner))
    instances_list = QListWidget(owner)
    root.addWidget(instances_list)

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

    display_tabs = QTabWidget(owner)
    root.addWidget(display_tabs, 1)

    return DisplayManagerUI(
        display_combo=display_combo,
        add_btn=add_btn,
        instances_list=instances_list,
        attach_btn=attach_btn,
        detach_btn=detach_btn,
        remove_btn=remove_btn,
        status_label=status_label,
        display_tabs=display_tabs,
    )
