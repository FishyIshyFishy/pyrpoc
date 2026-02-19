from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


@dataclass
class OptoControlManagerUI:
    type_combo: QComboBox
    add_btn: QPushButton
    status_label: QLabel
    instances_scroll: QScrollArea
    instances_content: QWidget
    instances_layout: QVBoxLayout


def build_opto_control_manager_ui(owner: QWidget) -> OptoControlManagerUI:
    root = QVBoxLayout(owner)

    add_row = QHBoxLayout()
    add_row.addWidget(QLabel("Opto-Control:", owner))
    type_combo = QComboBox(owner)
    add_row.addWidget(type_combo, 1)
    add_btn = QPushButton("Add", owner)
    add_row.addWidget(add_btn)
    root.addLayout(add_row)

    status_label = QLabel("Status: ready", owner)
    root.addWidget(status_label)

    instances_scroll = QScrollArea(owner)
    instances_scroll.setWidgetResizable(True)
    instances_content = QWidget(instances_scroll)
    instances_layout = QVBoxLayout(instances_content)
    instances_layout.setContentsMargins(8, 8, 8, 8)
    instances_layout.setSpacing(8)
    instances_layout.addStretch(1)
    instances_scroll.setWidget(instances_content)
    root.addWidget(instances_scroll, 1)

    return OptoControlManagerUI(
        type_combo=type_combo,
        add_btn=add_btn,
        status_label=status_label,
        instances_scroll=instances_scroll,
        instances_content=instances_content,
        instances_layout=instances_layout,
    )
