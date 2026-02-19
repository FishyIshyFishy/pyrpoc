from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


@dataclass
class OptoControlManagerUI:
    type_combo: QComboBox
    add_btn: QPushButton
    status_label: QLabel
    splitter: QSplitter
    instances_scroll: QScrollArea
    instances_content: QWidget
    instances_layout: QVBoxLayout
    editor_host_box: QGroupBox
    editor_host_layout: QVBoxLayout


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

    splitter = QSplitter(Qt.Orientation.Vertical, owner)
    splitter.setChildrenCollapsible(False)

    instances_container = QWidget(splitter)
    instances_container_layout = QVBoxLayout(instances_container)
    instances_container_layout.setContentsMargins(0, 0, 0, 0)
    instances_container_layout.setSpacing(6)
    instances_container_layout.addWidget(QLabel("Instances:", instances_container))

    instances_scroll = QScrollArea(instances_container)
    instances_scroll.setWidgetResizable(True)
    instances_content = QWidget(instances_scroll)
    instances_layout = QVBoxLayout(instances_content)
    instances_layout.setContentsMargins(8, 8, 8, 8)
    instances_layout.setSpacing(8)
    instances_layout.addStretch(1)
    instances_scroll.setWidget(instances_content)
    instances_container_layout.addWidget(instances_scroll, 1)
    instances_container.setMinimumHeight(120)

    editor_host_box = QGroupBox("Editor Host", splitter)
    editor_host_box.setMinimumHeight(80)
    editor_host_layout = QVBoxLayout(editor_host_box)
    editor_host_layout.setContentsMargins(8, 8, 8, 8)
    editor_host_layout.setSpacing(8)

    splitter.addWidget(instances_container)
    splitter.addWidget(editor_host_box)
    splitter.setSizes([600, 250])

    root.addWidget(splitter, 1)

    return OptoControlManagerUI(
        type_combo=type_combo,
        add_btn=add_btn,
        status_label=status_label,
        splitter=splitter,
        instances_scroll=instances_scroll,
        instances_content=instances_content,
        instances_layout=instances_layout,
        editor_host_box=editor_host_box,
        editor_host_layout=editor_host_layout,
    )
