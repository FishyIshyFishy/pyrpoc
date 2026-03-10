from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
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
    instances_scroll: QScrollArea
    instances_content: QWidget
    instances_layout: QVBoxLayout


def build_opto_control_manager_ui(owner: QWidget) -> OptoControlManagerUI:
    root = QVBoxLayout(owner)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(8)

    add_row = QHBoxLayout()
    add_row.setContentsMargins(0, 0, 0, 0)
    add_row.addWidget(QLabel("Opto-Control:", owner))
    type_combo = QComboBox(owner)
    add_row.addWidget(type_combo, 1)
    add_btn = QPushButton("Add", owner)
    add_row.addWidget(add_btn)
    root.addLayout(add_row)

    instances_scroll = QScrollArea(owner)
    instances_scroll.setWidgetResizable(True)
    instances_scroll.setFrameShape(QFrame.Shape.NoFrame)
    instances_scroll.setObjectName("optoControlScroll")
    instances_scroll.setStyleSheet(
        "#optoControlScroll {"
        "background: palette(alternate-base);"
        "border: 1px solid palette(mid);"
        "border-radius: 8px;"
        "}"
        "#optoControlScroll > QWidget > QWidget {"
        "background: palette(alternate-base);"
        "}"
    )
    instances_content = QWidget(instances_scroll)
    instances_layout = QVBoxLayout(instances_content)
    instances_layout.setContentsMargins(8, 8, 8, 8)
    instances_layout.setSpacing(4)
    instances_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    instances_scroll.setWidget(instances_content)
    root.addWidget(instances_scroll, 1)

    return OptoControlManagerUI(
        type_combo=type_combo,
        add_btn=add_btn,
        instances_scroll=instances_scroll,
        instances_content=instances_content,
        instances_layout=instances_layout,
    )
