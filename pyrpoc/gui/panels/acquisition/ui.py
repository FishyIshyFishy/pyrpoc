from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)


@dataclass
class AcquisitionManagerUI:
    modality_combo: QComboBox
    start_btn: QPushButton
    continuous_btn: QPushButton
    stop_btn: QPushButton
    status_label: QLabel
    params_container: QWidget
    params_layout: QVBoxLayout


def build_acquisition_manager_ui(owner: QWidget) -> AcquisitionManagerUI:
    root = QVBoxLayout(owner)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(8)

    top = QHBoxLayout()
    top.addWidget(QLabel("Modality:", owner))
    modality_combo = QComboBox(owner)
    top.addWidget(modality_combo, 1)
    root.addLayout(top)

    controls = QHBoxLayout()
    style = owner.style()
    start_btn = QPushButton(owner)
    start_btn.setToolTip("Start")
    continuous_btn = QPushButton(owner)
    continuous_btn.setToolTip("Continuous acquisition")
    stop_btn = QPushButton(owner)
    stop_btn.setToolTip("Stop")
    if style is not None:
        start_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        continuous_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        stop_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
    controls.addWidget(start_btn)
    controls.addWidget(continuous_btn)
    controls.addWidget(stop_btn)
    controls.addStretch(1)
    root.addLayout(controls)

    status_label = QLabel("Status: idle", owner)
    root.addWidget(status_label)

    params_container = QWidget(owner)
    params_layout = QVBoxLayout(params_container)
    params_layout.setContentsMargins(0, 0, 0, 0)
    params_layout.setSpacing(8)
    params_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
    params_layout.addStretch(1)

    scroll = QScrollArea(owner)
    scroll.setWidgetResizable(True)
    scroll.setWidget(params_container)
    root.addWidget(scroll, 1)

    return AcquisitionManagerUI(
        modality_combo=modality_combo,
        start_btn=start_btn,
        continuous_btn=continuous_btn,
        stop_btn=stop_btn,
        status_label=status_label,
        params_container=params_container,
        params_layout=params_layout,
    )
