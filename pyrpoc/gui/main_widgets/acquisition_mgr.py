from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Type, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QLineEdit, QFileDialog, QComboBox, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QScrollArea, QMessageBox
)

# from pyrpoc.modalities.mod_registry import modality_registry
from pyrpoc.backend_utils.data import BaseData
from pyrpoc.instruments import BaseInstrument
from pyrpoc.laser_modulations.base_laser_mod import BaseLaserModulation
from pyrpoc.modalities.base_modality import BaseModality
from pyrpoc.gui.signals.signals import UISignals

class AcquisitionManagerWidget(QWidget):
    def __init__(self, ui_signals: UISignals, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ui_signals = ui_signals

        self.modality_classes: Dict[str, Type[BaseModality]] = {}
        self.param_widgets: Dict[str, Dict[str, QWidget]] = {}

        self.build_ui()
        self.populate_modalities()

    def build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Row: modality selector
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel('Modality:', self))
        self.modality_combo = QComboBox(self)
        self.modality_combo.currentIndexChanged.connect(self.ui_signals.modality_changed) # TODO: make this a ui_signal
        top_row.addWidget(self.modality_combo, 1)
        root.addLayout(top_row)

        # Row: controls
        ctrl_row = QHBoxLayout()
        self.btn_start = QPushButton('Start', self)
        self.btn_cont = QPushButton('Continuous', self)
        self.btn_stop = QPushButton('Stop', self)
        self.btn_start.clicked.connect(self.ui_signals.start_clicked)
        self.btn_cont.clicked.connect(self.ui_signals.continuous_clicked)
        self.btn_stop.clicked.connect(self.ui_signals.stop_clicked)
        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_cont)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addStretch(1)
        root.addLayout(ctrl_row)

        # Row: save options
        save_row = QHBoxLayout()
        self.chk_save = QCheckBox('Save', self)
        self.save_path_edit = QLineEdit(self)
        self.save_path_edit.setPlaceholderText('choose a folder or file path…')
        self.btn_browse = QPushButton('Browse…', self)
        self.btn_browse.clicked.connect(self.ui_signals.browse_save)
        save_row.addWidget(self.chk_save)
        save_row.addWidget(self.save_path_edit, 1)
        save_row.addWidget(self.btn_browse)
        root.addLayout(save_row)

        # thing that calls the parameters functions and adds it to the widget

        # little spacer at bottom
        root.addStretch(1)

    def load_modalities(self) -> None | Dict[str, Type[BaseModality]]:
        # get rid of None
        # need to figure out how registries actually report things, should be 'str': class
        pass

    def populate_modalities(self) -> None:
        self.modality_combo.clear()
        names = sorted(self.modality_classes.keys())
        self.modality_combo.addItems(names)

    # function to create a group of parameters
    # function to create the whole parameters widget


