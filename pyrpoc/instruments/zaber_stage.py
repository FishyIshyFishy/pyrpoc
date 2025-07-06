import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from pyrpoc.instruments.base_instrument import Instrument

class ZaberStage(Instrument):
    def __init__(self, name="Zaber Stage"):
        super().__init__(name, "Zaber Stage")
        self.parameters = {
            'com_port': 'COM2',
            'baud_rate': 115200,
            'timeout': 1.0
        }

    def initialize(self):
        """Initialize Zaber stage connection"""
        try:
            # TODO: Add actual serial connection here
            return True
        except Exception as e:
            print(f"Failed to initialize Zaber stage: {e}")
            return False

    def get_widget(self):
        """Return unified Zaber stage widget"""
        return ZaberStageWidget(self)
    

class ZaberStageWidget(QWidget):
    def __init__(self, zaber_stage):
        super().__init__()
        self.zaber_stage = zaber_stage
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create tab widget for config and control
        self.tab_widget = QTabWidget()
        
        # Configuration tab
        self.config_widget = ZaberStageConfigWidget()
        self.tab_widget.addTab(self.config_widget, "Configuration")
        
        # Control tab
        self.control_widget = ZaberStageControlWidget(self.zaber_stage)
        self.tab_widget.addTab(self.control_widget, "Control")
        
        layout.addWidget(self.tab_widget)
        
        # Set current parameters
        if self.zaber_stage.parameters:
            self.config_widget.set_parameters(self.zaber_stage.parameters)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        """Get parameters from config widget"""
        return self.config_widget.get_parameters()
    
    def set_parameters(self, parameters):
        """Set parameters in config widget"""
        self.config_widget.set_parameters(parameters)

class ZaberStageConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_edit = QLineEdit("Zaber Stage")
        layout.addRow("Name:", self.name_edit)
        
        self.com_port_combo = QComboBox()
        self.com_port_combo.addItems(['COM1', 'COM2', 'COM3', 'COM4', 'COM5'])
        layout.addRow("COM Port:", self.com_port_combo)
        
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['9600', '19200', '38400', '57600', '115200'])
        self.baud_combo.setCurrentText('115200')
        layout.addRow("Baud Rate:", self.baud_combo)
        
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 10.0)
        self.timeout_spin.setValue(1.0)
        self.timeout_spin.setSuffix(" s")
        layout.addRow("Timeout:", self.timeout_spin)
        
        self.setLayout(layout)

    def set_parameters(self, parameters):
        """Set the widget values from parameters"""
        if 'name' in parameters:
            self.name_edit.setText(parameters['name'])
        if 'com_port' in parameters:
            index = self.com_port_combo.findText(parameters['com_port'])
            if index >= 0:
                self.com_port_combo.setCurrentIndex(index)
        if 'baud_rate' in parameters:
            baud_text = str(parameters['baud_rate'])
            index = self.baud_combo.findText(baud_text)
            if index >= 0:
                self.baud_combo.setCurrentIndex(index)
        if 'timeout' in parameters:
            self.timeout_spin.setValue(parameters['timeout'])

    def get_parameters(self):
        parameters = {
            'name': self.name_edit.text(),
            'com_port': self.com_port_combo.currentText(),
            'baud_rate': int(self.baud_combo.currentText()),
            'timeout': self.timeout_spin.value()
        }
        
        # Validate parameters before returning
        try:
            from pyrpoc.instruments.instrument_manager import validate_instrument_parameters
            validate_instrument_parameters("Zaber Stage", parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None

class ZaberStageControlWidget(QWidget):
    def __init__(self, zaber_stage):
        super().__init__()
        self.zaber_stage = zaber_stage
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Position control
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Position:"))
        self.pos_spin = QDoubleSpinBox()
        self.pos_spin.setRange(-1000, 1000)
        self.pos_spin.setSuffix(" mm")
        pos_layout.addWidget(self.pos_spin)
        
        self.move_btn = QPushButton("Move")
        self.move_btn.clicked.connect(self.move_stage)
        pos_layout.addWidget(self.move_btn)
        layout.addLayout(pos_layout)
        
        # Status display
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

    def move_stage(self):
        position = self.pos_spin.value()
        # TODO: Implement stage movement
        print(f"Moving Zaber stage to {position} mm")
        self.status_label.setText(f"Status: Moving to {position} mm")