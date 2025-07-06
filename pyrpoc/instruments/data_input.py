import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from pyrpoc.instruments.base_instrument import Instrument

class DataInput(Instrument):
    def __init__(self, name="Data Input"):
        super().__init__(name, "Data Input")
        self.parameters = {
            'input_channels': [0, 1],  # List of AI channels
            'voltage_range': 10.0,
            'sample_rate': 1000000,
            'device_name': 'Dev1'
        }

    def initialize(self):
        """Initialize data input connection"""
        try:
            # TODO: Add actual DAQ initialization here
            return True
        except Exception as e:
            print(f"Failed to initialize data input: {e}")
            return False

    def get_widget(self):
        """Return unified data input widget"""
        return DataInputWidget(self)
    

class DataInputWidget(QWidget):
    def __init__(self, data_input):
        super().__init__()
        self.data_input = data_input
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create tab widget for config and control
        self.tab_widget = QTabWidget()
        
        # Configuration tab
        self.config_widget = DataInputConfigWidget()
        self.tab_widget.addTab(self.config_widget, "Configuration")
        
        # Control tab
        self.control_widget = DataInputControlWidget(self.data_input)
        self.tab_widget.addTab(self.control_widget, "Control")
        
        layout.addWidget(self.tab_widget)
        
        # Set current parameters
        if self.data_input.parameters:
            self.config_widget.set_parameters(self.data_input.parameters)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        """Get parameters from config widget"""
        return self.config_widget.get_parameters()
    
    def set_parameters(self, parameters):
        """Set parameters in config widget"""
        self.config_widget.set_parameters(parameters)

class DataInputConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_edit = QLineEdit("Data Input")
        layout.addRow("Name:", self.name_edit)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(['Dev1', 'Dev2', 'Dev3'])
        layout.addRow("Device:", self.device_combo)
        
        self.channels_edit = QLineEdit("0,1")
        layout.addRow("Input Channels (comma-separated):", self.channels_edit)
        
        self.voltage_spin = QDoubleSpinBox()
        self.voltage_spin.setRange(1.0, 20.0)
        self.voltage_spin.setValue(10.0)
        self.voltage_spin.setSuffix(" V")
        layout.addRow("Voltage Range:", self.voltage_spin)
        
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(1000, 10000000)
        self.sample_rate_spin.setValue(1000000)
        self.sample_rate_spin.setSuffix(" Hz")
        layout.addRow("Sample Rate:", self.sample_rate_spin)
        
        self.setLayout(layout)

    def set_parameters(self, parameters):
        """Set the widget values from parameters"""
        if 'name' in parameters:
            self.name_edit.setText(parameters['name'])
        if 'device_name' in parameters:
            index = self.device_combo.findText(parameters['device_name'])
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
        if 'input_channels' in parameters:
            channels = parameters['input_channels']
            if isinstance(channels, list):
                self.channels_edit.setText(','.join(map(str, channels)))
        if 'voltage_range' in parameters:
            self.voltage_spin.setValue(parameters['voltage_range'])
        if 'sample_rate' in parameters:
            self.sample_rate_spin.setValue(parameters['sample_rate'])

    def get_parameters(self):
        channels_text = self.channels_edit.text()
        channels = [int(x.strip()) for x in channels_text.split(',') if x.strip().isdigit()]
        
        parameters = {
            'name': self.name_edit.text(),
            'device_name': self.device_combo.currentText(),
            'input_channels': channels,
            'voltage_range': self.voltage_spin.value(),
            'sample_rate': self.sample_rate_spin.value()
        }
        
        # Validate parameters before returning
        try:
            from pyrpoc.instruments.instrument_manager import validate_instrument_parameters
            validate_instrument_parameters("Data Input", parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None

class DataInputControlWidget(QWidget):
    def __init__(self, data_input):
        super().__init__()
        self.data_input = data_input
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Display current channels
        channels = self.data_input.parameters.get('input_channels', [])
        channels_text = ', '.join(map(str, channels))
        layout.addWidget(QLabel(f"Active Channels: {channels_text}"))
        
        # Status display
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)