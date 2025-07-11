import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal
from pyrpoc.instruments.base_instrument import Instrument

class DataInput(Instrument):
    def __init__(self, name="Data Input", console_callback=None):
        super().__init__(name, "data input", console_callback=console_callback)
        self.parameters = {
            'input_channels': [0, 1],
            'channel_names': {'0': 'ch0', '1': 'ch1'},  
            'sample_rate': 1000000,
            'device_name': 'Dev1'
        }

    def initialize(self):
        return True

    def get_widget(self):
        return DataInputWidget(self)
    
    def validate_parameters(self, parameters):
        required_params = ['input_channels', 'sample_rate', 'device_name']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for data input: {param}")

        if not isinstance(parameters['input_channels'], list) or len(parameters['input_channels']) == 0:
            raise ValueError("Input channels must be a non-empty list")
        if any(ch < 0 for ch in parameters['input_channels']):
            raise ValueError("Channel numbers must be non-negative")
        if parameters['sample_rate'] <= 0:
            raise ValueError("Sample rate must be positive")
    

class DataInputWidget(QWidget):
    def __init__(self, data_input):
        super().__init__()
        self.data_input = data_input
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        self.config_widget = DataInputConfigWidget(self.data_input)
        layout.addWidget(self.config_widget)

        if self.data_input.parameters:
            self.config_widget.set_parameters(self.data_input.parameters)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        return self.config_widget.get_parameters()
    
    def set_parameters(self, parameters):
        self.config_widget.set_parameters(parameters)

class DataInputConfigWidget(QWidget):
    def __init__(self, data_input):
        super().__init__()
        self.data_input = data_input
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
        
        self.channel_names_edit = QLineEdit("ch0,ch1")
        layout.addRow("Channel Names (comma-separated):", self.channel_names_edit)
        
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(1000, 10000000)
        self.sample_rate_spin.setValue(1000000)
        self.sample_rate_spin.setSuffix(" Hz")
        layout.addRow("Sample Rate:", self.sample_rate_spin)
        
        channels = self.data_input.parameters.get('input_channels', [])
        channel_names = self.data_input.parameters.get('channel_names', {})
        
        channels_info = []
        for ch in channels:
            ch_name = channel_names.get(str(ch), f'ch{ch}')
            channels_info.append(f"{ch_name} (AI{ch})")
        
        channels_text = ', '.join(channels_info)
        layout.addRow("Active Channels:", QLabel(channels_text))
        
        self.setLayout(layout)

    def set_parameters(self, parameters):
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
        if 'channel_names' in parameters:
            channel_names = parameters['channel_names']
            if isinstance(channel_names, dict):
                channels = parameters.get('input_channels', [])
                names_list = [channel_names.get(str(ch), f'ch{ch}') for ch in channels]
                self.channel_names_edit.setText(','.join(names_list))
        if 'sample_rate' in parameters:
            self.sample_rate_spin.setValue(parameters['sample_rate'])

    def get_parameters(self):
        channels_text = self.channels_edit.text()
        channels = [int(x.strip()) for x in channels_text.split(',') if x.strip().isdigit()]
        
        channel_names_text = self.channel_names_edit.text()
        channel_names_list = [x.strip() for x in channel_names_text.split(',') if x.strip()]
        
        channel_names = {}
        for i, ch in enumerate(channels):
            if i < len(channel_names_list):
                channel_names[str(ch)] = channel_names_list[i]
            else:
                channel_names[str(ch)] = f'ch{ch}'
        
        parameters = {
            'name': self.name_edit.text(),
            'device_name': self.device_combo.currentText(),
            'input_channels': channels,
            'channel_names': channel_names,
            'sample_rate': self.sample_rate_spin.value()
        }

        try:
            self.data_input.validate_parameters(parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None