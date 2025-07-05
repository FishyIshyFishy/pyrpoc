import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal

class Instrument(abc.ABC):
    def __init__(self, name, instrument_type):
        self.name = name
        self.instrument_type = instrument_type
        self.connected = False
        self.parameters = {}

    @abc.abstractmethod
    def initialize(self):
        """Initialize the instrument connection"""
        pass

    @abc.abstractmethod
    def get_control_widget(self):
        """Return a widget for controlling this instrument, or None if not controllable"""
        pass

    def get_instrument_info(self):
        return f'{self.instrument_type}: {self.name}'

    def is_connected(self):
        return self.connected

    def get_parameters(self):
        return self.parameters.copy()


class Galvo(Instrument):
    def __init__(self, name="Galvo Scanner"):
        super().__init__(name, "Galvo")
        self.parameters = {
            'slow_axis_channel': 0,
            'fast_axis_channel': 1,
            'voltage_range': 10.0,
            'sample_rate': 1000000,
            'device_name': 'Dev1'
        }

    def initialize(self):
        """Initialize galvo connection - for now just mark as connected"""
        try:
            # TODO: Add actual DAQ initialization here
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to initialize galvo: {e}")
            return False

    def get_control_widget(self):
        """Return galvo control widget"""
        return GalvoControlWidget(self)


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
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to initialize data input: {e}")
            return False

    def get_control_widget(self):
        """Return data input control widget"""
        return DataInputControlWidget(self)


class DelayStage(Instrument):
    def __init__(self, name="Delay Stage"):
        super().__init__(name, "Delay Stage")
        self.parameters = {
            'com_port': 'COM1',
            'baud_rate': 9600,
            'timeout': 1.0
        }

    def initialize(self):
        """Initialize delay stage connection"""
        try:
            # TODO: Add actual serial connection here
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to initialize delay stage: {e}")
            return False

    def get_control_widget(self):
        """Return delay stage control widget"""
        return DelayStageControlWidget(self)


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
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to initialize Zaber stage: {e}")
            return False

    def get_control_widget(self):
        """Return Zaber stage control widget"""
        return ZaberStageControlWidget(self)


# Dialog classes for instrument configuration
class InstrumentDialog(QDialog):
    def __init__(self, instrument_type, parent=None, current_parameters=None):
        super().__init__(parent)
        self.instrument_type = instrument_type
        self.parameters = {}
        self.current_parameters = current_parameters or {}
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(f"Configure {self.instrument_type}")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Create the appropriate configuration widget
        if self.instrument_type == "Galvo":
            self.config_widget = GalvoConfigWidget()
        elif self.instrument_type == "Data Input":
            self.config_widget = DataInputConfigWidget()
        elif self.instrument_type == "Delay Stage":
            self.config_widget = DelayStageConfigWidget()
        elif self.instrument_type == "Zaber Stage":
            self.config_widget = ZaberStageConfigWidget()
        else:
            self.config_widget = GenericConfigWidget(self.instrument_type)
        
        # Set current parameters if provided
        if hasattr(self.config_widget, 'set_parameters') and self.current_parameters:
            self.config_widget.set_parameters(self.current_parameters)
        
        layout.addWidget(self.config_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.connect_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def get_parameters(self):
        return self.config_widget.get_parameters()


class GalvoConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_edit = QLineEdit("Galvo Scanner")
        layout.addRow("Name:", self.name_edit)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(['Dev1', 'Dev2', 'Dev3'])
        layout.addRow("Device:", self.device_combo)
        
        self.slow_channel_spin = QSpinBox()
        self.slow_channel_spin.setRange(0, 7)
        self.slow_channel_spin.setValue(0)
        layout.addRow("Slow Axis Channel:", self.slow_channel_spin)
        
        self.fast_channel_spin = QSpinBox()
        self.fast_channel_spin.setRange(0, 7)
        self.fast_channel_spin.setValue(1)
        layout.addRow("Fast Axis Channel:", self.fast_channel_spin)
        
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
        if 'slow_axis_channel' in parameters:
            self.slow_channel_spin.setValue(parameters['slow_axis_channel'])
        if 'fast_axis_channel' in parameters:
            self.fast_channel_spin.setValue(parameters['fast_axis_channel'])
        if 'voltage_range' in parameters:
            self.voltage_spin.setValue(parameters['voltage_range'])
        if 'sample_rate' in parameters:
            self.sample_rate_spin.setValue(parameters['sample_rate'])

    def get_parameters(self):
        parameters = {
            'name': self.name_edit.text(),
            'device_name': self.device_combo.currentText(),
            'slow_axis_channel': self.slow_channel_spin.value(),
            'fast_axis_channel': self.fast_channel_spin.value(),
            'voltage_range': self.voltage_spin.value(),
            'sample_rate': self.sample_rate_spin.value()
        }
        
        # Validate parameters before returning
        try:
            validate_instrument_parameters("Galvo", parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None


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
            validate_instrument_parameters("Data Input", parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None


class DelayStageConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_edit = QLineEdit("Delay Stage")
        layout.addRow("Name:", self.name_edit)
        
        self.com_port_combo = QComboBox()
        self.com_port_combo.addItems(['COM1', 'COM2', 'COM3', 'COM4', 'COM5'])
        layout.addRow("COM Port:", self.com_port_combo)
        
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['9600', '19200', '38400', '57600', '115200'])
        self.baud_combo.setCurrentText('9600')
        layout.addRow("Baud Rate:", self.baud_combo)
        
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 10.0)
        self.timeout_spin.setValue(1.0)
        self.timeout_spin.setSuffix(" s")
        layout.addRow("Timeout:", self.timeout_spin)
        
        self.setLayout(layout)

    def get_parameters(self):
        parameters = {
            'name': self.name_edit.text(),
            'com_port': self.com_port_combo.currentText(),
            'baud_rate': int(self.baud_combo.currentText()),
            'timeout': self.timeout_spin.value()
        }
        
        # Validate parameters before returning
        try:
            validate_instrument_parameters("Delay Stage", parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None


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

    def get_parameters(self):
        parameters = {
            'name': self.name_edit.text(),
            'com_port': self.com_port_combo.currentText(),
            'baud_rate': int(self.baud_combo.currentText()),
            'timeout': self.timeout_spin.value()
        }
        
        # Validate parameters before returning
        try:
            validate_instrument_parameters("Zaber Stage", parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None


class GenericConfigWidget(QWidget):
    def __init__(self, instrument_type):
        super().__init__()
        self.instrument_type = instrument_type
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_edit = QLineEdit(f"{self.instrument_type}")
        layout.addRow("Name:", self.name_edit)
        
        self.setLayout(layout)

    def get_parameters(self):
        return {
            'name': self.name_edit.text()
        }


# Control widget classes for instrument control
class GalvoControlWidget(QWidget):
    def __init__(self, galvo):
        super().__init__()
        self.galvo = galvo
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Connected" if self.galvo.connected else "Disconnected")
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.home_btn = QPushButton("Home")
        self.home_btn.clicked.connect(self.home_galvo)
        control_layout.addWidget(self.home_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_galvo)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout)
        
        self.setLayout(layout)

    def home_galvo(self):
        # TODO: Implement galvo homing
        print("Homing galvo...")

    def stop_galvo(self):
        # TODO: Implement galvo stop
        print("Stopping galvo...")


class DataInputControlWidget(QWidget):
    def __init__(self, data_input):
        super().__init__()
        self.data_input = data_input
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Connected" if self.data_input.connected else "Disconnected")
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)
        
        # Channel info
        channels = self.data_input.parameters.get('input_channels', [])
        layout.addWidget(QLabel(f"Channels: {', '.join(map(str, channels))}"))
        
        self.setLayout(layout)


class DelayStageControlWidget(QWidget):
    def __init__(self, delay_stage):
        super().__init__()
        self.delay_stage = delay_stage
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Connected" if self.delay_stage.connected else "Disconnected")
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)
        
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
        
        self.setLayout(layout)

    def move_stage(self):
        position = self.pos_spin.value()
        # TODO: Implement stage movement
        print(f"Moving delay stage to {position} mm")


class ZaberStageControlWidget(QWidget):
    def __init__(self, zaber_stage):
        super().__init__()
        self.zaber_stage = zaber_stage
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Connected" if self.zaber_stage.connected else "Disconnected")
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)
        
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
        
        self.setLayout(layout)

    def move_stage(self):
        position = self.pos_spin.value()
        # TODO: Implement stage movement
        print(f"Moving Zaber stage to {position} mm")


# Factory function to create instruments
def create_instrument(instrument_type, name, parameters=None):
    """Create an instrument instance based on type, name, and parameters"""
    if parameters is None:
        parameters = {}
    
    if instrument_type == "Galvo":
        instrument = Galvo(name)
        instrument.parameters.update(parameters)
    elif instrument_type == "Data Input":
        instrument = DataInput(name)
        instrument.parameters.update(parameters)
    elif instrument_type == "Delay Stage":
        instrument = DelayStage(name)
        instrument.parameters.update(parameters)
    elif instrument_type == "Zaber Stage":
        instrument = ZaberStage(name)
        instrument.parameters.update(parameters)
    else:
        raise ValueError(f"Unknown instrument type: {instrument_type}")
    
    return instrument

def get_instruments_by_type(instruments_dict, instrument_type):
    """Get all instruments of a specific type from the instruments dictionary"""
    return [instrument for instrument in instruments_dict.values() 
            if instrument.instrument_type == instrument_type]

def validate_instrument_parameters(instrument_type, parameters):
    """Validate instrument parameters before creating or updating an instrument"""
    if instrument_type == "Galvo":
        required_params = ['slow_axis_channel', 'fast_axis_channel', 'voltage_range', 'sample_rate', 'device_name']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for Galvo: {param}")
        
        # Validate parameter values
        if parameters['slow_axis_channel'] < 0 or parameters['fast_axis_channel'] < 0:
            raise ValueError("Channel numbers must be non-negative")
        if parameters['voltage_range'] <= 0:
            raise ValueError("Voltage range must be positive")
        if parameters['sample_rate'] <= 0:
            raise ValueError("Sample rate must be positive")
            
    elif instrument_type == "Data Input":
        required_params = ['input_channels', 'voltage_range', 'sample_rate', 'device_name']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for Data Input: {param}")
        
        # Validate parameter values
        if not isinstance(parameters['input_channels'], list) or len(parameters['input_channels']) == 0:
            raise ValueError("Input channels must be a non-empty list")
        if any(ch < 0 for ch in parameters['input_channels']):
            raise ValueError("Channel numbers must be non-negative")
        if parameters['voltage_range'] <= 0:
            raise ValueError("Voltage range must be positive")
        if parameters['sample_rate'] <= 0:
            raise ValueError("Sample rate must be positive")
            
    elif instrument_type == "Delay Stage":
        required_params = ['com_port', 'baud_rate', 'timeout']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for Delay Stage: {param}")
        
        # Validate parameter values
        if not parameters['com_port'].startswith('COM'):
            raise ValueError("COM port must start with 'COM'")
        if parameters['baud_rate'] not in [9600, 19200, 38400, 57600, 115200]:
            raise ValueError("Invalid baud rate")
        if parameters['timeout'] <= 0:
            raise ValueError("Timeout must be positive")
            
    elif instrument_type == "Zaber Stage":
        required_params = ['com_port', 'baud_rate', 'timeout']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for Zaber Stage: {param}")
        
        # Validate parameter values
        if not parameters['com_port'].startswith('COM'):
            raise ValueError("COM port must start with 'COM'")
        if parameters['baud_rate'] not in [9600, 19200, 38400, 57600, 115200]:
            raise ValueError("Invalid baud rate")
        if parameters['timeout'] <= 0:
            raise ValueError("Timeout must be positive")
    
    else:
        raise ValueError(f"Unknown instrument type: {instrument_type}")

def initialize_instrument_with_retry(instrument, max_retries=3):
    """Initialize an instrument with retry logic for connection failures"""
    for attempt in range(max_retries):
        try:
            if instrument.initialize():
                return True
            else:
                print(f"Attempt {attempt + 1} failed for {instrument.name}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {instrument.name}: {e}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in 1 second...")
            import time
            time.sleep(1)
    
    print(f"Failed to initialize {instrument.name} after {max_retries} attempts")
    return False