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
            'device_name': 'Dev1',
            'dwell_time': 10e-6,  # Per pixel dwell time in seconds
            'extrasteps_left': 50,  # Extra steps left in fast direction
            'extrasteps_right': 50,  # Extra steps right in fast direction
            'amplitude_x': 0.5,  # Amplitude for X axis
            'amplitude_y': 0.5,  # Amplitude for Y axis
            'offset_x': 0.0,  # Offset for X axis
            'offset_y': 0.0,  # Offset for Y axis
            'numsteps_x': 512,  # Number of X pixels/steps
            'numsteps_y': 512   # Number of Y pixels/steps
        }

    def initialize(self):
        """Initialize galvo connection - for now just return True"""
        try:
            # TODO: Add actual DAQ initialization here
            return True
        except Exception as e:
            print(f"Failed to initialize galvo: {e}")
            return False

    def get_control_widget(self):
        """Return galvo control widget"""
        return GalvoControlWidget(self)
    
    def generate_raster_waveform(self, numsteps_x=None, numsteps_y=None):
        """
        Generate raster scan waveform based on current parameters
        
        Args:
            numsteps_x: Override X steps from parameters
            numsteps_y: Override Y steps from parameters
            
        Returns:
            numpy.ndarray: 2xN array with X and Y waveforms
        """
        import numpy as np
        
        # Use provided values or fall back to parameters
        x_steps = numsteps_x if numsteps_x is not None else self.parameters['numsteps_x']
        y_steps = numsteps_y if numsteps_y is not None else self.parameters['numsteps_y']
        
        # Extract parameters
        dwell = self.parameters['dwell_time']
        rate = self.parameters['sample_rate']
        extra_left = self.parameters['extrasteps_left']
        extra_right = self.parameters['extrasteps_right']
        amp_x = self.parameters['amplitude_x']
        amp_y = self.parameters['amplitude_y']
        offset_x = self.parameters['offset_x']
        offset_y = self.parameters['offset_y']
        
        # Calculate samples per pixel and total samples
        pixel_samples = max(1, int(dwell * rate))
        total_x = x_steps + extra_left + extra_right
        total_y = y_steps
        
        # Calculate step size and boundaries
        contained_rowsamples = pixel_samples * x_steps
        total_rowsamples = pixel_samples * total_x
        
        step_size = (2 * amp_x) / contained_rowsamples
        bottom = offset_x - amp_x - (step_size * extra_left)
        top = offset_x + amp_x + (step_size * extra_right)
        
        # Generate X waveform (fast axis)
        single_row_ramp = np.linspace(bottom, top, total_rowsamples, endpoint=False)
        x_waveform = np.tile(single_row_ramp, total_y)
        
        # Generate Y waveform (slow axis)
        y_steps_positions = np.linspace(
            offset_y + amp_y,
            offset_y - amp_y,
            total_y
        )
        y_waveform = np.repeat(y_steps_positions, total_rowsamples)
        
        # Create composite waveform
        composite = np.vstack([x_waveform, y_waveform])
        
        # Ensure correct size
        total_samples = total_x * total_y * pixel_samples
        if x_waveform.size < total_samples:
            x_waveform = np.pad(
                x_waveform,
                (0, total_samples - x_waveform.size),
                constant_values=x_waveform[-1]
            )
        else:
            x_waveform = x_waveform[:total_samples]
        composite[0] = x_waveform
        
        return composite
    
    def generate_variable_waveform(self, mask, dwell_multiplier=2.0, numsteps_x=None, numsteps_y=None):
        """
        Generate variable dwell waveform based on RPOC mask
        
        Args:
            mask: 2D numpy array representing the RPOC mask
            dwell_multiplier: Multiplier for dwell time in masked regions
            numsteps_x: Override X steps from parameters
            numsteps_y: Override Y steps from parameters
            
        Returns:
            tuple: (x_waveform, y_waveform, pixel_map)
        """
        import numpy as np
        
        # Use provided values or fall back to parameters
        x_steps = numsteps_x if numsteps_x is not None else self.parameters['numsteps_x']
        y_steps = numsteps_y if numsteps_y is not None else self.parameters['numsteps_y']
        
        # Extract parameters
        dwell = self.parameters['dwell_time']
        rate = self.parameters['sample_rate']
        extra_left = self.parameters['extrasteps_left']
        extra_right = self.parameters['extrasteps_right']
        amp_x = self.parameters['amplitude_x']
        amp_y = self.parameters['amplitude_y']
        offset_x = self.parameters['offset_x']
        offset_y = self.parameters['offset_y']
        
        total_x = x_steps + extra_left + extra_right
        
        # Validate mask dimensions
        mask_shape = np.shape(mask)
        if mask_shape[1] != y_steps or mask_shape[0] != (total_x - extra_left - extra_right):
            raise ValueError(f'Mask dimensions {mask_shape} do not match expected dimensions {(total_x - extra_left - extra_right)}x{y_steps}')
        
        dwell_on = dwell * dwell_multiplier
        dwell_off = dwell
        
        # Generate position arrays
        x_min = offset_x - amp_x
        x_max = offset_x + amp_x
        x_positions = np.linspace(x_min, x_max, total_x, endpoint=False)
        y_positions = np.linspace(offset_y + amp_y, offset_y - amp_y, y_steps)
        
        x_wave_list = []
        y_wave_list = []
        pixel_map = np.zeros((y_steps, total_x), dtype=int)
        
        for row_idx in range(y_steps):
            row_y = y_positions[row_idx]
            for col_idx in range(total_x):
                if col_idx < extra_left or col_idx >= (total_x - extra_right):
                    this_dwell = dwell_off
                elif mask[row_idx, col_idx - extra_left]:
                    this_dwell = dwell_on
                else:
                    this_dwell = dwell_off
                
                pixel_samps = max(1, int(this_dwell * rate))
                
                x_val = x_positions[col_idx]
                x_wave_list.append(np.full(pixel_samps, x_val))
                y_wave_list.append(np.full(pixel_samps, row_y))
                
                pixel_map[row_idx, col_idx] = pixel_samps
        
        x_wave = np.concatenate(x_wave_list)
        y_wave = np.concatenate(y_wave_list)
        
        return x_wave, y_wave, pixel_map
    



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
        
        # Galvo-specific parameters
        self.dwell_time_spin = QDoubleSpinBox()
        self.dwell_time_spin.setRange(1e-9, 1e-3)
        self.dwell_time_spin.setValue(10e-6)
        self.dwell_time_spin.setSuffix(" s")
        self.dwell_time_spin.setDecimals(9)
        layout.addRow("Dwell Time:", self.dwell_time_spin)
        
        self.extrasteps_left_spin = QSpinBox()
        self.extrasteps_left_spin.setRange(0, 1000)
        self.extrasteps_left_spin.setValue(50)
        layout.addRow("Extra Steps Left:", self.extrasteps_left_spin)
        
        self.extrasteps_right_spin = QSpinBox()
        self.extrasteps_right_spin.setRange(0, 1000)
        self.extrasteps_right_spin.setValue(50)
        layout.addRow("Extra Steps Right:", self.extrasteps_right_spin)
        
        self.amplitude_x_spin = QDoubleSpinBox()
        self.amplitude_x_spin.setRange(0.1, 10.0)
        self.amplitude_x_spin.setValue(0.5)
        self.amplitude_x_spin.setSuffix(" V")
        layout.addRow("Amplitude X:", self.amplitude_x_spin)
        
        self.amplitude_y_spin = QDoubleSpinBox()
        self.amplitude_y_spin.setRange(0.1, 10.0)
        self.amplitude_y_spin.setValue(0.5)
        self.amplitude_y_spin.setSuffix(" V")
        layout.addRow("Amplitude Y:", self.amplitude_y_spin)
        
        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-10.0, 10.0)
        self.offset_x_spin.setValue(0.0)
        self.offset_x_spin.setSuffix(" V")
        layout.addRow("Offset X:", self.offset_x_spin)
        
        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-10.0, 10.0)
        self.offset_y_spin.setValue(0.0)
        self.offset_y_spin.setSuffix(" V")
        layout.addRow("Offset Y:", self.offset_y_spin)
        
        # Pixel/step parameters
        self.numsteps_x_spin = QSpinBox()
        self.numsteps_x_spin.setRange(64, 4096)
        self.numsteps_x_spin.setValue(512)
        layout.addRow("X Pixels:", self.numsteps_x_spin)
        
        self.numsteps_y_spin = QSpinBox()
        self.numsteps_y_spin.setRange(64, 4096)
        self.numsteps_y_spin.setValue(512)
        layout.addRow("Y Pixels:", self.numsteps_y_spin)
        
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
        if 'dwell_time' in parameters:
            self.dwell_time_spin.setValue(parameters['dwell_time'])
        if 'extrasteps_left' in parameters:
            self.extrasteps_left_spin.setValue(parameters['extrasteps_left'])
        if 'extrasteps_right' in parameters:
            self.extrasteps_right_spin.setValue(parameters['extrasteps_right'])
        if 'amplitude_x' in parameters:
            self.amplitude_x_spin.setValue(parameters['amplitude_x'])
        if 'amplitude_y' in parameters:
            self.amplitude_y_spin.setValue(parameters['amplitude_y'])
        if 'offset_x' in parameters:
            self.offset_x_spin.setValue(parameters['offset_x'])
        if 'offset_y' in parameters:
            self.offset_y_spin.setValue(parameters['offset_y'])
        if 'numsteps_x' in parameters:
            self.numsteps_x_spin.setValue(parameters['numsteps_x'])
        if 'numsteps_y' in parameters:
            self.numsteps_y_spin.setValue(parameters['numsteps_y'])

    def get_parameters(self):
        parameters = {
            'name': self.name_edit.text(),
            'device_name': self.device_combo.currentText(),
            'slow_axis_channel': self.slow_channel_spin.value(),
            'fast_axis_channel': self.fast_channel_spin.value(),
            'voltage_range': self.voltage_spin.value(),
            'sample_rate': self.sample_rate_spin.value(),
            'dwell_time': self.dwell_time_spin.value(),
            'extrasteps_left': self.extrasteps_left_spin.value(),
            'extrasteps_right': self.extrasteps_right_spin.value(),
            'amplitude_x': self.amplitude_x_spin.value(),
            'amplitude_y': self.amplitude_y_spin.value(),
            'offset_x': self.offset_x_spin.value(),
            'offset_y': self.offset_y_spin.value(),
            'numsteps_x': self.numsteps_x_spin.value(),
            'numsteps_y': self.numsteps_y_spin.value()
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

def get_instruments_by_type(instruments_list, instrument_type):
    """Get all instruments of a specific type from the instruments list"""
    return [instrument for instrument in instruments_list 
            if instrument.instrument_type == instrument_type]

def validate_instrument_parameters(instrument_type, parameters):
    """Validate instrument parameters before creating or updating an instrument"""
    if instrument_type == "Galvo":
        required_params = ['slow_axis_channel', 'fast_axis_channel', 'voltage_range', 'sample_rate', 'device_name',
                          'dwell_time', 'extrasteps_left', 'extrasteps_right', 'amplitude_x', 'amplitude_y', 
                          'offset_x', 'offset_y', 'numsteps_x', 'numsteps_y']
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
        if parameters['dwell_time'] <= 0:
            raise ValueError("Dwell time must be positive")
        if parameters['extrasteps_left'] < 0 or parameters['extrasteps_right'] < 0:
            raise ValueError("Extra steps must be non-negative")
        if parameters['amplitude_x'] <= 0 or parameters['amplitude_y'] <= 0:
            raise ValueError("Amplitudes must be positive")
        if parameters['numsteps_x'] <= 0 or parameters['numsteps_x'] > 10000:
            raise ValueError("X pixels must be between 1 and 10000")
        if parameters['numsteps_y'] <= 0 or parameters['numsteps_y'] > 10000:
            raise ValueError("Y pixels must be between 1 and 10000")
            
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