import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from pyrpoc.instruments.base_instrument import Instrument

class Galvo(Instrument):
    def __init__(self, name="Galvo Scanner"):
        super().__init__(name, "Galvo")
        # Only connection parameters - acquisition parameters moved to GUI
        self.parameters = {
            'slow_axis_channel': 0,
            'fast_axis_channel': 1,
            'voltage_range': 10.0,
            'sample_rate': 1000000,
            'device_name': 'Dev1'
        }

    def initialize(self):
        """Initialize galvo connection - for now just return True"""
        try:
            # TODO: Add actual DAQ initialization here
            return True
        except Exception as e:
            print(f"Failed to initialize galvo: {e}")
            return False

    def get_widget(self):
        """Return unified galvo widget"""
        return GalvoWidget(self)
    
    def generate_raster_waveform(self, acquisition_parameters):
        """
        Generate raster scan waveform based on acquisition parameters
        
        Args:
            acquisition_parameters: Dict containing acquisition parameters like
                dwell_time, extrasteps_left, extrasteps_right, amplitude_x, 
                amplitude_y, offset_x, offset_y, numsteps_x, numsteps_y
            
        Returns:
            numpy.ndarray: 2xN array with X and Y waveforms
        """
        import numpy as np
        
        # Extract acquisition parameters
        dwell = acquisition_parameters['dwell_time']
        rate = self.parameters['sample_rate']
        extra_left = acquisition_parameters['extrasteps_left']
        extra_right = acquisition_parameters['extrasteps_right']
        amp_x = acquisition_parameters['amplitude_x']
        amp_y = acquisition_parameters['amplitude_y']
        offset_x = acquisition_parameters['offset_x']
        offset_y = acquisition_parameters['offset_y']
        x_steps = acquisition_parameters['numsteps_x']
        y_steps = acquisition_parameters['numsteps_y']
        
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

class GalvoWidget(QWidget):
    def __init__(self, galvo):
        super().__init__()
        self.galvo = galvo
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create tab widget for config and control
        self.tab_widget = QTabWidget()
        
        # Configuration tab
        self.config_widget = GalvoConfigWidget()
        self.tab_widget.addTab(self.config_widget, "Configuration")
        
        # Control tab
        self.control_widget = GalvoControlWidget(self.galvo)
        self.tab_widget.addTab(self.control_widget, "Control")
        
        layout.addWidget(self.tab_widget)
        
        # Set current parameters
        if self.galvo.parameters:
            self.config_widget.set_parameters(self.galvo.parameters)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        """Get parameters from config widget"""
        return self.config_widget.get_parameters()
    
    def set_parameters(self, parameters):
        """Set parameters in config widget"""
        self.config_widget.set_parameters(parameters)

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
            from pyrpoc.instruments.instrument_manager import validate_instrument_parameters
            validate_instrument_parameters("Galvo", parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None

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
        
        # Status display
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

    def home_galvo(self):
        # TODO: Implement galvo homing
        print("Homing galvo...")
        self.status_label.setText("Status: Homing...")

    def stop_galvo(self):
        # TODO: Implement galvo stop
        print("Stopping galvo...")
        self.status_label.setText("Status: Stopped")