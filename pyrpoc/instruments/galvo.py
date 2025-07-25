import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from pyrpoc.instruments.base_instrument import Instrument
import numpy as np

class Galvo(Instrument):
    def __init__(self, name="Galvo Scanner", console_callback=None):
        super().__init__(name, "galvo", console_callback=console_callback)

        self.parameters = {
            'slow_axis_channel': 0,
            'fast_axis_channel': 1,
            'sample_rate': 1000000,
            'device_name': 'Dev1'
        }

    def initialize(self):
        return True

    def get_widget(self):
        return GalvoWidget(self)
    
    def validate_parameters(self, parameters):
        required_params = ['slow_axis_channel', 'fast_axis_channel', 'sample_rate', 'device_name']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for galvo: {param}")

        if parameters['slow_axis_channel'] < 0 or parameters['fast_axis_channel'] < 0:
            raise ValueError("Channel numbers must be non-negative")
        if parameters['sample_rate'] <= 0:
            raise ValueError("Sample rate must be positive")
    
    def generate_raster_waveform(self, acquisition_parameters):
        dwell = acquisition_parameters['dwell_time']  # microseconds
        dwell_sec = dwell / 1e6  # convert to seconds
        rate = self.parameters['sample_rate']
        extra_left = acquisition_parameters['extrasteps_left']
        extra_right = acquisition_parameters['extrasteps_right']
        amp_x = acquisition_parameters['amplitude_x']
        amp_y = acquisition_parameters['amplitude_y']
        offset_x = acquisition_parameters['offset_x']
        offset_y = acquisition_parameters['offset_y']
        x_steps = acquisition_parameters['x_pixels']
        y_steps = acquisition_parameters['y_pixels']

        pixel_samples = max(1, int(dwell_sec * rate))
        total_x = x_steps + extra_left + extra_right
        total_y = y_steps

        contained_rowsamples = pixel_samples * x_steps
        total_rowsamples = pixel_samples * total_x
        
        step_size = (2 * amp_x) / contained_rowsamples
        bottom = offset_x - amp_x - (step_size * extra_left)
        top = offset_x + amp_x + (step_size * extra_right)

        single_row_ramp = np.linspace(bottom, top, total_rowsamples, endpoint=False)
        x_waveform = np.tile(single_row_ramp, total_y)

        y_steps_positions = np.linspace(
            offset_y + amp_y,
            offset_y - amp_y,
            total_y
        )
        y_waveform = np.repeat(y_steps_positions, total_rowsamples)

        composite = np.vstack([x_waveform, y_waveform])

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

        self.config_widget = GalvoConfigWidget(self.galvo)
        layout.addWidget(self.config_widget)

        if self.galvo.parameters:
            self.config_widget.set_parameters(self.galvo.parameters)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        return self.config_widget.get_parameters()
    
    def set_parameters(self, parameters):
        self.config_widget.set_parameters(parameters)

class GalvoConfigWidget(QWidget):
    def __init__(self, galvo):
        super().__init__()
        self.galvo = galvo
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
        if 'sample_rate' in parameters:
            self.sample_rate_spin.setValue(parameters['sample_rate'])

    def get_parameters(self):
        parameters = {
            'name': self.name_edit.text(),
            'device_name': self.device_combo.currentText(),
            'slow_axis_channel': self.slow_channel_spin.value(),
            'fast_axis_channel': self.fast_channel_spin.value(),
            'sample_rate': self.sample_rate_spin.value()
        }
        
        try:
            self.galvo.validate_parameters(parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None