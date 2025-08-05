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

        print(f'printing galvo test params')
        print(f'    dwell_time: {dwell} µs')
        print(f'    dwell_sec: {dwell_sec} s')
        print(f'    rate: {rate} Hz')
        print(f'    extra_left: {extra_left}')
        print(f'    extra_right: {extra_right}')
        print(f'    amp_x: {amp_x} V')
        print(f'    amp_y: {amp_y} V')
        print(f'    offset_x: {offset_x} V')
        print(f'    offset_y: {offset_y} V')
        print(f'    x_steps: {x_steps}')
        print(f'    y_steps: {y_steps}')
        
        pixel_samples = max(1, int(dwell_sec * rate))
        print(f'    pixel_samples: {pixel_samples}')
        total_x = x_steps + extra_left + extra_right
        print(f'    total_x: {total_x}')
        total_y = y_steps
        print(f'    total_y: {total_y}')

        contained_rowsamples = pixel_samples * x_steps
        print(f'    contained_rowsamples: {contained_rowsamples}')
        total_rowsamples = pixel_samples * total_x
        print(f'    total_rowsamples: {total_rowsamples}')
        
        step_size = (2 * amp_x) / contained_rowsamples
        print(f'    step_size: {step_size} V/sample')
        bottom = offset_x - amp_x - (step_size * extra_left)
        print(f'    bottom: {bottom} V')
        top = offset_x + amp_x + (step_size * extra_right)
        print(f'    top: {top} V')

        single_row_ramp = np.linspace(bottom, top, total_rowsamples, endpoint=False)
        print(f'    single_row_ramp shape: {single_row_ramp.shape}')
        print(f'    single_row_ramp range: {single_row_ramp[0]:.3f} to {single_row_ramp[-1]:.3f} V')
        x_waveform = np.tile(single_row_ramp, total_y)
        print(f'    x_waveform shape: {x_waveform.shape}')

        y_steps_positions = np.linspace(
            offset_y + amp_y,
            offset_y - amp_y,
            total_y
        )
        print(f'    y_steps_positions shape: {y_steps_positions.shape}')
        print(f'    y_steps_positions range: {y_steps_positions[0]:.3f} to {y_steps_positions[-1]:.3f} V')
        y_waveform = np.repeat(y_steps_positions, total_rowsamples)
        print(f'    y_waveform shape: {y_waveform.shape}')

        composite = np.vstack([x_waveform, y_waveform])
        print(f'    composite shape: {composite.shape}')

        total_samples = total_x * total_y * pixel_samples
        print(f'    total_samples: {total_samples}')
        if x_waveform.size < total_samples:
            print(f'    padding x_waveform from {x_waveform.size} to {total_samples}')
            x_waveform = np.pad(
                x_waveform,
                (0, total_samples - x_waveform.size),
                constant_values=x_waveform[-1]
            )
        else:
            print(f'    truncating x_waveform from {x_waveform.size} to {total_samples}')
            x_waveform = x_waveform[:total_samples]
        composite[0] = x_waveform
        
        print(f'    final composite shape: {composite.shape}')
        print(f'    final x_waveform range: {composite[0].min():.3f} to {composite[0].max():.3f} V')
        print(f'    final y_waveform range: {composite[1].min():.3f} to {composite[1].max():.3f} V')
        
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