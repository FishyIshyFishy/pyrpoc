from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QProgressBar, QComboBox
from PyQt6.QtCore import pyqtSignal, QTimer
from superqt import QSearchableComboBox

class LocalRPOCProgressDialog(QDialog):
    """Progress dialog for local RPOC treatment"""
    
    def __init__(self, total_repetitions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Local RPOC Treatment Progress")
        self.setModal(True)
        self.setFixedSize(400, 150)
        
        layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Preparing treatment...")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, total_repetitions)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Repetition counter
        self.repetition_label = QLabel(f"Repetition: 0/{total_repetitions}")
        layout.addWidget(self.repetition_label)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)
        
        self.setLayout(layout)
        
        self.total_repetitions = total_repetitions
        self.current_repetition = 0
    
    def update_progress(self, repetition_number):
        """Update progress bar and labels"""
        self.current_repetition = repetition_number
        self.progress_bar.setValue(repetition_number)
        self.repetition_label.setText(f"Repetition: {repetition_number}/{self.total_repetitions}")
        self.status_label.setText(f"Treatment in progress... (Repetition {repetition_number})")
    
    def set_completed(self):
        """Mark treatment as completed"""
        self.status_label.setText("Treatment completed!")
        self.progress_bar.setValue(self.total_repetitions)
        self.repetition_label.setText(f"Repetition: {self.total_repetitions}/{self.total_repetitions}")
        self.cancel_btn.setText("Close")
    
    def set_stopped(self):
        """Mark treatment as stopped"""
        self.status_label.setText("Treatment stopped.")
        self.cancel_btn.setText("Close")

class LocalRPOCDialog(QDialog):
    """Dialog for configuring local RPOC parameters"""
    
    def __init__(self, app_state, signals, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.signals = signals
        self.setWindowTitle("Local RPOC Configuration")
        self.setModal(True)
        self.resize(500, 600)
        
        layout = QVBoxLayout()
        
        # Acquisition parameters group
        self.add_acquisition_parameters(layout)
        
        # Drift offset parameters group
        self.add_drift_parameters(layout)
        
        # Repetitions parameter
        self.add_repetitions_parameter(layout)
        
        # TTL channel selection
        self.add_ttl_channel_selection(layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        start_btn = QPushButton("Start Local RPOC")
        start_btn.clicked.connect(self.start_local_rpoc)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(start_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def add_acquisition_parameters(self, layout):
        """Add acquisition parameters from app_state"""
        acq_group = QGroupBox("Acquisition Parameters")
        acq_layout = QFormLayout()
        
        # Dwell time
        self.dwell_time_spin = QDoubleSpinBox()
        self.dwell_time_spin.setRange(1, 1000)
        self.dwell_time_spin.setValue(self.app_state.acquisition_parameters.get('dwell_time', 10))
        self.dwell_time_spin.setSuffix(" μs")
        self.dwell_time_spin.setDecimals(0)
        self.dwell_time_spin.setSingleStep(1)
        acq_layout.addRow("Dwell Time:", self.dwell_time_spin)
        
        # Extra steps
        self.extrasteps_left_spin = QSpinBox()
        self.extrasteps_left_spin.setRange(0, 10000)
        self.extrasteps_left_spin.setValue(self.app_state.acquisition_parameters.get('extrasteps_left', 50))
        acq_layout.addRow("Extra Steps Left:", self.extrasteps_left_spin)
        
        self.extrasteps_right_spin = QSpinBox()
        self.extrasteps_right_spin.setRange(0, 10000)
        self.extrasteps_right_spin.setValue(self.app_state.acquisition_parameters.get('extrasteps_right', 50))
        acq_layout.addRow("Extra Steps Right:", self.extrasteps_right_spin)
        
        # Amplitude
        self.amplitude_x_spin = QDoubleSpinBox()
        self.amplitude_x_spin.setRange(0.0, 10.0)
        self.amplitude_x_spin.setDecimals(1)
        self.amplitude_x_spin.setSingleStep(0.1)
        self.amplitude_x_spin.setValue(self.app_state.acquisition_parameters.get('amplitude_x', 0.5))
        self.amplitude_x_spin.setSuffix(" V")
        acq_layout.addRow("Amplitude X:", self.amplitude_x_spin)
        
        self.amplitude_y_spin = QDoubleSpinBox()
        self.amplitude_y_spin.setRange(0.0, 10.0)
        self.amplitude_y_spin.setDecimals(1)
        self.amplitude_y_spin.setSingleStep(0.1)
        self.amplitude_y_spin.setValue(self.app_state.acquisition_parameters.get('amplitude_y', 0.5))
        self.amplitude_y_spin.setSuffix(" V")
        acq_layout.addRow("Amplitude Y:", self.amplitude_y_spin)
        
        # Offset
        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-10.0, 10.0)
        self.offset_x_spin.setDecimals(1)
        self.offset_x_spin.setSingleStep(0.1)
        self.offset_x_spin.setValue(self.app_state.acquisition_parameters.get('offset_x', 0.0))
        self.offset_x_spin.setSuffix(" V")
        acq_layout.addRow("Offset X:", self.offset_x_spin)
        
        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-10.0, 10.0)
        self.offset_y_spin.setDecimals(1)
        self.offset_y_spin.setSingleStep(0.1)
        self.offset_y_spin.setValue(self.app_state.acquisition_parameters.get('offset_y', 0.0))
        self.offset_y_spin.setSuffix(" V")
        acq_layout.addRow("Offset Y:", self.offset_y_spin)
        
        # Pixels
        self.x_pixels_spin = QSpinBox()
        self.x_pixels_spin.setRange(64, 4096)
        self.x_pixels_spin.setValue(self.app_state.acquisition_parameters.get('x_pixels', 512))
        acq_layout.addRow("X Pixels:", self.x_pixels_spin)
        
        self.y_pixels_spin = QSpinBox()
        self.y_pixels_spin.setRange(64, 4096)
        self.y_pixels_spin.setValue(self.app_state.acquisition_parameters.get('y_pixels', 512))
        acq_layout.addRow("Y Pixels:", self.y_pixels_spin)
        
        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)
    
    def add_drift_parameters(self, layout):
        """Add drift offset parameters"""
        drift_group = QGroupBox("Drift Offset Parameters")
        drift_layout = QFormLayout()
        
        self.offset_drift_x_spin = QDoubleSpinBox()
        self.offset_drift_x_spin.setRange(-10.0, 10.0)
        self.offset_drift_x_spin.setDecimals(3)
        self.offset_drift_x_spin.setSingleStep(0.001)
        self.offset_drift_x_spin.setValue(0.0)
        self.offset_drift_x_spin.setSuffix(" V")
        drift_layout.addRow("Offset Drift X:", self.offset_drift_x_spin)
        
        self.offset_drift_y_spin = QDoubleSpinBox()
        self.offset_drift_y_spin.setRange(-10.0, 10.0)
        self.offset_drift_y_spin.setDecimals(3)
        self.offset_drift_y_spin.setSingleStep(0.001)
        self.offset_drift_y_spin.setValue(0.0)
        self.offset_drift_y_spin.setSuffix(" V")
        drift_layout.addRow("Offset Drift Y:", self.offset_drift_y_spin)
        
        drift_group.setLayout(drift_layout)
        layout.addWidget(drift_group)
    
    def add_repetitions_parameter(self, layout):
        """Add number of repetitions parameter"""
        reps_layout = QHBoxLayout()
        reps_layout.addWidget(QLabel("Number of Repetitions:"))
        
        self.repetitions_spin = QSpinBox()
        self.repetitions_spin.setRange(1, 1000)
        self.repetitions_spin.setValue(1)
        reps_layout.addWidget(self.repetitions_spin)
        
        layout.addLayout(reps_layout)
    
    def add_ttl_channel_selection(self, layout):
        """Add TTL channel selection"""
        ttl_group = QGroupBox("TTL Channel Selection")
        ttl_layout = QFormLayout()
        
        # Device selection
        self.ttl_device_combo = QSearchableComboBox()
        self.ttl_device_combo.addItems(['Dev1', 'Dev2', 'Dev3', 'Dev4'])
        
        # Port/Line selection
        self.ttl_port_line_combo = QSearchableComboBox()
        self.ttl_port_line_combo.addItems([
            'port0/line0', 'port0/line1', 'port0/line2', 'port0/line3', 'port0/line4', 'port0/line5', 'port0/line6', 'port0/line7',
            'port0/line8', 'port0/line9', 'port0/line10', 'port0/line11', 'port0/line12', 'port0/line13', 'port0/line14', 'port0/line15',
            'port1/line0', 'port1/line1', 'port1/line2', 'port1/line3', 'port1/line4', 'port1/line5', 'port1/line6', 'port1/line7',
            'port1/line8', 'port1/line9', 'port1/line10', 'port1/line11', 'port1/line12', 'port1/line13', 'port1/line14', 'port1/line15'
        ])
        
        # Set default values from the parent widget (RPOC channel)
        if hasattr(self.parent(), 'channel_id') and hasattr(self.parent(), 'app_state'):
            channel_id = self.parent().channel_id
            if hasattr(self.parent().app_state, 'rpoc_mask_channels') and channel_id in self.parent().app_state.rpoc_mask_channels:
                channel_data = self.parent().app_state.rpoc_mask_channels[channel_id]
                default_device = channel_data.get('device', 'Dev1')
                default_port_line = channel_data.get('port_line', 'port0/line0')
                
                self.ttl_device_combo.setCurrentText(default_device)
                self.ttl_port_line_combo.setCurrentText(default_port_line)
        
        ttl_layout.addRow("Device:", self.ttl_device_combo)
        ttl_layout.addRow("Port/Line:", self.ttl_port_line_combo)
        
        ttl_group.setLayout(ttl_layout)
        layout.addWidget(ttl_group)
    
    def get_parameters(self):
        """Get all parameters from the dialog"""
        return {
            'dwell_time': self.dwell_time_spin.value(),
            'extrasteps_left': self.extrasteps_left_spin.value(),
            'extrasteps_right': self.extrasteps_right_spin.value(),
            'amplitude_x': self.amplitude_x_spin.value(),
            'amplitude_y': self.amplitude_y_spin.value(),
            'offset_x': self.offset_x_spin.value(),
            'offset_y': self.offset_y_spin.value(),
            'x_pixels': self.x_pixels_spin.value(),
            'y_pixels': self.y_pixels_spin.value(),
            'offset_drift_x': self.offset_drift_x_spin.value(),
            'offset_drift_y': self.offset_drift_y_spin.value(),
            'repetitions': self.repetitions_spin.value(),
            'ttl_device': self.ttl_device_combo.currentText(),
            'ttl_port_line': self.ttl_port_line_combo.currentText()
        }
    
    def start_local_rpoc(self):
        """Start the local RPOC treatment"""
        parameters = self.get_parameters()
        self.signals.local_rpoc_started.emit(parameters)
        self.accept()
