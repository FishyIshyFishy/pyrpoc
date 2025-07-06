import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from pyrpoc.instruments.base_instrument import Instrument

class PriorStage(Instrument):
    def __init__(self, name="Prior Stage"):
        super().__init__(name, "Prior Stage")
        # Connection and safety parameters
        self.parameters = {
            'port': 4,  # COM port number
            'max_z_height': 50000,  # Maximum Z height in µm
            'safe_move_distance': 10000  # Safe movement distance in µm
        }
        self._connected = False
        self._sdk_prior = None
        self._session_id = None

    def initialize(self):
        """Initialize Prior stage connection"""
        try:
            from ctypes import WinDLL, create_string_buffer
            import os
            
            # Load DLL
            dll_path = os.path.join(os.path.dirname(__file__), "..", "old", "helpers", "prior_stage", "PriorScientificSDK.dll")
            if not os.path.exists(dll_path):
                raise RuntimeError(f"PriorScientificSDK.dll not found at {dll_path}")
            
            self._sdk_prior = WinDLL(dll_path)
            
            # Initialize SDK
            ret = self._sdk_prior.PriorScientificSDK_Initialise()
            if ret != 0:
                raise RuntimeError(f"Failed to initialize Prior SDK. Error code: {ret}")
            
            # Open session
            self._session_id = self._sdk_prior.PriorScientificSDK_OpenNewSession()
            if self._session_id < 0:
                raise RuntimeError(f"Failed to open Prior SDK session. SessionID: {self._session_id}")
            
            # Connect to controller
            port = self.parameters['port']
            ret, _ = self._send_command(f"controller.connect {port}")
            if ret != 0:
                raise RuntimeError(f"Failed to connect to Prior stage on COM{port}")
            
            self._connected = True
            print(f"Prior stage initialized on COM{port}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize Prior stage: {e}")
            return False

    def _send_command(self, command):
        """Send command to Prior stage"""
        if not self._connected or self._sdk_prior is None:
            raise RuntimeError("Prior stage not initialized")
        
        from ctypes import create_string_buffer
        
        rx = create_string_buffer(1000)
        ret = self._sdk_prior.PriorScientificSDK_cmd(
            self._session_id, create_string_buffer(command.encode()), rx
        )
        response = rx.value.decode().strip()
        
        if ret != 0:
            print(f"Error executing command: {command} (Return Code: {ret})")
        
        return ret, response

    def _wait_for_z_motion(self):
        """Wait for Z motion to complete"""
        import time
        while True:
            _, response = self._send_command("controller.z.busy.get")
            
            if response:
                try:
                    status = int(response)
                    if status == 0:
                        break
                except ValueError:
                    print(f"Invalid response from controller: '{response}'")
            else:
                print("No response from controller, is it connected?")
            
            time.sleep(0.1)

    def move_z(self, z_height, max_z_height=None):
        """Move Z stage to specified height in µm"""
        if max_z_height is None:
            max_z_height = self.parameters.get('max_z_height', 50000)
        if not (0 <= z_height <= max_z_height):
            raise ValueError(f"Z height must be between 0 and {max_z_height} µm.")
        
        ret, _ = self._send_command(f"controller.z.goto-position {z_height}")
        if ret != 0:
            raise RuntimeError(f"Could not move Prior stage to {z_height} µm.")
        self._wait_for_z_motion()

    def move_xy(self, x, y, safe_move_distance=None):
        """Move XY stage to specified position in µm"""
        current_x, current_y = self.get_xy()
        if safe_move_distance is None:
            safe_move_distance = self.parameters.get('safe_move_distance', 10000)
        
        if not (current_x - safe_move_distance <= x <= current_x + safe_move_distance) or \
           not (current_y - safe_move_distance <= y <= current_y + safe_move_distance):
            raise ValueError(f"Entered position is more than {safe_move_distance} µm away, and may be unsafe. Cancelling...")
        
        ret, _ = self._send_command(f"controller.stage.goto-position {x} {y}")
        if ret != 0:
            raise RuntimeError(f"Could not move Prior stage to {x}, {y}.")

    def get_xy(self):
        """Get current XY position in µm"""
        ret, response = self._send_command("controller.stage.position.get")
        if ret != 0:
            raise RuntimeError("Failed to get XY position.")
        try:
            return tuple(map(int, response.split(",")))
        except ValueError:
            raise RuntimeError(f"Invalid XY position response: '{response}'")

    def get_z(self):
        """Get current Z position in µm"""
        ret, response = self._send_command("controller.z.position.get")
        if ret != 0:
            raise RuntimeError("Failed to get Z position.")
        try:
            return int(response)
        except ValueError:
            raise RuntimeError(f"Invalid Z position response: '{response}'")

    def get_widget(self):
        """Return unified Prior stage widget"""
        return PriorStageWidget(self)
    


class PriorStageWidget(QWidget):
    def __init__(self, prior_stage):
        super().__init__()
        self.prior_stage = prior_stage
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create tab widget for config and control
        self.tab_widget = QTabWidget()
        
        # Configuration tab
        self.config_widget = PriorStageConfigWidget()
        self.tab_widget.addTab(self.config_widget, "Configuration")
        
        # Control tab
        self.control_widget = PriorStageControlWidget(self.prior_stage)
        self.tab_widget.addTab(self.control_widget, "Control")
        
        layout.addWidget(self.tab_widget)
        
        # Set current parameters
        if self.prior_stage.parameters:
            self.config_widget.set_parameters(self.prior_stage.parameters)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        """Get parameters from config widget"""
        return self.config_widget.get_parameters()
    
    def set_parameters(self, parameters):
        """Set parameters in config widget"""
        self.config_widget.set_parameters(parameters)

class PriorStageConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_edit = QLineEdit("Prior Stage")
        layout.addRow("Name:", self.name_edit)
        
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 10)
        self.port_spin.setValue(4)
        layout.addRow("COM Port:", self.port_spin)
        
        # Safety parameters
        self.max_z_height_spin = QDoubleSpinBox()
        self.max_z_height_spin.setRange(10000, 100000)
        self.max_z_height_spin.setValue(50000)
        self.max_z_height_spin.setSuffix(" µm")
        layout.addRow("Max Z Height:", self.max_z_height_spin)
        
        self.safe_move_distance_spin = QDoubleSpinBox()
        self.safe_move_distance_spin.setRange(1000, 100000)
        self.safe_move_distance_spin.setValue(10000)
        self.safe_move_distance_spin.setSuffix(" µm")
        layout.addRow("Safe Move Distance:", self.safe_move_distance_spin)
        
        self.setLayout(layout)

    def set_parameters(self, parameters):
        """Set the widget values from parameters"""
        if 'name' in parameters:
            self.name_edit.setText(parameters['name'])
        if 'port' in parameters:
            self.port_spin.setValue(parameters['port'])
        if 'max_z_height' in parameters:
            self.max_z_height_spin.setValue(parameters['max_z_height'])
        if 'safe_move_distance' in parameters:
            self.safe_move_distance_spin.setValue(parameters['safe_move_distance'])

    def get_parameters(self):
        parameters = {
            'name': self.name_edit.text(),
            'port': self.port_spin.value(),
            'max_z_height': self.max_z_height_spin.value(),
            'safe_move_distance': self.safe_move_distance_spin.value()
        }
        
        # Validate parameters before returning
        try:
            from pyrpoc.instruments.instrument_manager import validate_instrument_parameters
            validate_instrument_parameters("Prior Stage", parameters)
            return parameters
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None

class PriorStageControlWidget(QWidget):
    def __init__(self, prior_stage):
        super().__init__()
        self.prior_stage = prior_stage
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # XY Position controls
        xy_group = QGroupBox("XY Position")
        xy_layout = QFormLayout()
        
        self.x_pos_spin = QDoubleSpinBox()
        self.x_pos_spin.setRange(-50000, 50000)
        self.x_pos_spin.setSuffix(" µm")
        xy_layout.addRow("X Position:", self.x_pos_spin)
        
        self.y_pos_spin = QDoubleSpinBox()
        self.y_pos_spin.setRange(-50000, 50000)
        self.y_pos_spin.setSuffix(" µm")
        xy_layout.addRow("Y Position:", self.y_pos_spin)
        
        self.move_xy_btn = QPushButton("Move XY")
        self.move_xy_btn.clicked.connect(self.move_xy)
        xy_layout.addRow("", self.move_xy_btn)
        
        xy_group.setLayout(xy_layout)
        layout.addWidget(xy_group)
        
        # Z Position controls
        z_group = QGroupBox("Z Position")
        z_layout = QFormLayout()
        
        self.z_pos_spin = QDoubleSpinBox()
        self.z_pos_spin.setRange(0, 50000)
        self.z_pos_spin.setSuffix(" µm")
        z_layout.addRow("Z Position:", self.z_pos_spin)
        
        self.move_z_btn = QPushButton("Move Z")
        self.move_z_btn.clicked.connect(self.move_z)
        z_layout.addRow("", self.move_z_btn)
        
        z_group.setLayout(z_layout)
        layout.addWidget(z_group)
        
        # Current position display
        pos_group = QGroupBox("Current Position")
        pos_layout = QFormLayout()
        
        self.current_x_label = QLabel("0 µm")
        pos_layout.addRow("Current X:", self.current_x_label)
        
        self.current_y_label = QLabel("0 µm")
        pos_layout.addRow("Current Y:", self.current_y_label)
        
        self.current_z_label = QLabel("0 µm")
        pos_layout.addRow("Current Z:", self.current_z_label)
        
        self.refresh_btn = QPushButton("Refresh Position")
        self.refresh_btn.clicked.connect(self.refresh_position)
        pos_layout.addRow("", self.refresh_btn)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        self.setLayout(layout)
        
        # Initial position refresh
        self.refresh_position()

    def move_xy(self):
        try:
            x = int(self.x_pos_spin.value())
            y = int(self.y_pos_spin.value())
            self.prior_stage.move_xy(x, y)
            self.refresh_position()
        except Exception as e:
            QMessageBox.warning(self, "Movement Error", f"Failed to move XY: {e}")

    def move_z(self):
        try:
            z = int(self.z_pos_spin.value())
            self.prior_stage.move_z(z)
            self.refresh_position()
        except Exception as e:
            QMessageBox.warning(self, "Movement Error", f"Failed to move Z: {e}")

    def refresh_position(self):
        try:
            x, y = self.prior_stage.get_xy()
            z = self.prior_stage.get_z()
            
            self.current_x_label.setText(f"{x} µm")
            self.current_y_label.setText(f"{y} µm")
            self.current_z_label.setText(f"{z} µm")
            
            # Update spin boxes to current position
            self.x_pos_spin.setValue(x)
            self.y_pos_spin.setValue(y)
            self.z_pos_spin.setValue(z)
            
        except Exception as e:
            self.current_x_label.setText("Error")
            self.current_y_label.setText("Error")
            self.current_z_label.setText("Error")
            print(f"Error refreshing position: {e}")