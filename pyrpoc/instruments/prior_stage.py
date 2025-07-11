import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from pyrpoc.instruments.base_instrument import Instrument
from ctypes import create_string_buffer, WinDLL
import os

class PriorStage(Instrument):
    def __init__(self, name="Prior Stage", console_callback=None):
        super().__init__(name, "prior stage")
        self.console_callback = console_callback

        self.parameters = {
            'port': 4,  # COM port number
            'max_z_height': 50000,  # Maximum Z height in µm
            'safe_move_distance': 10000  # Safe movement distance in µm
        }
        self.connected = False
        self.sdk = None
        self.session_id = None

    def log_message(self, message):
        if self.console_callback:
            self.console_callback(message)

    def initialize(self):
        try:
            if self.connected:
                return True

            # Clean up any existing connection first
            self.cleanup()

            current_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(current_dir, "PriorScientificSDK.dll")
            
            if not os.path.exists(dll_path):
                self.log_message(f"Error: PriorScientificSDK.dll not found. Searched at: {dll_path}")
                return False
            
            self.sdk = WinDLL(dll_path)

            # Only initialize SDK if not already done
            if not hasattr(self, '_sdk_initialized'):
                ret = self.sdk.PriorScientificSDK_Initialise()
                if ret != 0:
                    self.log_message(f"Error: Failed to initialize Prior SDK. Error code: {ret}")
                    return False
                self._sdk_initialized = True
            
            self.session_id = self.sdk.PriorScientificSDK_OpenNewSession()
            if self.session_id < 0:
                self.log_message(f"Error: Failed to open Prior SDK session. SessionID: {self.session_id}")
                return False
            
            port = self.parameters['port']
            # Send connection command directly to avoid recursion
            rx = create_string_buffer(1000)
            ret = self.sdk.PriorScientificSDK_cmd(
                self.session_id, create_string_buffer(f"controller.connect {port}".encode()), rx
            )
            response = rx.value.decode().strip()
            
            if ret == 0:
                self.connected = True
                self.log_message(f"Connected to Prior stage on COM{port}")
                return True
            else:
                self.log_message(f"Error: Failed to connect to Prior stage on COM{port} (Return Code: {ret})")
                return False
            
        except Exception as e:
            self.log_message(f"Error initializing Prior stage: {e}")
            return False

    def send_command(self, command):
        if self.sdk is None or self.session_id is None:
            raise RuntimeError("Prior stage not initialized. Call initialize() first.")
            
        rx = create_string_buffer(1000)
        ret = self.sdk.PriorScientificSDK_cmd(
            self.session_id, create_string_buffer(command.encode()), rx
        )
        response = rx.value.decode().strip()
        
        if ret != 0:
            self.log_message(f"Error executing command: {command} (Return Code: {ret})")
        
        return ret, response

    def wait_for_z_motion(self):
        import time
        while True:
            _, response = self.send_command("controller.z.busy.get")
            
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
        self.check_connection()
        
        if max_z_height is None:
            max_z_height = self.parameters.get('max_z_height', 50000)
        # z_height is in 0.1 µm units, so convert max_z_height from µm to 0.1 µm units
        max_z_height_units = int(max_z_height * 10)
        if not (0 <= z_height <= max_z_height_units):
            raise ValueError(f"Z height must be between 0 and {max_z_height} µm (0-{max_z_height_units} in 0.1 µm units).")
        
        ret, _ = self.send_command(f"controller.z.goto-position {z_height}")
        if ret != 0:
            raise RuntimeError(f"Could not move Prior stage to {z_height} (0.1 µm units).")
        self.wait_for_z_motion()

    def move_xy(self, x, y, safe_move_distance=None):
        self.check_connection()
        
        current_x, current_y = self.get_xy()
        if safe_move_distance is None:
            safe_move_distance = self.parameters.get('safe_move_distance', 10000)
        
        if not (current_x - safe_move_distance <= x <= current_x + safe_move_distance) or \
           not (current_y - safe_move_distance <= y <= current_y + safe_move_distance):
            raise ValueError(f"Entered position is more than {safe_move_distance} µm away, and may be unsafe. Cancelling...")
        
        ret, _ = self.send_command(f"controller.stage.goto-position {x} {y}")
        if ret != 0:
            raise RuntimeError(f"Could not move Prior stage to {x}, {y}.")
        
        self.wait_for_z_motion()  

    def get_xy(self):
        self.check_connection()
        
        ret, response = self.send_command("controller.stage.position.get")
        if ret != 0:
            raise RuntimeError("Failed to get XY position.")
        try:
            return tuple(map(int, response.split(",")))
        except ValueError:
            raise RuntimeError(f"Invalid XY position response: '{response}'")

    def get_z(self):
        self.check_connection()
        
        ret, response = self.send_command("controller.z.position.get")
        if ret != 0:
            raise RuntimeError("Failed to get Z position.")
        try:
            # Z position is returned in 0.1 µm units
            return int(response)
        except ValueError:
            raise RuntimeError(f"Invalid Z position response: '{response}'")

    def test_connection(self):
        try:
            x, y = self.get_xy()
            z = self.get_z()
            print(f"Prior stage test successful - Current position: X={x} µm, Y={y} µm, Z={z/10:.1f} µm")
            return True
        except Exception as e:
            print(f"Prior stage test failed: {e}")
            return False

    def cleanup(self):
        try:
            if self.connected and self.sdk is not None and self.session_id is not None:
                # Send disconnect command to the controller first
                ret, response = self.send_command("controller.disconnect")
                if ret != 0:
                    self.log_message(f"Warning: controller.disconnect returned {ret}: {response}")
                
                # Close the SDK session
                self.sdk.PriorScientificSDK_CloseSession(self.session_id)
                self.session_id = None
            self.connected = False
            if hasattr(self, '_sdk_initialized'):
                delattr(self, '_sdk_initialized')
        except Exception as e:
            self.log_message(f"Error during Prior stage cleanup: {e}")

    def disconnect(self):
        self.log_message("Prior stage connection closed")
        self.cleanup()

    def get_widget(self):
        return PriorStageWidget(self)
    
    def validate_parameters(self, parameters):
        required_params = ['port']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for prior stage: {param}")
        
        if parameters['port'] < 1 or parameters['port'] > 10:
            raise ValueError("COM port must be between 1 and 10")
    
    def check_connection(self):
        if not self.connected:
            self.initialize()
    


class PriorStageWidget(QWidget):
    def __init__(self, prior_stage):
        super().__init__()
        self.prior_stage = prior_stage
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        self.tab_widget = QTabWidget()
        
        self.config_widget = PriorStageConfigWidget(self.prior_stage)
        self.tab_widget.addTab(self.config_widget, "Configuration")
        
        self.control_widget = PriorStageControlWidget(self.prior_stage)
        self.tab_widget.addTab(self.control_widget, "Control")
        
        layout.addWidget(self.tab_widget)
        
        if self.prior_stage.parameters:
            self.config_widget.set_parameters(self.prior_stage.parameters)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        return self.config_widget.get_parameters()
    
    def set_parameters(self, parameters):
        self.config_widget.set_parameters(parameters)

class PriorStageConfigWidget(QWidget):
    def __init__(self, prior_stage):
        super().__init__()
        self.prior_stage = prior_stage
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_edit = QLineEdit("Prior Stage")
        layout.addRow("Name:", self.name_edit)
        
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 10)
        self.port_spin.setValue(4)
        layout.addRow("COM Port:", self.port_spin)
        
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
        
        try:
            self.prior_stage.validate_parameters(parameters)
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
        
        z_group = QGroupBox("Z Position")
        z_layout = QFormLayout()
        
        self.z_pos_spin = QDoubleSpinBox()
        self.z_pos_spin.setRange(0, 5000)  # 0 to 5000 µm
        self.z_pos_spin.setSuffix(" µm")
        self.z_pos_spin.setSingleStep(0.1)  # 0.1 µm steps
        self.z_pos_spin.setDecimals(1)  # Show 1 decimal place
        z_layout.addRow("Z Position:", self.z_pos_spin)
        
        self.move_z_btn = QPushButton("Move Z")
        self.move_z_btn.clicked.connect(self.move_z)
        z_layout.addRow("", self.move_z_btn)
        
        z_group.setLayout(z_layout)
        layout.addWidget(z_group)
        
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
        
        # Only refresh position if already connected
        if self.prior_stage.connected:
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
            z_microns = self.z_pos_spin.value()  # Get value in µm
            z_units = int(z_microns * 10)  # Convert to 0.1 µm units
            self.prior_stage.move_z(z_units)
            self.refresh_position()
        except Exception as e:
            QMessageBox.warning(self, "Movement Error", f"Failed to move Z: {e}")

    def refresh_position(self):
        try:
            x, y = self.prior_stage.get_xy()
            z_units = self.prior_stage.get_z()  # Z in 0.1 µm units
            z_microns = z_units / 10.0  # Convert to µm
            
            self.current_x_label.setText(f"{x} µm")
            self.current_y_label.setText(f"{y} µm")
            self.current_z_label.setText(f"{z_microns:.1f} µm")
            
            self.x_pos_spin.setValue(x)
            self.y_pos_spin.setValue(y)
            self.z_pos_spin.setValue(z_microns)
            
        except Exception as e:
            self.current_x_label.setText("Error")
            self.current_y_label.setText("Error")
            self.current_z_label.setText("Error")
            print(f"Error refreshing position: {e}")