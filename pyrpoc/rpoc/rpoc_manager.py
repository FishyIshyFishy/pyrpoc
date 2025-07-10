from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QDialogButtonBox, QWidget, QFileDialog
from PyQt6.QtCore import pyqtSignal
from superqt import QSearchableComboBox
import cv2
import numpy as np
from pyrpoc.rpoc.rpoc_mask_editor import RPOCMaskEditor

class RPOCChannelSelector(QDialog):
    """Dialog for selecting RPOC channel type"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add RPOC Channel")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Channel type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Channel Type:"))
        self.channel_type_combo = QComboBox()
        self.channel_type_combo.addItems(["Mask Channel", "Script Channel", "Static Channel"])
        type_layout.addWidget(self.channel_type_combo)
        layout.addLayout(type_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_selected_type(self):
        """Get the selected channel type"""
        selection = self.channel_type_combo.currentText()
        if selection == "Mask Channel":
            return "mask"
        elif selection == "Script Channel":
            return "script"
        elif selection == "Static Channel":
            return "static"
        return "mask"  # default

def show_rpoc_channel_selector(parent=None):
    """Show the RPOC channel selector dialog and return the selected type"""
    dialog = RPOCChannelSelector(parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_selected_type()
    return None

class BaseRPOCChannelWidget(QWidget):
    """Base class for RPOC channel widgets"""
    
    def __init__(self, channel_id, app_state, signals, parent=None):
        super().__init__(parent)
        self.channel_id = channel_id
        self.app_state = app_state
        self.signals = signals
        self.channel_type = self.get_channel_type()  # Store channel type
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with channel label and remove button
        header_layout = QHBoxLayout()
        self.channel_label = QLabel(f'Channel {channel_id}')
        header_layout.addWidget(self.channel_label)

        remove_btn = QPushButton('×')
        remove_btn.setMaximumWidth(20)
        remove_btn.clicked.connect(self.remove_channel)
        header_layout.addWidget(remove_btn)
        
        layout.addLayout(header_layout)

        # DAQ configuration
        daq_layout = QHBoxLayout()
        daq_layout.addWidget(QLabel('Device:'))
        self.device_edit = QSearchableComboBox()

        default_device = self.app_state.rpoc_channels.get(self.channel_id, {}).get('device', 'Dev1')
        self.device_edit.setCurrentText(default_device)
        self.device_edit.currentTextChanged.connect(self.on_daq_channel_changed)
        daq_layout.addWidget(self.device_edit)
        
        daq_layout.addWidget(QLabel('port#/line#:'))
        self.port_line_edit = QSearchableComboBox()
        self.port_line_edit.addItems(['port0/line0', 'port0/line1', 'port0/line2', 'port0/line3', 'port0/line4', 'port0/line5', 'port0/line6', 'port0/line7', 
                                      'port0/line8', 'port0/line9', 'port0/line10', 'port0/line11', 'port0/line12', 'port0/line13', 'port0/line14', 'port0/line15', 
                                      'port1/line0', 'port1/line1', 'port1/line2', 'port1/line3', 'port1/line4', 'port1/line5', 'port1/line6', 'port1/line7', 
                                      'port1/line8', 'port1/line9', 'port1/line10', 'port1/line11', 'port1/line12', 'port1/line13', 'port1/line14', 'port1/line15'])
        
        default_port_line = self.app_state.rpoc_channels.get(self.channel_id, {}).get('port_line', f'port0/line{3+channel_id}')
        self.port_line_edit.setCurrentText(default_port_line)
        self.port_line_edit.currentTextChanged.connect(self.on_daq_channel_changed)
        daq_layout.addWidget(self.port_line_edit)
        
        layout.addLayout(daq_layout)
        
        # Add channel-specific content
        self.add_channel_content(layout)
        
        self.setLayout(layout)
        self.on_daq_channel_changed()
    
    def get_channel_type(self):
        """Override in subclasses to return the channel type"""
        return "mask"  # Default
    
    def add_channel_content(self, layout):
        """Override in subclasses to add channel-specific content"""
        pass
    
    def on_daq_channel_changed(self):
        if not hasattr(self.app_state, 'rpoc_channels'):
            self.app_state.rpoc_channels = {}
        
        device = self.device_edit.currentText().strip()
        port_line = self.port_line_edit.currentText().strip()
        
        self.app_state.rpoc_channels[self.channel_id] = {
            'device': device,
            'port_line': port_line,
            'channel_type': self.channel_type  # Include channel type in saved data
        }
        self.signals.console_message.emit(f'RPOC channel {self.channel_id} set on {device}/{port_line}')
    
    def get_daq_channel_info(self):
        device = self.device_edit.currentText().strip()
        port_line = self.port_line_edit.currentText().strip()
        return {
            'device': device,
            'port_line': port_line
        }
    
    def remove_channel(self):
        # Remove channel-specific data from app_state
        self.remove_channel_data()
        
        # Remove DAQ channel info from app_state
        if hasattr(self.app_state, 'rpoc_channels') and self.channel_id in self.app_state.rpoc_channels:
            del self.app_state.rpoc_channels[self.channel_id]
        
        # Emit signal to remove this widget
        self.signals.rpoc_channel_removed.emit(self.channel_id)
        self.deleteLater()
    
    def remove_channel_data(self):
        """Override in subclasses to remove channel-specific data"""
        pass

class RPOCMaskChannelWidget(BaseRPOCChannelWidget):
    """Widget for mask-based RPOC channels"""
    
    def __init__(self, channel_id, app_state, signals, parent=None):
        self.mask_editor = None
        super().__init__(channel_id, app_state, signals, parent)
    
    def get_channel_type(self):
        return "mask"
    
    def add_channel_content(self, layout):
        # Status label
        self.mask_status = QLabel('No mask loaded')
        self.mask_status.setStyleSheet('color: #666; font-size: 10px;')
        layout.addWidget(self.mask_status)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.create_mask_btn = QPushButton('Create Mask')
        self.create_mask_btn.clicked.connect(self.create_mask)
        buttons_layout.addWidget(self.create_mask_btn)
        
        self.load_mask_btn = QPushButton('Load Mask')
        self.load_mask_btn.clicked.connect(self.load_mask)
        buttons_layout.addWidget(self.load_mask_btn)
        
        layout.addLayout(buttons_layout)
        
        self.update_mask_status()
    
    def create_mask(self):
        image_data = self.get_current_image_data()
        if image_data is None:
            self.signals.console_message.emit('No image data available for mask creation. Please acquire an image first.')
            return
            
        self.mask_editor = RPOCMaskEditor(image_data=image_data)
        self.mask_editor.mask_created.connect(self.handle_mask_created)
        self.mask_editor.show()
    
    def load_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Mask', '', 'Image files (*.png *.tif *.tiff);;All files (*)'
        )
        if file_path:
            try:
                mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    self.handle_mask_loaded(mask)
                else:
                    self.signals.console_message.emit(f"Failed to load mask from {file_path}")
            except Exception as e:
                self.signals.console_message.emit(f"Error loading mask: {e}")
    
    def get_current_image_data(self):
        widget = self
        while widget is not None:
            if hasattr(widget, 'app_state') and hasattr(widget, 'mid_layout'):
                # should be traversed up to main_window by here
                if hasattr(widget.mid_layout, 'image_display_widget'):
                    image_widget = widget.mid_layout.image_display_widget
                    if hasattr(image_widget, 'get_image_data_for_rpoc'):
                        return image_widget.get_image_data_for_rpoc()
                break
            widget = widget.parent()
        
        return None
    
    def handle_mask_created(self, mask):
        self.handle_mask_loaded(mask)
        self.signals.mask_created.emit(mask)
    
    def handle_mask_loaded(self, mask):
        # store the mask in app_state
        if not hasattr(self.app_state, 'rpoc_masks'):
            self.app_state.rpoc_masks = {}
        self.app_state.rpoc_masks[self.channel_id] = mask
        self.update_mask_status()
        self.signals.console_message.emit(f"Mask loaded for channel {self.channel_id} - shape: {mask.shape if hasattr(mask, 'shape') else 'unknown'}")
    
    def update_mask_status(self):
        if hasattr(self.app_state, 'rpoc_masks') and self.channel_id in self.app_state.rpoc_masks:
            mask = self.app_state.rpoc_masks[self.channel_id]
            if mask is not None:
                self.mask_status.setText(f'Mask: {mask.shape[1]}x{mask.shape[0]}')
                self.mask_status.setStyleSheet('color: #4CAF50; font-size: 10px;')
            else:
                self.mask_status.setText('No mask loaded')
                self.mask_status.setStyleSheet('color: #666; font-size: 10px;')
        else:
            self.mask_status.setText('No mask loaded')
            self.mask_status.setStyleSheet('color: #666; font-size: 10px;')
    
    def remove_channel_data(self):
        # Remove mask from app_state
        if hasattr(self.app_state, 'rpoc_masks') and self.channel_id in self.app_state.rpoc_masks:
            del self.app_state.rpoc_masks[self.channel_id]

class RPOCScriptChannelWidget(BaseRPOCChannelWidget):
    """Widget for script-based RPOC channels (placeholder)"""
    
    def get_channel_type(self):
        return "script"
    
    def add_channel_content(self, layout):
        # Placeholder content
        placeholder_label = QLabel('Script Channel - Coming Soon')
        placeholder_label.setStyleSheet('color: #666; font-style: italic;')
        layout.addWidget(placeholder_label)
    
    def remove_channel_data(self):
        # No specific data to remove for script channels yet
        pass

class RPOCStaticChannelWidget(BaseRPOCChannelWidget):
    """Widget for static RPOC channels"""
    
    def get_channel_type(self):
        return "static"
    
    def add_channel_content(self, layout):
        # Static level selector
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel('Static Level:'))
        self.static_level_combo = QComboBox()
        self.static_level_combo.addItems(['Static High', 'Static Low'])
        
        # Restore static level from config if available
        default_level = 'Static Low'
        if hasattr(self.app_state, 'rpoc_static_channels') and self.channel_id in self.app_state.rpoc_static_channels:
            saved_level = self.app_state.rpoc_static_channels[self.channel_id].get('level', default_level)
            self.static_level_combo.setCurrentText(saved_level)
        else:
            self.static_level_combo.setCurrentText(default_level)
        
        self.static_level_combo.currentTextChanged.connect(self.on_static_level_changed)
        level_layout.addWidget(self.static_level_combo)
        layout.addLayout(level_layout)
        
        # Initialize static level in app_state
        self.on_static_level_changed()
    
    def on_static_level_changed(self):
        if not hasattr(self.app_state, 'rpoc_static_channels'):
            self.app_state.rpoc_static_channels = {}
        
        static_level = self.static_level_combo.currentText()
        self.app_state.rpoc_static_channels[self.channel_id] = {
            'level': static_level
        }
        self.signals.console_message.emit(f'RPOC static channel {self.channel_id} set to {static_level}')
    
    def remove_channel_data(self):
        # Remove static channel data from app_state
        if hasattr(self.app_state, 'rpoc_static_channels') and self.channel_id in self.app_state.rpoc_static_channels:
            del self.app_state.rpoc_static_channels[self.channel_id]

def create_rpoc_channel_widget(channel_type, channel_id, app_state, signals, parent=None):
    """Factory function to create the appropriate RPOC channel widget"""
    if channel_type == "mask":
        return RPOCMaskChannelWidget(channel_id, app_state, signals, parent)
    elif channel_type == "script":
        return RPOCScriptChannelWidget(channel_id, app_state, signals, parent)
    elif channel_type == "static":
        return RPOCStaticChannelWidget(channel_id, app_state, signals, parent)
    else:
        # Default to mask channel
        return RPOCMaskChannelWidget(channel_id, app_state, signals, parent)
