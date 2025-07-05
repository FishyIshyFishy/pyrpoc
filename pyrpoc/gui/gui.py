import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, \
                             QLabel, QWidget, QComboBox, QSplitter, QPushButton, \
                             QPlainTextEdit, QStyle, QGroupBox, QSpinBox, QCheckBox, QLineEdit, QSlider, \
                             QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsLineItem, \
                             QFrame, QSizePolicy, QDockWidget, QFileDialog, QDialog
from PyQt6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter, QFont
from pyrpoc.gui.gui_handler import AppState, StateSignalBus
import sys
import pyqtgraph as pg
from pyrpoc.gui.image_widgets import ImageDisplayWidget
from pyrpoc.gui.dockable_widgets import LinesWidget
from pyrpoc.gui.rpoc_mask_editor import RPOCMaskEditor

# # ugly red border for outlines
# DEV_BORDER_STYLE = """ 
#     QWidget {
#         border: 2px solid #FF0000;
#         border-radius: 4px;
#         margin: 2px;
#     }
# """
DEV_BORDER_STYLE = """
    QWidget {
        margin: 2px;
    }
"""

class TopBar(QWidget):
    '''
    horizontal orientation with the important control widgets, most space given to SystemConsole
    AppConfigButtons | AcquisitionControls | SystemStatus
    '''
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        self.setFixedHeight(100) 
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # two buttons for loading exisitng AppState configs from .json, and saving
        config_widget = QWidget()
        config_layout = QVBoxLayout()

        load_btn = QPushButton('Load Config')
        load_btn.clicked.connect(signals.load_config_btn_clicked.emit)
        config_layout.addWidget(load_btn)

        save_btn = QPushButton('Save Config')
        save_btn.clicked.connect(signals.save_config_btn_clicked.emit)
        config_layout.addWidget(save_btn)

        config_widget.setLayout(config_layout)
        layout.addWidget(config_widget)

        # main acquisition controls: acq continuous, acq single, stop acq
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        single_btn = QPushButton()
        single_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        single_btn.setToolTip('Start Continuous Acquisition')
        single_btn.clicked.connect(signals.single_btn_clicked.emit)
        controls_layout.addWidget(single_btn)

        continuous_btn = QPushButton()
        continuous_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
        continuous_btn.setToolTip('Start Single Acquisition')
        continuous_btn.clicked.connect(signals.continuous_btn_clicked.emit)
        controls_layout.addWidget(continuous_btn)

        stop_btn = QPushButton()
        stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        stop_btn.setToolTip('Stop Acquisition')
        stop_btn.clicked.connect(signals.stop_btn_clicked.emit)
        controls_layout.addWidget(stop_btn)

        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)

        # system console to display status updates
        console_widget = QWidget()
        console_layout = QVBoxLayout()
        console_layout.setContentsMargins(0, 0, 0, 0)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        console_layout.addWidget(self.console)

        # tool buttons, eventually i will make these like imageJ icons but for now just text
        self.tool_buttons_widget = QWidget()
        tool_buttons_layout = QHBoxLayout()
        tool_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.lines_btn = QPushButton('Lines')
        self.lines_btn.setCheckable(True)
        self.lines_btn.setChecked(app_state.ui_state['lines_enabled'])
        self.lines_btn.toggled.connect(signals.lines_toggled.emit)
        tool_buttons_layout.addWidget(self.lines_btn)
        tool_buttons_layout.addStretch()
        self.tool_buttons_widget.setLayout(tool_buttons_layout)
        console_layout.addWidget(self.tool_buttons_widget)

        console_widget.setLayout(console_layout)
        layout.addWidget(console_widget, stretch=1)

        self.setLayout(layout)

    # called from a signal in gui_handler.py
    def add_console_message(self, message):
        self.console.appendPlainText(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

class AcquisitionParameters(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        
        main_layout = QVBoxLayout()
        self.group = QGroupBox('Acquisition Parameters')
        self.group.setCheckable(True)
        self.group.setChecked(app_state.ui_state['acquisition_parameters_visible'])
        self.group.toggled.connect(lambda checked: signals.ui_state_changed.emit('acquisition_parameters_visible', checked))
        
        self.container = QWidget()
        layout = QVBoxLayout()
        
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel('Number of Frames:'))
        self.frames_spinbox = QSpinBox()
        self.frames_spinbox.setRange(1, 10000)
        self.frames_spinbox.setValue(self.app_state.acquisition_parameters['num_frames'])
        self.frames_spinbox.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('num_frames', value))
        frames_layout.addWidget(self.frames_spinbox)
        layout.addLayout(frames_layout)
        
        x_pixels_layout = QHBoxLayout()
        x_pixels_layout.addWidget(QLabel('X Pixels:'))
        self.x_pixels_spinbox = QSpinBox()
        self.x_pixels_spinbox.setRange(64, 4096)
        self.x_pixels_spinbox.setValue(self.app_state.acquisition_parameters['x_pixels'])
        self.x_pixels_spinbox.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('x_pixels', value))
        x_pixels_layout.addWidget(self.x_pixels_spinbox)
        layout.addLayout(x_pixels_layout)
        
        y_pixels_layout = QHBoxLayout()
        y_pixels_layout.addWidget(QLabel('Y Pixels:'))
        self.y_pixels_spinbox = QSpinBox()
        self.y_pixels_spinbox.setRange(64, 4096)
        self.y_pixels_spinbox.setValue(self.app_state.acquisition_parameters['y_pixels'])
        self.y_pixels_spinbox.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('y_pixels', value))
        y_pixels_layout.addWidget(self.y_pixels_spinbox)
        layout.addLayout(y_pixels_layout)
        
        self.container.setLayout(layout)
        
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        
        self.group.toggled.connect(self.container.setVisible)
        
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)


class ModalityControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        ms_label = QLabel('Modality:')
        layout.addWidget(ms_label)
        ms_dropdown = QComboBox()
        ms_dropdown.addItems(['Simulated', 'Widefield', 'Confocal', 'Mosaic', 'ZScan'])
        current_modality = self.app_state.modality.capitalize()
        index = ms_dropdown.findText(current_modality)
        if index >= 0:
            ms_dropdown.setCurrentIndex(index)
        ms_dropdown.currentTextChanged.connect(self.signals.modality_dropdown_changed)
        layout.addWidget(ms_dropdown)
        self.setLayout(layout)


class InstrumentControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        self.instrument_widgets = {}  # instrument_id -> widget
        self.next_instrument_id = 1
        
        main_layout = QVBoxLayout()
        self.group = QGroupBox('Instruments')
        self.group.setCheckable(True)
        self.group.setChecked(app_state.ui_state['instrument_controls_visible'])
        self.group.toggled.connect(lambda checked: signals.ui_state_changed.emit('instrument_controls_visible', checked))
        
        self.container = QWidget()
        layout = QVBoxLayout()
        
        # Modality-specific instrument buttons
        self.modality_buttons_widget = QWidget()
        self.modality_buttons_layout = QVBoxLayout()
        self.modality_buttons_widget.setLayout(self.modality_buttons_layout)
        layout.addWidget(self.modality_buttons_widget)
        
        # Generic add instrument button
        add_btn = QPushButton('Add Instrument')
        add_btn.clicked.connect(self.signals.add_instrument_btn_clicked.emit)
        layout.addWidget(add_btn)
        
        # Instrument list
        self.instrument_list = QWidget()
        self.instrument_list_layout = QVBoxLayout()
        self.instrument_list.setLayout(self.instrument_list_layout)
        layout.addWidget(self.instrument_list)
        
        self.container.setLayout(layout)
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        self.group.toggled.connect(self.container.setVisible)
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)
        
        self.rebuild()
        
        # Connect signals
        self.signals.instrument_removed.connect(self.remove_instrument)
    
    def rebuild(self):
        # Clear modality-specific buttons
        while self.modality_buttons_layout.count():
            child = self.modality_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add modality-specific buttons (only if not already added)
        modality = self.app_state.modality.lower()
        if modality == 'confocal':
            if not self.has_instrument_type('Galvo'):
                galvo_btn = QPushButton('Add Galvos')
                galvo_btn.clicked.connect(lambda: self.signals.add_modality_instrument.emit('Galvo'))
                self.modality_buttons_layout.addWidget(galvo_btn)
            
            if not self.has_instrument_type('Data Input'):
                data_input_btn = QPushButton('Add Data Inputs')
                data_input_btn.clicked.connect(lambda: self.signals.add_modality_instrument.emit('Data Input'))
                self.modality_buttons_layout.addWidget(data_input_btn)
        elif modality == 'widefield':
            if not self.has_instrument_type('Data Input'):
                data_input_btn = QPushButton('Add Data Inputs')
                data_input_btn.clicked.connect(lambda: self.signals.add_modality_instrument.emit('Data Input'))
                self.modality_buttons_layout.addWidget(data_input_btn)
        elif modality == 'mosaic':
            if not self.has_instrument_type('Galvo'):
                galvo_btn = QPushButton('Add Galvos')
                galvo_btn.clicked.connect(lambda: self.signals.add_modality_instrument.emit('Galvo'))
                self.modality_buttons_layout.addWidget(galvo_btn)
            
            if not self.has_instrument_type('Data Input'):
                data_input_btn = QPushButton('Add Data Inputs')
                data_input_btn.clicked.connect(lambda: self.signals.add_modality_instrument.emit('Data Input'))
                self.modality_buttons_layout.addWidget(data_input_btn)
        
        # Rebuild instrument list
        self.rebuild_instrument_list()
    
    def has_instrument_type(self, instrument_type):
        """Check if we already have an instrument of the given type"""
        if hasattr(self.app_state, 'instruments'):
            for instrument in self.app_state.instruments.values():
                if instrument.instrument_type == instrument_type:
                    return True
        return False
    
    def update_modality_buttons(self):
        """Update modality-specific buttons based on current instruments"""
        self.rebuild()
    
    def rebuild_instrument_list(self):
        # Clear existing instrument widgets
        for widget in self.instrument_widgets.values():
            self.instrument_list_layout.removeWidget(widget)
            widget.deleteLater()
        self.instrument_widgets.clear()
        
        # Recreate widgets for existing instruments
        if hasattr(self.app_state, 'instruments'):
            for instrument_id, instrument in self.app_state.instruments.items():
                widget = InstrumentWidget(instrument_id, instrument, self.app_state, self.signals)
                self.instrument_widgets[instrument_id] = widget
                self.instrument_list_layout.addWidget(widget)
    
    def add_instrument(self, instrument_id, instrument):
        widget = InstrumentWidget(instrument_id, instrument, self.app_state, self.signals)
        self.instrument_widgets[instrument_id] = widget
        self.instrument_list_layout.addWidget(widget)
    
    def remove_instrument(self, instrument_id):
        if instrument_id in self.instrument_widgets:
            widget = self.instrument_widgets[instrument_id]
            self.instrument_list_layout.removeWidget(widget)
            del self.instrument_widgets[instrument_id]
            widget.deleteLater()
            
            # Update modality buttons after removal
            self.update_modality_buttons()
    
    def sync_ui_state(self):
        """Synchronize UI state with app_state"""
        # Update visibility based on app_state
        if hasattr(self.app_state, 'ui_state'):
            visible = self.app_state.ui_state.get('instrument_controls_visible', True)
            self.group.setChecked(visible)
        
        # Update modality buttons
        self.update_modality_buttons()
        
        # Update instrument list
        self.rebuild_instrument_list()
    
    def handle_instrument_connection_change(self, instrument_id, connected):
        """Handle instrument connection status changes"""
        if instrument_id in self.instrument_widgets:
            widget = self.instrument_widgets[instrument_id]
            widget.update_status()


class InstrumentWidget(QWidget):
    def __init__(self, instrument_id, instrument, app_state, signals, parent=None):
        super().__init__(parent)
        self.instrument_id = instrument_id
        self.instrument = instrument
        self.app_state = app_state
        self.signals = signals
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header with name and remove button
        header_layout = QHBoxLayout()
        # Ensure instrument name is a string
        instrument_name = instrument.name if hasattr(instrument, 'name') and isinstance(instrument.name, str) else 'Unknown Instrument'
        self.name_label = QLabel(instrument_name)
        header_layout.addWidget(self.name_label)
        
        remove_btn = QPushButton('×')
        remove_btn.setMaximumWidth(20)
        remove_btn.clicked.connect(self.remove_instrument)
        header_layout.addWidget(remove_btn)
        
        layout.addLayout(header_layout)
        
        # Status
        self.status_label = QLabel('Disconnected')
        self.status_label.setStyleSheet('color: #666; font-size: 10px;')
        layout.addWidget(self.status_label)
        
        # Control/Edit button
        self.control_btn = QPushButton('Edit/Control')
        self.control_btn.clicked.connect(self.edit_control_instrument)
        layout.addWidget(self.control_btn)
        
        self.setLayout(layout)
        self.update_status()
    
    def update_status(self):
        if self.instrument.connected:
            self.status_label.setText('Connected')
            self.status_label.setStyleSheet('color: #4CAF50; font-size: 10px;')
        else:
            self.status_label.setText('Disconnected')
            self.status_label.setStyleSheet('color: #666; font-size: 10px;')
        
        # Update name label if needed
        if hasattr(self.instrument, 'name') and isinstance(self.instrument.name, str):
            self.name_label.setText(self.instrument.name)
        else:
            self.name_label.setText('Unknown Instrument')
    
    def remove_instrument(self):
        # Remove from app_state
        if hasattr(self.app_state, 'instruments') and self.instrument_id in self.app_state.instruments:
            del self.app_state.instruments[self.instrument_id]
        
        # Emit signal to remove this widget
        self.signals.instrument_removed.emit(self.instrument_id)
        
        # Update modality buttons to show the button for the removed instrument type
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'update_modality_buttons'):
                parent.update_modality_buttons()
                break
            parent = parent.parent()
        
        self.deleteLater()
    
    def edit_control_instrument(self):
        # For galvos and data inputs, show configuration dialog to edit parameters
        if self.instrument.instrument_type in ['Galvo', 'Data Input']:
            from pyrpoc.imaging.instruments import InstrumentDialog
            
            # Prepare current parameters including the name
            current_params = self.instrument.parameters.copy()
            current_params['name'] = self.instrument.name
            
            dialog = InstrumentDialog(self.instrument.instrument_type, self, current_params)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                parameters = dialog.get_parameters()
                if parameters is not None:  # Check if validation passed
                    # Update instrument parameters
                    self.instrument.parameters.update(parameters)
                    new_name = parameters.get('name', 'Unknown Instrument')
                    self.instrument.name = new_name
                    self.name_label.setText(new_name)
                    self.signals.console_message.emit(f"Updated {new_name} parameters")
                else:
                    current_name = getattr(self.instrument, 'name', 'Unknown Instrument')
                    self.signals.console_message.emit(f"Failed to update {current_name} - invalid parameters")
        
        else:
            # For other instruments, check if they have a control widget
            control_widget = self.instrument.get_control_widget()
            if control_widget:
                # Show control widget in a dialog
                dialog = QDialog(self)
                dialog.setWindowTitle(f"Control {self.instrument.name}")
                dialog.setModal(True)
                
                layout = QVBoxLayout()
                layout.addWidget(control_widget)
                
                close_btn = QPushButton("Close")
                close_btn.clicked.connect(dialog.accept)
                layout.addWidget(close_btn)
                
                dialog.setLayout(layout)
                dialog.resize(300, 200)
                dialog.exec()
            else:
                # Reopen configuration dialog for other instruments
                from pyrpoc.imaging.instruments import InstrumentDialog
                dialog = InstrumentDialog(self.instrument.instrument_type, self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    parameters = dialog.get_parameters()
                    if parameters is not None:  # Check if validation passed
                        # Update instrument parameters
                        self.instrument.parameters.update(parameters)
                        new_name = parameters.get('name', 'Unknown Instrument')
                        self.instrument.name = new_name
                        self.name_label.setText(new_name)
                        self.signals.console_message.emit(f"Updated {new_name} parameters")
                    else:
                        current_name = getattr(self.instrument, 'name', 'Unknown Instrument')
                        self.signals.console_message.emit(f"Failed to update {current_name} - invalid parameters")


class DisplayControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        main_layout = QVBoxLayout()
        self.group = QGroupBox('Display Settings')
        self.group.setCheckable(True)
        self.group.setChecked(app_state.ui_state['display_controls_visible'])
        self.group.toggled.connect(lambda checked: signals.ui_state_changed.emit('display_controls_visible', checked))
        self.container = QWidget()
        layout = QVBoxLayout()
        self.container.setLayout(layout)
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        self.group.toggled.connect(self.container.setVisible)
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)


class RPOCChannelWidget(QWidget):
    def __init__(self, channel_id, app_state, signals, parent=None):
        super().__init__(parent)
        self.channel_id = channel_id
        self.app_state = app_state
        self.signals = signals
        self.mask_editor = None
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # channel header
        header_layout = QHBoxLayout()
        self.channel_label = QLabel(f'Channel {channel_id}')
        header_layout.addWidget(self.channel_label)
        
        # remove channel button
        remove_btn = QPushButton('×')
        remove_btn.setMaximumWidth(20)
        remove_btn.clicked.connect(self.remove_channel)
        header_layout.addWidget(remove_btn)
        
        layout.addLayout(header_layout)
        
        # mask status
        self.mask_status = QLabel('No mask loaded')
        self.mask_status.setStyleSheet('color: #666; font-size: 10px;')
        layout.addWidget(self.mask_status)
        
        # mask buttons
        buttons_layout = QHBoxLayout()
        
        self.create_mask_btn = QPushButton('Create Mask')
        self.create_mask_btn.clicked.connect(self.create_mask)
        buttons_layout.addWidget(self.create_mask_btn)
        
        self.load_mask_btn = QPushButton('Load Mask')
        self.load_mask_btn.clicked.connect(self.load_mask)
        buttons_layout.addWidget(self.load_mask_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
        # check if we already have a mask for this channel
        self.update_mask_status()
    
    def create_mask(self):
        # get current image data from the image display widget
        image_data = self.get_current_image_data()
        if image_data is None:
            self.signals.console_message.emit("No image data available for mask creation")
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
                import cv2
                mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    self.handle_mask_loaded(mask)
                else:
                    self.signals.console_message.emit(f"Failed to load mask from {file_path}")
            except Exception as e:
                self.signals.console_message.emit(f"Error loading mask: {e}")
    
    def get_current_image_data(self):
        """get current image data from the main image display widget"""
        # traverse up to find the main window and get image data
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'app_state'):
                # try to get data from the image display widget first
                if hasattr(parent, 'mid_layout') and hasattr(parent.mid_layout, 'image_display_widget'):
                    image_widget = parent.mid_layout.image_display_widget
                    if hasattr(image_widget, 'get_image_data_for_rpoc'):
                        return image_widget.get_image_data_for_rpoc()
                
                # fallback to app_state current_data
                if hasattr(parent.app_state, 'current_data'):
                    data = parent.app_state.current_data
                    if data is not None and isinstance(data, np.ndarray):
                        if data.ndim == 3:
                            # assume it's frames x height x width, return current frame as single channel
                            return data[0][np.newaxis, :, :] if data.shape[0] > 0 else None
                        elif data.ndim == 2:
                            return data[np.newaxis, :, :]
                return None
            parent = parent.parent()
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
        self.signals.console_message.emit(f"Mask loaded for channel {self.channel_id}")
    
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
    
    def remove_channel(self):
        # remove mask from app_state
        if hasattr(self.app_state, 'rpoc_masks') and self.channel_id in self.app_state.rpoc_masks:
            del self.app_state.rpoc_masks[self.channel_id]
        
        # emit signal to remove this widget
        self.signals.rpoc_channel_removed.emit(self.channel_id)
        self.deleteLater()

class RightPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        self.rpoc_channels = {}  # channel_id -> widget
        self.next_channel_id = 1
        
        self.rebuild()

    def rebuild(self):
        if self.layout():
            while self.layout().count():
                child = self.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
        layout = QVBoxLayout()
        
        self.add_modality_specific_controls(layout)
        self.add_common_controls(layout)
        
        self.setLayout(layout)

    def add_modality_specific_controls(self, layout):
        modality = self.app_state.modality.lower()
        
        if modality == 'simulated':
            group = QGroupBox('Simulation Mode')
            group_layout = QVBoxLayout()
            group_layout.addWidget(QLabel('Simulation Mode Active'))
            group.setLayout(group_layout)
            layout.addWidget(group)
        elif modality == 'widefield':
            group = QGroupBox('Widefield Mode')
            group_layout = QVBoxLayout()
            group_layout.addWidget(QLabel('Widefield Mode Active'))
            group.setLayout(group_layout)
            layout.addWidget(group)
        elif modality == 'confocal':
            group = QGroupBox('Confocal Mode')
            group_layout = QVBoxLayout()
            group_layout.addWidget(QLabel('Confocal Mode Active'))
            group.setLayout(group_layout)
            layout.addWidget(group)
        elif modality == 'mosaic':
            group = QGroupBox('Mosaic Mode')
            group_layout = QVBoxLayout()
            group_layout.addWidget(QLabel('Mosaic Mode Active'))
            group.setLayout(group_layout)
            layout.addWidget(group)
        elif modality == 'zscan':
            group = QGroupBox('ZScan Mode')
            group_layout = QVBoxLayout()
            group_layout.addWidget(QLabel('ZScan Mode Active'))
            group.setLayout(group_layout)
            layout.addWidget(group)

    def add_common_controls(self, layout):
        rpoc_group = QGroupBox('RPOC Controls')
        rpoc_layout = QVBoxLayout()
        
        rpoc_enabled_checkbox = QCheckBox('RPOC Enabled')
        rpoc_enabled_checkbox.setChecked(self.app_state.rpoc_enabled)
        rpoc_enabled_checkbox.toggled.connect(self.signals.rpoc_enabled_changed.emit)
        rpoc_layout.addWidget(rpoc_enabled_checkbox)
        
        # add rpoc channel button
        add_channel_btn = QPushButton('Add RPOC Channel')
        add_channel_btn.clicked.connect(self.add_rpoc_channel)
        rpoc_layout.addWidget(add_channel_btn)
        
        # container for rpoc channel widgets
        self.channels_container = QWidget()
        self.channels_layout = QVBoxLayout()
        self.channels_layout.setSpacing(5)
        self.channels_container.setLayout(self.channels_layout)
        rpoc_layout.addWidget(self.channels_container)
        
        rpoc_group.setLayout(rpoc_layout)
        layout.addWidget(rpoc_group)
        
        layout.addStretch()
        
        # connect signals
        self.signals.rpoc_channel_removed.connect(self.remove_rpoc_channel)
    
    def add_rpoc_channel(self):
        channel_id = self.next_channel_id
        self.next_channel_id += 1
        
        channel_widget = RPOCChannelWidget(channel_id, self.app_state, self.signals)
        self.rpoc_channels[channel_id] = channel_widget
        self.channels_layout.addWidget(channel_widget)
    
    def remove_rpoc_channel(self, channel_id):
        if channel_id in self.rpoc_channels:
            widget = self.rpoc_channels[channel_id]
            self.channels_layout.removeWidget(widget)
            del self.rpoc_channels[channel_id]
            widget.deleteLater()

class LeftPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.rebuild()

    def rebuild(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.modality_controls = ModalityControls(self.app_state, self.signals)
        self.layout.addWidget(self.modality_controls)
        
        self.acquisition_parameters = AcquisitionParameters(self.app_state, self.signals)
        self.layout.addWidget(self.acquisition_parameters)
        
        self.instrument_controls = InstrumentControls(self.app_state, self.signals)
        self.layout.addWidget(self.instrument_controls)
        
        self.display_controls = DisplayControls(self.app_state, self.signals)
        self.layout.addWidget(self.display_controls)
        
        self.layout.addStretch()


class DockableMiddlePanel(QMainWindow):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        
        self.rebuild()

    def rebuild(self):
        if self.centralWidget():
            self.centralWidget().deleteLater()

        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.image_display_widget = self.create_image_display_widget()
        layout.addWidget(self.image_display_widget)

        self.setCentralWidget(central_widget)

        self.lines_dock = QDockWidget('Lines', self)
        self.lines_widget = LinesWidget(self.app_state, self.signals)
        self.lines_dock.setWidget(self.lines_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.lines_dock)
        self.lines_dock.hide()
        
        self.signals.lines_toggled.connect(self.on_lines_toggled)
        self.on_lines_toggled(self.app_state.ui_state['lines_enabled'])

    def create_image_display_widget(self):
        return ImageDisplayWidget(self.app_state, self.signals)

    def on_lines_toggled(self, enabled):
        if enabled:
            self.lines_dock.show()
            self.lines_widget.update_status(True)
        else:
            self.lines_dock.hide()
            self.lines_widget.update_status(False)

    def set_image_display_widget(self, widget):
        layout = self.centralWidget().layout()
        if self.image_display_widget is not None:
            layout.removeWidget(self.image_display_widget)
            self.image_display_widget.deleteLater()
        self.image_display_widget = widget
        layout.addWidget(self.image_display_widget)


class MainWindow(QMainWindow):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setWindowTitle('pyrpoc - Development Mode')
        self.setGeometry(100, 100, 1400, 900)
        
        self.central_widget = None
        self.central_layout = None
        self.main_splitter = None
        self.left_widget = None
        self.mid_layout = None
        self.right_layout = None
        self.top_bar = None
        
        self.build_gui()

        self.signals.modality_dropdown_changed.connect(self.on_modality_changed)

    def build_gui(self):
        self._clear_existing_gui()
        self._create_central_widget()
        self._create_top_bar()
        self._create_main_splitter()
        self._setup_splitter_sizes()
        self._finalize_gui()

    def _clear_existing_gui(self):
        if self.centralWidget():
            self.centralWidget().deleteLater()

    def _create_central_widget(self):
        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout()
        self.central_layout.setContentsMargins(5, 5, 5, 5)
        self.central_layout.setSpacing(5)

    def _create_top_bar(self):
        self.top_bar = TopBar(self.app_state, self.signals)
        self.central_layout.addWidget(self.top_bar, stretch=0)

    def _create_main_splitter(self):
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.left_widget = LeftPanel(self.app_state, self.signals)
        self.mid_layout = DockableMiddlePanel(self.app_state, self.signals)
        self.right_layout = RightPanel(self.app_state, self.signals)
        
        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.mid_layout)
        self.main_splitter.addWidget(self.right_layout)

    def _setup_splitter_sizes(self):
        self.main_splitter.setSizes([200, 800, 200])
        if 'main_splitter_sizes' in self.app_state.ui_state:
            self.main_splitter.setSizes(self.app_state.ui_state['main_splitter_sizes'])
        self.main_splitter.splitterMoved.connect(lambda: self.save_splitter_sizes())

    def _finalize_gui(self):
        self.central_layout.addWidget(self.main_splitter, stretch=1)
        self.central_widget.setLayout(self.central_layout)
        self.setCentralWidget(self.central_widget)

    def rebuild_gui(self):
        if self.main_splitter:
            sizes = self.main_splitter.sizes()
            self.app_state.ui_state['main_splitter_sizes'] = sizes
        
        self.build_gui()
        
        self.signals.console_message.emit(f"GUI rebuilt for {self.app_state.modality} modality")

    def on_modality_changed(self, new_modality):
        self.app_state.modality = new_modality.lower()
        self.rebuild_gui()

    def save_splitter_sizes(self):
        if self.main_splitter:
            sizes = self.main_splitter.sizes()
            self.signals.ui_state_changed.emit('main_splitter_sizes', sizes)

if __name__ == '__main__':
    app_state = AppState()
    signals = StateSignalBus()

    app = QApplication(sys.argv)
    win = MainWindow(app_state, signals)
    
    signals.bind_controllers(app_state, win)
    win.show()
    sys.exit(app.exec())