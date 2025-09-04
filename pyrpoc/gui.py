import numpy as np
from typing import Dict, Any
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, \
                             QLabel, QWidget, QComboBox, QSplitter, QPushButton, \
                             QPlainTextEdit, QStyle, QGroupBox, QSpinBox, QCheckBox, QLineEdit, QSlider, \
                             QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsLineItem, \
                             QFrame, QSizePolicy, QDockWidget, QFileDialog, QDialog, QFormLayout, QDoubleSpinBox, \
                             QScrollArea
from PyQt6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, QEasingCurve, QSize
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter, QFont, QIcon
from pyrpoc.gui_handler import AppState, StateSignalBus
import sys
import pyqtgraph as pg
from pyrpoc.displays import *
from pyrpoc.displays.multichan_tiled import MultichannelDisplayParametersWidget
from pyrpoc.rpoc.rpoc_mask_editor import RPOCMaskEditor
from superqt import QSearchableComboBox
import cv2
from pathlib import Path


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

SPLITTER_STYLE = """
    QSplitter::handle {
        background-color: #666666;
        border: 1px solid #444444;
    }
    QSplitter::handle:hover {
        background-color: #888888;
    }
    QSplitter::handle:pressed {
        background-color: #aaaaaa;
    }
"""

class TopBar(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        # Remove fixed height to allow resizing by splitter
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

        self.single_btn = QPushButton()
        self.single_btn.setIcon(QIcon("pyrpoc/icons/single.svg"))
        self.single_btn.setIconSize(QSize(16, 16))
        self.single_btn.setFixedSize(32, 32)
        self.single_btn.setToolTip('Start Single Acquisition')
        self.single_btn.clicked.connect(signals.single_btn_clicked.emit)
        self.single_btn.setStyleSheet("QPushButton { color: white; } QPushButton::icon { color: white; }")
        controls_layout.addWidget(self.single_btn)

        self.continuous_btn = QPushButton()
        self.continuous_btn.setIcon(QIcon("pyrpoc/icons/multi.svg"))
        self.continuous_btn.setIconSize(QSize(16, 16))
        self.continuous_btn.setFixedSize(32, 32)
        self.continuous_btn.setToolTip('Start Continuous Acquisition')
        self.continuous_btn.clicked.connect(signals.continuous_btn_clicked.emit)
        self.continuous_btn.setStyleSheet("QPushButton { color: white; } QPushButton::icon { color: white; }")
        controls_layout.addWidget(self.continuous_btn)

        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(QIcon("pyrpoc/icons/stop.svg"))
        self.stop_btn.setIconSize(QSize(16, 16))
        self.stop_btn.setFixedSize(32, 32)
        self.stop_btn.setToolTip('Stop Acquisition')
        self.stop_btn.clicked.connect(signals.stop_btn_clicked.emit)
        self.stop_btn.setStyleSheet("QPushButton { color: white; } QPushButton::icon { color: white; }")
        controls_layout.addWidget(self.stop_btn)

        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)

        # system console to display status updates
        console_widget = QWidget()
        console_layout = QVBoxLayout()
        console_layout.setContentsMargins(0, 0, 0, 0)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(50)  # Ensure minimum height for console
        console_layout.addWidget(self.console)

        console_widget.setLayout(console_layout)
        layout.addWidget(console_widget, stretch=1)

        self.setLayout(layout)

    # called from a signal in gui_handler.py
    def add_console_message(self, message):
        self.console.appendPlainText(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())
    
    def on_acquisition_started(self):
        """disable start buttons and enable stop button when acquisition starts"""
        self.single_btn.setEnabled(False)
        self.continuous_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def on_acquisition_stopped(self):
        """enable start buttons and disable stop button when acquisition stops"""
        self.single_btn.setEnabled(True)
        self.continuous_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes without rebuilding the entire top bar"""
        # This method will be enhanced later to handle modality-specific top bar requirements
        # For now, it does nothing, keeping the top bar independent of modality changes
        pass


class ModalityControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)

        layout = QHBoxLayout()
    
        ms_label = QLabel('Modality:')
        layout.addWidget(ms_label)

        ms_dropdown = QComboBox()
        # Use registry instead of hard-coded list
        from pyrpoc.modalities import modality_registry
        ms_dropdown.addItems(modality_registry.get_modality_names())
        
        # Find the modality by key and get its display name
        from pyrpoc.modalities import modality_registry
        current_modality = modality_registry.get_modality(self.app_state.modality)
        if current_modality is not None:
            current_modality_name = current_modality.name
        else:
            current_modality_name = self.app_state.modality.capitalize()
        
        index = ms_dropdown.findText(current_modality_name)
        if index >= 0:
            ms_dropdown.setCurrentIndex(index)
        ms_dropdown.currentTextChanged.connect(self.signals.modality_dropdown_changed)
        layout.addWidget(ms_dropdown)

        self.setLayout(layout)
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes by updating the dropdown selection"""
        # Find the modality by key and get its display name
        from pyrpoc.modalities import modality_registry
        current_modality = modality_registry.get_modality(new_modality)
        if current_modality is not None:
            current_modality_name = current_modality.name
        else:
            current_modality_name = new_modality.capitalize()
        
        index = self.findChild(QComboBox).findText(current_modality_name)
        if index >= 0:
            self.findChild(QComboBox).setCurrentIndex(index)

'''
all subwidgets other than topbar in general get rebuilt upon modality changes
the widget needs to be built with a rebuild() method, which uses data from AppState to build
then within rebuild() there is an add_common_parameters() for universal settings and an add_specific_parameters() for modality-specific settings

TODO: make sure that handling of the parameters is done correctly prior to acquisition startup
'''

class AcquisitionParameters(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        '''
        acquisition parameters, contained in acquisition parameters dict in AppState
        signals.acquisition_parameter_changed.emit('param key', value)
        '''
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        
        self.rebuild() 

    def rebuild(self):
        # clear existing layout
        if self.layout():
            while self.layout().count():
                child = self.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
        main_layout = QVBoxLayout()

        self.group = QGroupBox('Acquisition Parameters')
        self.group.setCheckable(True)
        self.group.setChecked(self.app_state.ui_state['acquisition_parameters_visible'])
        self.group.toggled.connect(lambda checked: self.signals.ui_state_changed.emit('acquisition_parameters_visible', checked))
        
        self.container = QWidget()
        self.layout = QVBoxLayout()
        
        self.add_specific_parameters()
        self.add_common_parameters()
        
        self.container.setLayout(self.layout)
        
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        
        self.group.toggled.connect(self.container.setVisible)
        
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes by rebuilding the parameters"""
        self.rebuild()

    def add_specific_parameters(self):
        from pyrpoc.modalities import modality_registry
        
        modality = modality_registry.get_modality_by_name(self.app_state.modality.capitalize())
        if modality is None:
            return
        
        # Generate UI widgets based on modality requirements
        for param_name, param_meta in modality.required_parameters.items():
            self.add_parameter_widget(param_name, param_meta)
    
    def add_parameter_widget(self, param_name: str, param_meta: Dict[str, Any]):
        """Dynamically create parameter widgets based on metadata"""
        param_type = param_meta['type']
        default_value = param_meta.get('default', 0)
        
        # Get current value from app_state if it exists
        current_value = self.app_state.acquisition_parameters.get(param_name, default_value)
        
        if param_type == 'int':
            widget = QSpinBox()
            if 'range' in param_meta:
                widget.setRange(*param_meta['range'])
            widget.setValue(current_value)
        elif param_type == 'float':
            widget = QDoubleSpinBox()
            if 'range' in param_meta:
                widget.setRange(*param_meta['range'])
            widget.setValue(current_value)
        elif param_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(current_value)
        elif param_type == 'choice':
            widget = QComboBox()
            widget.addItems(param_meta['choices'])
            widget.setCurrentText(current_value)
        else:
            widget = QLineEdit()
            widget.setText(str(current_value))
        
        # Connect to signal
        widget.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit(param_name, value)
        )
        
        # Add to layout with label
        label = QLabel(f"{param_name.replace('_', ' ').title()}:")
        if 'unit' in param_meta:
            label.setText(f"{label.text()} ({param_meta['unit']})")
        
        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(widget)
        self.layout.addLayout(layout)








        

        

        

        


    def add_common_parameters(self):
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel('Number of Frames:'))
        self.frames_spinbox = QSpinBox()
        self.frames_spinbox.setRange(1, 10000)
        self.frames_spinbox.setValue(self.app_state.acquisition_parameters['num_frames'])
        self.frames_spinbox.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('num_frames', value))
        frames_layout.addWidget(self.frames_spinbox)
        self.layout.addLayout(frames_layout)
        
        save_layout = QVBoxLayout()
        
        save_enabled_layout = QHBoxLayout()
        self.save_enabled_checkbox = QCheckBox('Save Data')
        self.save_enabled_checkbox.setChecked(self.app_state.acquisition_parameters.get('save_enabled', False))
        self.save_enabled_checkbox.toggled.connect(
            lambda checked: self.signals.acquisition_parameter_changed.emit('save_enabled', checked))
        save_enabled_layout.addWidget(self.save_enabled_checkbox)
        save_layout.addLayout(save_enabled_layout)
        
        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(QLabel('Save Path:'))
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setText(self.app_state.acquisition_parameters.get('save_path', ''))
        self.save_path_edit.setPlaceholderText('Select save location and filename...')
        self.save_path_edit.textChanged.connect(
            lambda text: self.signals.save_path_changed.emit(text))
        save_path_layout.addWidget(self.save_path_edit)
        
        self.browse_btn = QPushButton('Browse')
        self.browse_btn.clicked.connect(self.browse_save_path)
        save_path_layout.addWidget(self.browse_btn)
        save_layout.addLayout(save_path_layout)

        self.layout.addLayout(save_layout)
    
    def browse_save_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Select Save Location', 'acquisition', 
            'All files (*)'
        )
        if file_path:
            base_filename = Path(file_path).stem
            full_path = Path(file_path).parent / base_filename
            self.save_path_edit.setText(str(full_path))
            self.signals.save_path_changed.emit(str(full_path))


class InstrumentControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        self.instrument_widgets = {}  # instrument -> widget
        
        main_layout = QVBoxLayout()
        self.group = QGroupBox('Instruments')
        self.group.setCheckable(True)
        self.group.setChecked(app_state.ui_state['instrument_controls_visible'])
        self.group.toggled.connect(lambda checked: signals.ui_state_changed.emit('instrument_controls_visible', checked))
        
        self.container = QWidget()
        layout = QVBoxLayout()
        
        self.modality_buttons_widget = QWidget()
        self.modality_buttons_layout = QVBoxLayout()
        self.modality_buttons_widget.setLayout(self.modality_buttons_layout)
        layout.addWidget(self.modality_buttons_widget)
        
        add_btn = QPushButton('Add Instrument')
        add_btn.clicked.connect(self.signals.add_instrument_btn_clicked.emit)
        layout.addWidget(add_btn)
        
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

        self.signals.instrument_removed.connect(self.remove_instrument)
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes by rebuilding the instrument controls"""
        self.rebuild()
    
    def rebuild(self):
        # clear any existing modality specific instrument buttons
        while self.modality_buttons_layout.count():
            child = self.modality_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Use modality registry to determine required instruments
        from pyrpoc.modalities import modality_registry
        
        modality = modality_registry.get_modality(self.app_state.modality)
        if modality is not None:
            for instrument_type in modality.required_instruments:
                if not self.has_instrument_type(instrument_type):
                    btn = QPushButton(f'Add {instrument_type.title()}')
                    btn.clicked.connect(lambda checked, it=instrument_type: self.signals.add_modality_instrument.emit(it))
                    self.modality_buttons_layout.addWidget(btn)
        
        self.rebuild_instrument_list()
    
    def has_instrument_type(self, instrument_type):
        if hasattr(self.app_state, 'instruments'):
            for instrument in self.app_state.instruments:
                if instrument.instrument_type == instrument_type:
                    return True
        return False
    
    def rebuild_instrument_list(self): # TODO: verify if this is useless or not
        for widget in self.instrument_widgets.values():
            self.instrument_list_layout.removeWidget(widget)
            widget.deleteLater()
        self.instrument_widgets.clear()
        
        for instrument in self.app_state.instruments:
            widget = InstrumentWidget(instrument, self.app_state, self.signals)
            self.instrument_widgets[instrument] = widget
            self.instrument_list_layout.addWidget(widget)
    
    def add_instrument(self, instrument):
        widget = InstrumentWidget(instrument, self.app_state, self.signals)
        self.instrument_widgets[instrument] = widget
        self.instrument_list_layout.addWidget(widget)
    
    def remove_instrument(self, instrument):
        if instrument in self.instrument_widgets:
            widget = self.instrument_widgets[instrument]
            self.instrument_list_layout.removeWidget(widget)
            del self.instrument_widgets[instrument]
            widget.deleteLater()


            parent = self.parent()
            while parent is not None and not hasattr(parent, 'rebuild'):
                parent = parent.parent()
            parent.rebuild()


class InstrumentWidget(QWidget):
    def __init__(self, instrument, app_state, signals, parent=None):
        super().__init__(parent)
        self.instrument = instrument
        self.app_state = app_state
        self.signals = signals
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        header_layout = QHBoxLayout()
        instrument_name = instrument.name if hasattr(instrument, 'name') and isinstance(instrument.name, str) else 'Unknown Instrument'
        self.name_label = QLabel(instrument_name)
        header_layout.addWidget(self.name_label)
        
        remove_btn = QPushButton('×')
        remove_btn.setMaximumWidth(20)
        remove_btn.clicked.connect(self.remove_instrument)
        header_layout.addWidget(remove_btn)
        
        layout.addLayout(header_layout)
        
        self.param_summary = QLabel()
        self.param_summary.setStyleSheet('color: #E0E0E0; font-size: 9px; background-color: #3F3F3F; padding: 2px; border-radius: 2px;')
        self.param_summary.setWordWrap(True)
        layout.addWidget(self.param_summary)
        
        self.control_btn = QPushButton('Edit/Control')
        self.control_btn.clicked.connect(self.edit_control_instrument)
        layout.addWidget(self.control_btn)
        
        self.setLayout(layout)
        self.update_status()
    
    def update_status(self):
        if hasattr(self.instrument, 'name') and isinstance(self.instrument.name, str):
            self.name_label.setText(self.instrument.name)
        else:
            self.name_label.setText('Unknown Instrument')
        self.update_parameter_summary()
    
    def update_parameter_summary(self):
        if not hasattr(self.instrument, 'parameters'):
            self.param_summary.setText('No parameters available')
            return
        params = self.instrument.parameters
        summary_lines = []
        if self.instrument.instrument_type == "galvo":
            summary_lines.append(f"Ch: {params.get('slow_axis_channel', '?')}/{params.get('fast_axis_channel', '?')}")
            summary_lines.append(f"Device: {params.get('device_name', '?')}")
            summary_lines.append(f"Rate: {params.get('sample_rate', 0)/1000:.0f}kHz")
        elif self.instrument.instrument_type == "data input":
            channels = params.get('input_channels', [])
            channel_names = params.get('channel_names', {})
            if isinstance(channels, list):
                channel_display = []
                for ch in channels:
                    ch_name = channel_names.get(str(ch), f'ch{ch}')
                    channel_display.append(f"{ch_name}(AI{ch})")
                summary_lines.append(f"Channels: {' | '.join(channel_display)}")
            summary_lines.append(f"Rate: {params.get('sample_rate', 0)/1000:.0f}kHz")
        elif self.instrument.instrument_type == "prior stage":
            summary_lines.append(f"Port: COM{params.get('port', '?')}")
        else:
            summary_lines.append(f"Type: {self.instrument.instrument_type}")
            param_items = list(params.items())[:3]
            for key, value in param_items:
                if isinstance(value, (int, float)):
                    summary_lines.append(f"{key}: {value}")
                else:
                    summary_lines.append(f"{key}: {str(value)[:10]}")
        self.param_summary.setText(' | '.join(summary_lines))
    
    def remove_instrument(self):
        self.signals.instrument_removed.emit(self.instrument)
        
        parent = self.parent()
        while parent is not None and not hasattr(parent, 'rebuild'):
            parent = parent.parent()
        if parent is not None and hasattr(parent, 'rebuild'):
            parent.rebuild()
        
        self.deleteLater()
    
    def edit_control_instrument(self):
        unified_widget = self.instrument.get_widget()
        if unified_widget:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Configure/Control {self.instrument.name}")
            dialog.setModal(True)
            
            layout = QVBoxLayout()
            layout.addWidget(unified_widget)

            button_layout = QHBoxLayout()
            save_btn = QPushButton("Save")
            save_btn.clicked.connect(self.save_instrument_parameters)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(save_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            dialog.resize(400, 300)

            self.current_unified_widget = unified_widget
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.save_instrument_parameters()
        else:
            self.signals.console_message.emit(f"Failed to get widget for {self.instrument.name}")
    
    def save_instrument_parameters(self):
        if hasattr(self, 'current_unified_widget'):
            parameters = self.current_unified_widget.get_parameters()
            if parameters is not None:
                self.instrument.parameters.update(parameters)
                new_name = parameters.get('name', 'Unknown Instrument')
                self.instrument.name = new_name
                self.name_label.setText(new_name)
                self.update_status() 
                self.signals.console_message.emit(f"Updated {new_name} parameters")
                self.signals.instrument_updated.emit(self.instrument)
            else:
                current_name = getattr(self.instrument, 'name', 'Unknown Instrument')
                self.signals.console_message.emit(f"Failed to update {current_name} - invalid parameters")
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes without rebuilding the instrument widget"""
        # This method will be enhanced later to handle modality-specific instrument requirements
        # For now, it does nothing, keeping the instrument widget independent of modality changes
        pass

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
        self.layout = QVBoxLayout()
        self.container.setLayout(self.layout)
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        self.group.toggled.connect(self.container.setVisible)
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)
        self.display_params_widget = None
        
        # Add display selection dropdown
        self.add_display_selection_dropdown()
        
        # Show placeholder after adding the dropdown
        self.show_placeholder()

    def show_placeholder(self):
        # Clear only the display parameters widget, not the entire layout
        # The display selection dropdown should remain
        if self.display_params_widget and hasattr(self.display_params_widget, 'parent'):
            if self.display_params_widget.parent():
                self.display_params_widget.parent().layout().removeWidget(self.display_params_widget)
            self.display_params_widget.deleteLater()
        
        placeholder = QLabel('No display settings available for this display type.')
        placeholder.setStyleSheet('color: #888; font-style: italic;')
        self.display_params_widget = placeholder
        self.layout.addWidget(self.display_params_widget)
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes by updating the display selection dropdown"""
        # Rebuild the display selection dropdown for the new modality
        self.rebuild_display_selection_dropdown()
    
    def rebuild_display_selection_dropdown(self):
        """Rebuild the display selection dropdown for the current modality"""
        from pyrpoc.modalities import modality_registry
        
        # Get current modality
        modality = modality_registry.get_modality(self.app_state.modality)
        if modality is None:
            return
        
        # Clear existing dropdown items
        if hasattr(self, 'display_dropdown'):
            self.display_dropdown.clear()
            
            # Get compatible displays for current modality
            compatible_displays = modality.compatible_displays
            display_names = [display.__name__ for display in compatible_displays]
            
            # Add display options
            self.display_dropdown.addItems(display_names)
            
            # Set current selection (try to keep current if compatible, otherwise use first)
            current_display = self.app_state.selected_display
            index = self.display_dropdown.findText(current_display)
            if index >= 0:
                self.display_dropdown.setCurrentIndex(index)
            else:
                # Current display not compatible with new modality, use first compatible one
                self.app_state.selected_display = display_names[0]
                self.display_dropdown.setCurrentIndex(0)

    def add_display_selection_dropdown(self):
        """Add display selection dropdown to the display controls"""
        from pyrpoc.modalities import modality_registry
        
        # Get current modality
        modality = modality_registry.get_modality(self.app_state.modality)
        if modality is None:
            return
        
        # Create display selection group
        display_selection_group = QGroupBox('Display Type')
        display_selection_layout = QVBoxLayout()
        
        # Create dropdown
        self.display_dropdown = QComboBox()
        
        # Get compatible displays for current modality
        compatible_displays = modality.compatible_displays
        display_names = [display.__name__ for display in compatible_displays]
        
        # Add display options
        self.display_dropdown.addItems(display_names)
        
        # Set current selection
        current_display = self.app_state.selected_display
        index = self.display_dropdown.findText(current_display)
        if index >= 0:
            self.display_dropdown.setCurrentIndex(index)
        
        # Connect signal
        self.display_dropdown.currentTextChanged.connect(self.on_display_selection_changed)
        
        display_selection_layout.addWidget(self.display_dropdown)
        display_selection_group.setLayout(display_selection_layout)
        
        # Add to main layout (before the placeholder)
        self.layout.addWidget(display_selection_group)
    
    def on_display_selection_changed(self, display_name):
        """Handle display selection change"""
        self.app_state.selected_display = display_name
        self.signals.console_message.emit(f"Display type changed to {display_name}")
    
    def update_display_selection(self, display_name):
        """Update the display selection dropdown to reflect external changes"""
        if hasattr(self, 'display_dropdown'):
            index = self.display_dropdown.findText(display_name)
            if index >= 0:
                self.display_dropdown.setCurrentIndex(index)
    
    def set_display_params_widget(self, display_widget):
        # Remove only the display parameters widget, not the entire layout
        if self.display_params_widget and hasattr(self.display_params_widget, 'parent'):
            if self.display_params_widget.parent():
                self.display_params_widget.parent().layout().removeWidget(self.display_params_widget)
            self.display_params_widget.deleteLater()
        
        self.display_params_widget = None

        if display_widget is not None and display_widget.__class__.__name__ == 'TiledChannelsWidget':
            from pyrpoc.displays.multichan_tiled import MultichannelDisplayParametersWidget
            self.display_params_widget = MultichannelDisplayParametersWidget(display_widget)
        else:
            self.show_placeholder()
            return
        self.layout.addWidget(self.display_params_widget)



class RightPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        self.rpoc_channels = {}  # channel_id -> widget
        self.next_channel_id = 1
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)

        self.scroll_area.setWidget(self.content_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)
        
        self.rebuild()

    def rebuild(self):
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.add_modality_specific_controls(self.content_layout)
        self.add_common_controls(self.content_layout)
        
        # Restore RPOC channels from config
        self.restore_rpoc_channels()

    def add_modality_specific_controls(self, layout):
        pass

    def add_common_controls(self, layout):
        rpoc_group = QGroupBox('RPOC Controls')
        rpoc_layout = QVBoxLayout()
        
        add_channel_btn = QPushButton('Add RPOC Channel')
        add_channel_btn.clicked.connect(self.add_rpoc_channel)
        rpoc_layout.addWidget(add_channel_btn)
        
        self.channels_container = QWidget()
        self.channels_layout = QVBoxLayout()
        self.channels_layout.setSpacing(5)
        self.channels_container.setLayout(self.channels_layout)
        rpoc_layout.addWidget(self.channels_container)
        
        rpoc_group.setLayout(rpoc_layout)
        layout.addWidget(rpoc_group)
        
        layout.addStretch()
        
        self.signals.rpoc_channel_removed.connect(self.remove_rpoc_channel)
    
    def add_rpoc_channel(self):
        from pyrpoc.rpoc.rpoc_gui import show_rpoc_channel_selector, create_rpoc_channel_widget
        
        # Show channel type selector
        channel_type = show_rpoc_channel_selector(self)
        if channel_type is None:
            return  # User cancelled
        
        channel_id = self.next_channel_id
        self.next_channel_id += 1
        
        # Create the appropriate channel widget
        channel_widget = create_rpoc_channel_widget(channel_type, channel_id, self.app_state, self.signals)
        self.rpoc_channels[channel_id] = channel_widget
        self.channels_layout.addWidget(channel_widget)
    
    def remove_rpoc_channel(self, channel_id):
        if channel_id in self.rpoc_channels:
            widget = self.rpoc_channels[channel_id]
            self.channels_layout.removeWidget(widget)
            del self.rpoc_channels[channel_id]
            widget.deleteLater()

    def restore_rpoc_channels(self):
        """Restore RPOC channels from the app_state config"""
        # Clear existing channels
        self.rpoc_channels.clear()
        while self.channels_layout.count():
            child = self.channels_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Find the highest channel ID to set next_channel_id
        max_channel_id = 0
        
        # Restore mask channels
        if hasattr(self.app_state, 'rpoc_mask_channels'):
            for channel_id, channel_data in self.app_state.rpoc_mask_channels.items():
                max_channel_id = max(max_channel_id, channel_id)
                from pyrpoc.rpoc.rpoc_gui import create_rpoc_channel_widget
                channel_widget = create_rpoc_channel_widget('mask', channel_id, self.app_state, self.signals)
                self.rpoc_channels[channel_id] = channel_widget
                self.channels_layout.addWidget(channel_widget)
        
        # Restore static channels
        if hasattr(self.app_state, 'rpoc_static_channels'):
            for channel_id, channel_data in self.app_state.rpoc_static_channels.items():
                max_channel_id = max(max_channel_id, channel_id)
                from pyrpoc.rpoc.rpoc_gui import create_rpoc_channel_widget
                channel_widget = create_rpoc_channel_widget('static', channel_id, self.app_state, self.signals)
                self.rpoc_channels[channel_id] = channel_widget
                self.channels_layout.addWidget(channel_widget)
        
        # Restore script channels
        if hasattr(self.app_state, 'rpoc_script_channels'):
            for channel_id, channel_data in self.app_state.rpoc_script_channels.items():
                max_channel_id = max(max_channel_id, channel_id)
                from pyrpoc.rpoc.rpoc_gui import create_rpoc_channel_widget
                channel_widget = create_rpoc_channel_widget('script', channel_id, self.app_state, self.signals)
                self.rpoc_channels[channel_id] = channel_widget
                self.channels_layout.addWidget(channel_widget)
        
        # Update next_channel_id to be higher than any existing channel
        self.next_channel_id = max_channel_id + 1
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes without rebuilding RPOC channels"""
        # This method will be enhanced later to handle modality-specific RPOC requirements
        # For now, it does nothing, keeping RPOC channels independent of modality changes
        pass

class LeftPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)
        
        self.scroll_area.setWidget(self.content_widget)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)
        
        self.rebuild()

    def rebuild(self):
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.modality_controls = ModalityControls(self.app_state, self.signals)
        self.content_layout.addWidget(self.modality_controls)
        
        self.acquisition_parameters = AcquisitionParameters(self.app_state, self.signals)
        self.content_layout.addWidget(self.acquisition_parameters)
        
        self.instrument_controls = InstrumentControls(self.app_state, self.signals)
        self.content_layout.addWidget(self.instrument_controls)
        
        self.display_controls = DisplayControls(self.app_state, self.signals)
        self.content_layout.addWidget(self.display_controls)
        
        self.content_layout.addStretch()
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes by rebuilding only modality-specific components"""
        self.rebuild_modality_specific()

    def rebuild_modality_specific(self):
        """Rebuild only the components that change with modality changes"""
        # Rebuild modality controls
        if hasattr(self, 'modality_controls'):
            self.modality_controls.deleteLater()
        self.modality_controls = ModalityControls(self.app_state, self.signals)
        self.content_layout.insertWidget(0, self.modality_controls)
        
        # Rebuild acquisition parameters
        if hasattr(self, 'acquisition_parameters'):
            self.acquisition_parameters.deleteLater()
        self.acquisition_parameters = AcquisitionParameters(self.app_state, self.signals)
        self.content_layout.insertWidget(1, self.acquisition_parameters)
        
        # Rebuild instrument controls since they are modality-specific
        if hasattr(self, 'instrument_controls'):
            self.instrument_controls.deleteLater()
        self.instrument_controls = InstrumentControls(self.app_state, self.signals)
        self.content_layout.insertWidget(2, self.instrument_controls)


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



    def create_image_display_widget(self):
        # For now, use a default display that works with all modalities
        # This will be enhanced later to handle modality-specific display requirements
        # without requiring a full rebuild
        return TiledChannelsWidget(self.app_state, self.signals)

    def set_image_display_widget(self, widget):
        layout = self.centralWidget().layout()
        if self.image_display_widget is not None:
            layout.removeWidget(self.image_display_widget)
            self.image_display_widget.deleteLater()
        self.image_display_widget = widget
        layout.addWidget(self.image_display_widget)
    
    def on_modality_changed(self, new_modality):
        """Handle modality changes without rebuilding the entire display"""
        # This method will be enhanced later to handle modality-specific display requirements
        # For now, it does nothing, keeping the display independent of modality changes
        pass


class MainWindow(QMainWindow):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setWindowTitle('pyrpoc - Development Mode')
        self.setGeometry(100, 100, 1400, 900)
        
        self.central_widget = None
        self.central_layout = None
        self.vertical_splitter = None
        self.main_splitter = None
        self.left_widget = None
        self.mid_layout = None
        self.right_layout = None
        self.top_bar = None
        
        self.build_gui()

        self.signals.modality_dropdown_changed.connect(self.on_modality_changed)

    def build_gui(self):
        self.clear_existing_gui()
        self.create_central_widget()
        self.create_vertical_splitter()
        self.create_top_bar()
        self.create_main_splitter()
        self.setup_splitter_sizes()
        self.finalize_gui()

    def clear_existing_gui(self):
        if self.centralWidget():
            self.centralWidget().deleteLater()

    def create_central_widget(self):
        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout()
        self.central_layout.setContentsMargins(5, 5, 5, 5)
        self.central_layout.setSpacing(5)

    def create_vertical_splitter(self):
        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.setStyleSheet(SPLITTER_STYLE)

    def create_top_bar(self):
        self.top_bar = TopBar(self.app_state, self.signals)
        # Remove fixed height constraint to allow resizing
        self.top_bar.setMinimumHeight(50)
        self.top_bar.setMaximumHeight(300)
        self.vertical_splitter.addWidget(self.top_bar)

    def create_main_splitter(self):
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setStyleSheet(SPLITTER_STYLE)
        
        self.left_widget = LeftPanel(self.app_state, self.signals)
        self.mid_layout = DockableMiddlePanel(self.app_state, self.signals)
        self.right_layout = RightPanel(self.app_state, self.signals)

        # Set the display parameters widget - this will be enhanced later to handle
        # modality-specific display requirements without requiring a full rebuild
        self.left_widget.display_controls.set_display_params_widget(self.mid_layout.image_display_widget)
        
        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.mid_layout)
        self.main_splitter.addWidget(self.right_layout)
        
        self.vertical_splitter.addWidget(self.main_splitter)

    def setup_splitter_sizes(self):
        # Set initial sizes for the vertical splitter (top bar vs main content)
        if 'vertical_splitter_sizes' in self.app_state.ui_state:
            self.vertical_splitter.setSizes(self.app_state.ui_state['vertical_splitter_sizes'])
        else:
            # Default: top bar takes 100px, rest goes to main content
            # Use a reasonable default if window height is not available yet
            default_height = 900  # Default window height
            self.vertical_splitter.setSizes([100, default_height - 100])
        
        # Set initial sizes for the main horizontal splitter
        self.main_splitter.setSizes([200, 800, 200])
        if 'main_splitter_sizes' in self.app_state.ui_state:
            self.main_splitter.setSizes(self.app_state.ui_state['main_splitter_sizes'])
        
        # Connect splitter movement signals
        self.vertical_splitter.splitterMoved.connect(lambda: self.save_splitter_sizes())
        self.main_splitter.splitterMoved.connect(lambda: self.save_splitter_sizes())

    def finalize_gui(self):
        self.central_layout.addWidget(self.vertical_splitter, stretch=1)
        self.central_widget.setLayout(self.central_layout)
        self.setCentralWidget(self.central_widget)



    def rebuild_display(self):
        try:
            # Get the selected display class
            from pyrpoc.modalities import modality_registry
            modality = modality_registry.get_modality(self.app_state.modality)
            if modality is None:
                return
            
            # Find the selected display class
            selected_display_class = None
            for display_class in modality.compatible_displays:
                if display_class.__name__ == self.app_state.selected_display:
                    selected_display_class = display_class
                    break
            
            if selected_display_class is None:
                # Fallback to first compatible display
                selected_display_class = modality.compatible_displays[0]
                self.app_state.selected_display = selected_display_class.__name__
            
            # Create new display widget
            new_display_widget = selected_display_class(self.app_state, self.signals)
            
            # Update the middle layout with the new display widget
            if self.mid_layout:
                self.mid_layout.set_image_display_widget(new_display_widget)
                
                # Update the display parameters widget in the left panel
                if self.left_widget and hasattr(self.left_widget, 'display_controls'):
                    self.left_widget.display_controls.set_display_params_widget(new_display_widget)
            
            
        except Exception as e:
            self.signals.console_message.emit(f"Error rebuilding display: {e}")
    
    def on_modality_changed(self, new_modality):
        self.app_state.modality = new_modality.lower()
        # Notify all widgets of the modality change instead of rebuilding the entire GUI
        self.notify_modality_changed(new_modality.lower())
        self.signals.console_message.emit(f"Modality changed to {self.app_state.modality}")
    
    def notify_modality_changed(self, new_modality):
        """Notify all relevant widgets of modality changes without rebuilding the entire GUI"""
        # Notify top bar
        if self.top_bar:
            self.top_bar.on_modality_changed(new_modality)
        
        # Notify left panel components
        if self.left_widget:
            self.left_widget.on_modality_changed(new_modality)
        
        # Notify middle panel (display)
        if self.mid_layout:
            self.mid_layout.on_modality_changed(new_modality)
        
        # Notify right panel (RPOC)
        if self.right_layout:
            self.right_layout.on_modality_changed(new_modality)

    def save_splitter_sizes(self):
        if self.vertical_splitter:
            vertical_sizes = self.vertical_splitter.sizes()
            self.signals.ui_state_changed.emit('vertical_splitter_sizes', vertical_sizes)
        if self.main_splitter:
            horizontal_sizes = self.main_splitter.sizes()
            self.signals.ui_state_changed.emit('main_splitter_sizes', horizontal_sizes)

if __name__ == '__main__':
    app_state = AppState()
    signals = StateSignalBus()

    app = QApplication(sys.argv)
    win = MainWindow(app_state, signals)
    
    signals.bind_controllers(app_state, win)
    win.show()
    sys.exit(app.exec())