import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, \
                             QLabel, QWidget, QComboBox, QSplitter, QPushButton, \
                             QPlainTextEdit, QStyle, QGroupBox, QSpinBox, QCheckBox, QLineEdit, QSlider, \
                             QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsLineItem, \
                             QFrame, QSizePolicy, QDockWidget, QFileDialog, QDialog, QFormLayout, QDoubleSpinBox, \
                             QScrollArea
from PyQt6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter, QFont
from pyrpoc.gui_handler import AppState, StateSignalBus
import sys
import pyqtgraph as pg
from pyrpoc.displays import *
from pyrpoc.dockable_widgets import LinesWidget
from pyrpoc.rpoc_mask_editor import RPOCMaskEditor
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
        ms_dropdown.addItems(['Simulated', 'Confocal', 'Split Data Stream'])
        current_modality = self.app_state.modality.capitalize()
        index = ms_dropdown.findText(current_modality)
        if index >= 0:
            ms_dropdown.setCurrentIndex(index)
        ms_dropdown.currentTextChanged.connect(self.signals.modality_dropdown_changed)
        layout.addWidget(ms_dropdown)

        self.setLayout(layout)

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

    def add_specific_parameters(self):
        modality = self.app_state.modality.lower()
        
        if modality == 'confocal':
            self.add_galvo_parameters()

        elif modality == 'split data stream':
            # Split percentage parameter for split data stream modality
            split_layout = QHBoxLayout()
            split_layout.addWidget(QLabel('Split Percentage:'))
            self.split_percentage_spinbox = QSpinBox()
            self.split_percentage_spinbox.setRange(1, 99)
            self.split_percentage_spinbox.setValue(self.app_state.acquisition_parameters.get('split_percentage', 50))
            self.split_percentage_spinbox.setSuffix('%')
            self.split_percentage_spinbox.valueChanged.connect(
                lambda value: self.signals.acquisition_parameter_changed.emit('split_percentage', value))
            split_layout.addWidget(self.split_percentage_spinbox)
            self.layout.addLayout(split_layout)
            

            self.add_galvo_parameters()
            self.add_prior_stage_parameters()

        elif modality == 'simulated':
            self.add_pixel_parameters()

        else:
            self.add_galvo_parameters()

    def add_galvo_parameters(self):
        galvo_group = QGroupBox("Galvo Parameters")
        galvo_layout = QFormLayout()
        
        self.dwell_time_spin = QDoubleSpinBox()
        self.dwell_time_spin.setRange(1e-6, 1e-3)
        self.dwell_time_spin.setValue(self.app_state.acquisition_parameters.get('dwell_time', 10e-6))
        self.dwell_time_spin.setSuffix(" s")
        self.dwell_time_spin.setDecimals(6)
        self.dwell_time_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('dwell_time', value))
        galvo_layout.addRow("Dwell Time:", self.dwell_time_spin)
        
        self.extrasteps_left_spin = QSpinBox()
        self.extrasteps_left_spin.setRange(0, 10000)
        self.extrasteps_left_spin.setValue(self.app_state.acquisition_parameters.get('extrasteps_left', 50))
        self.extrasteps_left_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('extrasteps_left', value))
        galvo_layout.addRow("Extra Steps Left:", self.extrasteps_left_spin)
        
        self.extrasteps_right_spin = QSpinBox()
        self.extrasteps_right_spin.setRange(0, 10000)
        self.extrasteps_right_spin.setValue(self.app_state.acquisition_parameters.get('extrasteps_right', 50))
        self.extrasteps_right_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('extrasteps_right', value))
        galvo_layout.addRow("Extra Steps Right:", self.extrasteps_right_spin)

        self.amplitude_x_spin = QDoubleSpinBox()
        self.amplitude_x_spin.setRange(0.01, 10.0)
        self.amplitude_x_spin.setValue(self.app_state.acquisition_parameters.get('amplitude_x', 0.5))
        self.amplitude_x_spin.setSuffix(" V")
        self.amplitude_x_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('amplitude_x', value))
        galvo_layout.addRow("Amplitude X:", self.amplitude_x_spin)
        
        self.amplitude_y_spin = QDoubleSpinBox()
        self.amplitude_y_spin.setRange(0.01, 10.0)
        self.amplitude_y_spin.setValue(self.app_state.acquisition_parameters.get('amplitude_y', 0.5))
        self.amplitude_y_spin.setSuffix(" V")
        self.amplitude_y_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('amplitude_y', value))
        galvo_layout.addRow("Amplitude Y:", self.amplitude_y_spin)

        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-10.0, 10.0)
        self.offset_x_spin.setValue(self.app_state.acquisition_parameters.get('offset_x', 0.0))
        self.offset_x_spin.setSuffix(" V")
        self.offset_x_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('offset_x', value))
        galvo_layout.addRow("Offset X:", self.offset_x_spin)
        
        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-10.0, 10.0)
        self.offset_y_spin.setValue(self.app_state.acquisition_parameters.get('offset_y', 0.0))
        self.offset_y_spin.setSuffix(" V")
        self.offset_y_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('offset_y', value))
        galvo_layout.addRow("Offset Y:", self.offset_y_spin)
        
        self.x_pixels_spin = QSpinBox()
        self.x_pixels_spin.setRange(64, 4096)
        self.x_pixels_spin.setValue(self.app_state.acquisition_parameters.get('x_pixels', 512))
        self.x_pixels_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('x_pixels', value))
        galvo_layout.addRow("X Pixels:", self.x_pixels_spin)
        
        self.y_pixels_spin = QSpinBox()
        self.y_pixels_spin.setRange(64, 4096)
        self.y_pixels_spin.setValue(self.app_state.acquisition_parameters.get('y_pixels', 512))
        self.y_pixels_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('y_pixels', value))
        galvo_layout.addRow("Y Pixels:", self.y_pixels_spin)
        
        galvo_group.setLayout(galvo_layout)
        self.layout.addWidget(galvo_group)

    def add_pixel_parameters(self):
        """Add pixel parameters for modalities that don't use galvo scanning"""
        pixel_group = QGroupBox("Image Parameters")
        pixel_layout = QFormLayout()
        
        self.x_pixels_spin = QSpinBox()
        self.x_pixels_spin.setRange(64, 4096)
        self.x_pixels_spin.setValue(self.app_state.acquisition_parameters.get('x_pixels', 512))
        self.x_pixels_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('x_pixels', value))
        pixel_layout.addRow("X Pixels:", self.x_pixels_spin)
        
        self.y_pixels_spin = QSpinBox()
        self.y_pixels_spin.setRange(64, 4096)
        self.y_pixels_spin.setValue(self.app_state.acquisition_parameters.get('y_pixels', 512))
        self.y_pixels_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('y_pixels', value))
        pixel_layout.addRow("Y Pixels:", self.y_pixels_spin)
        
        pixel_group.setLayout(pixel_layout)
        self.layout.addWidget(pixel_group)

    def add_prior_stage_parameters(self):
        prior_group = QGroupBox("Prior Stage Parameters")
        prior_layout = QFormLayout()

        self.numtiles_x_spin = QSpinBox()
        self.numtiles_x_spin.setRange(1, 1000)
        self.numtiles_x_spin.setValue(self.app_state.acquisition_parameters.get('numtiles_x', 10))
        self.numtiles_x_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('numtiles_x', value))
        prior_layout.addRow("X Tiles:", self.numtiles_x_spin)
        
        self.numtiles_y_spin = QSpinBox()
        self.numtiles_y_spin.setRange(1, 1000)
        self.numtiles_y_spin.setValue(self.app_state.acquisition_parameters.get('numtiles_y', 10))
        self.numtiles_y_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('numtiles_y', value))
        prior_layout.addRow("Y Tiles:", self.numtiles_y_spin)
        
        self.numtiles_z_spin = QSpinBox()
        self.numtiles_z_spin.setRange(1, 1000)
        self.numtiles_z_spin.setValue(self.app_state.acquisition_parameters.get('numtiles_z', 5))
        self.numtiles_z_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('numtiles_z', value))
        prior_layout.addRow("Z Tiles:", self.numtiles_z_spin)
        
        self.tile_size_x_spin = QDoubleSpinBox()
        self.tile_size_x_spin.setRange(0.1, 10000)
        self.tile_size_x_spin.setValue(self.app_state.acquisition_parameters.get('tile_size_x', 100))
        self.tile_size_x_spin.setSuffix(" µm")
        self.tile_size_x_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('tile_size_x', value))
        prior_layout.addRow("X Tile Size:", self.tile_size_x_spin)
        
        self.tile_size_y_spin = QDoubleSpinBox()
        self.tile_size_y_spin.setRange(0.1, 10000)
        self.tile_size_y_spin.setValue(self.app_state.acquisition_parameters.get('tile_size_y', 100))
        self.tile_size_y_spin.setSuffix(" µm")
        self.tile_size_y_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('tile_size_y', value))
        prior_layout.addRow("Y Tile Size:", self.tile_size_y_spin)
        
        self.tile_size_z_spin = QDoubleSpinBox()
        self.tile_size_z_spin.setRange(0.1, 10000)
        self.tile_size_z_spin.setValue(self.app_state.acquisition_parameters.get('tile_size_z', 50))
        self.tile_size_z_spin.setSuffix(" µm")
        self.tile_size_z_spin.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('tile_size_z', value))
        prior_layout.addRow("Z Tile Size:", self.tile_size_z_spin)
        
        prior_group.setLayout(prior_layout)
        self.layout.addWidget(prior_group)

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
    
    def rebuild(self):
        # clear any existing modality specific instrument buttons
        while self.modality_buttons_layout.count():
            child = self.modality_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

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
        elif modality == 'split data stream':
            if not self.has_instrument_type('Galvo'):
                galvo_btn = QPushButton('Add Galvos')
                galvo_btn.clicked.connect(lambda: self.signals.add_modality_instrument.emit('Galvo'))
                self.modality_buttons_layout.addWidget(galvo_btn)
            
            if not self.has_instrument_type('Data Input'):
                data_input_btn = QPushButton('Add Data Inputs')
                data_input_btn.clicked.connect(lambda: self.signals.add_modality_instrument.emit('Data Input'))
                self.modality_buttons_layout.addWidget(data_input_btn)
            
            if not self.has_instrument_type('Prior Stage'):
                prior_stage_btn = QPushButton('Add Prior Stage')
                prior_stage_btn.clicked.connect(lambda: self.signals.add_modality_instrument.emit('Prior Stage'))
                self.modality_buttons_layout.addWidget(prior_stage_btn)
        self.rebuild_instrument_list()
    
    def has_instrument_type(self, instrument_type):
        if hasattr(self.app_state, 'instruments'):
            for instrument in self.app_state.instruments:
                if instrument.instrument_type == instrument_type:
                    return True
        return False
    
    def rebuild_instrument_list(self):
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
        if self.instrument.instrument_type == "Galvo":
            summary_lines.append(f"Ch: {params.get('slow_axis_channel', '?')}/{params.get('fast_axis_channel', '?')}")
            summary_lines.append(f"Device: {params.get('device_name', '?')}")
            summary_lines.append(f"Rate: {params.get('sample_rate', 0)/1000:.0f}kHz")
        elif self.instrument.instrument_type == "Data Input":
            channels = params.get('input_channels', [])
            channel_names = params.get('channel_names', {})
            if isinstance(channels, list):
                channel_display = []
                for ch in channels:
                    ch_name = channel_names.get(str(ch), f'ch{ch}')
                    channel_display.append(f"{ch_name}(AI{ch})")
                summary_lines.append(f"Channels: {' | '.join(channel_display)}")
            summary_lines.append(f"Rate: {params.get('sample_rate', 0)/1000:.0f}kHz")
        elif self.instrument.instrument_type == "Prior Stage":
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
        if self.instrument in self.app_state.instruments:
            self.app_state.instruments.remove(self.instrument)
        
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

        header_layout = QHBoxLayout()
        self.channel_label = QLabel(f'Channel {channel_id}')
        header_layout.addWidget(self.channel_label)

        remove_btn = QPushButton('×')
        remove_btn.setMaximumWidth(20)
        remove_btn.clicked.connect(self.remove_channel)
        header_layout.addWidget(remove_btn)
        
        layout.addLayout(header_layout)

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

        self.mask_status = QLabel('No mask loaded')
        self.mask_status.setStyleSheet('color: #666; font-size: 10px;')
        layout.addWidget(self.mask_status)
        
        buttons_layout = QHBoxLayout()
        
        self.create_mask_btn = QPushButton('Create Mask')
        self.create_mask_btn.clicked.connect(self.create_mask)
        buttons_layout.addWidget(self.create_mask_btn)
        
        self.load_mask_btn = QPushButton('Load Mask')
        self.load_mask_btn.clicked.connect(self.load_mask)
        buttons_layout.addWidget(self.load_mask_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)

        self.on_daq_channel_changed()

        self.update_mask_status()
    
    def on_daq_channel_changed(self):
        if not hasattr(self.app_state, 'rpoc_channels'):
            self.app_state.rpoc_channels = {}
        
        device = self.device_edit.currentText().strip()
        port_line = self.port_line_edit.currentText().strip()
        
        self.app_state.rpoc_channels[self.channel_id] = {
            'device': device,
            'port_line': port_line
        }
        self.signals.console_message.emit(f'RPOC channel {self.channel_id} set on {device}/{port_line}')
    
    def get_daq_channel_info(self):
        device = self.device_edit.currentText().strip()
        port_line = self.port_line_edit.currentText().strip()
        return {
            'device': device,
            'port_line': port_line
        }
    
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
        print(f"RPOC mask stored for channel {self.channel_id}: {mask.shape if hasattr(mask, 'shape') else 'check file type'}")
    
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
        
        # remove DAQ channel info from app_state
        if hasattr(self.app_state, 'rpoc_channels') and self.channel_id in self.app_state.rpoc_channels:
            del self.app_state.rpoc_channels[self.channel_id]
        
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

    def add_modality_specific_controls(self, layout):
        pass

    def add_common_controls(self, layout):
        rpoc_group = QGroupBox('RPOC Controls')
        rpoc_layout = QVBoxLayout()
        
        rpoc_enabled_checkbox = QCheckBox('RPOC Enabled')
        rpoc_enabled_checkbox.setChecked(self.app_state.rpoc_enabled)
        rpoc_enabled_checkbox.toggled.connect(lambda checked: self.signals.rpoc_enabled_changed.emit(checked))
        rpoc_layout.addWidget(rpoc_enabled_checkbox)
        
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
        
        self.on_lines_toggled(self.app_state.ui_state['lines_enabled'])


    def create_image_display_widget(self):
        modality = self.app_state.modality.lower()
        
        if modality in ['confocal', 'split data stream']:
            return MultichannelImageDisplayWidget(self.app_state, self.signals)
        else:
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
        self.clear_existing_gui()
        self.create_central_widget()
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

    def create_top_bar(self):
        self.top_bar = TopBar(self.app_state, self.signals)
        self.central_layout.addWidget(self.top_bar, stretch=0)

    def create_main_splitter(self):
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setStyleSheet(SPLITTER_STYLE)
        
        self.left_widget = LeftPanel(self.app_state, self.signals)
        self.mid_layout = DockableMiddlePanel(self.app_state, self.signals)
        self.right_layout = RightPanel(self.app_state, self.signals)
        
        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.mid_layout)
        self.main_splitter.addWidget(self.right_layout)

    def setup_splitter_sizes(self):
        self.main_splitter.setSizes([200, 800, 200])
        if 'main_splitter_sizes' in self.app_state.ui_state:
            self.main_splitter.setSizes(self.app_state.ui_state['main_splitter_sizes'])
        self.main_splitter.splitterMoved.connect(lambda: self.save_splitter_sizes())

    def finalize_gui(self):
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