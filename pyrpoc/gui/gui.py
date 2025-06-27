import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, \
                             QLabel, QWidget, QComboBox, QSplitter, QPushButton, \
                             QPlainTextEdit, QStyle, QGroupBox, QSpinBox, QCheckBox, QLineEdit, QSlider, \
                             QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsLineItem, \
                             QFrame, QSizePolicy, QDockWidget
from PyQt6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter, QFont
from pyrpoc.gui.gui_handler import AppState, StateSignalBus
import sys
import pyqtgraph as pg
from pyrpoc.gui.image_widgets import ImageDisplayWidget
from pyrpoc.gui.dockable_widgets import LinesWidget

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
        main_layout = QVBoxLayout()
        self.group = QGroupBox('Instrument Settings')
        self.group.setCheckable(True)
        self.group.setChecked(app_state.ui_state['instrument_controls_visible'])
        self.group.toggled.connect(lambda checked: signals.ui_state_changed.emit('instrument_controls_visible', checked))
        self.container = QWidget()
        layout = QVBoxLayout()

        self.instrument_list = QWidget()
        instrument_list_layout = QVBoxLayout()
        self.instrument_list.setLayout(instrument_list_layout)
        layout.addWidget(self.instrument_list)
        add_btn = QPushButton('Add Instrument')
        add_btn.clicked.connect(self.signals.add_instrument_btn_clicked.emit)
        layout.addWidget(add_btn)
        self.container.setLayout(layout)
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        self.group.toggled.connect(self.container.setVisible)
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)


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


class RightPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        
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
        
        rpoc_layout.addWidget(QPushButton('Load Mask'))
        rpoc_layout.addWidget(QPushButton('Save Mask'))
        rpoc_group.setLayout(rpoc_layout)
        layout.addWidget(rpoc_group)
        
        layout.addStretch()


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