import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, \
                             QLabel, QWidget, QComboBox, QSplitter, QPushButton, \
                             QPlainTextEdit, QStyle, QGroupBox
from PyQt6.QtCore import Qt
from pyrpoc.gui.gui_handler import AppState, StateSignalBus
import sys

DEV_BORDER_STYLE = """
    QWidget {
        border: 2px solid #FF0000;
        border-radius: 4px;
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
        self.setFixedHeight(100)  # Set a fixed height for the top bar
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins to make it more compact

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

        continuous_btn = QPushButton()
        continuous_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        continuous_btn.setToolTip('Start Continuous Acquisition')
        continuous_btn.clicked.connect(signals.continuous_btn_clicked.emit)
        controls_layout.addWidget(continuous_btn)

        single_btn = QPushButton()
        single_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
        single_btn.setToolTip('Start Single Acquisition')
        single_btn.clicked.connect(signals.single_btn_clicked.emit)
        controls_layout.addWidget(single_btn)

        stop_btn = QPushButton()
        stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        stop_btn.setToolTip('Stop Acquisition')
        stop_btn.clicked.connect(signals.stop_btn_clicked.emit)
        controls_layout.addWidget(stop_btn)

        # Set fixed size for icon buttons to make them square
        for btn in [continuous_btn, single_btn, stop_btn]:
            btn.setFixedSize(32, 32)

        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)

        # system console to display status updates
        console_widget = QWidget()
        console_layout = QVBoxLayout()
        console_layout.setContentsMargins(0, 0, 0, 0)

        console = QPlainTextEdit()
        console.setReadOnly(True)

        console_layout.addWidget(console)
        console_widget.setLayout(console_layout)
        layout.addWidget(console_widget, stretch=1)  # Give console widget stretch to take remaining space

        self.setLayout(layout)


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
        ms_dropdown.currentTextChanged.connect(signals.modality_dropdown_changed)
        layout.addWidget(ms_dropdown)

        self.setLayout(layout)


class InstrumentControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        
        # Create the group box
        self.group = QGroupBox('Instrument Settings')
        self.group.setCheckable(True)
        self.group.setChecked(True)
        
        # Create container for the contents
        self.container = QWidget()
        layout = QVBoxLayout()
        
        # List for instruments
        self.instrument_list = QWidget()
        instrument_list_layout = QVBoxLayout()
        self.instrument_list.setLayout(instrument_list_layout)
        layout.addWidget(self.instrument_list)

        # Add instrument button
        add_btn = QPushButton('Add Instrument')
        add_btn.clicked.connect(signals.add_instrument_btn_clicked.emit)
        layout.addWidget(add_btn)

        self.container.setLayout(layout)
        
        # Set up the group box layout
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        
        # Connect toggle signal
        self.group.toggled.connect(self.container.setVisible)
        
        # Main layout for this widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)


class DisplayControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        
        self.group = QGroupBox('Display Settings')
        self.group.setCheckable(True)
        self.group.setChecked(True)

        self.container = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel('Display controls to be implemented'))
        
        self.container.setLayout(layout)

        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)

        self.group.toggled.connect(self.container.setVisible)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)


class LeftPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Add each control section
        self.modality_controls = ModalityControls(app_state, signals)
        layout.addWidget(self.modality_controls)

        self.instrument_controls = InstrumentControls(app_state, signals)
        layout.addWidget(self.instrument_controls)

        self.display_controls = DisplayControls(app_state, signals)
        layout.addWidget(self.display_controls)

        # Add stretch to push all widgets to the top
        layout.addStretch()

        self.setLayout(layout)


########################################################
######### MIDDLE (DISPLAY) PANEL STUFF #################
########################################################
class MiddlePanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        layout = QVBoxLayout()

        self.label = QLabel('Image Display Here')
        layout.addWidget(self.label)


        self.setLayout(layout)

########################################################
######### RIGHT (RPOC) PANEL STUFF #################
########################################################
class RightPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        layout = QVBoxLayout()

        layout.addWidget(QLabel('RPOC Controls placeholder'))
        self.setLayout(layout)
        
#########################################
######### MAIN WINDOW #################
#########################################



class MainWindow(QMainWindow):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setWindowTitle('pyrpoc - Development Mode')
        self.setGeometry(100, 100, 1200, 800)

        # top bar config
        top_bar = TopBar(app_state, signals)

        # main section of GUI config
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = LeftPanel(app_state, signals) 
        splitter.addWidget(left_widget) # left section of the splitter - global settings
        mid_layout = MiddlePanel(app_state, signals)
        splitter.addWidget(mid_layout) # mid section of splitter - display stuff
        right_layout = RightPanel(app_state, signals)
        splitter.addWidget(right_layout) # right section of splitter - rpoc settings
        splitter.setSizes([200,800,200])

        # main gui organization - top bar above the splitter
        wrapper = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  
        layout.setSpacing(5)
        
        layout.addWidget(top_bar, stretch=0)
        layout.addWidget(splitter, stretch=1)
        
        wrapper.setLayout(layout)
        self.setCentralWidget(wrapper)

if __name__ == '__main__':
    app_state = AppState() # can initialize GUI configs with this
    signals = StateSignalBus()
    signals.bind_controllers(app_state)

    app = QApplication(sys.argv)
    win = MainWindow(app_state, signals)
    win.show()
    sys.exit(app.exec())