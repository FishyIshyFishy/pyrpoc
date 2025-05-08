import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, \
                             QLabel, QWidget, QComboBox, QSplitter, QPushButton, \
                             QPlainTextEdit
from PyQt6.QtCore import Qt
from pyrpoc.gui.gui_handler import AppState, StateSignalBus
import sys



class TopBar(QWidget):
    '''
    horizontal orientation with the important control widgets, most space given to SystemConsole
    AppConfigButtons | AcquisitionControls | SystemStatus
    '''
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        layout = QHBoxLayout()

        # two buttons for loading exisitng AppState configs from .json, and saving
        config_widget = QWidget()
        config_layout = QVBoxLayout()

        load_btn = QPushButton('Load Config')
        load_btn.clicked.connect(signals.load_config_btn_clicked.emit)
        config_layout.addWidget(load_btn)

        save_btn = QPushButton('Save Config')
        save_btn.clicked.connect(signals.save_config_btn_clicked.emit)
        config_layout.addWidget(save_btn)

        config_layout.addStretch()
        config_widget.setLayout(config_layout)

        layout.addWidget(config_widget)



        # main acquisition controls: acq continuous, acq single, stop acq
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()

        continuous_btn = QPushButton('Start Continuous')
        continuous_btn.clicked.connect(signals.continuous_btn_clicked.emit)
        controls_layout.addWidget(continuous_btn)

        single_btn = QPushButton('Start')
        single_btn.clicked.connect(signals.single_btn_clicked.emit)
        controls_layout.addWidget(single_btn)

        stop_btn = QPushButton('Stop')
        stop_btn.clicked.connect(signals.stop_btn_clicked.emit)
        controls_layout.addWidget(stop_btn)

        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)



        # system console to display status updates, probably with a "system ready" label above it
        console_widget = QWidget()
        console_layout = QVBoxLayout()

        console = QPlainTextEdit()
        console.setReadOnly(True)

        console_layout.addWidget(console)
        layout.addWidget(console_widget)



        self.setLayout(layout)




###################################################
######### LEFT (SETTINGS) PANEL STUFF ###############
###################################################

# class AcquisitionPlanDisplay(QWidget):
#     def __init__(self, app_state: AppState, signals: StateSignalBus):
#         super().__init__()
#         self.app_state = app_state
#         self.signals = signals

# class InstrumentSettings(QWidget):

# class DisplaySettings(QWidget):

# class FullSettings(QWidget):

class LeftPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        layout = QVBoxLayout()

        # modality selector
        modality_selector_layout = QHBoxLayout()
        modality_selector_widget = QWidget()

        ms_label = QLabel('Modality:')
        modality_selector_layout.addWidget(ms_label)

        ms_dropdown = QComboBox()
        ms_dropdown.addItems(['Widefield', 'Confocal', 'Mosaic', 'ZScan'])
        ms_dropdown.currentTextChanged.connect(signals.modality_dropdown_changed)
        modality_selector_layout.addWidget(ms_dropdown)

        modality_selector_widget.setLayout(modality_selector_layout)
        layout.addWidget(modality_selector_widget)



        # acquisition plan display



        # instrument settings



        # display settings



        self.setLayout(layout)


########################################################
######### MIDDLE (DISPLAY) PANEL STUFF #################
########################################################
class MiddlePanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
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
        self.setWindowTitle('pyrpoc')
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
        layout.addWidget(top_bar)
        layout.addWidget(splitter)
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