import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, \
                             QLabel, QWidget, QComboBox, QSplitter
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from pyrpoc.gui.gui_handler import AppState, StateSignalBus
import sys


########################################
######### GUI SUBWIDGETS ###############
########################################
class ModalitySelector(QWidget):
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals

        layout = QHBoxLayout()
        self.label = QLabel('Modality:')
        self.dropdown = QComboBox()
        self.dropdown.addItems(['Widefield', 'Confocal', 'Mosaic', 'ZScan'])
        self.dropdown.currentTextChanged.connect(self.on_modality_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.dropdown)
        layout.addStretch()
        self.setLayout(layout)

    def on_modality_changed(self, text):
        self.signals.modality_changed.emit(text)

class StatusBar(QWidget):
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals

        self.label = QLabel('Idle')
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addStretch()
        self.setLayout(layout)

        self.signals.modality_changed.connect(self.update_status)

    def update_status(self, new_modality):
        self.label.setText(f'Modality changed to {new_modality}')






########################################
######### GUI MAIN PANELS ###############
########################################

class TopPanel(QWidget):
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals

        layout = QHBoxLayout()

        self.status = StatusBar(app_state, signals)
        layout.addWidget(self.status)
        self.setLayout(layout)

class SettingsPanel(QWidget):
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals

        layout = QVBoxLayout()

        self.selector = ModalitySelector(app_state, signals)
        layout.addWidget(self.selector)
        self.setLayout(layout)

class DisplayPanel(QWidget):
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals

        layout = QVBoxLayout()

        self.label = QLabel('Image Display Here')
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.signals.data_updated.connect(self.update_display)

    def update_display(self, data):
        self.label.setText('New Data Received')

class RPOCPanel(QWidget):
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        
        layout = QVBoxLayout()

        layout.addWidget(QLabel('RPOC Controls placeholder'))
        self.setLayout(layout)
        




#####################################
########## MAIN GUI WINDOW ###########
#####################################
class MainWindow(QMainWindow):
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setWindowTitle('pyrpoc')
        self.setGeometry(100, 100, 1200, 800)

        # top bar config
        top_bar = TopPanel(app_state, signals)

        # main section of GUI config
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = SettingsPanel(app_state, signals) 
        splitter.addWidget(left_widget) # left section of the splitter - global settings
        mid_layout = DisplayPanel(app_state, signals)
        splitter.addWidget(mid_layout) # mid section of splitter - display stuff
        right_layout = RPOCPanel(app_state, signals)
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
    signals.bind_controllers()

    app = QApplication(sys.argv)
    win = MainWindow(app_state, signals)
    win.show()
    sys.exit(app.exec())