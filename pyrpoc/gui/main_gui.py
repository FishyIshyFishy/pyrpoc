import numpy as np
from PyQt6.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QMainWindow,
                             QLabel, QWidget, QComboBox, QSplitter)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
import sys

######### HELPER CLASSES ###############3
class AppState:
    '''
    APP STATE CLASS
    holds all of the important signaled variables
    '''
    def __init__(self):
        self.modality = 'widefield'
        self.current_data = None

class StateSignalBus(QObject):
    '''
    SIGNAL COMMUNICATION CLASS
    create all the signals that will be transmitted multi-functionally
    '''
    modality_changed = pyqtSignal(str) # emits when the dropdown for the modality is changed
    data_updated = pyqtSignal(object) # emits when data acquisition is complete

app_state = AppState()
signals = StateSignalBus()

######### GUI SUBWIDGETS ###############3
class ModalitySelector(QWidget):
    def __init__(self):
        super().__init__()
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
        app_state.modality = text
        signals.modality_changed.emit(text)

class StatusBar(QWidget):
    def __init__(self):
        super().__init__()
        self.label = QLabel('Idle')
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addStretch()
        self.setLayout(layout)
        signals.modality_changed.connect(self.update_status)

    def update_status(self, new_modality):
        self.label.setText(f'Modality changed to {new_modality}')

class GlobalControls(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Global Controls Placeholder'))
        layout.addStretch()
        self.setLayout(layout)

class DisplayPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel('Image Display Here')
        layout.addWidget(self.label)
        self.setLayout(layout)
        signals.data_updated.connect(self.update_display)

    def update_display(self, data):
        self.label.setText('New Data Received')

class RPOCPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel('RPOC Control Panel'))
        layout.addStretch()
        self.setLayout(layout)
        

########## MAIN GUI WINDOW ##########3
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('pyrpoc')
        self.setGeometry(100, 100, 1200, 800)

        top_bar = QHBoxLayout()
        self.selector = ModalitySelector()
        self.status = StatusBar()
        top_bar.addWidget(self.selector)
        top_bar.addStretch()
        top_bar.addWidget(self.status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_panel = GlobalControls()
        self.center_display = DisplayPanel()
        self.right_panel = RPOCPanel()

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.center_display)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([200,800,200])

        wrapper = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(splitter)
        wrapper.setLayout(layout)

        self.setCentralWidget(wrapper)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())