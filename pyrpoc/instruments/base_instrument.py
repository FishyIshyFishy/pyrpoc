import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal

class Instrument(abc.ABC):
    def __init__(self, name, instrument_type):
        self.name = name
        self.instrument_type = instrument_type
        self.parameters = {}

    @abc.abstractmethod
    def initialize(self):
        """Initialize the instrument connection"""
        pass

    @abc.abstractmethod
    def get_widget(self):
        """Return a unified widget for configuring and controlling this instrument"""
        pass

    def get_instrument_info(self):
        return f'{self.instrument_type}: {self.name}'

    def get_parameters(self):
        return self.parameters.copy()

