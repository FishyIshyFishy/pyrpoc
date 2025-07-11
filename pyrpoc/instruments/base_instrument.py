import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal

class Instrument(abc.ABC):
    def __init__(self, name, instrument_type, console_callback=None):
        self.name = name
        self.instrument_type = instrument_type
        self.parameters = {}
        self.console_callback = console_callback

    def log_message(self, message):
        if self.console_callback:
            self.console_callback(message)

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

    @abc.abstractmethod
    def validate_parameters(self, parameters):
        """Validate instrument communication parameters"""
        pass

    def disconnect(self):
        """Disconnect the instrument. Override in subclasses if needed."""
        pass

