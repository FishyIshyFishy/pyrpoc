import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal
from .galvo import Galvo
from .data_input import DataInput
from .prior_stage import PriorStage

def create_instrument(instrument_type, name, parameters=None, console_callback=None):
    if parameters is None:
        parameters = {}
    
    if instrument_type == "galvo":
        instrument = Galvo(name, console_callback=console_callback)
    elif instrument_type == "data input":
        instrument = DataInput(name, console_callback=console_callback)
    elif instrument_type == "prior stage":
        instrument = PriorStage(name, console_callback=console_callback)
    else:
        raise ValueError(f"Unknown instrument type: {instrument_type}")

    if parameters:
        instrument.validate_parameters(parameters)
        instrument.parameters.update(parameters)
    
    return instrument

def get_instruments_by_type(instruments_list, instrument_type):
    return [instrument for instrument in instruments_list 
            if instrument.instrument_type == instrument_type]



def initialize_instrument_with_retry(instrument, max_retries=3):
    for attempt in range(max_retries):
        try:
            if instrument.initialize():
                return True
            else:
                print(f"Attempt {attempt + 1} failed for {instrument.name}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {instrument.name}: {e}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in 1 second...")
            import time
            time.sleep(1)
    
    print(f"Failed to initialize {instrument.name} after {max_retries} attempts")
    return False

def show_add_instrument_dialog(parent=None):
    dialog = QDialog(parent)
    dialog.setWindowTitle("Add Instrument")
    dialog.setModal(True)
    
    layout = QVBoxLayout()
    
    layout.addWidget(QLabel("Select instrument type:"))
    
    combo = QComboBox()
    combo.addItems(['prior stage'])
    layout.addWidget(combo)
    
    button_layout = QHBoxLayout()
    ok_btn = QPushButton("OK")
    cancel_btn = QPushButton("Cancel")
    
    ok_btn.clicked.connect(dialog.accept)
    cancel_btn.clicked.connect(dialog.reject)
    
    button_layout.addWidget(ok_btn)
    button_layout.addWidget(cancel_btn)
    layout.addLayout(button_layout)
    
    dialog.setLayout(layout)
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return combo.currentText()
    else:
        return None

def show_configure_instrument_dialog(instrument_type, parent=None, console_callback=None):
    instrument = create_instrument(instrument_type.lower(), instrument_type, console_callback=console_callback)
    unified_widget = instrument.get_widget()
    if unified_widget:
        dialog = QDialog(parent)
        dialog.setWindowTitle(f"Configure {instrument_type}")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        layout.addWidget(unified_widget)
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.resize(400, 300)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            parameters = unified_widget.get_parameters()
            
            if parameters is not None:  
                instrument_name = parameters.get('name', f'{instrument_type}')
                instrument = create_instrument(instrument_type.lower(), instrument_name, parameters, console_callback=console_callback)
                if hasattr(instrument, 'name') and instrument.name:
                    display_name = instrument.name
                else:
                    display_name = instrument_name
                    
                return instrument, display_name
            else:
                if console_callback:
                    console_callback(f"Failed to create {instrument_type} - invalid parameters")
                return None, None
        else:
            return None, None
    else:
        if console_callback:
            console_callback(f"Failed to get configuration widget for {instrument_type}")
        return None, None 