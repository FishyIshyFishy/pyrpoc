import numpy as np
import abc
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QSpinBox, QPushButton, 
                             QGroupBox, QFormLayout, QMessageBox, QWidget,
                             QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal

# Import the actual instrument classes
from pyrpoc.instruments.galvo import Galvo
from pyrpoc.instruments.data_input import DataInput
from pyrpoc.instruments.prior_stage import PriorStage

# Factory function to create instruments
def create_instrument(instrument_type, name, parameters=None):
    """Create an instrument instance based on type, name, and parameters"""
    if parameters is None:
        parameters = {}
    
    if instrument_type == "Galvo":
        instrument = Galvo(name)
        instrument.parameters.update(parameters)
    elif instrument_type == "Data Input":
        instrument = DataInput(name)
        instrument.parameters.update(parameters)
    elif instrument_type == "Prior Stage":
        instrument = PriorStage(name)
        instrument.parameters.update(parameters)
    else:
        raise ValueError(f"Unknown instrument type: {instrument_type}")
    
    return instrument

def get_instruments_by_type(instruments_list, instrument_type):
    """Get all instruments of a specific type from the instruments list"""
    return [instrument for instrument in instruments_list 
            if instrument.instrument_type == instrument_type]

def validate_instrument_parameters(instrument_type, parameters):
    """Validate instrument parameters before creating or updating an instrument"""
    if instrument_type == "Galvo":
        required_params = ['slow_axis_channel', 'fast_axis_channel', 'sample_rate', 'device_name']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for Galvo: {param}")
        
        # Validate parameter values
        if parameters['slow_axis_channel'] < 0 or parameters['fast_axis_channel'] < 0:
            raise ValueError("Channel numbers must be non-negative")
        if parameters['sample_rate'] <= 0:
            raise ValueError("Sample rate must be positive")
            
    elif instrument_type == "Data Input":
        required_params = ['input_channels', 'sample_rate', 'device_name']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for Data Input: {param}")
        
        # Validate parameter values
        if not isinstance(parameters['input_channels'], list) or len(parameters['input_channels']) == 0:
            raise ValueError("Input channels must be a non-empty list")
        if any(ch < 0 for ch in parameters['input_channels']):
            raise ValueError("Channel numbers must be non-negative")
        if parameters['sample_rate'] <= 0:
            raise ValueError("Sample rate must be positive")
            
    elif instrument_type == "Prior Stage":
        required_params = ['port']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter for Prior Stage: {param}")
        
        # Validate parameter values
        if parameters['port'] < 1 or parameters['port'] > 10:
            raise ValueError("COM port must be between 1 and 10")
    
    else:
        raise ValueError(f"Unknown instrument type: {instrument_type}")

def initialize_instrument_with_retry(instrument, max_retries=3):
    """Initialize an instrument with retry logic for connection failures"""
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