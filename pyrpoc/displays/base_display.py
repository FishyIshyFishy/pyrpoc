import numpy as np
from PyQt6.QtWidgets import QWidget

class BaseImageDisplayWidget(QWidget):   
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals

    def build(self):
        raise NotImplementedError('Subdisplay class MUST implement build()')
    
    def handle_display_setup(self):
        raise NotImplementedError('Subdisplay class MUST implement handle_display_setup()')
    
    def handle_acquisition_started(self):
        raise NotImplementedError('Subdisplay class MUST implement handle_acquisition_started()')
    
    def handle_data_received(self, data):
        raise NotImplementedError('Subdisplay class MUST implement handle_data_received()')
        
    def handle_acquisition_complete(self):
        raise NotImplementedError('Subdisplay class MUST implement handle_acquisition_complete()')
    
    def get_image_data_for_rpoc(self):
        raise NotImplementedError('Subdisplay class MUST implement get_image_data_for_rpoc()')
    
