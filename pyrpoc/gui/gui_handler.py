from PyQt6.QtCore import QObject, pyqtSignal
from pyrpoc.imaging.acquisitions import *

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
    load_config_btn_clicked = pyqtSignal()
    save_config_btn_clicked = pyqtSignal()

    continuous_btn_clicked = pyqtSignal()
    single_btn_clicked = pyqtSignal()
    stop_btn_clicked = pyqtSignal()

    modality_dropdown_changed = pyqtSignal(str) # emits when the dropdown for the modality is changed
    add_instrument_btn_clicked = pyqtSignal()

    data_updated = pyqtSignal(object) # emits when data acquisition is complete
    
    def bind_controllers(self, app_state):
        self.load_config_btn_clicked.connect(lambda: handle_load_config(app_state))
        self.save_config_btn_clicked.connect(lambda: handle_save_config(app_state))

        self.continuous_btn_clicked.connect(lambda: handle_continuous_acquisition(app_state))
        self.single_btn_clicked.connect(lambda: handle_single_acquisition(app_state))
        self.stop_btn_clicked.connect(lambda: handle_stop_acquisition(app_state))

        self.modality_dropdown_changed.connect(lambda text: handle_modality_changed(text, app_state))
        self.add_instrument_btn_clicked.connect(lambda: handle_add_instrument(app_state))

        self.data_updated.connect(lambda data: handle_data_updated(data, app_state))

def handle_load_config(app_state):
    return 0

def handle_save_config(app_state):
    return 0

def handle_continuous_acquisition(app_state):
    return 0

def handle_single_acquisition(app_state):
    modality = app_state.modality
    match modality:
        case 'widefield':
            acquisition = Widefield()
        case 'confocal':
            acquisition = Confocal()
        case 'mosaic':
            acquisition = Mosaic()
        case 'zscan':
            acquisition = ZScan()
        case 'simulated':
            acquisition = Simulated()
        case 'custom':
            acquisition = Custom()
        case _:
            print('Warning: invalid modality, defaulting to simulation')
            acquisition = Simulated()

    acquisition.verify_acquisition()
    acquisition.configure_rpoc()
    acquisition.configure_imaging()
    acquisition.perform_acquisition()
    StateSignalBus.data_updated.emit(acquisition.finalize_acquisition()) # emits the data
    pass

def handle_stop_acquisition(app_state):
    return 0

def handle_modality_changed(text, app_state):
    app_state.modality = text
    return 0

def handle_data_updated(data, app_state):
    return 0

def handle_add_instrument(app_state):
    return 0