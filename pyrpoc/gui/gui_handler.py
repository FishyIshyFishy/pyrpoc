from PyQt6.QtCore import QObject, pyqtSignal
from pyrpoc.imaging.acquisitions import *

class AppState:
    '''
    APP STATE CLASS
    holds all of the important signaled variables
    '''
    def __init__(self):
        self.modality = 'widefield' 
        self.instruments = { # list of possible instruments, will add as needed
            'galvo': None,
            'daq_inputs': [],
            'tcspc_inputs': []
            } 
        self.acquisition_parameters = {
            'num_frames': None,
            'x_pixels': 512,
            'y_pixels': 512,
        }
        self.display_parameters = {
            'overlay': True,
        }
            

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
    # TODO: figure out how i need to handle copying of stuff, use deepcopy?
    # TODO: make the prints just statusbar updates
    modality = app_state.modality
    instruments = app_state.instruments
    parameters = app_state.parameters

    # for each case, the structure is:
    # try: copy the variables from the current app_state
    # try: create the acquisition class instance with those variables
    match modality:
        case 'simulated':
            try: 
                x_pixels = parameters.x_pixels
                y_pixels = parameters.y_pixels
            except Exception as e:
                print(f'Error getting parameter values: {e}')    
            
            try:
                acquisition = Simulated()
            except Exception as e:
                print(f'Error in instantiating Simulated(): {e}')


        case 'widefield':
            try:
                x_pixels = parameters.x_pixels
                y_pixels = parameters.y_pixels
                daq_inputs = instruments['daq_inputs']
                tcspc_inputs = instruments['tcspc_inputs']
            except Exception as e:
                print(f'Error getting parameter values: {e}')
                
            acquisition = Widefield()
        case 'confocal':
            acquisition = Confocal()
        case 'mosaic':
            acquisition = Mosaic()
        case 'zscan':
            acquisition = ZScan()
        case 'custom':
            acquisition = Custom()
        case _:
            print('Warning: invalid modality, defaulting to simulation')
            acquisition = Simulated()

    acquisition.configure_acquisition()
    acquisition.configure_rpoc()
    acquisition.perform_acquisition()
    StateSignalBus.data_updated.emit(acquisition.finalize_acquisition()) # emits the data
    pass

def handle_stop_acquisition(app_state):
    # this needs to stop all processes except the main thread i guess
    return 0

def handle_modality_changed(text, app_state):
    app_state.modality = text
    return 0

def handle_data_updated(data, app_state):
    # show new data for display
    return 0

def handle_add_instrument(app_state):
    # open the add instrument popout in a side thread and record the necessary params
    return 0