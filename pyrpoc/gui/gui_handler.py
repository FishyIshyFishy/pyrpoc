from PyQt6.QtCore import QObject, pyqtSignal

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

    data_updated = pyqtSignal(object) # emits when data acquisition is complete
    
    def bind_controllers(self, app_state):
        self.load_config_btn_clicked.connect(lambda: handle_load_config(app_state))
        self.save_config_btn_clicked.connect(lambda: handle_save_config(app_state))

        self.continuous_btn_clicked.connect(lambda: handle_continuous_acquisition(app_state))
        self.single_btn_clicked.connect(lambda: handle_single_acquisition(app_state))
        self.stop_btn_clicked.connect(lambda: handle_stop_acquisition(app_state))

        self.modality_dropdown_changed.connect(lambda text: handle_modality_changed(text, app_state))

        self.data_updated.connect(lambda data: handle_data_updated(data, app_state))

def handle_load_config(app_state):
    return 0

def handle_save_config(app_state):
    return 0

def handle_continuous_acquisition(app_state):
    return 0

def handle_single_acquisition(app_state):
    return 0

def handle_stop_acquisition(app_state):
    return 0

def handle_modality_changed(text, app_state):
    return 0

def handle_data_updated(data, app_state):
    return 0