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
    modality_changed = pyqtSignal(str) # emits when the dropdown for the modality is changed
    data_updated = pyqtSignal(object) # emits when data acquisition is complete

    def __init__():
        super().__init__()
    
    def bind_controllers(self, app_state):
        self.modality_changed.connect(lambda text: handle_modality_changed(text, app_state))
        self.data_updated.connect(lambda data: handle_data_updated(data, app_state))


def handle_modality_changed(text, app_state):
    return 0

def handle_data_updated(data, app_state):
    return 0