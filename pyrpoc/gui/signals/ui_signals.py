from PyQt6.QtCore import QObject, pyqtSignal
from pyrpoc.backend_utils.data import BaseData

class UISignals(QObject):
    style_selected = pyqtSignal(str) # theme name

    # from acquisition manager
    modality_changed = pyqtSignal()
    start_clicked = pyqtSignal()
    continuous_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    browse_save = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)