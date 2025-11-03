from PyQt6.QtCore import QObject, pyqtSignal
from pyrpoc.utils.data import BaseData


class AppStateSignals(QObject):
    load_config = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
