from PyQt6.QtCore import QObject, pyqtSignal
from pyrpoc.utils.data import BaseData

class InstrumentSignals(QObject):
    add_instrument = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
