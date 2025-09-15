from PyQt6.QtCore import QObject, pyqtSignal
from pyrpoc.backend_utils.data import BaseData

class AcquisitionSignals(QObject):
    start = pyqtSignal()
    continuous = pyqtSignal()
    stop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.start.connect(self.handle_start_acquisition)
        self.continuous.connect(self.handle_continuous_acquisition)
        self.stop.connect(self.handle_stop_acquisition)

    def handle_start_acquisition(self):
        pass

    def handle_continuous_acquisition(self):
        pass

    def handle_stop_acquisition(self):
        pass