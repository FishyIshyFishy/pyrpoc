from PyQt6.QtCore import QObject, pyqtSignal
from typing import Any

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.modalities import BaseModality
from pyrpoc.instruments import BaseInstrument
from pyrpoc.laser_modulations import BaseLaserModulation

class AcquisitionSignals(QObject):
    start = pyqtSignal(dict[str, Any]) 
    stop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.start.connect(self.handle_start_acquisition)
        self.stop.connect(self.handle_stop_acquisition)

    def handle_start_acquisition(self, modality: BaseModality, context: dict[str, Any]):
        modality.start_acquisition(context)

    def handle_stop_acquisition(self, modality: BaseModality):
        modality.stop_acquisition()