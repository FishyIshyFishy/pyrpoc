from PyQt6.QtCore import QObject
from pyrpoc.gui.signals.acq_signals import AcquisitionSignals
from pyrpoc.gui.signals.instr_signals import InstrumentSignals
from pyrpoc.gui.signals.app_state_signals import AppStateSignals
from pyrpoc.gui.signals.laser_mod_signals import LaserModulationSignals
from pyrpoc.gui.signals.ui_signals import UISignals


class SignalManager(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

        # instantiate groups
        self.ui = UISignals()
        self.acquisition = AcquisitionSignals()
        self.instrument = InstrumentSignals()
        self.app_state = AppStateSignals()
        self.laser = LaserModulationSignals()

        # connect UI events to backend signals
        self.wire_signals()

    def wire_signals(self):
        # acquisition
        self.ui.start_clicked.connect(self.acquisition.start)
        self.ui.stop_clicked.connect(self.acquisition.stop)

        
