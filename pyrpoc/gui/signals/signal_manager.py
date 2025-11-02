from PyQt6.QtCore import QObject
from pyrpoc.gui.signals.acq_signals import AcquisitionSignals
from pyrpoc.gui.signals.instr_signals import InstrumentSignals
from pyrpoc.gui.signals.app_state_signals import AppStateSignals
from pyrpoc.gui.signals.laser_mod_signals import LaserModulationSignals
from pyrpoc.gui.signals.ui_signals import UISignals


class SignalManager(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

        # low level
        self.acq_signals = AcquisitionSignals()
        self.instr_signals = InstrumentSignals()
        self.app_state_signals = AppStateSignals()
        self.laser_mod_signals = LaserModulationSignals()

        # high level
        self.ui_signals = UISignals(acq=self.acq_signals, instr=self.instr_signals, app_state=self.app_state_signals, mod=self.laser_mod_signals)

