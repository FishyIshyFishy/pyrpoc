from PyQt6.QtCore import QObject, pyqtSignal
from typing import Any
from dataclasses import dataclass

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.gui.signals.acq_signals import AcquisitionSignals
from pyrpoc.gui.signals.instr_signals import InstrumentSignals
from pyrpoc.gui.signals.app_state_signals import AppStateSignals
from pyrpoc.gui.signals.laser_mod_signals import LaserModulationSignals
from pyrpoc.gui.signals.ui_signals import UISignals

class UISignalMediator:
    acq: AcquisitionSignals
    instr: InstrumentSignals
    app_state: AppStateSignals
    mod: LaserModulationSignals

class UISignals(QObject):
    style_selected = pyqtSignal(str) # theme name

    # from acquisition manager
    modality_changed = pyqtSignal()
    start_clicked = pyqtSignal()
    continuous_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self, acq: AcquisitionSignals, instr: InstrumentSignals, app_state: AppStateSignals, mod: LaserModulationSignals, parent=None):
        super().__init__(parent)
        self.acq = acq
        self.instr = instr
        self.app_state = app_state
        self.mod = mod

    def handle_modality_changed(self):
        pass

    def handle_start_clicked(self):
        context = self.prepare_acquisition_context()
        self.acq.start.emit(context)
        
    def handle_continuous_clicked(self):
        pass
        

    def prepare_acquisition_context(self) -> dict[str, Any]:
        return {'params': 1, 'instruments': 2}
        
    def handle_stop_clicked(self):
        pass