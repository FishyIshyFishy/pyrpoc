from PyQt6.QtCore import QObject, pyqtSignal
from typing import Any
from dataclasses import dataclass

from pyrpoc.utils.datas import BaseData
from pyrpoc.utils.parameters import BaseParameter
from pyrpoc.utils.contexts import AcquisitionContext

from pyrpoc.utils.base_types.base_display import BaseDisplay
from pyrpoc.gui.signals.acq_signals import AcquisitionSignals
from pyrpoc.gui.signals.instr_signals import InstrumentSignals
from pyrpoc.gui.signals.app_state_signals import AppStateSignals
from pyrpoc.gui.signals.laser_mod_signals import LaserModulationSignals
from pyrpoc.gui.signals.ui_signals import UISignals


class UISignals(QObject):
    '''
    interface between GUI and backend
    all functions implemented here will solely read things from the GUI to the backend
    there should be no functional code here - that must be from other signal types.
    '''
    style_selected = pyqtSignal(str) # theme name

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
        

    def prepare_acquisition_context(self) -> AcquisitionContext:
        '''
        read everything from GUI
        
        need to implement readers on frontend
        '''
        temp = AcquisitionContext(
            params=[
                BaseParameter(name='placeholder', value=0)
            ],
            
            instruments = [],
            mods = [],
            display_type=BaseDisplay(),
            save=False
            )
        
        return temp
        
    def handle_stop_clicked(self):
        pass