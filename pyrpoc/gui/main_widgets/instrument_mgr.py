from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from pyrpoc.instruments import BaseInstrument

class InstrumentManagerWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        label = QLabel("Instrument Manager Placeholder", self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def get_connected_instruments(self) -> list[BaseInstrument]:
        '''
        description:
            function to get currently CONNECTED instruments from the GUI and return them to the ui_signal

            eventually params gets put into a context object which is sent with params/laser_mods 
            to acq_signals
        '''
        connected_instruments = []
        for instr in self.instruments:
            if instr.connected:
                connected_instruments.append(instr)
        return instr