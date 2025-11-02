from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from pyrpoc.laser_modulations.base_laser_mod import BaseLaserModulation

class LaserModManagerWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        label = QLabel("Laser Modulation Manager Placeholder", self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def get_enabled_modulation(self) -> list[BaseLaserModulation]:
        '''
        description:
            function to get currently CONNECTED instruments from the GUI and return them to the ui_signal

            eventually params gets put into a context object which is sent with params/laser_mods 
            to acq_signals
        '''
        enabled_mods = []
        for mod in self.modulations:
            if mod.enabled: # read from GUI, not from modulation object, because it won't have a parameter for that
                enabled_mods.append(mod)
        return enabled_mods