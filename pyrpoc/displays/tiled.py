import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QSlider, QHBoxLayout, QComboBox, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from .base_display import BaseImageDisplayWidget
from typing import List

class TiledChannelsWidget(BaseImageDisplayWidget):
    def __init__(self, app_state, signals):
        super().__init__(app_state, signals)

        # self.channel_views: List[ImageView] = []

        self.build()

    def build(self):
        return 0

        # create display widget

    def handle_display_setup(self):
        return 0
    
    def handle_data_received(self):
        return 0
    
    def handle_acquisition_complete(self):
        return 0