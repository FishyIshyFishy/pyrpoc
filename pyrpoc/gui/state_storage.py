from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .signals import AppStateSignals

@dataclass
class AppState:
    config_path: str
    
    # INSTRUMENTS
    # need a way of saying which instruments are currently in the GUI and connected
    # also need a way of handling instruments upon startup
    # maybe the BaseInstrument forces an on_startup() method

    # 