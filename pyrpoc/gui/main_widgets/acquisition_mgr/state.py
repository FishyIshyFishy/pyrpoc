from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.contracts import Parameter


@dataclass
class AcquisitionManagerState:
    param_widgets: dict[str, QWidget] = field(default_factory=dict)
    param_defs: dict[str, Parameter] = field(default_factory=dict)
    continuous_timer: QTimer | None = None
