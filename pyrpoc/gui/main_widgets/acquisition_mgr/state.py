from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.parameter_utils import BaseParameter


@dataclass
class AcquisitionManagerState:
    param_widgets: dict[str, tuple[BaseParameter, QWidget]] = field(default_factory=dict)
    param_defs: dict[str, BaseParameter] = field(default_factory=dict)
