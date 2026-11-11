"""Programs that acquire from the simulated dish, for working without instruments. Only
``pyrpoc --simulated`` imports this package, and importing it registers them."""

from __future__ import annotations

from .confocal.program import SimulatedConfocal
from .galvo_calibration.program import SimulatedGalvoCalibration
from .longitudinal_monitor.program import SimulatedLongitudinalMonitor
from .survey.program import SimulatedSurvey

__all__ = [
    "SimulatedConfocal",
    "SimulatedGalvoCalibration",
    "SimulatedLongitudinalMonitor",
    "SimulatedSurvey",
]
