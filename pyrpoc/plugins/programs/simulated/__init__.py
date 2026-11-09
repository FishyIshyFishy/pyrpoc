"""Programs that acquire from the simulated dish or fake their data, for working without
instruments, with the program inputs and fake detector only they use. Only
``pyrpoc --simulated`` imports this package."""

from __future__ import annotations

from pyrpoc.core.program import program_registry

from .confocal.program import SimulatedConfocal
from .galvo_calibration.program import SimulatedGalvoCalibration
from .longitudinal_monitor.program import SimulatedLongitudinalMonitor
from .mosaic_simulation.program import MosaicSimulation
from .simulation.program import Simulation
from .survey.program import SimulatedSurvey


def register() -> None:
    """Add the simulated programs to the program list."""
    # Registered here rather than where each is defined, so they exist only
    # when this is called.
    program_registry.add("simulation", Simulation)
    program_registry.add("mosaic_simulation", MosaicSimulation)
    program_registry.add("simulated_confocal", SimulatedConfocal)
    program_registry.add("simulated_galvo_calibration", SimulatedGalvoCalibration)
    program_registry.add("simulated_survey", SimulatedSurvey)
    program_registry.add("simulated_longitudinal_monitor", SimulatedLongitudinalMonitor)
