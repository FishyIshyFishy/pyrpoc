"""One file per experiment, plus the modules those files share.

The thing changed most often, so it sits at the bottom with everything else
ignorant of it: nothing imports programs/ except shell/ (to launch them) and the
registry (to collect them). Any program can be deleted outright -- its file and
its row in shell/catalog.py.

Two modules here are not programs. ``hardware/`` holds the NI task setup and
scan arithmetic that three programs share; ``synthetic.py`` holds the frame
generators that ``simulation.py`` alone uses. Both were a top-level
``operations/`` folder, which read as a second hardware layer beside
``devices/`` and invited the question of which one controlled the instrument.
Neither is a layer. They are shared code, filed with the only thing that calls
them.
"""

from .registry import program_registry
from .confocal import Confocal, ConfocalParams
from .split_confocal import SplitConfocal, SplitConfocalParams
from .flim import FLIM, FlimParams
from .simulation import Simulation, SimulationParams

__all__ = [
    "program_registry",
    "Confocal",
    "ConfocalParams",
    "SplitConfocal",
    "SplitConfocalParams",
    "FLIM",
    "FlimParams",
    "Simulation",
    "SimulationParams",
]
