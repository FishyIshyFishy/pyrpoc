"""One file per experiment, plus the components they are assembled from.

The thing changed most often, so it sits at the bottom with everything else
ignorant of it: nothing imports programs/ except shell/ (to launch them) and the
registry (to collect them). Any program can be deleted outright -- its file and
its row in shell/catalog.py.

A program is a composition. It declares the parameter blocks it wants from
``components/`` and writes the loop that drives them; the scan code it needs to
touch hardware lives in its own file, because that code is what the modality
*is*. Two modalities running similar scans therefore hold near-identical copies of
the waveform arithmetic. Those copies are meant to stay identical: change one
and change the others, or say in the docstring why they now differ.
"""

from .registry import program_registry
from .confocal import Confocal
from .split_confocal import SplitConfocal
from .flim import FLIM
from .simulation import Simulation

__all__ = [
    "program_registry",
    "Confocal",
    "SplitConfocal",
    "FLIM",
    "Simulation",
]
