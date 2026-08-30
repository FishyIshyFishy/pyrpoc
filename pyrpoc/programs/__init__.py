"""One file per experiment.

The thing changed most often, so it sits at the bottom with everything else
ignorant of it: nothing imports programs/ except shell/ (to launch them) and the
registry (to collect them). Any program can be deleted outright -- its file and
its row in shell/catalog.py.
"""

from .registry import program_registry
from .confocal import Confocal, ConfocalParams

__all__ = ["program_registry", "Confocal", "ConfocalParams"]
