"""The acquisition panel: pick a program, name it, set it up, run it.

Split into a subpackage because it is not one widget: ``panel.py`` is the
program picker, play/stop and save controls; ``mask_table.py`` and
``point_picker.py`` are the two domain-specific field widgets a program's
parameters can pull in. Those two register themselves into the generic form
engine's ``BUILDERS`` on import (see their own docstrings), so they must be
imported -- for the side effect, not the names -- before ``LauncherPanel``
builds a form that might contain a ``MasksField`` or ``PointField``.
"""

from __future__ import annotations

from . import mask_table as _mask_table  # noqa: F401 - registers build_masks
from . import point_picker as _point_picker  # noqa: F401 - registers build_point
from .panel import LauncherPanel

__all__ = ["LauncherPanel"]
