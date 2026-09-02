"""The hardware layer's calling convention, enforced.

Replaces ``tests/devices/test_unpacking.py``, which compared the keyword sets of
four splatted config dicts against a fifteen-wide signature so that a renamed
field failed at import time rather than, in its own words, "on the instrument,
at night". The entry points take the device and the parameter group now, so a
renamed field is an attribute error a type checker sees, and that file had
nothing left to do.

What a type checker cannot check is the two rules that keep the convention:
everything is passed by name, and nothing has a default.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import pyrpoc.programs.hardware as hardware
from pyrpoc.programs.hardware.raster import raster_scan
from pyrpoc.programs.hardware.split_raster import split_raster_scan
from pyrpoc.programs.hardware.tagger import flim_scan

HARDWARE_DIR = Path(hardware.__file__).parent
MODULES = {"modulation.py", "raster.py", "split_raster.py", "tagger.py"}

ENTRY_POINTS = [raster_scan, split_raster_scan, flim_scan]


@pytest.mark.parametrize("func", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_entry_points_take_keyword_arguments_only(func):
    """Passed by name, so ``daq`` and ``galvo`` cannot be transposed silently."""
    positional = [
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert not positional, f"{func.__name__} accepts {positional} positionally"


def _functions():
    for path in sorted(HARDWARE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                yield path.name, node


def test_no_hardware_function_has_a_default_argument():
    """Hardware control gets no default behaviour.

    A default here is a value that reaches the instrument without anyone having
    chosen it. ``raster_scan`` and ``split_raster_scan`` defaulted ``ttl=None``,
    which conflated "no mask is bound" with "the caller forgot"; ``flim_scan``
    defaulted ``ai_channels=()`` purely so that splatting the whole DAQ config
    would not raise, and deleted the value unread.
    """
    offenders = []
    for filename, node in _functions():
        defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
        if defaults:
            offenders.append(f"{filename}:{node.name}")
    assert not offenders, f"default arguments in programs/hardware/: {sorted(offenders)}"


def test_the_check_looks_at_every_hardware_module():
    """Guards the guard: a glob that matched nothing would pass vacuously."""
    assert {name for name, _ in _functions()} == MODULES
