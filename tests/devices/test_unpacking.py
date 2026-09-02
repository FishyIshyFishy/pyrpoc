"""The groups must unpack into the hardware functions exactly.

Programs call ``raster_scan(**p.scan, **p.daq, **daq.config, **galvo.config)``.
If a field is renamed on either side, that call breaks at runtime with a
TypeError on the first frame -- on the instrument, at night. This checks the
keyword sets line up, which is cheap and catches it at import time instead.
"""

from __future__ import annotations

import inspect

import pytest

from pyrpoc.core.params import DaqGroup, FlimDaqGroup, ScanGroup, TriggerGroup
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.programs.hardware.raster import raster_scan
from pyrpoc.programs.hardware.split_raster import split_raster_scan
from pyrpoc.programs.hardware.tagger import flim_scan
from pyrpoc.core.params import SplitGroup


def keywords(func) -> set[str]:
    return {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }


def supplied(*groups) -> set[str]:
    out: set[str] = set()
    for group in groups:
        out |= set(group.keys())
    return out


def test_raster_scan_signature_is_exactly_what_the_groups_supply():
    needed = keywords(raster_scan) - {"ttl"}
    have = supplied(ScanGroup(), DaqGroup(), DAQ().config, Galvo().config)
    assert needed == have


def test_split_raster_scan_signature_matches_its_groups():
    needed = keywords(split_raster_scan) - {"ttl"}
    have = supplied(ScanGroup(), DaqGroup(), DAQ().config, Galvo().config, SplitGroup())
    assert needed == have


def test_flim_scan_signature_matches_its_groups():
    needed = keywords(flim_scan)
    have = supplied(ScanGroup(), FlimDaqGroup(), DAQ().config, Galvo().config, TriggerGroup())
    assert needed == have


@pytest.mark.parametrize("func", [raster_scan, split_raster_scan, flim_scan])
def test_hardware_functions_take_keyword_arguments_only(func):
    """Positional arguments across four unpacked groups would be unreadable."""
    positional = {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }
    assert not positional
