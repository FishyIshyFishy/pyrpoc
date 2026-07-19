"""Typed registry identities for every plugin kind.

Plugins are referenced by these enums instead of bare strings. The enum *values*
are stable strings used only when a routine or session is serialized.
"""

from __future__ import annotations

from enum import Enum


class ModalityKey(Enum):
    """Registry identity for an acquisition modality."""

    confocal = "confocal"
    split_confocal = "split_confocal"
    flim = "flim"
    simulated = "simulated"
    simulated_flim = "simulated_flim"


class ModifierKey(Enum):
    """Registry identity for an acquisition modifier."""

    mask = "mask"


class DisplayKey(Enum):
    """Registry identity for a display."""

    streamed = "streamed"
    tiled = "tiled"
    multichannel = "multichannel"
    flim = "flim"


class InstrumentKey(Enum):
    """Registry identity for an instrument."""

    ni_daq = "ni_daq"
    time_tagger = "time_tagger"
    prior_stage = "prior_stage"
    zaber_stage = "zaber_stage"
    simulated_daq = "simulated_daq"
    simulated_tagger = "simulated_tagger"
