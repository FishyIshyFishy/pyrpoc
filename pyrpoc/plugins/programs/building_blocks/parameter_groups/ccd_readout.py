"""How the CCD reads a spectrum: exposure, binning and amplifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.plugins.devices import SpectrumReadout
from pyrpoc.plugins.devices.andor_ccd.device import Amplifier, ReadMode
from pyrpoc.structs.plugins.params import Group, block, choice_field, float_field, int_field

# Menu labels mapped to the device's values; the labels are what sessions save.
READ_MODES: dict[str, ReadMode] = {"Full vertical binning": "fvb", "Single track": "single_track"}
AMPLIFIERS: dict[str, Amplifier] = {"Conventional": "conventional", "EM": "em"}
READOUT_RATES: dict[str, float] = {"0.05 MHz": 0.05, "1 MHz": 1.0, "3 MHz": 3.0}
PREAMP_GAINS: dict[str, float] = {"1x": 1.0, "2x": 2.0, "4x": 4.0}


@block
@dataclass
class CcdReadoutGroup(Group):
    label: ClassVar[str] = "CCD Readout"

    exposure_s: float = float_field(
        "Exposure (s)", 0.1, minimum=0.0, step=0.01, decimals=4, tooltip="Per accumulation"
    )
    accumulations: int = int_field(
        "Accumulations",
        1,
        minimum=1,
        tooltip="Reads summed on the camera into one spectrum",
    )
    read_mode: str = choice_field(
        "Read Mode",
        "Full vertical binning",
        choices=READ_MODES,
        tooltip="Sum every row, or only a band of rows around the track centre",
    )
    track_centre: int = int_field(
        "Track Centre (row)", 100, minimum=1, tooltip="Single track only; rows count from 1"
    )
    track_height: int = int_field("Track Height (rows)", 20, minimum=1, tooltip="Single track only")
    amplifier: str = choice_field(
        "Amplifier", "Conventional", choices=AMPLIFIERS, tooltip="EM multiplies weak signals"
    )
    em_gain: int = int_field("EM Gain", 1, minimum=1, maximum=255, tooltip="EM amplifier only")
    readout_rate: str = choice_field(
        "Readout Rate",
        "0.05 MHz",
        choices=READOUT_RATES,
        tooltip="Slower reads with less noise",
    )
    preamp_gain: str = choice_field("Preamp Gain", "1x", choices=PREAMP_GAINS)

    def to_readout(self) -> SpectrumReadout:
        return SpectrumReadout(
            exposure_s=self.exposure_s,
            accumulations=self.accumulations,
            read_mode=READ_MODES[self.read_mode],
            track_centre=self.track_centre,
            track_height=self.track_height,
            amplifier=AMPLIFIERS[self.amplifier],
            em_gain=self.em_gain,
            readout_mhz=READOUT_RATES[self.readout_rate],
            preamp_gain=PREAMP_GAINS[self.preamp_gain],
        )
