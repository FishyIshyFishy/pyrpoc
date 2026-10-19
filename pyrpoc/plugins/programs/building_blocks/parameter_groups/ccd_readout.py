"""How the CCD reads a Raman spectrum: exposure and frames to average."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.plugins.devices import SpectrumReadout
from pyrpoc.plugins.devices.andor_ccd.device import Amplifier, ReadMode
from pyrpoc.structs.plugins.params import Group, block, float_field, int_field

# The rest of the readout is fixed at what a Raman spectrum wants: every row
# binned, read slowly through the conventional amplifier for the least noise.
# A track is unused under full vertical binning.
RAMAN_READ_MODE: ReadMode = "fvb"
RAMAN_AMPLIFIER: Amplifier = "conventional"
RAMAN_READOUT_MHZ = 0.05
RAMAN_PREAMP_GAIN = 1.0


@block
@dataclass
class CcdReadoutGroup(Group):
    label: ClassVar[str] = "CCD Readout"

    exposure_s: float = float_field(
        "Exposure (s)", 0.1, minimum=0.0, step=0.01, decimals=4, tooltip="Per frame"
    )
    frames_to_average: int = int_field(
        "Frames to Average",
        1,
        minimum=1,
        tooltip="Reads accumulated on the camera and averaged into one spectrum",
    )

    def to_readout(self) -> SpectrumReadout:
        return SpectrumReadout(
            exposure_s=self.exposure_s,
            accumulations=self.frames_to_average,
            read_mode=RAMAN_READ_MODE,
            track_centre=1,
            track_height=1,
            amplifier=RAMAN_AMPLIFIER,
            em_gain=1,
            readout_mhz=RAMAN_READOUT_MHZ,
            preamp_gain=RAMAN_PREAMP_GAIN,
        )
