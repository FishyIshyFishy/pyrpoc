"""The CCD readout: exposure and averaging, and ``spectrum_readout``, which
turns them into the camera's full readout."""

from __future__ import annotations

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import ProgramInputValues, float_input, int_input
from pyrpoc.plugins.devices import SpectrumReadout
from pyrpoc.plugins.devices.andor_ccd.device import Amplifier, ReadMode

# The rest of the readout is fixed at what a Raman spectrum wants: every row
# binned, read slowly through the conventional amplifier for the least noise.
# A track is unused under full vertical binning.
RAMAN_READ_MODE: ReadMode = "fvb"
RAMAN_AMPLIFIER: Amplifier = "conventional"
RAMAN_READOUT_MHZ = 0.05
RAMAN_PREAMP_GAIN = 1.0

EXPOSURE_S = float_input(
    "ccd_readout.exposure_s",
    "Exposure (s)",
    0.1,
    minimum=0.0,
    step=0.01,
    decimals=4,
    tooltip="Per frame",
)
FRAMES_TO_AVERAGE = int_input(
    "ccd_readout.frames_to_average",
    "Frames to Average",
    1,
    minimum=1,
    tooltip="Reads accumulated on the camera and averaged into one spectrum",
)

CCD_READOUT_SECTION = InputSection("CCD Readout", (EXPOSURE_S, FRAMES_TO_AVERAGE))


def spectrum_readout(values: ProgramInputValues) -> SpectrumReadout:
    """The readout a Raman spectrum is taken with."""
    return SpectrumReadout(
        exposure_s=values[EXPOSURE_S],
        accumulations=values[FRAMES_TO_AVERAGE],
        read_mode=RAMAN_READ_MODE,
        track_centre=1,
        track_height=1,
        amplifier=RAMAN_AMPLIFIER,
        em_gain=1,
        readout_mhz=RAMAN_READOUT_MHZ,
        preamp_gain=RAMAN_PREAMP_GAIN,
    )
