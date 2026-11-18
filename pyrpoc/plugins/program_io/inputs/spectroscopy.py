"""The program inputs of spectroscopy: the CCD readout, FLIM's decay histograms, split
confocal's windows, and hyperspectral's delay sweep."""

from __future__ import annotations

from pyrpoc.utils.value_rules import float_input, int_input

CCD_EXPOSURE_S = float_input(
    "ccd_exposure_s",
    "Exposure (s)",
    0.1,
    minimum=0.0,
    step=0.01,
    decimals=4,
    tooltip="Per frame",
)
CCD_FRAMES_TO_AVERAGE = int_input(
    "ccd_frames_to_average",
    "Frames to Average",
    1,
    minimum=1,
    tooltip="Reads accumulated on the camera and averaged into one spectrum",
)

LASER_FREQUENCY_MHZ = float_input(
    "laser_frequency_mhz",
    "Laser Frequency MHz",
    80.0,
    minimum=0.001,
    tooltip="Laser repetition rate in MHz",
)
HISTOGRAM_BINS = int_input(
    "histogram_bins",
    "Histogram Bins",
    125,
    minimum=2,
    tooltip="Number of decay-histogram bins per pixel",
)
HISTOGRAM_BINWIDTH_PS = int_input(
    "histogram_binwidth_ps",
    "Histogram Bin Width (ps)",
    100,
    minimum=1,
    tooltip="Bin width in ps (bins x width should span one laser period)",
)
FRAME_SETTLE_S = float_input(
    "frame_settle_s",
    "Frame Settle (s)",
    5e-3,
    minimum=0.0,
    step=1e-3,
    tooltip="Wait after the scan so the last photons reach the measurement",
)

T0_SAMPLES = int_input(
    "t0_samples",
    "t0 Samples",
    1,
    minimum=1,
    tooltip="Number of samples in the first subpixel window",
)
T1_SAMPLES = int_input(
    "t1_samples",
    "t1 Samples",
    0,
    minimum=0,
    tooltip="Number of samples to discard between t0 and t2",
)

DELAY_START_UM = float_input(
    "delay_start_um",
    "Start (um)",
    20000.0,
    step=100.0,
    decimals=3,
    tooltip="Delay stage position of the first image",
)
DELAY_STOP_UM = float_input(
    "delay_stop_um",
    "Stop (um)",
    30000.0,
    step=100.0,
    decimals=3,
    tooltip="Delay stage position of the last image; below Start, the sweep runs backwards",
)
DELAY_STEPS = int_input(
    "delay_steps",
    "Steps",
    10,
    minimum=1,
    tooltip="Images in the sweep, evenly spaced from Start to Stop, both included",
)
