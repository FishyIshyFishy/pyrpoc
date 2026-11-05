"""The program inputs programs and panels share, one topic per file, and the value types built
from them. Importing this package declares them all."""

# So a session can name any of them.

from __future__ import annotations

from .camera import CAMERA_BINNED, CAMERA_EXPOSURE_MS, CAMERA_GAIN_PCT, CameraSettings
from .ccd_readout import CCD_EXPOSURE_S, CCD_FRAMES_TO_AVERAGE
from .daq import SAMPLE_RATE_MHZ, rate_hz
from .focus_search import (
    FOCUS_FINAL_STEP_UM,
    FOCUS_INITIAL_STEP_UM,
    FOCUS_SEARCH_RANGE_UM,
    FocusSearch,
)
from .frame_count import NUM_FRAMES
from .histogram import (
    FRAME_SETTLE_S,
    HISTOGRAM_BINS,
    HISTOGRAM_BINWIDTH_PS,
    LASER_FREQUENCY_MHZ,
    HistogramSettings,
)
from .mask import Mask
from .masks import MASKS, MasksField, thumbnail
from .modulation import MASK_LINES, MaskLine, lines_for
from .pixel_pick import PICK, PickField, PixelPick
from .point import GALVO_TARGET, Point, PointField
from .roi import CENTER_REGION_FRACTION
from .scan import (
    DWELL_TIME_US,
    EXTRA_LEFT,
    EXTRA_RIGHT,
    SCAN_GEOMETRY_INPUTS,
    X_PIXELS,
    Y_PIXELS,
    ScanGeometry,
)
from .split import T0_SAMPLES, T1_SAMPLES, SplitWindows
from .survey import FIELD_PIXELS, FIELD_UM, SURVEY_HEIGHT_UM, SURVEY_WIDTH_UM, TILE_OVERLAP
from .trigger import FRAME_TRIGGER_PFI, PIXEL_CLOCK_CTR, PIXEL_CLOCK_PFI, TriggerLines

__all__ = [
    "CAMERA_BINNED",
    "CAMERA_EXPOSURE_MS",
    "CAMERA_GAIN_PCT",
    "CCD_EXPOSURE_S",
    "CCD_FRAMES_TO_AVERAGE",
    "CENTER_REGION_FRACTION",
    "DWELL_TIME_US",
    "EXTRA_LEFT",
    "EXTRA_RIGHT",
    "FIELD_PIXELS",
    "FIELD_UM",
    "FOCUS_FINAL_STEP_UM",
    "FOCUS_INITIAL_STEP_UM",
    "FOCUS_SEARCH_RANGE_UM",
    "FRAME_SETTLE_S",
    "FRAME_TRIGGER_PFI",
    "GALVO_TARGET",
    "HISTOGRAM_BINS",
    "HISTOGRAM_BINWIDTH_PS",
    "LASER_FREQUENCY_MHZ",
    "MASKS",
    "MASK_LINES",
    "NUM_FRAMES",
    "PICK",
    "PIXEL_CLOCK_CTR",
    "PIXEL_CLOCK_PFI",
    "SAMPLE_RATE_MHZ",
    "SCAN_GEOMETRY_INPUTS",
    "SURVEY_HEIGHT_UM",
    "SURVEY_WIDTH_UM",
    "T0_SAMPLES",
    "TILE_OVERLAP",
    "T1_SAMPLES",
    "X_PIXELS",
    "Y_PIXELS",
    "CameraSettings",
    "FocusSearch",
    "HistogramSettings",
    "Mask",
    "MaskLine",
    "MasksField",
    "Point",
    "PickField",
    "PixelPick",
    "PointField",
    "ScanGeometry",
    "SplitWindows",
    "TriggerLines",
    "lines_for",
    "rate_hz",
    "thumbnail",
]
