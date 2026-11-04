"""The program inputs programs and panels share, one topic per file, and the value types built
from them. Importing this package declares them all."""

# So a session can name any of them.

from __future__ import annotations

from .camera import CAMERA_BINNED, CAMERA_EXPOSURE_MS, CAMERA_GAIN_PCT, CameraSettings
from .ccd_readout import CCD_EXPOSURE_S, CCD_FRAMES_TO_AVERAGE
from .daq import SAMPLE_RATE_HZ
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
from .masks import MASKS, MaskList, MasksField, thumbnail
from .modulation import MASK_LINES, MaskLine, lines_for
from .mosaic import TILE_SPACING_UM, X_TILES, Y_TILES, MosaicGrid
from .pixel_pick import PICK, PickField, PixelPick
from .point import GALVO_TARGET, Point, PointField
from .roi import CENTER_REGION_FRACTION
from .scan import SCAN_GEOMETRY_INPUTS, X_PIXELS, Y_PIXELS, ScanGeometry
from .split import T0_SAMPLES, T1_SAMPLES, SplitWindows
from .trigger import FRAME_TRIGGER_PFI, PIXEL_CLOCK_CTR, PIXEL_CLOCK_PFI, TriggerLines

__all__ = [
    "CAMERA_BINNED",
    "CAMERA_EXPOSURE_MS",
    "CAMERA_GAIN_PCT",
    "CCD_EXPOSURE_S",
    "CCD_FRAMES_TO_AVERAGE",
    "CENTER_REGION_FRACTION",
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
    "SAMPLE_RATE_HZ",
    "SCAN_GEOMETRY_INPUTS",
    "T0_SAMPLES",
    "T1_SAMPLES",
    "TILE_SPACING_UM",
    "X_PIXELS",
    "X_TILES",
    "Y_PIXELS",
    "Y_TILES",
    "CameraSettings",
    "FocusSearch",
    "HistogramSettings",
    "Mask",
    "MaskLine",
    "MaskList",
    "MasksField",
    "MosaicGrid",
    "Point",
    "PickField",
    "PixelPick",
    "PointField",
    "ScanGeometry",
    "SplitWindows",
    "TriggerLines",
    "lines_for",
    "thumbnail",
]
