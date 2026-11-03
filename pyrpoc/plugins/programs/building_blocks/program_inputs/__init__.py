"""Program inputs a program can declare, one topic per file, each with its input section
and any value type built from them. Importing this package declares them all."""

# So a session can name any of them.

from __future__ import annotations

from .camera import CAMERA_SECTION, CameraSettings
from .ccd_readout import CCD_READOUT_SECTION, spectrum_readout
from .daq import DAQ_SECTION, SAMPLE_RATE_HZ
from .focus_search import FOCUS_SEARCH_SECTION, FocusSearch
from .frame_count import FRAME_COUNT_SECTION, NUM_FRAMES
from .histogram import HISTOGRAM_SECTION, HistogramSettings
from .modulation import MASKS, MODULATION_SECTION, MaskBinding, MasksField
from .mosaic import MOSAIC_SECTION, MosaicGrid
from .point import POINT_SECTION, TARGET, Point, PointField
from .roi import REGION_SECTION, ScanRegion
from .scan import SCAN_SECTION, ScanGeometry
from .split import SPLIT_SECTION, SplitWindows
from .trigger import TRIGGER_SECTION, TriggerLines

__all__ = [
    "CAMERA_SECTION",
    "CCD_READOUT_SECTION",
    "DAQ_SECTION",
    "FOCUS_SEARCH_SECTION",
    "FRAME_COUNT_SECTION",
    "HISTOGRAM_SECTION",
    "MASKS",
    "MODULATION_SECTION",
    "MOSAIC_SECTION",
    "NUM_FRAMES",
    "POINT_SECTION",
    "REGION_SECTION",
    "SAMPLE_RATE_HZ",
    "SCAN_SECTION",
    "SPLIT_SECTION",
    "TARGET",
    "TRIGGER_SECTION",
    "CameraSettings",
    "FocusSearch",
    "HistogramSettings",
    "MaskBinding",
    "MasksField",
    "MosaicGrid",
    "Point",
    "PointField",
    "ScanGeometry",
    "ScanRegion",
    "SplitWindows",
    "TriggerLines",
    "spectrum_readout",
]
