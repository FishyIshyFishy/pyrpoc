"""The ROI spectral series without a rig: the sweep it loops, and what it
records of that sweep."""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.plugins.programs.building_blocks.functions.roi_scan import roi_waveform, row_spans
from pyrpoc.plugins.programs.building_blocks.parameter_groups import MaskRef, RoiGroup, ScanGroup
from pyrpoc.plugins.programs.roi_spectral_series.program import RoiSpectralSeries, plan_sweep
from pyrpoc.structs.plugins.programs.program import program_registry


def test_it_is_listed_under_custom() -> None:
    assert program_registry.entries["roi_spectral_series"] is RoiSpectralSeries
    assert RoiSpectralSeries.group == "Custom"


def test_the_sweep_is_the_region_scan_and_records_its_period() -> None:
    scan = ScanGroup(x_pixels=12, y_pixels=10, extra_left=2, extra_right=1, dwell_time_us=2.5)
    region = np.zeros((10, 12), dtype=np.uint8)
    region[3, 5:7] = 255
    region[4, 4:9] = 255
    roi = RoiGroup(mask=MaskRef(source_id="m", source_label="cell", array=region))

    waveform, sweep = plan_sweep(scan, roi, 1_000_000.0)

    # 2.5 us at 1 MHz truncates to 2 samples per pixel.
    spans = row_spans(region > 0)
    np.testing.assert_array_equal(waveform, roi_waveform(scan, spans, 2))
    swept = (2 + 2 + 1) + (5 + 2 + 1)
    assert waveform.shape == (2, swept * 2)
    assert sweep["source"] == "cell"
    assert (sweep["top"], sweep["left"], sweep["height"], sweep["width"]) == (3, 4, 2, 5)
    assert sweep["pixels_swept"] == swept
    assert sweep["dwell_s"] == pytest.approx(2e-6)
    assert sweep["period_s"] == pytest.approx(swept * 2e-6)
