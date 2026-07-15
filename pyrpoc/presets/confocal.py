from __future__ import annotations

from typing import Any

from pyrpoc.acquisition.setup import Setup
from pyrpoc.acquisition.source import CommandSource, FiniteScanSource
from pyrpoc.optocontrols.mask import MaskOptoControl
from pyrpoc.presets.base import Preset, preset_registry
from pyrpoc.structs.acquired_data import DataKind
from pyrpoc.structs.commands import RasterScanCommand
from pyrpoc.structs.config.confocal import ConfocalParameters, parameter_groups


@preset_registry.register("confocal")
class ConfocalPreset(Preset):
    key = "confocal"
    display_name = "Confocal"
    parameter_groups = parameter_groups
    required_instruments = []
    allowed_optocontrols = [MaskOptoControl]
    allowed_displays = ["streamed_image", "tiled_2d", "multichan_overlay"]
    emitted_kinds = [DataKind.INTENSITY_FRAME]

    def build_source_and_setup(
        self,
        *,
        params: dict[str, Any],
        instruments: dict[type, Any],
        frame_limit: int | None,
    ) -> tuple[CommandSource, Setup]:
        p = ConfocalParameters.from_dict(params)
        labels = [f"ai{ch}" for ch in p.active_ai_channels]

        def factory(index: int) -> RasterScanCommand:
            return RasterScanCommand(
                expected_kinds=[DataKind.INTENSITY_FRAME],
                frame_index=index,
                device_name=p.device_name,
                sample_rate_hz=p.sample_rate_hz,
                fast_axis_ao=p.fast_axis_ao,
                slow_axis_ao=p.slow_axis_ao,
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                dwell_time_us=p.dwell_time_us,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
                active_ai_channels=tuple(p.active_ai_channels),
                channel_labels=list(labels),
            )

        return FiniteScanSource(factory, frame_limit), Setup()
