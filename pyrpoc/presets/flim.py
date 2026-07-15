from __future__ import annotations

from typing import Any

from pyrpoc.acquisition.setup import FlimRunState, FlimSetup, Setup
from pyrpoc.acquisition.source import CommandSource, FiniteScanSource
from pyrpoc.instruments.time_tagger import TimeTaggerInstrument
from pyrpoc.optocontrols.mask import MaskOptoControl
from pyrpoc.presets.base import Preset, preset_registry
from pyrpoc.structs.acquired_data import DataKind
from pyrpoc.structs.commands import FlimScanCommand
from pyrpoc.structs.config.flim import FlimParameters, parameter_groups


@preset_registry.register("flim")
class FlimPreset(Preset):
    key = "flim"
    display_name = "FLIM"
    parameter_groups = parameter_groups
    required_instruments = [TimeTaggerInstrument]
    allowed_optocontrols = [MaskOptoControl]
    allowed_displays = ["streamed_image", "flim_display", "tiled_2d", "multichan_overlay"]
    emitted_kinds = [DataKind.INTENSITY_FRAME, DataKind.FLIM_RAW_FRAME]

    def build_source_and_setup(
        self,
        *,
        params: dict[str, Any],
        instruments: dict[type, Any],
        frame_limit: int | None,
    ) -> tuple[CommandSource, Setup]:
        p = FlimParameters.from_dict(params)
        tagger = instruments.get(TimeTaggerInstrument)
        if tagger is None:
            raise RuntimeError("TimeTaggerInstrument is required for the FLIM preset")

        total_x = p.x_pixels + p.extra_left + p.extra_right
        run_state = FlimRunState()
        setup = FlimSetup(
            run_state=run_state,
            tagger_instrument=tagger,
            config={
                "laser_channel": p.laser_channel,
                "detector_channel": p.detector_channel,
                "pixel_channel": p.pixel_channel,
                "frame_channel": p.frame_channel,
                "laser_trigger_v": p.laser_trigger_v,
                "detector_trigger_v": p.detector_trigger_v,
                "pixel_trigger_v": p.pixel_trigger_v,
                "frame_trigger_v": p.frame_trigger_v,
                "laser_input_delay_ps": p.laser_input_delay_ps,
                "n_pixels": total_x * p.y_pixels,
                "n_bins": p.histogram_bins,
                "binwidth_ps": p.histogram_binwidth_ps,
            },
        )
        laser_period_ps = int(round(1e6 / p.laser_frequency_mhz))

        def factory(index: int) -> FlimScanCommand:
            return FlimScanCommand(
                expected_kinds=[DataKind.INTENSITY_FRAME, DataKind.FLIM_RAW_FRAME],
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
                frame_trigger_pfi_line=p.frame_trigger_pfi_line,
                pixel_clock_ctr=p.pixel_clock_ctr,
                pixel_clock_pfi_line=p.pixel_clock_pfi_line,
                histogram_bins=p.histogram_bins,
                histogram_binwidth_ps=p.histogram_binwidth_ps,
                laser_period_ps=laser_period_ps,
                flim=run_state.flim,
                simulated=run_state.simulated,
            )

        return FiniteScanSource(factory, frame_limit), setup
