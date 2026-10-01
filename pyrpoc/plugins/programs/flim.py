"""FLIM: scan the galvo emitting tagger markers, read back a histogram cube.

The tagger is set up once per run, outside the frame loop, and torn down in a
``finally`` that also runs on a stop. The waveform arithmetic is a copy of
``confocal.py``'s and must stay identical. What is FLIM's own is the
counter-derived pixel clock, the exported start trigger, and reading no analog
input: the image comes from the photon stream.
"""

from __future__ import annotations

import nidaqmx as nx
import numpy as np
from nidaqmx.constants import AcquisitionType, Signal

from pyrpoc.plugins.devices import DAQ, DaqError, FlimMeasurement, Galvo, TimeTagger
from pyrpoc.structs.data_library.data import Cube3D, Image2D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from .components.param_groups import (
    DaqGroup,
    FrameCountGroup,
    HistogramGroup,
    ScanGroup,
    TriggerGroup,
)
from .components.runners import Continuous, Single


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel: rounding, floor 2, since the counter needs a high and
    a low tick. The raster path truncates with floor 1; do not unify them."""
    return max(2, int(round(dwell_time_us * 1e-6 * sample_rate_hz)))


def generate_raster_waveform(
    x_pixels: int,
    extra_left: int,
    extra_right: int,
    y_pixels: int,
    pixel_samples: int,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
) -> np.ndarray:
    """The (fast, slow) AO waveform for one frame: the fast axis sweeps the
    overscan-padded width each line, holding ``pixel_samples`` per pixel."""
    total_x = extra_left + x_pixels + extra_right
    fast_step = 2.0 * fast_axis_amplitude / x_pixels
    fast_start = -fast_axis_amplitude - extra_left * fast_step
    fast_axis = fast_start + np.arange(total_x, dtype=np.float32) * fast_step + fast_axis_offset
    slow_axis = (
        np.linspace(-1.0, 1.0, y_pixels, endpoint=False, dtype=np.float32) * slow_axis_amplitude
        + slow_axis_offset
    )
    fast_raster = np.tile(np.repeat(fast_axis, pixel_samples), y_pixels)
    slow_raster = np.repeat(slow_axis, total_x * pixel_samples)
    return np.vstack((fast_raster, slow_raster)).astype(np.float64)


def waveform_for_scan(scan: ScanGroup, pixel_samples: int) -> np.ndarray:
    return generate_raster_waveform(
        x_pixels=scan.x_pixels,
        extra_left=scan.extra_left,
        extra_right=scan.extra_right,
        y_pixels=scan.y_pixels,
        pixel_samples=pixel_samples,
        fast_axis_offset=scan.fast_axis_offset,
        fast_axis_amplitude=scan.fast_axis_amplitude,
        slow_axis_offset=scan.slow_axis_offset,
        slow_axis_amplitude=scan.slow_axis_amplitude,
    )


def run_flim_scan(
    device_name: str,
    sample_rate_hz: float,
    fast_ao: int,
    slow_ao: int,
    raster_waveform: np.ndarray,
    n_pixels: int,
    pixel_samples: int,
    frame_trigger_pfi: int,
    pixel_clock_ctr: int,
    pixel_clock_pfi: int,
) -> None:
    """Drive one galvo raster while emitting the two markers the TimeTagger
    needs: a frame-start trigger (the exported AO start trigger) and a pixel
    clock (a counter pulse every ``pixel_samples`` AO sample-clock ticks).

    The pixel clock is divided down from the AO sample clock, so pixel
    boundaries stay locked to galvo position with no drift. No analog input is
    read — the FLIM image and lifetimes come from the photon stream.
    """
    total_samples = int(raster_waveform.shape[1])
    timeout = total_samples / sample_rate_hz + 5.0
    try:
        with nx.Task() as ao_task, nx.Task() as co_task:
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{int(fast_ao)}")
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{int(slow_ao)}")
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=total_samples,
            )
            ao_task.export_signals.export_signal(
                Signal.START_TRIGGER, f"/{device_name}/PFI{int(frame_trigger_pfi)}"
            )

            co_channel = co_task.co_channels.add_co_pulse_chan_ticks(
                f"{device_name}/ctr{int(pixel_clock_ctr)}",
                source_terminal=f"/{device_name}/ao/SampleClock",
                high_ticks=1,
                low_ticks=int(pixel_samples) - 1,
            )
            co_channel.co_pulse_term = f"/{device_name}/PFI{int(pixel_clock_pfi)}"
            co_task.timing.cfg_implicit_timing(
                sample_mode=AcquisitionType.FINITE, samps_per_chan=int(n_pixels)
            )
            co_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{device_name}/ao/StartTrigger"
            )

            # Many samples per channel, so write() does not auto-start.
            ao_task.write(np.asarray(raster_waveform, dtype=np.float64))
            co_task.start()  # arms and waits for the AO start trigger
            ao_task.start()
            ao_task.wait_until_done(timeout=timeout)
            co_task.wait_until_done(timeout=timeout)
    except Exception as exc:
        raise DaqError(f"NI-DAQ FLIM scan failed: {exc}") from exc


def flim_scan(
    *,
    daq: DAQ,
    galvo: Galvo,
    scan: ScanGroup,
    sample_rate_hz: float,
    triggers: TriggerGroup,
) -> None:
    """Build the raster waveform and run one FLIM scan. The DAQ supplies only
    its device name: FLIM reads no analog input."""
    samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)
    n_pixels = scan.total_x * scan.y_pixels

    run_flim_scan(
        device_name=daq.config.device_name,
        sample_rate_hz=sample_rate_hz,
        fast_ao=galvo.config.fast_ao,
        slow_ao=galvo.config.slow_ao,
        raster_waveform=waveform_for_scan(scan, samples_per_pixel),
        n_pixels=n_pixels,
        pixel_samples=samples_per_pixel,
        frame_trigger_pfi=triggers.frame_trigger_pfi,
        pixel_clock_ctr=triggers.pixel_clock_ctr,
        pixel_clock_pfi=triggers.pixel_clock_pfi,
    )


def reshape_flim_frame(
    histograms: np.ndarray,
    n_bins: int,
    y_pixels: int,
    total_x_pixels: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    """Fold the flat ``(n_pixels, n_bins)`` Flim histogram into a
    ``(y_pixels, x_pixels, n_bins)`` float32 cube with the overscan columns
    clipped off."""
    cube = np.asarray(histograms, dtype=np.float32).reshape(y_pixels, total_x_pixels, n_bins)
    return cube[:, extra_left : extra_left + x_pixels, :]


def flim_intensity(hist_frame: np.ndarray) -> np.ndarray:
    """Collapse a ``(H, W, n_bins)`` histogram cube to a ``(H, W)`` photon-count
    intensity image."""
    return np.asarray(hist_frame, dtype=np.float32).sum(axis=2)


def read_flim_frame(flim: FlimMeasurement, n_bins: int, scan: ScanGroup) -> np.ndarray:
    """The just-scanned Flim frame as a clipped ``(H, W, n_bins)`` cube."""
    histograms = flim.getCurrentFrameEx().getHistograms()
    return reshape_flim_frame(
        histograms, n_bins, scan.y_pixels, scan.total_x, scan.extra_left, scan.x_pixels
    )


@program_registry.register("flim")
class FLIM(Program):
    display_name = "FLIM"
    uses = [Galvo, DAQ, TimeTagger]
    params = [ScanGroup, FrameCountGroup, DaqGroup, TriggerGroup, HistogramGroup]
    emits = {"intensity": Image2D, "histogram": Cube3D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        scan = ctx.params[ScanGroup]
        num_frames = ctx.params[FrameCountGroup].num_frames
        histogram = ctx.params[HistogramGroup]
        tagger = ctx.devices[TimeTagger]
        ctx.describe(
            "histogram",
            laser_period_ps=histogram.laser_period_ps,
            binwidth_ps=histogram.histogram_binwidth_ps,
            n_bins=histogram.histogram_bins,
        )

        ctx.status("starting the time tagger")
        tagger.create_tagger()
        tagger.configure_for_flim()
        flim = tagger.start_flim_measurement(
            n_pixels=scan.total_x * scan.y_pixels,
            n_bins=histogram.histogram_bins,
            binwidth_ps=histogram.histogram_binwidth_ps,
        )
        try:
            for index in range(num_frames):
                ctx.check_cancel()
                ctx.status(f"frame {index + 1}/{num_frames}")
                self.acquire_frame(ctx, flim)
        finally:
            tagger.stop_flim_measurement(flim)

    @staticmethod
    def acquire_frame(ctx: RunContext, flim: FlimMeasurement) -> None:
        scan = ctx.params[ScanGroup]
        histogram = ctx.params[HistogramGroup]
        flim_scan(
            daq=ctx.devices[DAQ],
            galvo=ctx.devices[Galvo],
            scan=scan,
            sample_rate_hz=ctx.params[DaqGroup].sample_rate_hz,
            triggers=ctx.params[TriggerGroup],
        )
        ctx.sleep(histogram.frame_settle_s)
        cube = read_flim_frame(flim, histogram.histogram_bins, scan)
        ctx.publish("histogram", cube)
        ctx.publish("intensity", flim_intensity(cube)[np.newaxis], channels=["intensity"])
