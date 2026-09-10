"""FLIM: scan the galvo emitting tagger markers, read back a histogram cube.

The clearest demonstration of "the program owns the loop". In v3.0
``FlimModality.acquire_once`` called ``setup_tagger()`` at the top and
``teardown_tagger()`` in a ``finally`` -- **per frame** -- because
``acquire_once`` had to be self-contained and there was nowhere else for per-run
setup to go. A ten-frame run created and freed the TimeTagger ten times.

Here setup is simply outside the loop, and the ``finally`` runs on a stop
because cancellation is an exception raised out through ``run()``.

The waveform arithmetic is duplicated from ``confocal.py`` rather than shared,
and is meant to stay identical to it. What is genuinely FLIM's is the
counter-derived pixel clock, the exported start trigger, and the fact that no
analog input is read at all -- the image comes from the photon stream.
"""

from __future__ import annotations

import numpy as np
import nidaqmx as nx
from nidaqmx.constants import AcquisitionType, Signal

from pyrpoc.core.errors import DaqError
from pyrpoc.core.streams import Cube3D, Image2D
from pyrpoc.devices import DAQ, Galvo, TimeTagger
from pyrpoc.run.program import Program

from .components import (
    DaqGroup,
    FramesGroup,
    HistogramGroup,
    ScanGroup,
    TriggerGroup,
)
from .registry import program_registry


# --------------------------------------------------------------------------- #
# Waveform arithmetic                                                          #
# --------------------------------------------------------------------------- #


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel for a FLIM scan.

    Rounds, floor 2 -- the counter needs at least one high tick and one low
    tick. The raster path truncates and has a floor of 1. The two formulas
    genuinely differ -- do not unify them.
    """
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
    total_x = extra_left + x_pixels + extra_right
    fast_amp = max(float(fast_axis_amplitude), 1e-6)
    slow_amp = max(float(slow_axis_amplitude), 1e-6)
    fast_step = (2.0 * fast_amp) / float(x_pixels)
    fast_start = -fast_amp - (float(extra_left) * fast_step)
    fast_axis = fast_start + (np.arange(total_x, dtype=np.float32) * fast_step) + float(fast_axis_offset)
    slow_axis = (
        np.linspace(-1.0, 1.0, y_pixels, endpoint=False, dtype=np.float32) * slow_amp
        + float(slow_axis_offset)
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


# --------------------------------------------------------------------------- #
# The scan                                                                     #
# --------------------------------------------------------------------------- #


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

            ao_task.write(np.asarray(raster_waveform, dtype=np.float64), auto_start=False)  # pyright:ignore
            co_task.start()   # arms and waits for the AO start trigger
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
    """Build the raster waveform and run one FLIM scan.

    Takes the DAQ for its device name only. FLIM reads no analog input, so it
    does not ask for any.
    """
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


# --------------------------------------------------------------------------- #
# Reading a frame back                                                         #
# --------------------------------------------------------------------------- #


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


def read_flim_frame(
    flim_measurement,
    n_bins: int,
    y_pixels: int,
    total_x_pixels: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    """Read the current (just-scanned) Flim frame and return its clipped
    ``(y_pixels, x_pixels, n_bins)`` histogram cube."""
    frame = flim_measurement.getCurrentFrameEx()
    return reshape_flim_frame(
        frame.getHistograms(), n_bins, y_pixels, total_x_pixels, extra_left, x_pixels
    )


# --------------------------------------------------------------------------- #
# The program                                                                  #
# --------------------------------------------------------------------------- #


@program_registry.register("flim")
class FLIM(Program):
    uses = [Galvo, DAQ, TimeTagger]
    params = [ScanGroup, DaqGroup, TriggerGroup, HistogramGroup, FramesGroup]
    emits = {"intensity": Image2D, "histogram": Cube3D}

    def run(self, ctx) -> None:
        scan = ctx.params[ScanGroup]
        daq_params = ctx.params[DaqGroup]
        triggers = ctx.params[TriggerGroup]
        histogram = ctx.params[HistogramGroup]
        num_frames = ctx.params[FramesGroup].num_frames

        daq: DAQ = ctx.devices[DAQ]
        galvo: Galvo = ctx.devices[Galvo]
        tagger: TimeTagger = ctx.devices[TimeTagger]

        total_x = scan.total_x
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
            n_pixels=total_x * scan.y_pixels,
            n_bins=histogram.histogram_bins,
            binwidth_ps=histogram.histogram_binwidth_ps,
        )
        try:
            total = "" if ctx.continuous else f"/{num_frames}"
            for index in ctx.frames(num_frames):
                ctx.status(f"frame {index + 1}{total}")
                flim_scan(
                    daq=daq,
                    galvo=galvo,
                    scan=scan,
                    sample_rate_hz=daq_params.sample_rate_hz,
                    triggers=triggers,
                )
                ctx.sleep(histogram.frame_settle_s)
                cube = read_flim_frame(
                    flim,
                    n_bins=histogram.histogram_bins,
                    y_pixels=scan.y_pixels,
                    total_x_pixels=total_x,
                    extra_left=scan.extra_left,
                    x_pixels=scan.x_pixels,
                )
                ctx.publish("histogram", cube)
                ctx.publish(
                    "intensity",
                    flim_intensity(cube)[np.newaxis],
                    channels=["intensity"],
                )
        finally:
            tagger.stop_flim_measurement(flim)
