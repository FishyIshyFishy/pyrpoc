"""FLIM's scan: confocal's raster waveform, emitting the two markers the tagger
needs instead of reading analog input, since the image comes from the photon
stream. These tasks only fail on the instrument, so change them with a rig."""

from __future__ import annotations

import nidaqmx as nx
import nidaqmx.errors
from nidaqmx.constants import AcquisitionType, Signal

from pyrpoc.plugins.devices import DAQ, DaqError, Galvo

from ..building_blocks.functions.daq_tasks import clock_galvos, play, scan_timeout
from ..building_blocks.functions.galvo_raster import raster_waveform
from ..building_blocks.parameter_groups import ScanGroup, TriggerGroup


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel: rounding, floor 2, since the counter needs a high and
    a low tick. The raster path truncates with floor 1; do not unify them."""
    return max(2, round(dwell_time_us * 1e-6 * sample_rate_hz))


def add_pixel_clock(
    task: nx.Task, device_name: str, triggers: TriggerGroup, samples_per_pixel: int, pixels: int
) -> None:
    """A pulse every ``samples_per_pixel`` galvo-clock ticks, armed on the galvo
    start. Divided down from that clock, pixel boundaries stay locked to galvo
    position with no drift."""
    channel = task.co_channels.add_co_pulse_chan_ticks(
        f"{device_name}/ctr{triggers.pixel_clock_ctr}",
        source_terminal=f"/{device_name}/ao/SampleClock",
        high_ticks=1,
        low_ticks=samples_per_pixel - 1,
    )
    channel.co_pulse_term = f"/{device_name}/PFI{triggers.pixel_clock_pfi}"
    task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=pixels)
    task.triggers.start_trigger.cfg_dig_edge_start_trig(f"/{device_name}/ao/StartTrigger")


def flim_scan(
    daq: DAQ, galvo: Galvo, scan: ScanGroup, sample_rate_hz: float, triggers: TriggerGroup
) -> None:
    """One raster, exporting the galvo start trigger as the frame marker and
    driving the pixel clock alongside."""
    device = daq.config.device_name
    samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)
    waveform = raster_waveform(scan, samples_per_pixel)
    total = waveform.shape[1]
    try:
        with nx.Task() as galvos, nx.Task() as pixel_clock:
            clock_galvos(galvos, device, galvo, sample_rate_hz, total)
            galvos.export_signals.export_signal(
                Signal.START_TRIGGER, f"/{device}/PFI{triggers.frame_trigger_pfi}"
            )
            add_pixel_clock(
                pixel_clock, device, triggers, samples_per_pixel, scan.total_x * scan.y_pixels
            )
            play(galvos, [pixel_clock], waveform, scan_timeout(total, sample_rate_hz))
    except nidaqmx.errors.Error as exc:
        raise DaqError(f"NI-DAQ FLIM scan failed: {exc}") from exc
