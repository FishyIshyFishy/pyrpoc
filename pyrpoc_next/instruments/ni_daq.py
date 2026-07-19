"""The NI-DAQ card as an instrument: galvo raster scan with analog read and TTL out.

Ported from the old modality acquisition core. Runs real hardware only — it has no
simulation path. Without a connected card, run() raises DaqError.
"""

from __future__ import annotations

import numpy as np

from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.instruments.scanning import RasterScan
from pyrpoc_next.structs.keys import InstrumentKey
from pyrpoc_next.instruments.registry import instrument_registry


class DaqError(RuntimeError):
    """Raised when an NI-DAQ operation fails (typically: no card connected)."""


def raster_waveform(scan: RasterScan) -> np.ndarray:
    """Build the (2, samples) galvo waveform — fast row then slow row — for one scan."""
    fast_amp = max(scan.fast_axis_amplitude, 1e-6)
    slow_amp = max(scan.slow_axis_amplitude, 1e-6)
    fast_step = (2.0 * fast_amp) / scan.x_pixels
    fast_start = -fast_amp - scan.extra_left * fast_step
    fast_axis = fast_start + np.arange(scan.total_x, dtype=np.float32) * fast_step + scan.fast_axis_offset
    slow_axis = np.linspace(-1.0, 1.0, scan.y_pixels, endpoint=False, dtype=np.float32) * slow_amp
    slow_axis = slow_axis + scan.slow_axis_offset
    fast_raster = np.tile(np.repeat(fast_axis, scan.pixel_samples), scan.y_pixels)
    slow_raster = np.repeat(slow_axis, scan.total_x * scan.pixel_samples)
    return np.vstack((fast_raster, slow_raster)).astype(np.float64)


def mask_to_ttl(mask: np.ndarray, device_name: str, daq_port: int, daq_line: int, scan: RasterScan,
                active_samples: int | None = None) -> dict[str, np.ndarray]:
    """Turn a 2D mask into a per-pixel boolean TTL waveform on one digital line.

    ``active_samples`` gates the pulse to the first N samples of each pixel; a
    modality passes its measurement window (whole dwell for intensity, t0 for split).
    """
    window = scan.pixel_samples if active_samples is None else active_samples
    binary = np.asarray(mask, dtype=np.uint8) > 0
    grid = resize_mask(binary, scan.y_pixels, scan.x_pixels)
    padded = np.zeros((scan.y_pixels, scan.total_x), dtype=bool)
    padded[:, scan.extra_left : scan.extra_left + scan.x_pixels] = grid
    if not padded.any():
        return {}
    ttl = np.zeros((scan.y_pixels, scan.total_x, scan.pixel_samples), dtype=bool)
    ttl[padded, :window] = True
    channel = f"{device_name}/port{daq_port}/line{daq_line}"
    return {channel: ttl.reshape(-1)}


def resize_mask(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask to the scan grid."""
    if mask_bool.shape == (target_h, target_w):
        return mask_bool
    source_h, source_w = mask_bool.shape
    y_idx = np.minimum((np.arange(target_h) * source_h) // target_h, source_h - 1)
    x_idx = np.minimum((np.arange(target_w) * source_w) // target_w, source_w - 1)
    return mask_bool[np.ix_(y_idx, x_idx)]


@instrument_registry.register
class NIDAQ(Instrument):
    """National Instruments DAQ: drives the galvos and reads the photodetectors."""

    key = InstrumentKey.ni_daq
    display_name = "NI-DAQ"

    def __init__(self, device_name: str = "Dev1"):
        super().__init__()
        self.device_name = device_name

    def run(self, scan: RasterScan, ttl_signals: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Run one raster scan; return the per-pixel sample cube (channels, H, W, samples)."""
        import nidaqmx as nx
        from nidaqmx.constants import AcquisitionType

        total_samples = scan.total_x * scan.y_pixels * scan.pixel_samples
        waveform = raster_waveform(scan)
        ai_names = [f"{self.device_name}/ai{index}" for index in scan.ai_channels]
        do_task = None
        try:
            with nx.Task() as ao_task, nx.Task() as ai_task:
                ao_task.ao_channels.add_ao_voltage_chan(f"{self.device_name}/ao{scan.fast_axis_ao}")
                ao_task.ao_channels.add_ao_voltage_chan(f"{self.device_name}/ao{scan.slow_axis_ao}")
                for name in ai_names:
                    ai_task.ai_channels.add_ai_voltage_chan(name)
                ao_task.timing.cfg_samp_clk_timing(
                    rate=scan.sample_rate_hz, sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples
                )
                ai_task.timing.cfg_samp_clk_timing(
                    rate=scan.sample_rate_hz, source=f"/{self.device_name}/ao/SampleClock",
                    sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples,
                )
                do_task = self.arm_ttl(nx, AcquisitionType, ttl_signals, total_samples, scan)
                ao_task.write(waveform, auto_start=False)  # pyright: ignore
                ai_task.start()
                if do_task is not None:
                    do_task.start()
                ao_task.start()
                timeout = total_samples / scan.sample_rate_hz + 5
                ao_task.wait_until_done(timeout=timeout)
                ai_task.wait_until_done(timeout=timeout)
                raw = np.asarray(ai_task.read(number_of_samples_per_channel=total_samples), dtype=np.float32)  # pyright: ignore
                return self.reshape_cube(raw, scan)
        except Exception as exc:
            raise DaqError(f"NI-DAQ acquisition failed: {exc}") from exc
        finally:
            if do_task is not None:
                do_task.close()

    def arm_ttl(self, nx, acquisition_type, ttl_signals, total_samples, scan):
        """Create and load the hardware-timed digital-out task for mask TTLs, if any."""
        if not ttl_signals:
            return None
        do_task = nx.Task()
        payload = []
        for channel, ttl in ttl_signals.items():
            do_task.do_channels.add_do_chan(channel)
            payload.append(ttl.tolist())
        do_task.timing.cfg_samp_clk_timing(
            rate=scan.sample_rate_hz, source=f"/{self.device_name}/ao/SampleClock",
            sample_mode=acquisition_type.FINITE, samps_per_chan=total_samples,
        )
        do_task.write(payload[0] if len(payload) == 1 else payload, auto_start=False)
        return do_task

    def reshape_cube(self, raw: np.ndarray, scan: RasterScan) -> np.ndarray:
        """Reshape flat AI samples into (channels, H, W, samples), dropping overscan columns."""
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        cube = raw.reshape(len(scan.ai_channels), scan.y_pixels, scan.total_x, scan.pixel_samples)
        return cube[:, :, scan.extra_left : scan.extra_left + scan.x_pixels, :].astype(np.float32, copy=False)
