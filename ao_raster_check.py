"""Hardware check: play the confocal raster waveform on AO 0/1 and, identically,
on AO 2/3, then read every AO channel back through the card's internal
``_aoN_vs_aognd`` channels to show what each one actually output.

    uv run python ao_raster_check.py
    uv run python ao_raster_check.py --frames 0      # loop until Ctrl+C, for a scope
    uv run python ao_raster_check.py --no-readback   # skip the internal readback
"""

from __future__ import annotations

import argparse

import nidaqmx as nx
import numpy as np
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader

from pyrpoc.plugins.programs.building_blocks.functions.galvo_raster import (
    pixel_samples,
    raster_waveform,
)
from pyrpoc.plugins.programs.building_blocks.parameter_groups.scan import ScanGroup

AO_CHANNELS = [0, 1, 2, 3]
ROLES = ["pair A fast", "pair A slow", "pair B fast", "pair B slow"]
# A readback this far from the commanded voltage means the channel is not
# outputting what was written.
TOLERANCE_V = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="Dev1")
    # 4 readback channels share the 6363's 1 MS/s aggregate AI rate.
    parser.add_argument("--sample-rate", type=float, default=200_000.0)
    parser.add_argument("--frames", type=int, default=3, help="0 loops until Ctrl+C")
    parser.add_argument("--no-readback", action="store_true")
    return parser.parse_args()


def build_waveform(sample_rate_hz: float) -> tuple[ScanGroup, np.ndarray]:
    """The app's raster with default scan settings, on both pairs."""
    scan = ScanGroup()
    pair = raster_waveform(scan, pixel_samples(scan.dwell_time_us, sample_rate_hz))
    return scan, np.vstack((pair, pair))


def describe(scan: ScanGroup, waveform: np.ndarray, sample_rate_hz: float) -> None:
    total = waveform.shape[1]
    print(f"Scan: {scan.x_pixels}x{scan.y_pixels} px, total_x {scan.total_x}, "
          f"fast amp {scan.fast_axis_amplitude} V, slow amp {scan.slow_axis_amplitude} V")
    print(f"Waveform: {waveform.shape[0]} channels x {total} samples "
          f"= {total / sample_rate_hz:.3f} s per frame at {sample_rate_hz:,.0f} S/s")
    print(f"Pair B identical to pair A: {np.array_equal(waveform[:2], waveform[2:])}\n")
    print(f"{'channel':<8}{'role':<14}{'min V':>9}{'max V':>9}{'first':>9}{'last':>9}")
    for index, role, row in zip(AO_CHANNELS, ROLES, waveform):
        print(f"ao{index:<6}{role:<14}{row.min():>9.3f}{row.max():>9.3f}{row[0]:>9.3f}{row[-1]:>9.3f}")
    print()


def play_frame(device: str, waveform: np.ndarray, sample_rate_hz: float, readback: bool) -> np.ndarray | None:
    """One frame on all four AO channels; the internal readback of each, if asked."""
    total = waveform.shape[1]
    timeout = total / sample_rate_hz + 5.0
    with nx.Task() as ao, nx.Task() as ai:
        for index in AO_CHANNELS:
            ao.ao_channels.add_ao_voltage_chan(f"{device}/ao{index}")
        ao.timing.cfg_samp_clk_timing(
            rate=sample_rate_hz, sample_mode=AcquisitionType.FINITE, samps_per_chan=total
        )
        ao.write(waveform)
        if readback:
            for index in AO_CHANNELS:
                ai.ai_channels.add_ai_voltage_chan(f"{device}/_ao{index}_vs_aognd")
            ai.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                source=f"/{device}/ao/SampleClock",
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=total,
            )
            ai.start()
        ao.start()
        ao.wait_until_done(timeout=timeout)
        if not readback:
            return None
        measured = np.empty((len(AO_CHANNELS), total), dtype=np.float64)
        AnalogMultiChannelReader(ai.in_stream).read_many_sample(
            measured, number_of_samples_per_channel=total, timeout=timeout
        )
        return measured


def report(waveform: np.ndarray, measured: np.ndarray) -> None:
    """Commanded vs read back, per channel. The AI samples on the same clock
    edge the AO updates on, so mid-flyback a reading can be the old value, the
    new one, or in between, and the multiplexed AI reads later channels later:
    each reading only has to fall within the commanded samples two either side."""
    total = waveform.shape[1]
    neighbours = np.stack([waveform[:, shift : total - 4 + shift] for shift in range(5)])
    low, high = neighbours.min(axis=0), neighbours.max(axis=0)
    actual = measured[:, 2:-2]
    outside = np.maximum(low - actual, actual - high).clip(min=0.0)
    print(f"  {'channel':<8}{'role':<14}{'read min':>10}{'read max':>10}{'max err':>10}  result")
    for i, (index, role) in enumerate(zip(AO_CHANNELS, ROLES)):
        error = outside[i].max()
        verdict = "OK" if error < TOLERANCE_V else "MISMATCH"
        print(f"  ao{index:<6}{role:<14}{actual[i].min():>10.3f}{actual[i].max():>10.3f}"
              f"{error:>10.3f}  {verdict}")
    pair_gap = np.abs(measured[:2] - measured[2:])
    print(f"  pair A vs pair B: median difference {np.median(pair_gap):.4f} V "
          f"(max {pair_gap.max():.3f} V, at flyback)\n")


def main() -> None:
    args = parse_args()
    scan, waveform = build_waveform(args.sample_rate)
    describe(scan, waveform, args.sample_rate)
    frame = 0
    try:
        while args.frames == 0 or frame < args.frames:
            frame += 1
            print(f"Frame {frame}: playing ao0-ao3 ...")
            measured = play_frame(args.device, waveform, args.sample_rate, not args.no_readback)
            print(f"Frame {frame}: done.")
            if measured is not None:
                report(waveform, measured)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        with nx.Task() as park:
            for index in AO_CHANNELS:
                park.ao_channels.add_ao_voltage_chan(f"{args.device}/ao{index}")
            park.write([0.0] * len(AO_CHANNELS), auto_start=True)
        print("All AO channels parked at 0 V.")


if __name__ == "__main__":
    main()
