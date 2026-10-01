"""The NI tasks behind a galvo scan: AO drives the mirrors and is the clock
every other task follows. These only fail on the instrument, so change them
with a rig to test on."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import nidaqmx as nx
import nidaqmx.errors
import numpy as np
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader

from pyrpoc.plugins.devices import DAQ, DaqError, Galvo

# Allowed beyond the scan's own length before a wait gives up.
TIMEOUT_MARGIN_S = 5.0


def add_galvo_channels(task: nx.Task, device_name: str, galvo: Galvo) -> None:
    task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{galvo.config.fast_ao}")
    task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{galvo.config.slow_ao}")


def clock_galvos(
    task: nx.Task, device_name: str, galvo: Galvo, sample_rate_hz: float, total_samples: int
) -> None:
    """The galvo channels on a finite sample clock of their own."""
    add_galvo_channels(task, device_name, galvo)
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate_hz, sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples
    )


def follow_galvo_clock(
    task: nx.Task, device_name: str, sample_rate_hz: float, total_samples: int
) -> None:
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate_hz,
        source=f"/{device_name}/ao/SampleClock",
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=total_samples,
    )


def play(galvos: nx.Task, followers: list[nx.Task], waveform: np.ndarray, timeout: float) -> None:
    """Start the followers first, so they are armed when the galvo clock starts."""
    # Many samples per channel, so write() does not auto-start.
    galvos.write(waveform)
    for task in followers:
        task.start()
    galvos.start()
    galvos.wait_until_done(timeout=timeout)
    for task in followers:
        task.wait_until_done(timeout=timeout)


def scan_timeout(total_samples: int, sample_rate_hz: float) -> float:
    return total_samples / sample_rate_hz + TIMEOUT_MARGIN_S


def is_clocked(channel: str) -> bool:
    """Only port 0 is hardware-timed on these cards; other ports can only hold a level."""
    return "/port0/" in channel.lower()


@contextmanager
def held_lines(levels: dict[str, bool]) -> Iterator[None]:
    """Unclocked lines held at ``levels`` for the scan, then inverted."""
    with nx.Task() as task:
        for channel in levels:
            task.do_channels.add_do_chan(channel)
        # One sample per line, so write() auto-starts.
        task.write(list(levels.values()))
        try:
            yield
        finally:
            task.write([not level for level in levels.values()])


def add_clocked_lines(
    task: nx.Task,
    device_name: str,
    signals: dict[str, np.ndarray],
    sample_rate_hz: float,
    total_samples: int,
) -> None:
    for channel in signals:
        task.do_channels.add_do_chan(channel)
    follow_galvo_clock(task, device_name, sample_rate_hz, total_samples)
    payload = [signal.tolist() for signal in signals.values()]
    # Many samples per line, so write() does not auto-start.
    task.write(payload[0] if len(payload) == 1 else payload)


def acquire(
    daq: DAQ, galvo: Galvo, waveform: np.ndarray, ttl: dict[str, np.ndarray], sample_rate_hz: float
) -> np.ndarray:
    """Play ``waveform`` on the galvos and read every analog input on its clock,
    with ``ttl`` on the digital lines: the raw ``(channels, samples)`` stream."""
    device = daq.config.device_name
    total = waveform.shape[1]
    clocked = {channel: signal for channel, signal in ttl.items() if is_clocked(channel)}
    held = {channel: bool(signal[0]) for channel, signal in ttl.items() if not is_clocked(channel)}
    try:
        with ExitStack() as stack:
            if held:
                stack.enter_context(held_lines(held))
            galvos = stack.enter_context(nx.Task())
            clock_galvos(galvos, device, galvo, sample_rate_hz, total)
            inputs = stack.enter_context(nx.Task())
            for index in daq.config.ai_channels:
                inputs.ai_channels.add_ai_voltage_chan(f"{device}/ai{index}")
            follow_galvo_clock(inputs, device, sample_rate_hz, total)
            followers = [inputs]
            if clocked:
                lines = stack.enter_context(nx.Task())
                add_clocked_lines(lines, device, clocked, sample_rate_hz, total)
                followers.append(lines)
            play(galvos, followers, waveform, scan_timeout(total, sample_rate_hz))
            return read_inputs(inputs, total)
    except nidaqmx.errors.Error as exc:
        raise DaqError(f"NI-DAQ acquisition failed: {exc}") from exc


def read_inputs(task: nx.Task, total_samples: int) -> np.ndarray:
    # Task.read is typed to accept no sample count; the stream reader takes
    # one, and always fills (channels, samples).
    raw = np.empty((task.number_of_channels, total_samples), dtype=np.float64)
    AnalogMultiChannelReader(task.in_stream).read_many_sample(
        raw, number_of_samples_per_channel=total_samples
    )
    return raw
