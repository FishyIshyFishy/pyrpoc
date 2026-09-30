"""Pinpoint Raman: park the galvos at one point and take a spectrum there.

The point comes from ``PointGroup``, typed or set by an ``ArmAndRun`` runner
from a clicked pixel; ``run`` does not know a display exists. The parking is
real and is left in place when the run ends, so the beam stays on the spot for
a spectrometer read by external software. The spectrum published here is
synthetic until there is a CCD device: a function of (seed, point, frame
index), so re-clicking a pixel reproduces it.
"""

from __future__ import annotations

import nidaqmx as nx
import nidaqmx.errors
import numpy as np

from pyrpoc.plugins.devices.daq.device import DAQ, DaqError
from pyrpoc.plugins.devices.galvo.device import Galvo
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.plugins.params import BlockMap
from pyrpoc.structs.plugins.programs.picks import Pick, PixelPick
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from .components.param_groups import Point, PointGroup, SpectrumGroup
from .components.runners import ArmAndRun, Continuous, Single

# Keeps the noise generator off the band generator's stream, so changing the
# frame index cannot shift a band centre.
NOISE_STREAM = 0x5EED


def park_galvos(daq: DAQ, galvo: Galvo, point: Point) -> None:
    """Move the galvos to ``point`` and leave them there.

    An un-clocked task, so the write is one on-demand sample per channel.
    Closing the task frees the AO channels for the next scan; the card holds
    its last written voltage, so the mirrors stay parked after the run ends.
    """
    device_name = daq.config.device_name
    try:
        with nx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{galvo.config.fast_ao}")
            task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{galvo.config.slow_ao}")
            # One sample per channel, so write() auto-starts.
            task.write([point.fast_v, point.slow_v])
    except nidaqmx.errors.Error as exc:
        raise DaqError(f"could not park the galvos: {exc}") from exc


def synthetic_spectrum(spec: SpectrumGroup, point: Point, *, frame_index: int) -> np.ndarray:
    """A ``(1, n_points)`` spectrum of gaussian bands, keyed to the position.

    The position enters the seed quantised to 0.1 mV, so a re-click reproduces
    the trace and a click one pixel over does not. Only the noise varies per
    frame, so a long run looks like a detector integrating.
    """
    bands = np.random.default_rng(
        [
            spec.seed % (2**32),
            round(point.fast_v * 1e4) % (2**32),
            round(point.slow_v * 1e4) % (2**32),
        ]
    )
    xs = np.arange(spec.n_points, dtype=np.float32)
    spectrum = np.zeros(spec.n_points, dtype=np.float32)
    centres = bands.uniform(0.05, 0.95, spec.n_peaks) * spec.n_points
    widths = bands.uniform(0.004, 0.02, spec.n_peaks) * spec.n_points
    heights = bands.uniform(0.2, 1.0, spec.n_peaks)
    for centre, width, height in zip(centres, widths, heights, strict=True):
        spectrum += height * np.exp(-((xs - centre) ** 2) / (2.0 * width**2))

    # A broad fluorescence background, so the bands sit on something.
    spectrum += 0.12 * np.exp(-xs / (0.6 * spec.n_points))

    noise = np.random.default_rng([spec.seed % (2**32), frame_index % (2**32), NOISE_STREAM])
    spectrum = spectrum + noise.normal(0.0, spec.noise_level, spec.n_points)
    return np.clip(spectrum, 0.0, None).astype(np.float32)[np.newaxis]


def aim_at_pick(pick: Pick, params: BlockMap) -> None:
    """Point the galvos at the picked pixel: its volts become the target."""
    # Narrows for the type checker: the runner asks for a PixelPick.
    if not isinstance(pick, PixelPick):
        raise TypeError(f"expected a PixelPick, got {type(pick).__name__}")
    params[PointGroup].target = Point.from_pixel(pick.dataset, pick.x, pick.y)


@program_registry.register("pinpoint_raman")
class PinpointRaman(Program):
    display_name = "Pinpoint Raman"
    uses = [Galvo]
    params = [PointGroup, SpectrumGroup]
    emits = {"spectrum": Spectrum1D}
    runners = [
        Single(),
        Continuous(),
        ArmAndRun(
            PixelPick,
            aim_at_pick,
            label="Acquire at point…",
            icon=None,
            tooltip="Arm, then provide what this run needs to start it",
        ),
    ]

    def run(self, ctx: RunContext) -> None:
        point = ctx.params[PointGroup].target
        spec = ctx.params[SpectrumGroup]

        ctx.status(f"parking at {point.fast_v:.3f} / {point.slow_v:.3f} V")
        # Both devices are bound though ``uses`` names only the galvo: claims
        # follow ``backed_by``, and the mirrors are voltages on the card.
        park_galvos(ctx.devices[DAQ], ctx.devices[Galvo], point)

        for index in range(spec.num_frames):
            ctx.check_cancel()
            ctx.status(f"spectrum {index + 1}/{spec.num_frames}")
            ctx.sleep(spec.integration_ms / 1000.0)
            spectrum = synthetic_spectrum(spec, point, frame_index=index)
            ctx.publish("spectrum", spectrum, channels=["raman"])
