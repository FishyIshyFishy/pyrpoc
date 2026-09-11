"""Pinpoint Raman: park the galvos at one point and take a spectrum there.

The first program configured by clicking rather than typing. That changes
nothing about the program: it declares ``PointGroup`` the way confocal declares
``ScanGroup``, reads volts out of it, and has no idea a display exists. Which is
the point -- a click is a way of filling in a parameter, so the acquisition side
needed no new concept to gain one.

The spectrometer is synthetic for now and the parking is a documented stub, so
this runs on a laptop with no card in it. What is real is everything above the
hardware boundary: claims over the galvo, the runner's thread, dataset creation
from ``emits``, publishing, the save policy and the spectrum view.

Deterministic by construction, like ``simulation.py``: a spectrum is a function
of (seed, point, frame index), so clicking the same pixel twice gives the same
trace and two different pixels visibly differ. That is what makes the fake data
worth looking at -- a picker that returned noise would not show you whether the
point actually changed.
"""

from __future__ import annotations

import numpy as np

from pyrpoc.core.streams import Spectrum1D
from pyrpoc.devices.daq.device import DAQ
from pyrpoc.devices.galvo.device import Galvo
from pyrpoc.run.program import Program

from .components import Point, PointGroup, SpectrumGroup
from .registry import program_registry

#: Keeps the noise generator off the band generator's stream, so changing the
#: frame index cannot shift a band centre.
NOISE_STREAM = 0x5EED


def park_galvos(daq: DAQ, galvo: Galvo, point: Point) -> None:
    """Hold the galvos at one position. Not implemented yet.

    When the DAQ logic lands this writes two AO samples -- ``point.fast_v`` on
    ``galvo.config.fast_ao`` and ``point.slow_v`` on ``galvo.config.slow_ao`` --
    on an un-clocked task and leaves them asserted for the duration of the
    spectrum, then returns the mirrors to their resting position.

    It is a real function with its real signature rather than an inline comment
    because the seam is the interesting part: it is the only place in this file
    that will ever touch hardware, and the synthetic detector below does not
    care whether it did. Both devices are bound even though ``uses`` names only
    the galvo, since claims propagate along ``backed_by`` and the mirrors are
    voltages on the card's AO channels.
    """
    del daq, galvo, point


def synthetic_spectrum(
    spec: SpectrumGroup, point: Point, *, frame_index: int
) -> np.ndarray:
    """A ``(1, n_points)`` spectrum of gaussian bands, keyed to the position.

    The position enters the seed quantised to 0.1 mV, so that a re-click on the
    same pixel reproduces the trace exactly while a click one pixel over does
    not. Band centres, widths and heights are drawn once from that seed; only
    the noise varies per frame, which is what makes a continuous run at one spot
    look like a detector integrating rather than a new sample each time.
    """
    n_points = int(spec.n_points)
    bands = np.random.default_rng(
        [
            int(spec.seed) % (2**32),
            int(round(point.fast_v * 1e4)) % (2**32),
            int(round(point.slow_v * 1e4)) % (2**32),
        ]
    )
    xs = np.arange(n_points, dtype=np.float32)

    spectrum = np.zeros(n_points, dtype=np.float32)
    peaks = int(spec.n_peaks)
    if peaks > 0:
        centres = bands.uniform(0.05, 0.95, peaks) * n_points
        widths = bands.uniform(0.004, 0.02, peaks) * n_points
        heights = bands.uniform(0.2, 1.0, peaks)
        for centre, width, height in zip(centres, widths, heights):
            spectrum += height * np.exp(-((xs - centre) ** 2) / (2.0 * width**2))

    # A broad fluorescence background, so the bands sit on something.
    spectrum += 0.12 * np.exp(-xs / (0.6 * n_points))

    noise = np.random.default_rng(
        [int(spec.seed) % (2**32), int(frame_index) % (2**32), NOISE_STREAM]
    )
    spectrum = spectrum + noise.normal(0.0, max(float(spec.noise_level), 0.0), n_points)
    return np.clip(spectrum, 0.0, None).astype(np.float32)[np.newaxis]


@program_registry.register("pinpoint_raman")
class PinpointRaman(Program):
    uses = [Galvo]
    params = [PointGroup, SpectrumGroup]
    emits = {"spectrum": Spectrum1D}

    def run(self, ctx) -> None:
        point = ctx.params[PointGroup].target
        spec = ctx.params[SpectrumGroup]

        daq: DAQ = ctx.devices[DAQ]
        galvo: Galvo = ctx.devices[Galvo]

        ctx.status(f"parking at {point.fast_v:.3f} / {point.slow_v:.3f} V")
        park_galvos(daq, galvo, point)

        total = "" if ctx.continuous else f"/{spec.num_frames}"
        for index in ctx.frames(spec.num_frames):
            ctx.status(f"spectrum {index + 1}{total}")
            ctx.sleep(spec.integration_ms / 1000.0)
            spectrum = synthetic_spectrum(spec, point, frame_index=index)
            ctx.publish("spectrum", spectrum, channels=["raman"])
