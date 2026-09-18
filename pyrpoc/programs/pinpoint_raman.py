"""Pinpoint Raman: park the galvos at one point and take a spectrum there.

The first program configured by clicking rather than typing. That changes
nothing about the program: it declares ``PointGroup`` the way confocal declares
``ScanGroup``, reads volts out of it, and has no idea a display exists. Which is
the point -- a click is a way of filling in a parameter, so the acquisition side
needed no new concept to gain one.

Three things happen in order and each undoes itself on the way out: the camera
opens, the mirrors park, spectra are read. A stop at any point unwinds all
three, because cancellation is an exception raised out through ``run()``.

The readout loop is a copy of the one in ``spectrum.py`` rather than an import,
the way ``split_confocal.py`` copies confocal's waveform arithmetic. The two are
meant to stay identical.

What is genuinely this program's, and lives nowhere else, is ``park_galvos`` and
``release_galvos``: an un-clocked AO task held open for the duration of the
measurement. Everything else about driving this rig writes a waveform and lets
a sample clock walk through it; this is the one place that asserts two voltages
and leaves them there.
"""

from __future__ import annotations

import nidaqmx as nx
import numpy as np

from pyrpoc.core.errors import CcdError, DaqError
from pyrpoc.core.streams import Spectrum1D
from pyrpoc.devices import CCD, DAQ, Galvo
from pyrpoc.run.program import Program

from .components import Point, PointGroup, ReadoutGroup
from .registry import program_registry

#: Where the mirrors are left when the measurement ends. Zero on both axes is
#: the centre of the field for a scan with no offset, and it is a definite
#: position rather than wherever the last spectrum happened to be.
REST_VOLTS = (0.0, 0.0)


def park_galvos(daq: DAQ, galvo: Galvo, point: Point) -> nx.Task:
    """Hold the mirrors at one position, and return the task holding them there.

    An un-clocked task: no ``cfg_samp_clk_timing``, so this is a software-timed
    on-demand write of *one sample per channel*, which is why the payload is
    two numbers and not two lists. A clocked multi-sample write takes the
    nested form, and confusing the two is how this fails on the instrument
    rather than here.

    The task is returned still open, because closing it is what ends the park.
    An X-series card holds its last written voltage after the task closes, so
    the mirrors stay where they were put until something says otherwise --
    ``release_galvos`` is that something.

    No clamp on the voltage. ``GalvoConfig`` carries no per-axis limits, so
    there is nothing to clamp against, and inventing a range here would be a
    limit that looks authoritative and came from nowhere.
    """
    device_name = daq.config.device_name
    task = nx.Task()
    try:
        task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{int(galvo.config.fast_ao)}")
        task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{int(galvo.config.slow_ao)}")
        task.write([float(point.fast_v), float(point.slow_v)], auto_start=True)  # pyright:ignore
    except Exception as exc:
        task.close()
        raise DaqError(f"could not park the galvos: {exc}") from exc
    return task


def release_galvos(task: nx.Task) -> None:
    """Return the mirrors to rest and hand the AO channels back.

    Closing the task is what releases the channels, so a confocal scan started
    straight afterwards can claim them. The write comes first because the card
    holds its last value: closing without it would leave the beam parked on the
    sample indefinitely.

    The write is allowed to fail without stopping the close. A task that cannot
    be written is a task that must still be closed, or the channels stay
    claimed for the life of the process.
    """
    try:
        task.write([REST_VOLTS[0], REST_VOLTS[1]], auto_start=True)  # pyright:ignore
    except Exception:
        pass
    finally:
        task.close()


def point_metadata(point: Point) -> dict:
    """Which image, which pixel, which volts.

    Provenance by reference. The volts are what the hardware was told and are
    reproducible from this alone; the rest says where the instruction came
    from. ``source_scan`` is the geometry of the image that was clicked,
    snapshotted when the pixel was resolved into volts -- carried rather than
    looked up because the image is very often a preview that was never saved,
    and a reference to a file that does not exist is not provenance.

    A point typed by hand has no source, and the empty fields say so rather
    than leaving the question open.
    """
    return {
        "park_fast_v": point.fast_v,
        "park_slow_v": point.slow_v,
        "source_picked": point.picked,
        "source_dataset_id": point.source_id,
        "source_label": point.source_label,
        "source_pixel": [point.pixel_x, point.pixel_y],
        "source_started_at": point.source_started_at,
        "source_scan": dict(point.source_scan),
    }


@program_registry.register("pinpoint_raman")
class PinpointRaman(Program):
    uses = [Galvo, DAQ, CCD]
    params = [PointGroup, ReadoutGroup]
    emits = {"spectrum": Spectrum1D}

    def run(self, ctx) -> None:
        point = ctx.params[PointGroup].target
        readout = ctx.params[ReadoutGroup]

        daq: DAQ = ctx.devices[DAQ]
        galvo: Galvo = ctx.devices[Galvo]
        ccd: CCD = ctx.devices[CCD]

        ctx.status("opening the camera")
        ccd.open()
        try:
            ccd.configure_for_spectrum(exposure_s=readout.exposure_s)
            ccd.temperature()
            ctx.describe(
                "spectrum", **ccd.instrument_metadata(), **point_metadata(point)
            )

            ctx.status(f"parking at {point.fast_v:.3f} / {point.slow_v:.3f} V")
            task = park_galvos(daq, galvo, point)
            try:
                for _ in ctx.frames(1):
                    spectrum = read_averaged(ctx, ccd, readout.num_acquisitions)
                    ctx.publish("spectrum", spectrum, channels=["raman"])
            finally:
                release_galvos(task)
        finally:
            # Idle, not closed. The handle outlives the run on purpose.
            ccd.abort()


def read_averaged(ctx, ccd: CCD, num_acquisitions: int) -> np.ndarray:
    """Read ``num_acquisitions`` spectra off the camera and average them.

    A copy of the loop in ``spectrum.py`` rather than an import, the way
    ``split_confocal.py`` copies confocal's waveform arithmetic. The two are
    meant to stay identical.
    """
    total = max(int(num_acquisitions), 1)
    accumulated: np.ndarray | None = None
    for shot in range(total):
        if total > 1:
            ctx.status(f"acquisition {shot + 1}/{total}")
        else:
            ctx.status("spectrum")

        spectrum = ccd.read_spectrum(should_stop=ctx.cancelled)
        if spectrum is None:
            ctx.check_cancel()
            raise CcdError("the camera returned no data and was not stopped")

        if ccd.saturated:
            ctx.status(f"acquisition {shot + 1}/{total} - saturated at {ccd.last_max_counts} counts")
        accumulated = spectrum if accumulated is None else accumulated + spectrum

    assert accumulated is not None
    return (accumulated / total).astype(np.float32, copy=False)
