"""Simulation: frames out of thin air, no instruments required.

``uses = []``, so it runs on any machine: no DAQ, no galvo, no tagger, nothing
to claim. Everything above the hardware boundary is the real thing -- the
runner's thread, dataset creation from ``emits``, publishing, the save policy,
views and their source picker -- so this is how you exercise the software
itself, and how the UI can be looked at away from the rig.

Deterministic by construction: a pattern is a function of (seed, channel,
frame_index), so the same parameters give the same frames on every run and a
test can assert on pixels.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pyrpoc.core.streams import Image2D
from pyrpoc.run.program import Program

from .components import (
    FrameGroup,
    ModulationGroup,
    PacingGroup,
    SignalGroup,
)
from .registry import program_registry

#: Blobs per channel in the "cells" pattern.
BLOB_COUNT = 14


# --------------------------------------------------------------------------- #
# Plane generators                                                             #
# --------------------------------------------------------------------------- #


def _rng(*parts: int) -> np.random.Generator:
    """A generator keyed by whatever identifies this plane."""
    return np.random.default_rng([int(part) % (2**32) for part in parts])


def _axes(y_pixels: int, x_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Column and row vectors, so plane maths broadcasts to ``(H, W)``."""
    ys = np.arange(y_pixels, dtype=np.float32)[:, None]
    xs = np.arange(x_pixels, dtype=np.float32)[None, :]
    return ys, xs


def cells_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """Gaussian blobs drifting across the field, each channel its own set.

    Separable: the two 1-D exponentials multiply into the 2-D blob, which keeps
    a 512x512 frame at a few milliseconds instead of a few hundred. Distances
    wrap, so a blob leaving one edge comes back in the other and a long
    continuous run never empties the field.
    """
    generator = _rng(seed, channel, 0xB10B)
    centre_y = generator.uniform(0.0, y_pixels, BLOB_COUNT)
    centre_x = generator.uniform(0.0, x_pixels, BLOB_COUNT)
    extent = max(y_pixels, x_pixels)
    sigma = generator.uniform(0.02, 0.055, BLOB_COUNT) * extent
    amplitude = generator.uniform(0.4, 1.0, BLOB_COUNT)
    heading = generator.uniform(0.0, 2.0 * np.pi, BLOB_COUNT)

    travelled = float(drift) * float(frame_index)
    centre_y = (centre_y + np.sin(heading) * travelled) % y_pixels
    centre_x = (centre_x + np.cos(heading) * travelled) % x_pixels

    ys, xs = _axes(y_pixels, x_pixels)
    plane = np.zeros((y_pixels, x_pixels), dtype=np.float32)
    for index in range(BLOB_COUNT):
        spread = 2.0 * sigma[index] ** 2
        dy = np.abs(ys - centre_y[index])
        dy = np.minimum(dy, y_pixels - dy)
        dx = np.abs(xs - centre_x[index])
        dx = np.minimum(dx, x_pixels - dx)
        plane += amplitude[index] * np.exp(-(dy**2) / spread) * np.exp(-(dx**2) / spread)
    return np.clip(plane, 0.0, 1.0)


def rings_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """Concentric sine rings, breathing outwards. Good for spotting resampling."""
    del seed
    ys, xs = _axes(y_pixels, x_pixels)
    radius = np.sqrt((ys - y_pixels / 2.0) ** 2 + (xs - x_pixels / 2.0) ** 2)
    period = max(4.0, min(y_pixels, x_pixels) / 12.0)
    phase = channel * (np.pi / 3.0) + float(drift) * float(frame_index) * 0.25
    return 0.5 * (1.0 + np.sin(2.0 * np.pi * radius / period - phase)).astype(np.float32)


def gradient_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """A linear ramp whose direction differs per channel. Shows orientation.

    It sweeps as a triangle wave rather than wrapping, so the drift never puts a
    hard seam across the frame that could be read as an artifact.
    """
    del seed
    ys, xs = _axes(y_pixels, x_pixels)
    angle = channel * (np.pi / 3.0)
    ramp = np.cos(angle) * (xs / max(1, x_pixels - 1)) + np.sin(angle) * (
        ys / max(1, y_pixels - 1)
    )
    shift = (float(drift) * float(frame_index)) / max(1, max(y_pixels, x_pixels))
    swept = (ramp + shift) % 2.0
    return np.abs(1.0 - swept).astype(np.float32)


def checkerboard_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """Hard-edged squares that march diagonally. Shows pixel alignment."""
    del seed
    square = max(2, min(y_pixels, x_pixels) // 8)
    offset = int(round(float(drift) * float(frame_index))) + channel * (square // 2)
    ys, xs = _axes(y_pixels, x_pixels)
    board = (((ys + offset) // square) + ((xs + offset) // square)) % 2.0
    return np.broadcast_to(board, (y_pixels, x_pixels)).astype(np.float32)


def flat_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """A uniform field. With noise turned up it is a pure noise source."""
    del channel, seed, drift, frame_index
    return np.full((y_pixels, x_pixels), 0.5, dtype=np.float32)


PLANES = {
    "cells": cells_plane,
    "rings": rings_plane,
    "gradient": gradient_plane,
    "checkerboard": checkerboard_plane,
    "flat": flat_plane,
}


# --------------------------------------------------------------------------- #
# Masks                                                                        #
# --------------------------------------------------------------------------- #


def resize_mask_nearest(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    source_h, source_w = mask_bool.shape
    if source_h <= 0 or source_w <= 0:
        return np.zeros((target_h, target_w), dtype=bool)
    y_idx = np.minimum((np.arange(target_h, dtype=np.int64) * source_h) // target_h, source_h - 1)
    x_idx = np.minimum((np.arange(target_w, dtype=np.int64) * source_w) // target_w, source_w - 1)
    return mask_bool[np.ix_(y_idx, x_idx)]


def combine_masks(
    masks: Sequence[np.ndarray], *, y_pixels: int, x_pixels: int
) -> np.ndarray | None:
    """Every bound mask resized to the frame and OR-ed into one boolean plane.

    The simulated program has no digital lines, so a mask's port and line are
    ignored here; what the mask still says is *which pixels are illuminated*,
    which is the half of optocontrol that can be checked without a DAQ.
    """
    combined: np.ndarray | None = None
    for mask in masks:
        array = np.asarray(mask)
        if array.ndim != 2 or array.size == 0:
            continue
        resized = resize_mask_nearest(array > 0, target_h=y_pixels, target_w=x_pixels)
        combined = resized if combined is None else (combined | resized)
    return combined


# --------------------------------------------------------------------------- #
# One frame                                                                    #
# --------------------------------------------------------------------------- #


def synthetic_frame(
    *,
    frame_shape: FrameGroup,
    signal: SignalGroup,
    frame_index: int,
    mask: np.ndarray | None,
) -> np.ndarray:
    """One ``(C, H, W)`` float32 frame, as a real scan would have returned it.

    Masked pixels are brightened by ``signal.mask_gain``, standing in for the
    photostimulation the TTL lines would have driven. Noise is added after the
    mask and the result is clipped at zero, because a detector cannot read
    negative.

    ``mask`` has no default: the caller says "no mask is bound" by passing
    ``None`` rather than by leaving the argument off.
    """
    x_pixels, y_pixels, channels = frame_shape.x_pixels, frame_shape.y_pixels, frame_shape.channels
    pattern = signal.pattern
    signal_level = signal.signal_level
    noise_level = signal.noise_level
    drift_pixels_per_frame = signal.drift_pixels_per_frame
    mask_gain = signal.mask_gain
    seed = signal.seed

    if channels <= 0:
        raise ValueError("channels must be at least 1")
    if pattern not in PLANES:
        raise ValueError(f"unknown pattern {pattern!r}; expected one of {list(PLANES)}")

    plane_of = PLANES[pattern]
    frame = np.stack(
        [
            plane_of(
                y_pixels,
                x_pixels,
                channel=channel,
                seed=int(seed),
                drift=float(drift_pixels_per_frame),
                frame_index=int(frame_index),
            )
            for channel in range(channels)
        ]
    ).astype(np.float32)
    frame *= float(signal_level)

    if mask is not None and mask_gain:
        illuminated = resize_mask_nearest(np.asarray(mask) > 0, y_pixels, x_pixels)
        frame = frame * (1.0 + float(mask_gain) * illuminated.astype(np.float32))

    if noise_level:
        generator = _rng(seed, frame_index, 0x0125E)
        frame = frame + generator.standard_normal(frame.shape).astype(np.float32) * float(
            noise_level
        )

    return np.clip(frame, 0.0, None).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# The program                                                                  #
# --------------------------------------------------------------------------- #


def channel_labels(frame_shape: FrameGroup) -> list[str]:
    return [f"sim{index}" for index in range(frame_shape.channels)]


def build_mask(frame_shape: FrameGroup, modulation: ModulationGroup):
    """Load the bound masks and flatten them onto the frame grid.

    Same shape as confocal's ``build_ttl``: once before the loop.
    """
    if not modulation.masks:
        return None
    loaded = [mask.load() for mask in modulation.masks]
    return combine_masks(
        loaded, y_pixels=frame_shape.y_pixels, x_pixels=frame_shape.x_pixels
    )


@program_registry.register("simulation")
class Simulation(Program):
    uses = []
    params = [FrameGroup, SignalGroup, ModulationGroup, PacingGroup]
    emits = {"intensity": Image2D}

    def run(self, ctx) -> None:
        frame_shape = ctx.params[FrameGroup]
        signal = ctx.params[SignalGroup]
        modulation = ctx.params[ModulationGroup]
        num_frames = frame_shape.num_frames
        interval_ms = ctx.params[PacingGroup].frame_interval_ms

        mask = build_mask(frame_shape, modulation)
        labels = channel_labels(frame_shape)
        total = "" if ctx.continuous else f"/{num_frames}"

        for index in ctx.frames(num_frames):
            ctx.status(f"frame {index + 1}{total}")
            frame = synthetic_frame(
                frame_shape=frame_shape,
                signal=signal,
                frame_index=index,
                mask=mask,
            )
            ctx.publish("intensity", frame, channels=labels)
            ctx.sleep(interval_ms / 1000.0)
