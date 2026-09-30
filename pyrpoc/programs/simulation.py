"""Simulation: frames out of thin air, no instruments required.

Everything above the hardware boundary is real (the executor's thread, datasets,
publishing, saving, the panels), so this exercises the software itself on any
machine. A plane is a function of (seed, channel, frame index). Each run draws
its own seed, so repeated runs show a different sample rather than a replay.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyrpoc.structs.data import Image2D
from pyrpoc.structs.program import Program, RunContext
from pyrpoc.structs.registries import program_registry

from .components.param_groups import (
    FrameGroup,
    Mask,
    ModulationGroup,
    PacingGroup,
    SignalGroup,
)
from .components.runners import Continuous, Single

# Blobs per channel in the "cells" pattern.
BLOB_COUNT = 14


def _rng(*parts: int) -> np.random.Generator:
    """A generator keyed by whatever identifies this plane."""
    return np.random.default_rng([part % (2**32) for part in parts])


def _axes(y_pixels: int, x_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Column and row vectors, so plane maths broadcasts to ``(H, W)``."""
    ys = np.arange(y_pixels, dtype=np.float32)[:, None]
    xs = np.arange(x_pixels, dtype=np.float32)[None, :]
    return ys, xs


def cells_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """Gaussian blobs drifting across the field, each channel its own set.

    Separable exponentials keep a 512x512 frame at milliseconds; distances wrap
    so a long run never empties the field.
    """
    generator = _rng(seed, channel, 0xB10B)
    centre_y = generator.uniform(0.0, y_pixels, BLOB_COUNT)
    centre_x = generator.uniform(0.0, x_pixels, BLOB_COUNT)
    extent = max(y_pixels, x_pixels)
    sigma = generator.uniform(0.02, 0.055, BLOB_COUNT) * extent
    amplitude = generator.uniform(0.4, 1.0, BLOB_COUNT)
    heading = generator.uniform(0.0, 2.0 * np.pi, BLOB_COUNT)

    travelled = drift * frame_index
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
    phase = channel * (np.pi / 3.0) + drift * frame_index * 0.25
    return 0.5 * (1.0 + np.sin(2.0 * np.pi * radius / period - phase)).astype(np.float32)


def gradient_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """A linear ramp whose direction differs per channel. Shows orientation.
    It sweeps as a triangle wave, so the drift never puts a seam across the frame."""
    del seed
    ys, xs = _axes(y_pixels, x_pixels)
    angle = channel * (np.pi / 3.0)
    ramp = np.cos(angle) * (xs / max(1, x_pixels - 1)) + np.sin(angle) * (ys / max(1, y_pixels - 1))
    shift = drift * frame_index / max(y_pixels, x_pixels)
    swept = (ramp + shift) % 2.0
    return np.abs(1.0 - swept).astype(np.float32)


def checkerboard_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """Hard-edged squares that march diagonally. Shows pixel alignment."""
    del seed
    square = max(2, min(y_pixels, x_pixels) // 8)
    offset = int(round(drift * frame_index)) + channel * (square // 2)
    ys, xs = _axes(y_pixels, x_pixels)
    board = (((ys + offset) // square) + ((xs + offset) // square)) % 2.0
    return np.broadcast_to(board, (y_pixels, x_pixels)).astype(np.float32)


def flat_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """A uniform field. With noise turned up it is a pure noise source."""
    del channel, seed, drift, frame_index
    return np.full((y_pixels, x_pixels), 0.5, dtype=np.float32)


# Keyed by the choices ``SignalGroup.pattern`` offers.
PLANES = {
    "cells": cells_plane,
    "rings": rings_plane,
    "gradient": gradient_plane,
    "checkerboard": checkerboard_plane,
    "flat": flat_plane,
}


def resize_mask_nearest(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    source_h, source_w = mask_bool.shape
    y_idx = np.minimum((np.arange(target_h, dtype=np.int64) * source_h) // target_h, source_h - 1)
    x_idx = np.minimum((np.arange(target_w, dtype=np.int64) * source_w) // target_w, source_w - 1)
    return mask_bool[np.ix_(y_idx, x_idx)]


def combine_masks(masks: Sequence[Mask], frame_shape: FrameGroup) -> np.ndarray | None:
    """Every bound mask resized to the frame and OR-ed into one plane, or None
    for no masks. Ports and lines are ignored: there are no digital lines here,
    but which pixels are illuminated can still be checked."""
    combined: np.ndarray | None = None
    for mask in masks:
        # Narrows for the type checker: a run's masks are resolved at start.
        if mask.array is None:
            raise ValueError(f"mask '{mask.describe()}' was not resolved")
        resized = resize_mask_nearest(mask.array > 0, frame_shape.y_pixels, frame_shape.x_pixels)
        combined = resized if combined is None else (combined | resized)
    return combined


def synthetic_frame(
    *,
    frame_shape: FrameGroup,
    signal: SignalGroup,
    seed: int,
    frame_index: int,
    mask: np.ndarray | None,
) -> np.ndarray:
    """One ``(C, H, W)`` float32 frame, as a real scan would have returned it.

    Masked pixels are brightened by ``mask_gain``, standing in for the
    stimulation the TTL lines would drive. Noise goes on after the mask and the
    result is clipped at zero, since a detector cannot read negative.
    """
    y_pixels, x_pixels = frame_shape.y_pixels, frame_shape.x_pixels
    plane_of = PLANES[signal.pattern]
    frame = np.stack(
        [
            plane_of(
                y_pixels,
                x_pixels,
                channel=channel,
                seed=seed,
                drift=signal.drift_pixels_per_frame,
                frame_index=frame_index,
            )
            for channel in range(frame_shape.channels)
        ]
    ).astype(np.float32)
    frame *= signal.signal_level

    if mask is not None and signal.mask_gain:
        frame = frame * (1.0 + signal.mask_gain * mask.astype(np.float32))

    if signal.noise_level:
        noise = _rng(seed, frame_index, 0x0125E).standard_normal(frame.shape)
        frame = frame + noise.astype(np.float32) * signal.noise_level

    return np.clip(frame, 0.0, None).astype(np.float32, copy=False)


def channel_labels(frame_shape: FrameGroup) -> list[str]:
    return [f"sim{index}" for index in range(frame_shape.channels)]


@program_registry.register("simulation")
class Simulation(Program):
    display_name = "Simulation"
    uses = []
    params = [FrameGroup, SignalGroup, ModulationGroup, PacingGroup]
    emits = {"intensity": Image2D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        frame_shape = ctx.params[FrameGroup]
        signal = ctx.params[SignalGroup]
        num_frames = frame_shape.num_frames
        interval_ms = ctx.params[PacingGroup].frame_interval_ms

        # Built once before the loop; the pixels arrived with the parameter.
        mask = combine_masks(ctx.params[ModulationGroup].masks, frame_shape)
        labels = channel_labels(frame_shape)
        # Fixed for the run so its frames stay one drifting sample.
        seed = int(np.random.default_rng().integers(2**32))

        for index in range(num_frames):
            ctx.check_cancel()
            ctx.status(f"frame {index + 1}/{num_frames}")
            frame = synthetic_frame(
                frame_shape=frame_shape, signal=signal, seed=seed, frame_index=index, mask=mask
            )
            ctx.publish("intensity", frame, channels=labels)
            ctx.sleep(interval_ms / 1000.0)
