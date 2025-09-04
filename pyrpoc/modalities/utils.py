import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple, Optional, Union

def create_ttl_from_mask(mask: np.ndarray, numsteps_x: int, numsteps_y: int,
                         extrasteps_left: int, extrasteps_right: int,
                         dwell_time: float, sample_rate: float,
                         thresh: float = 0,) -> np.ndarray:
    '''
    create the TTL signal from a mask to scan

    mask: 2D image mask
    numsteps_x, numsteps_y, extrasteps_left, extrasteps_right: galvo scanning params
    dwell_time: time in us spent per pixel
    sample_rate: DAQ sample rate for output task in MHz (samples per pixel = dwell_time * sample_rate)
    thresh: minimum value relative to maximum in the mask to take as an active pixel, default is 0
    '''

    mask = mask > (thresh * np.max(mask))
    total_y = numsteps_y
    total_x = extrasteps_left + extrasteps_right + numsteps_x
    pixel_samples = int(dwell_time * sample_rate) # us * MHz cancels out

    padded_mask = []
    for row in range(numsteps_y):
        padded_row = np.concatenate((
            np.zeros(extrasteps_left, dtype=bool),
            mask[row, :] if row < mask.shape[0] else np.zeros(numsteps_x, dtype=bool),
            np.zeros(extrasteps_right, dtype=bool)
        ))
        padded_mask.append(padded_row)

    padded_mask = np.array(padded_mask)
    ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)

    for y in range(total_y):
        for x in range(total_x):
            if padded_mask[y, x]:
                ttl[y, x, :] = True
    ttl = ttl.ravel()
    return ttl


def create_static_ttl(level: bool, numsteps_x: int, numsteps_y: int,
                      extrasteps_left: int, extrasteps_right: int,
                      dwell_time: float, sample_rate: float) -> np.ndarray:
    '''
    create a static TTL signal from a mask to scan

    level: static level for the signal as a boolean
    numsteps_x, numsteps_y, extrasteps_left, extrasteps_right: galvo scanning params
    dwell_time: time in us spent per pixel
    sample_rate: DAQ sample rate for output task in MHz (samples per pixel = dwell_time * sample_rate)
    '''

    total_x = numsteps_x + extrasteps_left + extrasteps_right
    total_y = numsteps_y
    pixel_samples = dwell_time * sample_rate

    ttl = np.full(total_x * total_y * pixel_samples, level, dtype=bool)
    return ttl


def process_ai_read(acq_data: np.ndarray, num_ai_channels: int,
                    total_y: int, total_x: int, pixel_samples: int,
                    extra_left: int, numsteps_x: int) -> List[np.ndarray]:
    '''
    reshape [samples] or [channels, samples] to per-channel images with cropping
    '''
    results: List[np.ndarray] = []
    for i in range(num_ai_channels):
        channel_data = acq_data if num_ai_channels == 1 else acq_data[i]
        reshaped = channel_data.reshape(total_y, total_x, pixel_samples)
        pixel_values = np.mean(reshaped, axis=2)
        cropped = pixel_values[:, extra_left:extra_left + numsteps_x]
        results.append(cropped)
    return results


def mosaic_stage_steps(numtiles_x: int, numtiles_y: int, numtiles_z: int,
                       tile_size_x: int, tile_size_y: int, tile_size_z: float) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    '''
    compute stage step sequence and tile indices for mosaic scans
    '''
    steps: List[Tuple[int, int, int]] = []
    indices: List[Tuple[int, int, int]] = []
    for z_idx in range(int(numtiles_z)):
        for y_idx in range(int(numtiles_y)):
            for x_idx in range(int(numtiles_x)):
                if x_idx == 0 and y_idx == 0 and z_idx == 0:
                    x_step = 0
                    y_step = 0
                    z_step = 0
                else:
                    if x_idx > 0:
                        x_step = int(tile_size_x)
                        y_step = 0
                        z_step = 0
                    elif y_idx > 0:
                        x_step = -x_idx * int(tile_size_x)
                        y_step = int(tile_size_y)
                        z_step = 0
                    elif z_idx > 0:
                        x_step = -x_idx * int(tile_size_x)
                        y_step = -y_idx * int(tile_size_y)
                        z_step = int(tile_size_z * 10)
                    else:
                        x_step = 0
                        y_step = 0
                        z_step = 0
                steps.append((x_step, y_step, z_step))
                indices.append((x_idx, y_idx, z_idx))
    return steps, indices
