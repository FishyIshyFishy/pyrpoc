from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass

@dataclass
class FLIMConfig:
    """Container for acquisition settings to simplify function signatures."""
    x_pixels: int
    y_pixels: int
    extra_left: int
    extra_right: int
    pixel_dwell_ps: int
    laser_ch: int
    detector_ch: int
    pixel_ch: int
    stream_buffer_size: int = 4_000_000
    
    @property
    def total_x_pixels(self) -> int:
        return self.x_pixels + self.extra_left + self.extra_right

    @property
    def total_pixels(self) -> int:
        return self.total_x_pixels * self.y_pixels

    @property
    def frame_duration_ps(self) -> int:
        return self.total_pixels * self.pixel_dwell_ps


def collect_timetagger_data(tagger, config: FLIMConfig, n_frames: int):
    """
    Module 1: Hardware Interaction.
    Handles the stream, polling loop, and basic event filtering.
    """
    from Swabian import TimeTagger
    
    stream = TimeTagger.TimeTagStream(
        tagger=tagger,
        n_max_events=config.stream_buffer_size,
        channels=[config.laser_ch, config.detector_ch, config.pixel_ch]
    )
    
    all_frames_data = []
    frames_captured = 0
    stream.start()

    print(f"Acquisition started. Waiting for {n_frames} frames...")

    try:
        while frames_captured < n_frames:
            data = stream.getData()
            if data.size == 0:
                time.sleep(0.001)
                continue

            # Basic filtering for valid time tags
            event_types = data.getEventTypes()
            valid_mask = event_types == 0
            
            if np.any(valid_mask):
                ts = data.getTimestamps()[valid_mask].astype(np.int64)
                ch = data.getChannels()[valid_mask].astype(np.int64)
                
                # In a real production scenario, you'd handle the 'arming' logic 
                # here to split continuous stream data into discrete frame chunks.
                # For this version, we pass the raw chunk to the processor.
                all_frames_data.append((ts, ch))
                
                # Logic to increment frames_captured based on pixel_ch tags
                frames_captured += np.count_nonzero(ch == config.pixel_ch)

    finally:
        stream.stop()
    
    return all_frames_data


def process_photon_lifetimes(timestamps: np.ndarray, channels: np.ndarray, config: FLIMConfig, last_laser_ps: int | None):
    """
    Module 2: The Math Engine.
    Calculates Δt (lifetime) and maps photons to 1D pixel indices.
    """
    laser_ts = timestamps[channels == config.laser_ch]
    detector_ts = timestamps[channels == config.detector_ch]
    
    if detector_ts.size == 0:
        return np.array([]), np.array([]), last_laser_ps

    # Build reference laser array
    if last_laser_ps is not None:
        laser_ref = np.concatenate(([last_laser_ps], laser_ts))
    else:
        laser_ref = laser_ts

    if laser_ref.size == 0:
        return np.array([]), np.array([]), None

    # Find preceding laser pulse for each photon
    indices = np.searchsorted(laser_ref, detector_ts, side="right") - 1
    valid_mask = indices >= 0
    
    valid_det_ts = detector_ts[valid_mask]
    lifetimes = valid_det_ts - laser_ref[indices[valid_mask]]
    
    # Map to pixel index (relative to start of chunk)
    # Note: frame_start_ps would ideally be passed here for perfect sync
    pixel_indices = (valid_det_ts - timestamps[0]) // config.pixel_dwell_ps
    
    new_last_laser = int(laser_ts[-1]) if laser_ts.size > 0 else last_laser_ps
    
    return lifetimes, pixel_indices, new_last_laser


def reshape_to_frame(lifetimes: np.ndarray, pixel_indices: np.ndarray, config: FLIMConfig):
    """
    Module 3: Geometry & Clipping.
    Groups photon list into 2D image and removes overscan.
    """
    # Create empty list-of-lists for the full scan area (including extra margins)
    pixel_lists = [[] for _ in range(config.total_pixels)]
    
    # Filter for photons that actually fall within the frame bounds
    in_bounds = (pixel_indices >= 0) & (pixel_indices < config.total_pixels)
    active_indices = pixel_indices[in_bounds].astype(int)
    active_lifetimes = lifetimes[in_bounds]

    # Populate the lists
    for idx, lt in zip(active_indices, active_lifetimes):
        pixel_lists[idx].append(lt)

    # Convert to 2D object array (Y, X)
    full_frame = np.empty((config.y_pixels, config.total_x_pixels), dtype=object)
    for i, delays in enumerate(pixel_lists):
        y, x = divmod(i, config.total_x_pixels)
        full_frame[y, x] = np.array(delays, dtype=np.int64)

    # CLIP: Remove extra_left and extra_right to return the requested ROI
    return full_frame[:, config.extra_left : config.extra_left + config.x_pixels]