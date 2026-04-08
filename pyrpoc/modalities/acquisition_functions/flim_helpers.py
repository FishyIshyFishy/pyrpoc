from __future__ import annotations

import time

import numpy as np


def generate_pixel_clock_signal(
    total_x: int,
    y_pixels: int,
    pixel_samples: int,
) -> np.ndarray:
    """Return a 1D bool array with True at the first sample of each pixel.

    The array is suitable for output on a DAQ digital line sample-clocked to
    the AO clock so that the TimeTagger receives one rising edge per pixel.
    """
    n_total = total_x * y_pixels * pixel_samples
    signal = np.zeros(n_total, dtype=bool)
    signal[::pixel_samples] = True
    return signal

# create tagger
# creat
def collect_timetagger_data(
        stream: object,
):
    try:
        polling = True
        while polling:
            data = stream.getData() #pyright:ignore

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

def reshape_timetagger_data():
    ...


def poll_one_flim_frame(
    stream: object,
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    pixel_dwell_ps: int,
    laser_ch: int,
    detector_ch: int,
    pixel_ch: int,
    final_pixel_margin_s: float = 1e-3,
    poll_sleep_s: float = 1e-4,
) -> np.ndarray:
    """Poll a running TimeTagStream until one complete FLIM frame is collected.

    This is the single-frame inner loop extracted from acquire_raw_flim in
    swabian_raw_flim.py. The stream must already be created and started by
    the caller (via TimeTaggerInstrument.create_flim_stream / stream.start()).

    Returns a (y_pixels, x_pixels) object array where each cell is an int64
    array of photon arrival delays in picoseconds relative to the preceding
    laser pulse.
    """
    total_x_pixels = int(x_pixels) + int(extra_left) + int(extra_right)
    total_pixels = total_x_pixels * int(y_pixels)
    frame_duration_ps = total_pixels * int(pixel_dwell_ps)

    def finalize_frame(pixel_lists):
        frame = np.empty((y_pixels, total_x_pixels), dtype=object)
        for flat_index, delays in enumerate(pixel_lists):
            iy, ix = divmod(flat_index, total_x_pixels)
            frame[iy, ix] = np.asarray(delays, dtype=np.int64)
        return frame[:, extra_left : extra_left + x_pixels]

    armed = False
    pixel_lists = None
    frame_start_ps = None
    frame_stop_ps = None
    last_laser_time_ps = None
    finalize_deadline = None

    while True:
        data = stream.getData()  # type: ignore[attr-defined]

        if data.size > 0:
            event_types = data.getEventTypes()
            valid_mask = event_types == 0

            if np.any(valid_mask):
                timestamps = data.getTimestamps()[valid_mask].astype(np.int64, copy=False)
                channels = data.getChannels()[valid_mask].astype(np.int64, copy=False)

                if not armed:
                    pixel_positions = np.flatnonzero(channels == pixel_ch)
                    if pixel_positions.size:
                        arm_pos = int(pixel_positions[0])
                        pre_arm_lasers = timestamps[:arm_pos][channels[:arm_pos] == laser_ch]
                        if pre_arm_lasers.size:
                            last_laser_time_ps = int(pre_arm_lasers[-1])

                        timestamps = timestamps[arm_pos:]
                        channels = channels[arm_pos:]
                        armed = True
                        pixel_lists = [[] for _ in range(total_pixels)]
                        frame_start_ps = int(timestamps[0])
                        frame_stop_ps = frame_start_ps + frame_duration_ps
                        finalize_deadline = time.monotonic() + (
                            frame_duration_ps / 1e12
                        ) + final_pixel_margin_s

                if armed and timestamps.size:
                    laser_ts = timestamps[channels == laser_ch]
                    detector_ts = timestamps[channels == detector_ch]

                    if frame_stop_ps is not None:
                        laser_ts = laser_ts[laser_ts < frame_stop_ps]
                        detector_ts = detector_ts[detector_ts < frame_stop_ps]

                    if detector_ts.size:
                        if laser_ts.size:
                            if last_laser_time_ps is None:
                                laser_reference = laser_ts
                            else:
                                laser_reference = np.empty(laser_ts.size + 1, dtype=np.int64)
                                laser_reference[0] = last_laser_time_ps
                                laser_reference[1:] = laser_ts

                            laser_indices = np.searchsorted(laser_reference, detector_ts, side="right") - 1
                            valid_det_mask = laser_indices >= 0
                            valid_det_ts = detector_ts[valid_det_mask]
                            delays_ps = valid_det_ts - laser_reference[laser_indices[valid_det_mask]]
                        elif last_laser_time_ps is not None:
                            valid_det_ts = detector_ts
                            delays_ps = detector_ts - last_laser_time_ps
                            valid_det_mask = np.ones(detector_ts.shape, dtype=bool)
                        else:
                            valid_det_ts = detector_ts[:0]
                            delays_ps = detector_ts[:0]
                            valid_det_mask = np.zeros(detector_ts.shape, dtype=bool)

                        if valid_det_ts.size:
                            pixel_indices = (
                                (valid_det_ts - frame_start_ps) // int(pixel_dwell_ps)
                            ).astype(np.int64, copy=False)
                            in_frame = (pixel_indices >= 0) & (pixel_indices < total_pixels)

                            if np.any(in_frame):
                                pix = pixel_indices[in_frame]
                                dlys = delays_ps[in_frame]
                                order = np.argsort(pix, kind="stable")
                                pix = pix[order]
                                dlys = dlys[order]
                                splits = np.flatnonzero(np.diff(pix)) + 1
                                for pix_grp, dly_grp in zip(
                                    np.split(pix, splits),
                                    np.split(dlys, splits),
                                ):
                                    pixel_lists[int(pix_grp[0])].extend(dly_grp.tolist())

                    if laser_ts.size:
                        last_laser_time_ps = int(laser_ts[-1])

                    crossed = bool(np.any(timestamps >= frame_stop_ps))
                    if crossed or (finalize_deadline is not None and time.monotonic() >= finalize_deadline):
                        return finalize_frame(pixel_lists)

        if armed and finalize_deadline is not None and time.monotonic() >= finalize_deadline:
            return finalize_frame(pixel_lists)

        if data.size == 0:
            time.sleep(poll_sleep_s)
