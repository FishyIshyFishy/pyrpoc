import numpy as np
import nidaqmx
import abc
from pyrpoc.instruments.instrument_manager import *
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType
import tifffile
from pathlib import Path
from .base_acquisition import Acquisition
from datetime import datetime
from PIL import Image


class Confocal(Acquisition):
    def __init__(self, galvo=None, data_inputs=None, num_frames=1, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.galvo = galvo
        self.data_inputs = data_inputs or []
        self.num_frames = num_frames
        self.signal_bus = signal_bus
        self.verified = False
        self.acquisition_parameters = acquisition_parameters or {}
        
        self.rpoc_enabled = False
        self.rpoc_mask_channels = {}
        self.rpoc_static_channels = {}
        self.rpoc_script_channels = {}
        self.rpoc_ttl_signals = {}  # channel_id -> flat TTL array

    def configure_rpoc(self, rpoc_enabled, rpoc_mask_channels=None, rpoc_static_channels=None, rpoc_script_channels=None, **kwargs):
        self.rpoc_enabled = rpoc_enabled
        self.rpoc_mask_channels = rpoc_mask_channels or {}
        self.rpoc_static_channels = rpoc_static_channels or {}
        self.rpoc_script_channels = rpoc_script_channels or {}
        self.rpoc_ttl_signals = {}  # channel_id -> flat TTL array

        # Get acquisition parameters
        dwell_time = self.acquisition_parameters.get('dwell_time', 10)  # microseconds
        rate = self.galvo.parameters.get('sample_rate', 1000000)
        dwell_time_sec = dwell_time / 1e6
        pixel_samples = max(1, int(dwell_time_sec * rate))
        numsteps_x = self.acquisition_parameters.get('x_pixels', 512)
        numsteps_y = self.acquisition_parameters.get('y_pixels', 512)
        extra_left = self.acquisition_parameters.get('extrasteps_left', 50)
        extra_right = self.acquisition_parameters.get('extrasteps_right', 50)
        total_x = numsteps_x + extra_left + extra_right
        total_y = numsteps_y

        # Handle mask-based channels
        for channel_id, channel_data in (self.rpoc_mask_channels or {}).items():
            mask = channel_data.get('mask_data')
            if mask is None:
                continue
            # Prepare mask
            if isinstance(mask, np.ndarray):
                mask_array = mask
            else:
                try:
                    from PIL import Image
                    if isinstance(mask, Image.Image):
                        mask_array = np.array(mask)
                    else:
                        mask_array = mask
                except:
                    mask_array = mask
            mask_array = mask_array > 0
            # Pad mask to match scan size
            padded_mask = []
            for row in range(numsteps_y):
                padded_row = np.concatenate((
                    np.zeros(extra_left, dtype=bool),
                    mask_array[row, :] if row < mask_array.shape[0] else np.zeros(numsteps_x, dtype=bool),
                    np.zeros(extra_right, dtype=bool)
                ))
                padded_mask.append(padded_row)
            padded_mask = np.array(padded_mask)
            # For each pixel, create TTL: high for all samples if mask is high, else all low
            ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
            for y in range(total_y):
                for x in range(total_x):
                    if padded_mask[y, x]:
                        ttl[y, x, :] = True
            flat_ttl = ttl.ravel()
            self.rpoc_ttl_signals[channel_id] = flat_ttl

        # Handle static channels
        for channel_id, channel_data in (self.rpoc_static_channels or {}).items():
            level = channel_data.get('level', 'Static Low').lower()
            # All high or all low
            value = True if 'high' in level else False
            flat_ttl = np.full(total_x * total_y * pixel_samples, value, dtype=bool)
            self.rpoc_ttl_signals[channel_id] = flat_ttl

        if self.signal_bus:
            n_masks = len(self.rpoc_mask_channels) if self.rpoc_mask_channels else 0
            n_static = len(self.rpoc_static_channels) if self.rpoc_static_channels else 0
            n_script = len(self.rpoc_script_channels) if self.rpoc_script_channels else 0
            n_total = len(self.rpoc_ttl_signals)
            self.signal_bus.console_message.emit(f"RPOC Configured - enabled: {rpoc_enabled}, masks: {n_masks}, static: {n_static}, script: {n_script}, total: {n_total}")

    def perform_acquisition(self):     
        self.save_metadata()

        ai_channels = []
        for data_input in self.data_inputs:
            channels = data_input.parameters.get('input_channels', [])
            device = data_input.parameters.get('device_name', 'Dev1')
            for ch in channels:
                ai_channels.append(f"{device}/ai{ch}")
        
        if not ai_channels:
            ai_channels = ["Dev1/ai0"] 

        all_frames = []
        for frame_idx in range(self.num_frames):
            if self._stop_flag and self._stop_flag():
                break
            
            # frame_data = self.generate_simulated_confocal()
            frame_data = self.collect_data(self.galvo, ai_channels)
            
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame_idx, self.num_frames, False)
            all_frames.append(frame_data)
        
        if all_frames:
            final_data = np.stack(all_frames)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(all_frames)-1, self.num_frames, True)
            self.save_data(final_data)
            return final_data
        else:
            return None
    
    def generate_simulated_confocal(self):
        x_pixels = self.acquisition_parameters.get('x_pixels', 512)
        y_pixels = self.acquisition_parameters.get('y_pixels', 512)
        
        num_channels = 1
        if self.data_inputs:
            total_channels = 0
            for data_input in self.data_inputs:
                channels = data_input.parameters.get('input_channels', [])
                total_channels += len(channels)
            num_channels = max(1, total_channels)

        frame = np.zeros((num_channels, y_pixels, x_pixels))
        
        for ch in range(num_channels):
            num_circles = np.random.randint(2, 6)
            for _ in range(num_circles):
                center_x = np.random.randint(50, x_pixels - 50)
                center_y = np.random.randint(50, y_pixels - 50)
                radius = np.random.randint(10, 30)
                intensity = np.random.uniform(0.3, 1.0)
                
                y, x = np.ogrid[:y_pixels, :x_pixels]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                frame[ch, mask] += intensity

            frame[ch] += np.random.normal(0, 0.1, frame[ch].shape)
            frame[ch] = np.clip(frame[ch], 0, 1)

        if self.signal_bus:
            n_masks = len(self.rpoc_mask_channels) if self.rpoc_mask_channels else 0
            n_static = len(self.rpoc_static_channels) if self.rpoc_static_channels else 0
            n_script = len(self.rpoc_script_channels) if self.rpoc_script_channels else 0
            self.signal_bus.console_message.emit(f"RPOC Debug - enabled: {self.rpoc_enabled}, masks: {n_masks}, static: {n_static}, script: {n_script}")
        
        if self.rpoc_enabled and (self.rpoc_mask_channels or self.rpoc_static_channels):
            # Handle mask channels
            if self.rpoc_mask_channels:
                combined_mask = np.zeros((y_pixels, x_pixels), dtype=bool)
                
                for channel_id, channel_data in self.rpoc_mask_channels.items():
                    mask = channel_data.get('mask_data')
                    if mask is None:
                        continue
                    if isinstance(mask, np.ndarray):
                        mask_array = mask
                    else:
                        try:
                            if isinstance(mask, Image.Image):
                                mask_array = np.array(mask)
                            else:
                                mask_array = mask
                        except:
                            mask_array = mask
                    
                    mask_array = mask_array > 0
                    if mask_array.shape != (y_pixels, x_pixels):
                        from scipy.ndimage import zoom
                        zoom_factors = (y_pixels / mask_array.shape[0], x_pixels / mask_array.shape[1])
                        mask_array = zoom(mask_array, zoom_factors, order=0).astype(bool)
                    
                    combined_mask |= mask_array
                
                for ch in range(num_channels):
                    frame[ch, combined_mask] = 0
            
            # Handle static channels (simulation only shows they're configured)
            if self.rpoc_static_channels:
                for ch in range(num_channels):
                    if np.max(frame[ch]) > 0:
                        frame[ch] = frame[ch] / np.max(frame[ch])
            
            if self.signal_bus:
                rpoc_channels_info = []
                for channel_id, channel_data in self.rpoc_mask_channels.items():
                    device = channel_data.get('device', 'Dev1')
                    port_line = channel_data.get('port_line', f'port0/line{4+channel_id-1}')
                    rpoc_channels_info.append(f"Channel {channel_id}: {device}/{port_line}")
                
                static_channels_info = []
                for channel_id, channel_data in self.rpoc_static_channels.items():
                    device = channel_data.get('device', 'Dev1')
                    port_line = channel_data.get('port_line', f'port0/line{4+channel_id-1}')
                    level = channel_data.get('level', 'Static Low')
                    static_channels_info.append(f"Channel {channel_id}: {device}/{port_line} ({level})")
                
                total_info = rpoc_channels_info + static_channels_info
                self.signal_bus.console_message.emit(f"RPOC Active - {len(self.rpoc_mask_channels)} masks, {len(self.rpoc_static_channels)} static channels: {', '.join(total_info)}")

        time.sleep(1)
        
        return frame
    
    def collect_data(self, galvo, ai_channels):
        try:
            rate = galvo.parameters.get('sample_rate', 1000000)
            dwell_time = self.acquisition_parameters.get('dwell_time', 10)  # microseconds
            dwell_time_sec = dwell_time / 1e6  # convert to seconds
            extra_left = self.acquisition_parameters.get('extrasteps_left', 50)
            extra_right = self.acquisition_parameters.get('extrasteps_right', 50)
            numsteps_x = self.acquisition_parameters.get('x_pixels', 512)
            numsteps_y = self.acquisition_parameters.get('y_pixels', 512)
            slow_channel = galvo.parameters.get('slow_axis_channel', 0)
            fast_channel = galvo.parameters.get('fast_axis_channel', 1)
            device_name = galvo.parameters.get('device_name', 'Dev1')

            pixel_samples = max(1, int(dwell_time_sec * rate))
            total_x = numsteps_x + extra_left + extra_right
            total_y = numsteps_y
            total_samples = total_x * total_y * pixel_samples

            waveform = galvo.generate_raster_waveform(self.acquisition_parameters)

            rpoc_do_channels = []
            rpoc_ttl_signals = []

            if self.rpoc_enabled and self.rpoc_ttl_signals and (self.rpoc_mask_channels or self.rpoc_static_channels or self.rpoc_script_channels):
                for channel_id, flat_ttl in self.rpoc_ttl_signals.items():
                    # Find the channel info from the appropriate storage
                    channel_info = None
                    if channel_id in self.rpoc_mask_channels:
                        channel_info = self.rpoc_mask_channels[channel_id]
                    elif channel_id in self.rpoc_static_channels:
                        channel_info = self.rpoc_static_channels[channel_id]
                    elif channel_id in self.rpoc_script_channels:
                        channel_info = self.rpoc_script_channels[channel_id]
                    
                    if channel_info:
                        device = channel_info.get('device', device_name)
                        # Convert channel_id to int for arithmetic if it's a string
                        channel_id_int = int(channel_id) if isinstance(channel_id, str) else channel_id
                        port_line = channel_info.get('port_line', f'port0/line{4+channel_id_int-1}')
                        rpoc_do_channels.append(f"{device}/{port_line}")
                        rpoc_ttl_signals.append(flat_ttl)
            
            # Calculate timeout once
            timeout = total_samples / rate + 5
            
            # Separate dynamic (port0) and static (port1+) channels
            dyn_chans = []
            dyn_ttls = []
            stat_chans = []
            stat_vals = []
            
            for chan, flat_ttl in zip(rpoc_do_channels, rpoc_ttl_signals):
                if '/port0/' in chan.lower():
                    # anything on port0 → dynamic, clocked DO
                    dyn_chans.append(chan)
                    dyn_ttls.append(flat_ttl)
                else:
                    # anything else (e.g. port1) → static DO
                    stat_chans.append(chan)
                    # take the first value as constant level
                    stat_vals.append(bool(flat_ttl.flat[0]))

            # -- static, immediate DO task (on port1 or others) --
            if stat_chans:
                with nidaqmx.Task() as static_do:
                    for c in stat_chans:
                        static_do.do_channels.add_do_chan(c)
                    # write a constant level (list of booleans matching each line)
                    # auto_start=True so it drives immediately
                    static_do.write(stat_vals, auto_start=True)

            # -- dynamic, hardware-timed DO task (on port0) --
            if dyn_chans:
                with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
                    # 1) Add AO & AI channels
                    ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                    ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                    for ch in ai_channels:
                        ai_task.ai_channels.add_ai_voltage_chan(ch)
                    
                    # 2) Clock AO
                    ao_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples
                    )

                    print("AO clock terminal is:", ao_task.timing.samp_clk_term)
                    print(f'timing tied to /{device_name}/ao/SampleClock')
                    
                    # 3) Clock AI off of AO's internal clock
                    ai_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        source=f"/{device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples
                    )
                    
                    # 4) Clock DO off of AO's internal clock (AO still open!)
                    for c in dyn_chans:
                        do_task.do_channels.add_do_chan(c)
                    do_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        source=f"/{device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples
                    )
                    
                    # 5) Write waveforms, then start in order:
                    ao_task.write(waveform, auto_start=False)
                    
                    # write the pattern(s)
                    if len(dyn_chans) == 1:
                        do_task.write(dyn_ttls[0].tolist(), auto_start=False)
                    else:
                        data_to_write = [arr.tolist() for arr in dyn_ttls]
                        do_task.write(data_to_write, auto_start=False)
                    
                    # 6) Start in order: AI, AO, DO
                    ai_task.start()
                    do_task.start()
                    ao_task.start()
                    
                    
                    # 7) Wait and tear down all three
                    ao_task.wait_until_done(timeout=timeout)
                    do_task.wait_until_done(timeout=timeout)
                    ai_task.wait_until_done(timeout=timeout)
                    
                    
                    acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samples))
            else:
                # No dynamic DO channels, just AO and AI
                with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                    # 1) Add AO & AI channels
                    ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                    ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                    for ch in ai_channels:
                        ai_task.ai_channels.add_ai_voltage_chan(ch)
                    
                    # 2) Clock AO
                    ao_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples
                    )
                    
                    # 3) Clock AI off of AO's internal clock
                    ai_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        source=f"/{device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples
                    )
                    
                    # 4) Write waveforms, then start in order:
                    ao_task.write(waveform, auto_start=False)
                    
                    # 5) Start in order: AI, AO
                    ai_task.start()
                    ao_task.start()
                    
                    # 6) Wait and tear down
                   
                    ai_task.wait_until_done(timeout=timeout)
                    ao_task.wait_until_done(timeout=timeout)
                    
                    acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samples))
            
            results = []
            for i in range(len(ai_channels)):
                channel_data = acq_data if len(ai_channels) == 1 else acq_data[i]
                reshaped = channel_data.reshape(total_y, total_x, pixel_samples)
                pixel_values = np.mean(reshaped, axis=2)
                cropped = pixel_values[:, extra_left:extra_left + numsteps_x]
                results.append(cropped)
            
            if len(results) == 1:
                return results[0]
            else:
                return np.stack(results)
                    
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f'Error in DAQ acquisition: {e}')
            return self.generate_simulated_confocal()
    
    def save_data(self, data):
        if not self.save_enabled or not self.save_path:
            return
        
        try:
            save_dir = Path(self.save_path).parent
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            if len(data.shape) == 4:
                num_frames, num_channels, height, width = data.shape
            elif len(data.shape) == 3:
                num_frames, height, width = data.shape
                num_channels = 1
                data = data.reshape(num_frames, 1, height, width)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
            
            channel_names = self._get_channel_names()
            
            for ch in range(num_channels):
                channel_data = data[:, ch, :, :]
                
                channel_name = channel_names.get(ch, f"ch{ch:02d}")
                filename = f"{Path(self.save_path).stem}_{channel_name}.tiff"
                filepath = save_dir / filename
                
                tifffile.imwrite(filepath, channel_data)
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving confocal data: {e}")
    
    def _get_channel_names(self):
        channel_names = {}
        channel_idx = 0
        
        if hasattr(self, 'data_inputs') and self.data_inputs:
            for data_input in self.data_inputs:
                if hasattr(data_input, 'parameters'):
                    input_channels = data_input.parameters.get('input_channels', [])
                    channel_names_param = data_input.parameters.get('channel_names', {})
                    
                    for ch in input_channels:
                        channel_name = channel_names_param.get(str(ch), f"ch{ch}")
                        channel_names[channel_idx] = channel_name
                        channel_idx += 1
        
        if not channel_names:
            for ch in range(channel_idx):
                channel_names[ch] = f"ch{ch:02d}"
        
        return channel_names
    


def _write_port1(self, lines, levels):
    """Write a list of boolean levels to the given port1 lines, un-timed."""
    with nidaqmx.Task() as t:
        for ln in lines:
            t.do_channels.add_do_chan(ln)
        t.write(levels, auto_start=True)

def collect_data(self, galvo, ai_channels):
    stat_chans, stat_vals = …        # as before
    # 1) Raise port1 lines before imaging:
    if stat_chans:
        self._write_port1(stat_chans, stat_vals)

    try:
        # … your AO/AI (and port0) timed acquisition block …
        return final_image
    finally:
        # 2) Always clear them after, even if an exception occurred:
        if stat_chans:
            self._write_port1(stat_chans, [False]*len(stat_chans))