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


class SplitDataStream(Acquisition):
    def __init__(self, galvo=None, data_inputs=None, num_frames=1, split_percentage=50, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.galvo = galvo
        self.data_inputs = data_inputs or []
        self.num_frames = num_frames
        self.split_percentage = split_percentage
        self.signal_bus = signal_bus
        self.verified = False
        self.acquisition_parameters = acquisition_parameters or {}
        
        if self.galvo is None:
            raise ValueError("Galvo is required for SplitDataStream acquisition")
        if not self.data_inputs:
            raise ValueError("At least one DataInput is required for SplitDataStream acquisition")

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

        # Get split percentage and acquisition parameters
        split_percentage = self.acquisition_parameters.get('split_percentage', 50)
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

        split_point = int(split_percentage / 100.0 * pixel_samples)

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
            # For each pixel, create TTL: high for split_point samples, low for the rest if mask is high, else all low
            ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
            for y in range(total_y):
                for x in range(total_x):
                    if padded_mask[y, x]:
                        ttl[y, x, :split_point] = True
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
        
        current_position = 0
        for _ in range(self.num_frames):
            frame_data = self.collect_split_data(self.galvo, ai_channels)
                
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, current_position, self.num_frames, False)
            all_frames.append(frame_data)
            current_position += 1
        
        if all_frames:
            final_data = np.stack(all_frames)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(all_frames)-1, self.num_frames, True)
         
            self.save_data(final_data)
            return final_data
        else:
            return None
    
    def collect_split_data(self, galvo, ai_channels):
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

            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                for ch in ai_channels:
                    ai_task.ai_channels.add_ai_voltage_chan(ch)
                ao_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samples
                )
                ai_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    source=f"/{device_name}/ao/SampleClock",
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samples
                )
                do_task = None
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

                # -- dynamic, hardware-timed DO task (on port0) --
                do_task = None
                if dyn_chans:
                    do_task = nidaqmx.Task()
                    # add all port0 lines as one buffered channel
                    for c in dyn_chans:
                        do_task.do_channels.add_do_chan(c)
                    do_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        source=f"/{device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples
                    )

                    # write the pattern(s)
                    if len(dyn_chans) == 1:
                        do_task.write(dyn_ttls[0].tolist(), auto_start=False)
                    else:
                        data_to_write = [arr.tolist() for arr in dyn_ttls]
                        do_task.write(data_to_write, auto_start=False)

                # -- static, immediate DO task (on port1 or others) --
                static_do = None
                if stat_chans:
                    static_do = nidaqmx.Task()
                    for c in stat_chans:
                        static_do.do_channels.add_do_chan(c)
                    # write a constant level (list of booleans matching each line)
                    # auto_start=True so it drives immediately
                    static_do.write(stat_vals, auto_start=True)
                # now write AO waveform, start AI, start DO, start AO
                ao_task.write(waveform, auto_start=False)
                ai_task.start()

                if do_task:
                    do_task.start()

                ao_task.start()

                timeout = total_samples / rate + 5
                ao_task.wait_until_done(timeout=timeout)
                ai_task.wait_until_done(timeout=timeout)
                if do_task:
                    do_task.wait_until_done(timeout=timeout)

                if static_do:
                    # off_vals = [not v for v in stat_vals]
                    # static_do.write(off_vals, auto_start=True)
                    static_do.write([not v for v in stat_vals])
                    static_do.close()
                    
                
                acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samples))
                
                input_results = []
                split_point = int(self.acquisition_parameters.get('split_percentage', 50) / 100.0 * pixel_samples)
                
                # Calculate AOM delay in samples
                aom_delay_us = self.acquisition_parameters.get('aom_delay', 0)
                aom_delay_samples = max(0, int(aom_delay_us / dwell_time * pixel_samples))
                
                # Validate that we have enough samples for both portions
                split_percentage = self.acquisition_parameters.get('split_percentage', 50)
                if split_point + aom_delay_samples >= pixel_samples:
                    raise ValueError(f"AOM delay too large: {aom_delay_us} µs with {dwell_time} µs dwell time and {split_percentage}% split leaves no samples for second portion")
                
                for i in range(len(ai_channels)):
                    channel_data = acq_data if len(ai_channels) == 1 else acq_data[i]
                    reshaped = channel_data.reshape(total_y, total_x, pixel_samples)
                    
                    # First portion: from start to split_point
                    first_portion = np.mean(reshaped[:, :, :split_point], axis=2)
                    
                    # Second portion: from split_point + aom_delay_samples to end
                    second_start = split_point + aom_delay_samples
                    if second_start < pixel_samples:
                        second_portion = np.mean(reshaped[:, :, second_start:], axis=2)
                    else:
                        # If no samples left, create zero array
                        second_portion = np.zeros_like(first_portion)
                    
                    cropped_first = first_portion[:, extra_left:extra_left + numsteps_x]
                    cropped_second = second_portion[:, extra_left:extra_left + numsteps_x]
                    input_results.append(cropped_first)
                    input_results.append(cropped_second)
                
                # Output shape: (N*2, height, width)
                return np.stack(input_results)
        except Exception as e:
            self.signal_bus.console_message.emit(f'Error in DAQ acquisition: {e}')
            return None
    
    def save_data(self, data):
        if not self.save_enabled or not self.save_path:
            return
        
        try:
            save_dir = Path(self.save_path).parent
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            if len(data.shape) == 4:
                num_total_images, num_channels, height, width = data.shape
            elif len(data.shape) == 3:
                num_total_images, height, width = data.shape
                num_channels = 1
                data = data.reshape(num_total_images, 1, height, width)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
            
            num_input_channels = num_channels // 2
            
            input_channel_names = self._get_input_channel_names()
            
            for input_ch_idx in range(num_input_channels):
                input_channel_name = input_channel_names.get(input_ch_idx, f"input{input_ch_idx}")
                
                first_ch_idx = input_ch_idx * 2
                second_ch_idx = input_ch_idx * 2 + 1
  
                first_data = data[:, first_ch_idx, :, :]
                first_filename = f"{Path(self.save_path).stem}_{input_channel_name}_first.tiff"
                first_filepath = save_dir / first_filename
                tifffile.imwrite(first_filepath, first_data)
                
                second_data = data[:, second_ch_idx, :, :]
                second_filename = f"{Path(self.save_path).stem}_{input_channel_name}_second.tiff"
                second_filepath = save_dir / second_filename
                tifffile.imwrite(second_filepath, second_data)
                
                if self.signal_bus:
                    self.signal_bus.console_message.emit(f"Saved split stream data for {input_channel_name}: first, second, and full channels")
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving split stream data: {e}")
            else:
                print(f"Error saving split stream data: {e}")
    
    def _get_input_channel_names(self):

        input_channel_names = {}
        input_ch_idx = 0
        
        if hasattr(self, 'data_inputs') and self.data_inputs:
            for data_input in self.data_inputs:
                if hasattr(data_input, 'parameters'):
                    input_channels = data_input.parameters.get('input_channels', [])
                    channel_names_param = data_input.parameters.get('channel_names', {})
                    
                    for ch in input_channels:
                        channel_name = channel_names_param.get(str(ch), f"ch{ch}")
                        input_channel_names[input_ch_idx] = channel_name
                        input_ch_idx += 1
        
        if not input_channel_names:
            for ch in range(input_ch_idx):
                input_channel_names[ch] = f"input{ch}"
        
        return input_channel_names