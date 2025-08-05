import numpy as np
import cv2
from .base_acquisition import Acquisition
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType
from scipy.ndimage import zoom

class LocalRPOC(Acquisition):   
    def __init__(self, galvo=None, mask_data=None, treatment_parameters=None, signal_bus=None, **kwargs):
        super().__init__(**kwargs)
        self.galvo = galvo
        self.mask_data = mask_data
        self.treatment_parameters = treatment_parameters or {}
        self.signal_bus = signal_bus
        self.verified = False
        
        # Extract parameters from treatment_parameters (not acquisition_parameters)
        self.dwell_time = self.treatment_parameters.get('dwell_time', 10)
        self.extrasteps_left = self.treatment_parameters.get('extrasteps_left', 50)
        self.extrasteps_right = self.treatment_parameters.get('extrasteps_right', 50)
        self.amplitude_x = self.treatment_parameters.get('amplitude_x', 0.5)
        self.amplitude_y = self.treatment_parameters.get('amplitude_y', 0.5)
        self.offset_x = self.treatment_parameters.get('offset_x', 0.0)
        self.offset_y = self.treatment_parameters.get('offset_y', 0.0)
        self.x_pixels = self.treatment_parameters.get('x_pixels', 512)
        self.y_pixels = self.treatment_parameters.get('y_pixels', 512)
        self.offset_drift_x = self.treatment_parameters.get('offset_drift_x', 0.0)
        self.offset_drift_y = self.treatment_parameters.get('offset_drift_y', 0.0)
        self.repetitions = self.treatment_parameters.get('repetitions', 1)
        
        # Calculate treatment region from mask
        self.treatment_region = self._calculate_treatment_region()
        
    def _calculate_treatment_region(self):
        '''get a bounding box for the region'''
        if self.mask_data is None:
            return None
            
        # Find contours in the mask
        contours, _ = cv2.findContours(self.mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the bounding box that encompasses all contours
        x_min, y_min = self.x_pixels, self.y_pixels
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            
        return (x_min, y_min, x_max, y_max)
    
    def _calculate_scan_parameters(self):
        if self.treatment_region is None:
            return None
            
        x_min, y_min, x_max, y_max = self.treatment_region
        
        region_width = x_max - x_min
        region_height = y_max - y_min
        
        voltage_per_pixel_x = self.amplitude_x / self.x_pixels
        voltage_per_pixel_y = self.amplitude_y / self.y_pixels

        center_x_pixel = (x_min + x_max) / 2.0
        center_y_pixel = (y_min + y_max) / 2.0
        
        center_x_voltage = (center_x_pixel - self.x_pixels / 2.0) * voltage_per_pixel_x
        center_y_voltage = (center_y_pixel - self.y_pixels / 2.0) * voltage_per_pixel_y

        final_offset_x = self.offset_x + self.offset_drift_x
        final_offset_y = self.offset_y + self.offset_drift_y
        
        # Calculate region-specific amplitude
        region_amplitude_x = region_width * voltage_per_pixel_x
        region_amplitude_y = region_height * voltage_per_pixel_y
        
        return {
            'amplitude_x': region_amplitude_x,
            'amplitude_y': region_amplitude_y,
            'offset_x': final_offset_x + center_x_voltage,
            'offset_y': final_offset_y + center_y_voltage,
            'region_width': region_width,
            'region_height': region_height,
            'region_center_x': center_x_pixel,
            'region_center_y': center_y_pixel
        }
    
    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass
    
    def perform_acquisition(self):
        if self.signal_bus:
            self.signal_bus.console_message.emit("Starting local RPOC treatment...")
        
        # calculate scan parameters
        scan_params = self._calculate_scan_parameters()
        if scan_params is None:
            if self.signal_bus:
                self.signal_bus.console_message.emit("Error: No valid treatment region found")
            return None
        
        if self.signal_bus:
            self.signal_bus.console_message.emit(f"Treatment region: {scan_params['region_width']}x{scan_params['region_height']} pixels")
            self.signal_bus.console_message.emit(f"Center: ({scan_params['region_center_x']:.1f}, {scan_params['region_center_y']:.1f})")
        
        # Perform treatment for specified number of repetitions
        for rep in range(self.repetitions):
            if self._stop_flag and self._stop_flag():
                if self.signal_bus:
                    self.signal_bus.console_message.emit("Local RPOC treatment stopped")
                break
                
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Treatment repetition {rep + 1}/{self.repetitions}")
            
            # Perform the actual treatment scan
            self._perform_treatment_scan(scan_params)
            
            # Emit progress signal after treatment scan completes
            if self.signal_bus:
                self.signal_bus.local_rpoc_progress.emit(rep + 1)
            
            if rep < self.repetitions - 1:  # Don't wait after the last repetition
                time.sleep(0.1)  # Brief pause between repetitions
        
        if self.signal_bus:
            self.signal_bus.console_message.emit("Local RPOC treatment completed")
        
        return None  # No data to return
    
    def _perform_treatment_scan(self, scan_params):       
        try:
            rate = self.galvo.parameters.get('sample_rate', 1000000)
            dwell_time_sec = self.dwell_time / 1e6  # convert to seconds
            pixel_samples = max(1, int(dwell_time_sec * rate))
            slow_channel = self.galvo.parameters.get('slow_axis_channel', 0)
            fast_channel = self.galvo.parameters.get('fast_axis_channel', 1)
            device_name = self.galvo.parameters.get('device_name', 'Dev1')
            
            voltage_per_pixel_x = self.amplitude_x / self.x_pixels
            voltage_per_pixel_y = self.amplitude_y / self.y_pixels
            
            center_x = self.offset_x + self.offset_drift_x
            center_y = self.offset_y + self.offset_drift_y
            
            treatment_waveform, treatment_ttl = self._generate_treatment_waveform(
                center_x, center_y, voltage_per_pixel_x, voltage_per_pixel_y,
                pixel_samples, rate
            )
            
            if treatment_waveform is None:
                if self.signal_bus:
                    self.signal_bus.console_message.emit("Error: No valid treatment regions found in mask")
                return
            
                        # Send signals to galvo (similar to confocal.py)
            with nidaqmx.Task() as ao_task:                
                ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                
                # Configure timing for analog output
                ao_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=treatment_waveform.shape[1]
                )
                
                # exported sample clock for the digital task
                # there is no AI task here so i have to explicitly wire the DO task to the AO task
                # DO tasks (apparently) do not have any implicit timing
                ao_task.export_signals.samp_clk_output_term = f"/{device_name}/PFI0"

                do_task = None
                if treatment_ttl is not None:
                    try:
                        do_task = nidaqmx.Task()

                        ttl_device = self.treatment_parameters.get('ttl_device', device_name)
                        ttl_port_line = self.treatment_parameters.get('ttl_port_line', 'port0/line0')
                        ttl_channel = f"{ttl_device}/{ttl_port_line}"
                        
                        if self.signal_bus:
                            self.signal_bus.console_message.emit(f"Debug: Configuring TTL channel: {ttl_channel}")
                        
                        do_task.do_channels.add_do_chan(ttl_channel)
                        do_task.timing.cfg_samp_clk_timing(
                            rate=rate,
                            source=f"/{device_name}/PFI0",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=treatment_waveform.shape[1]
                        )
                        do_task.write(treatment_ttl.tolist(), auto_start=False)
                        
                        if self.signal_bus:
                            self.signal_bus.console_message.emit(f"TTL output configured on {ttl_channel}")
                    except Exception as e:
                        if self.signal_bus:
                            self.signal_bus.console_message.emit(f"Warning: Could not set up TTL output: {e}")
                        if do_task:
                            try:
                                do_task.close()
                            except:
                                pass
                        do_task = None

                ao_task.write(treatment_waveform, auto_start=False)

                if do_task:
                    do_task.start()
                ao_task.start()
                

                timeout = treatment_waveform.shape[1] / rate + 5
                ao_task.wait_until_done(timeout=timeout)
                if do_task:
                    do_task.wait_until_done(timeout=timeout)

                if do_task:
                    try:
                        do_task.close()
                    except:
                        pass
                
            if self.signal_bus:
                self.signal_bus.console_message.emit("Treatment scan completed successfully")
                
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error during treatment scan: {e}")
            time.sleep(1)
    
    def _generate_treatment_waveform(self, center_x, center_y, voltage_per_pixel_x, voltage_per_pixel_y, pixel_samples, rate):
        if self.mask_data is None:
            return None, None
        
        # Ensure mask is the right size
        mask = self.mask_data
        if mask.shape != (self.y_pixels, self.x_pixels):
            # Resize mask to match pixel dimensions
            zoom_factors = (self.y_pixels / mask.shape[0], self.x_pixels / mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0).astype(bool)
        
        mask = mask > 0  # Ensure boolean mask
        
        # Find active rows (rows with any mask pixels)
        active_rows = np.any(mask, axis=1)
        if not np.any(active_rows):
            return None, None
        
        # Generate waveforms for each active row
        x_waveforms = []
        y_waveforms = []
        ttl_signals = []
        
        for row_idx in range(self.y_pixels):
            if not active_rows[row_idx]:
                continue  # Skip rows with no mask pixels
            
            # Find leftmost and rightmost mask pixels in this row
            row_mask = mask[row_idx, :]
            if not np.any(row_mask):
                continue
            
            leftmost_pixel = np.where(row_mask)[0][0]
            rightmost_pixel = np.where(row_mask)[0][-1]
            
            # Calculate voltage positions
            leftmost_voltage = center_x + (leftmost_pixel - self.x_pixels/2) * voltage_per_pixel_x
            rightmost_voltage = center_x + (rightmost_pixel - self.x_pixels/2) * voltage_per_pixel_x
            
            # Add extra steps
            extra_left_voltage = self.extrasteps_left * voltage_per_pixel_x
            extra_right_voltage = self.extrasteps_right * voltage_per_pixel_x
            
            start_voltage = leftmost_voltage - extra_left_voltage
            stop_voltage = rightmost_voltage + extra_right_voltage
            
            # Calculate number of steps for this row
            row_pixels = rightmost_pixel - leftmost_pixel + 1 + self.extrasteps_left + self.extrasteps_right
            row_samples = row_pixels * pixel_samples
            
            # Generate X waveform for this row
            x_waveform = np.linspace(start_voltage, stop_voltage, row_samples, endpoint=False)
            x_waveforms.append(x_waveform)
            
            # Generate Y position for this row
            y_position = center_y + (row_idx - self.y_pixels/2) * voltage_per_pixel_y
            y_waveform = np.full(row_samples, y_position)
            y_waveforms.append(y_waveform)
            
            # Generate TTL signal for this row
            # TTL is high only for the actual mask pixels, not extra steps
            ttl_row = np.zeros(row_samples, dtype=bool)
            
            # Calculate indices for TTL high regions
            extra_left_samples = self.extrasteps_left * pixel_samples
            mask_start_sample = extra_left_samples
            mask_end_sample = extra_left_samples + (rightmost_pixel - leftmost_pixel + 1) * pixel_samples
            
            # Set TTL high for mask pixels only
            ttl_row[mask_start_sample:mask_end_sample] = True
            
            ttl_signals.append(ttl_row)
        
        if not x_waveforms:
            return None, None
        
        # Concatenate all waveforms
        x_composite = np.concatenate(x_waveforms)
        y_composite = np.concatenate(y_waveforms)
        ttl_composite = np.concatenate(ttl_signals)
        
        # Create final waveform
        treatment_waveform = np.vstack([x_composite, y_composite])
        
        if self.signal_bus:
            total_samples = treatment_waveform.shape[1]
            total_time = total_samples / rate
            self.signal_bus.console_message.emit(f"Treatment waveform: {len(x_waveforms)} active rows, {total_samples} samples, {total_time:.3f}s")
        
        return treatment_waveform, ttl_composite
    
    def save_data(self, data):
        """
        Local RPOC doesn't save data - this is a placeholder
        """
        pass
