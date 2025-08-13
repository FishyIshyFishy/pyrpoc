import numpy as np
import cv2
from pyrpoc.acquisitions.base_acquisition import Acquisition
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType
from scipy.ndimage import zoom, label

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
        
    
    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass
    
    def perform_acquisition(self):
        if self.signal_bus:
            self.signal_bus.console_message.emit("Starting local RPOC treatment...")
        
        for rep in range(self.repetitions):
            if self._stop_flag and self._stop_flag():
                if self.signal_bus:
                    self.signal_bus.console_message.emit("Local RPOC treatment stopped")
                break

            
            # Perform the actual treatment scan
            self._perform_treatment_scan()
            
            # Emit progress signal after treatment scan completes
            if self.signal_bus:
                self.signal_bus.local_rpoc_progress.emit(rep + 1)
            
            if rep < self.repetitions - 1:  # Don't wait after the last repetition
                time.sleep(0.1)  # Brief pause between repetitions
        
        if self.signal_bus:
            self.signal_bus.console_message.emit("Local RPOC treatment completed")
        
        return None  # No data to return
    
    def _perform_treatment_scan(self):       
        try:
            rate = self.galvo.parameters.get('sample_rate', 1000000)
            slow_channel = self.galvo.parameters.get('slow_axis_channel', 0)
            fast_channel = self.galvo.parameters.get('fast_axis_channel', 1)
            device_name = self.galvo.parameters.get('device_name', 'Dev1')
            
            treatment_waveform, treatment_ttl = self._generate_treatment_waveform(rate)
            
            if treatment_waveform is None:
                if self.signal_bus:
                    self.signal_bus.console_message.emit("Error: No valid treatment regions found in mask")
                return
            
            # Send signals to galvo (similar to confocal.py)
            # Calculate timeout once
            timeout = treatment_waveform.shape[1] / rate + 5

            # Check if PFI line is specified for timing
            pfi_line = self.treatment_parameters.get('pfi_line', 'None')
            
            # Configure and start tasks
            if treatment_ttl is not None:
                # With DO task
                with nidaqmx.Task() as ao_task, nidaqmx.Task() as do_task:
                    # 1) Add AO channels
                    ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                    ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                    
                    # 2) Clock AO
                    ao_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=treatment_waveform.shape[1]
                    )

                    # 3) Export AO sample clock to PFI line if specified
                    if pfi_line and pfi_line != 'None':
                        try:
                            ao_task.export_signals.samp_clk_output_term = f"/{device_name}/{pfi_line}"
                            if self.signal_bus:
                                self.signal_bus.console_message.emit(f"Exported AO sample clock to {pfi_line}")
                        except Exception as e:
                            if self.signal_bus:
                                self.signal_bus.console_message.emit(f"Warning: Could not export sample clock to {pfi_line}: {e}")

                    # 4) Configure DO task
                    ttl_device = self.treatment_parameters.get('ttl_device', device_name)
                    ttl_port_line = self.treatment_parameters.get('ttl_port_line', 'port0/line0')
                    ttl_channel = f"{ttl_device}/{ttl_port_line}"
                    
                    if self.signal_bus:
                        self.signal_bus.console_message.emit(f"Debug: Configuring TTL channel: {ttl_channel}")
                    
                    do_task.do_channels.add_do_chan(ttl_channel)
                    
                    # Determine timing source
                    if pfi_line and pfi_line != 'None':
                        # Use PFI line for timing if specified
                        timing_source = f"/{device_name}/{pfi_line}"
                        if self.signal_bus:
                            self.signal_bus.console_message.emit(f"Using PFI line {pfi_line} for DO task timing")
                    else:
                        # Use AO sample clock as default (internal wiring)
                        timing_source = f"/{device_name}/ao/SampleClock"
                        if self.signal_bus:
                            self.signal_bus.console_message.emit("Using AO sample clock for DO task timing (internal wiring)")
                    
                    # 5) Clock DO off of AO's internal clock (AO still open!)
                    do_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        source=timing_source,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=treatment_waveform.shape[1]
                    )
                    
                    # 6) Write waveforms, then start in order:
                    ao_task.write(treatment_waveform, auto_start=False)
                    do_task.write(treatment_ttl.tolist(), auto_start=False)
                    
                    if self.signal_bus:
                        self.signal_bus.console_message.emit(f"TTL output configured on {ttl_channel}")
                    
                    # 7) Start in order: AO, DO
                    do_task.start()
                    ao_task.start()
                    
                    
                    # 8) Wait and tear down both
                    ao_task.wait_until_done(timeout=timeout)
                    do_task.wait_until_done(timeout=timeout)
            else:
                # No DO task, just AO
                with nidaqmx.Task() as ao_task:
                    # 1) Add AO channels
                    ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                    ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                    
                    # 2) Clock AO
                    ao_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=treatment_waveform.shape[1]
                    )
                    
                    # 3) Write waveform and start
                    ao_task.write(treatment_waveform, auto_start=False)
                    ao_task.start()
                    
                    # 4) Wait and tear down
                    ao_task.wait_until_done(timeout=timeout)
                
            if self.signal_bus:
                self.signal_bus.console_message.emit("Treatment scan completed successfully")
                
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error during treatment scan: {e}")
            time.sleep(1)
    
    def _generate_treatment_waveform(self, rate):
        mask = self.mask_data
        if mask.shape != (self.y_pixels, self.x_pixels):
            factors = (self.y_pixels / mask.shape[0],
                    self.x_pixels / mask.shape[1])
            mask = zoom(mask, factors, order=0).astype(bool)
        mask = mask.astype(bool)

        # 2) Ensure exactly one connected region
        comps, ncomp = label(mask)
        if ncomp != 1:
            raise RuntimeError(f"Expected single connected mask region, found {ncomp}")

        # 3) Precompute scales and offsets
        pixel_samples = max(1, int(self.dwell_time / 1e6 * rate))

        # Global offsets (includes drift if any)
        base_x = self.offset_x + getattr(self, "offset_drift_x", 0.0)
        base_y = self.offset_y + getattr(self, "offset_drift_y", 0.0)

        x_wfs, y_wfs, ttls = [], [], []

        # 4) Row-by-row scan
        for row in range(self.y_pixels):
            row_mask = mask[row, :]
            if not row_mask.any():
                continue  

            # pixel indices
            left_px  = np.argmax(row_mask)

            right_px = self.x_pixels - 1 - np.argmax(row_mask[::-1])
            n_mask   = right_px - left_px + 1
            n_scan   = n_mask + self.extrasteps_left + self.extrasteps_right
            total_samps = n_scan * pixel_samples


            step_size = (2 * self.amplitude_x) / self.x_pixels
            
            # base_x is the 0 position
            # base_x - amp_x is the position of the leftmost pixel in volts
            # left_pix*step_size is how far from that position the ROI starts
            # move backwards by extrasteps_left to get to where the scan should start
            v_start = base_x - self.amplitude_x + left_px*step_size - self.extrasteps_left*step_size
            v_stop = base_x - self.amplitude_x + right_px*step_size + self.extrasteps_right*step_size 
            x_row = np.linspace(v_start, v_stop, total_samps, endpoint=False)

            # galvo coordinates go high to low
            y_val = base_y + self.amplitude_y - (row / self.y_pixels) * (2 * self.amplitude_y)
            y_row = np.full(total_samps, y_val)

            # extrasteps need to be 0 in sample space
            ttl = np.zeros(total_samps, dtype=bool)
            start_idx = int(self.extrasteps_left * pixel_samples)
            end_idx = start_idx + int(n_mask * pixel_samples)
            ttl[start_idx:end_idx] = True

            x_wfs.append(x_row)
            y_wfs.append(y_row)
            ttls.append(ttl)

        if not x_wfs:
            raise RuntimeError("No active mask rows found for scan.")

        # 5) Concatenate rows
        X   = np.concatenate(x_wfs)
        Y   = np.concatenate(y_wfs)
        TTL = np.concatenate(ttls)
        waveform = np.vstack((X, Y))   # shape (2, N)

        return waveform, TTL

    
    def save_data(self, data):
        pass

if __name__ == '__main__':
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # 1) Treatment parameters
    treatment_params = {
        'dwell_time':       10,      # μs per pixel
        'extrasteps_left':  10,
        'extrasteps_right': 1,
        'amplitude_x':      1.0,     # total voltage span
        'amplitude_y':      1.0,
        'offset_x':         0.0,
        'offset_y':         0.0,
        'offset_drift_x':   0.0,
        'offset_drift_y':   0.0,
        'x_pixels':         512,
        'y_pixels':         512,
        'repetitions':      1
    }

    # 2) Load + binarize mask
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("mask.png not found.")
    _, mask_bin = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
    mask_bin = mask_bin.astype(bool)

    # 3) Instantiate + generate
    obj = LocalRPOC(
        galvo=None,
        mask_data=mask_bin,
        treatment_parameters=treatment_params,
        signal_bus=None
    )
    sr = 500_000  # Hz
    waveform, ttl = obj._generate_treatment_waveform(sr)

    # 4) Compute voltage‐space extents
    ax = treatment_params['amplitude_x'] / 2
    ay = treatment_params['amplitude_y'] / 2
    ox = treatment_params['offset_x']
    oy = treatment_params['offset_y']
    extent = [ox - ax, ox + ax, oy - ay, oy + ay]

    # 5) Plot
    fig, (ax_mask, ax_path) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: mask in voltage coordinates
    im = ax_mask.imshow(
        mask_bin,
        origin='lower',
        cmap='gray',
        extent=extent,
        aspect='equal'
    )
    ax_mask.set_title('Mask in Voltage Space')
    ax_mask.set_xlabel('X Voltage (V)')
    ax_mask.set_ylabel('Y Voltage (V)')
    fig.colorbar(im, ax=ax_mask, label='Mask')

    # Right: galvo path (X vs Y)
    X = waveform[0]
    Y = waveform[1]
    ax_path.plot(X, Y, lw=0.5, color='C2')
    ax_path.set_aspect('equal')
    ax_path.set_title('Galvo Scan Path')
    ax_path.set_xlabel('X Voltage (V)')
    ax_path.set_ylabel('Y Voltage (V)')

    plt.tight_layout()
    plt.show()