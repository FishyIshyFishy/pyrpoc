import numpy as np
import cv2
from .base_acquisition import Acquisition
import time

class LocalRPOC(Acquisition):
    """
    Local RPOC treatment class that scans only the region defined by a mask
    without collecting any data - only performs treatment laser output
    """
    
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
        """
        Calculate the smallest bounding box for the mask region
        Returns: (x_min, y_min, x_max, y_max) in pixel coordinates
        """
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
    
    def _apply_drift_offset(self, base_offset_x, base_offset_y):
        """
        Apply drift offset to the base offset values
        """
        return base_offset_x + self.offset_drift_x, base_offset_y + self.offset_drift_y
    
    def _calculate_scan_parameters(self):
        """
        Calculate scan parameters for the treatment region
        Returns: scan parameters for the galvo
        """
        if self.treatment_region is None:
            return None
            
        x_min, y_min, x_max, y_max = self.treatment_region
        
        # Calculate region dimensions
        region_width = x_max - x_min
        region_height = y_max - y_min
        
        # Calculate voltage ranges for the region
        # Assuming linear mapping from pixel coordinates to voltage
        voltage_per_pixel_x = self.amplitude_x / self.x_pixels
        voltage_per_pixel_y = self.amplitude_y / self.y_pixels
        
        # Calculate center of the region in voltage space
        center_x_pixel = (x_min + x_max) / 2.0
        center_y_pixel = (y_min + y_max) / 2.0
        
        # Convert to voltage coordinates
        center_x_voltage = (center_x_pixel - self.x_pixels / 2.0) * voltage_per_pixel_x
        center_y_voltage = (center_y_pixel - self.y_pixels / 2.0) * voltage_per_pixel_y
        
        # Apply base offset and drift offset
        final_offset_x, final_offset_y = self._apply_drift_offset(self.offset_x, self.offset_y)
        
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
        """
        Local RPOC doesn't use traditional RPOC - this is a placeholder
        """
        if self.signal_bus:
            self.signal_bus.console_message.emit("Local RPOC treatment configured")
    
    def perform_acquisition(self):
        """
        Perform the local RPOC treatment without data collection
        This is the main method called by the worker
        """
        if self.signal_bus:
            self.signal_bus.console_message.emit("Starting local RPOC treatment...")
        
        # Calculate scan parameters
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
                # Emit progress signal
                self.signal_bus.local_rpoc_progress.emit(rep + 1)
            
            # Perform the actual treatment scan
            self._perform_treatment_scan(scan_params)
            
            if rep < self.repetitions - 1:  # Don't wait after the last repetition
                time.sleep(0.1)  # Brief pause between repetitions
        
        if self.signal_bus:
            self.signal_bus.console_message.emit("Local RPOC treatment completed")
        
        return None  # No data to return
    
    def _perform_treatment_scan(self, scan_params):
        """
        Perform the actual treatment scan using galvo control
        """
        if self.galvo is None:
            if self.signal_bus:
                self.signal_bus.console_message.emit("Warning: No galvo instrument available - simulating treatment")
            # Simulate treatment timing
            # time.sleep(self.dwell_time * scan_params['region_width'] * scan_params['region_height'] / 1e6)
            time.sleep(1)
            return
        
        # TODO: Implement actual galvo control for treatment
        # This would involve:
        # 1. Setting up the galvo with the calculated parameters
        # 2. Running the scan pattern for the treatment region
        # 3. Controlling the treatment laser output during the scan
        
        if self.signal_bus:
            self.signal_bus.console_message.emit("Treatment scan completed (galvo control not yet implemented)")
    
    def save_data(self, data):
        """
        Local RPOC doesn't save data - this is a placeholder
        """
        pass
