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
from scipy.spatial import KDTree
from scipy import ndimage
from skimage.filters import gaussian
from skimage.morphology import erosion, dilation, disk
from skimage.segmentation import find_boundaries
import cv2

class Simulated(Acquisition):
    def __init__(self, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.signal_bus = signal_bus
        self.acquisition_parameters = acquisition_parameters or {}
        
        # Extract parameters from acquisition_parameters dict
        self.x_pixels = self.acquisition_parameters.get('x_pixels', 512)
        self.y_pixels = self.acquisition_parameters.get('y_pixels', 512)
        self.num_frames = self.acquisition_parameters.get('num_frames', 1)

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # Save metadata before starting acquisition
        self.save_metadata()
        
        frames = []
        for frame in range(self.num_frames):
            if self._stop_flag and self._stop_flag():
                break
                
            frame_data = generate_cells(self.y_pixels, self.x_pixels)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, self.num_frames, False)
            time.sleep(1)
        
        if frames:
            final_data = np.stack(frames)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(frames)-1, self.num_frames, True)

            self.save_data(final_data)
            return final_data
        else:
            return None
        
    


    
    def save_data(self, data):
        """
        Save simulated data as a single TIFF file
        data shape: (num_frames, height, width) or (height, width)
        """
        if not self.save_enabled or not self.save_path:
            return
        
        try:
            save_dir = Path(self.save_path).parent
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{Path(self.save_path).stem}_{timestamp}.tiff"
            filepath = save_dir / filename
            
            # Save as TIFF
            tifffile.imwrite(filepath, data)
            
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Saved simulated data to {filepath}")
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving simulated data: {e}")
    
@staticmethod
def generate_cells(
    x_pixels,
    y_pixels,
    n_cells=20,                  # actual visible cells
    oversample_factor=2,          # Voronoi seeds = n_cells * this
    intensity_range=(0.4, 1.0),
    background_level=0.05,
    noise_sigma=0.02,
    nucleus_dim=0.25,
    texture_sigma=8.0,
    blur_sigma=1.2,
    cell_fill_fraction=0.65,      # how much of Voronoi region to keep
    boundary_noise_sigma=10.0,    # membrane roughness
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    yy, xx = np.mgrid[0:y_pixels, 0:x_pixels]
    coords = np.column_stack([yy.ravel(), xx.ravel()])

    # oversampled Voronoi seeds
    n_seeds = n_cells * oversample_factor
    seeds = np.column_stack([
        np.random.uniform(0, y_pixels, n_seeds),
        np.random.uniform(0, x_pixels, n_seeds),
    ])

    tree = KDTree(seeds)
    _, labels = tree.query(coords)
    labels = labels.reshape((y_pixels, x_pixels))

    # randomly choose which Voronoi regions become cells
    chosen = np.random.choice(
        np.arange(n_seeds),
        size=n_cells,
        replace=False
    )

    img = np.full((y_pixels, x_pixels), background_level, dtype=np.float32)

    intensities = np.random.uniform(*intensity_range, n_cells)

    # boundary noise field (shared across cells → coherent roughness)
    boundary_noise = gaussian(
        np.random.randn(y_pixels, x_pixels),
        sigma=boundary_noise_sigma,
        preserve_range=True,
    )
    boundary_noise = boundary_noise / (np.ptp(boundary_noise) + 1e-9)

    for idx, seed_idx in enumerate(chosen):
        region = labels == seed_idx
        if region.sum() < 80:
            continue

        # distance inside this Voronoi region
        dist = ndimage.distance_transform_edt(region)
        maxd = dist.max()
        if maxd <= 0:
            continue

        # irregular cell shape cutoff
        cutoff = cell_fill_fraction * maxd * (1.0 + 0.25 * boundary_noise)
        cell_mask = dist > cutoff
        if cell_mask.sum() < 50:
            continue

        cell = np.full((y_pixels, x_pixels), float(intensities[idx]), dtype=np.float32)
        img[cell_mask] += cell[cell_mask]


    img += np.random.normal(0, noise_sigma, img.shape)

    img = np.clip(img, 0, 1)
    return img
