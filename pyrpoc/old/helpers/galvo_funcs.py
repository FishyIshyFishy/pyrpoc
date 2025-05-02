import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from PIL import Image
from tkinter import messagebox

class Galvo:
    def __init__(self, config, rpoc_mask=None, rpoc_do_chan=None, rpoc_mode=None, dwell_multiplier=2.0, **kwargs):
        defaults = {
            "numsteps_x": 400,
            "numsteps_y": 400,
            "extrasteps_left": 50,
            "extrasteps_right": 50,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "dwell": 10e-6,
            "amp_x": 0.5,
            "amp_y": 0.5,
            "rate": 100000,
            "device": 'Dev1',
            "ao_chans": ['ao1', 'ao0']
        }
        if config:
            defaults.update(config)
        defaults.update(kwargs)
        for key, val in defaults.items():
            setattr(self, key, val)

        self.rpoc_mask = rpoc_mask
        self.rpoc_do_chan = rpoc_do_chan
        self.rpoc_mode = rpoc_mode  # "standard" or "variable" so far
        self.dwell_multiplier = dwell_multiplier

        self.pixel_samples = max(1, int(self.dwell * self.rate))
        self.total_x = self.numsteps_x + self.extrasteps_left + self.extrasteps_right
        self.total_y = self.numsteps_y
        self.total_samples = self.total_x * self.total_y * self.pixel_samples

        if self.rpoc_mode == "variable":
            self.waveform = None  # or maybe generate it here...
        else:
            self.waveform = self.gen_raster()

    def gen_raster(self):
        contained_rowsamples = self.pixel_samples * self.numsteps_x
        total_rowsamples = self.pixel_samples * self.total_x
        self.total_samples = total_rowsamples * self.total_y

        step_size = (2 * self.amp_x)/(contained_rowsamples)
        bottom = self.offset_x - self.amp_x - (step_size * self.extrasteps_left)
        top = self.offset_x + self.amp_x + (step_size * self.extrasteps_right)

        single_row_ramp = np.linspace(bottom, top, total_rowsamples, endpoint=False)
        x_waveform = np.tile(single_row_ramp, self.total_y)

        y_steps = np.linspace(
            self.offset_y + self.amp_y,
            self.offset_y - self.amp_y,
            self.total_y
        )
        y_waveform = np.repeat(y_steps, total_rowsamples)

        composite = np.vstack([x_waveform, y_waveform])

        if x_waveform.size < self.total_samples:
            x_waveform = np.pad(
                x_waveform,
                (0, self.total_samples - x_waveform.size),
                constant_values=x_waveform[-1]
            )
        else:
            x_waveform = x_waveform[:self.total_samples]
        composite[0] = x_waveform

        return composite


    def gen_variable_waveform(self, mask, dwell_multiplier):
        dwell = self.dwell
        rate = self.rate
        num_y = self.numsteps_y
        num_x = self.numsteps_x + self.extrasteps_left + self.extrasteps_right

        mask_shape = np.shape(mask)
        if mask_shape[1] != num_y or mask_shape[0] != (num_x - self.extrasteps_left - self.extrasteps_right):
            raise ValueError(f'Error in galvo_funcs.gen_variable_waveform(). Mask is not the right size. Load a mask of dimensions {num_x - self.extrasteps_left - self.extrasteps_right} by {num_y}.')


        dwell_on = dwell * dwell_multiplier
        dwell_off = dwell


        x_min = self.offset_x - self.amp_x
        x_max = self.offset_x + self.amp_x
        x_positions = np.linspace(x_min, x_max, num_x, endpoint=False)
        y_positions = np.linspace(self.offset_y + self.amp_y,
                                self.offset_y - self.amp_y,
                                num_y)

        x_wave_list = []
        y_wave_list = []
        pixel_map = np.zeros((num_y, num_x), dtype=int)

        for row_idx in range(num_y):
            row_y = y_positions[row_idx]
            for col_idx in range(num_x):
                if col_idx < self.extrasteps_left or col_idx >= (num_x - self.extrasteps_right): # for clarity im leaving this out of the else - we need to remember to account for extrasteps here
                    this_dwell = dwell_off
                elif mask[row_idx, col_idx - self.extrasteps_left]:
                    this_dwell = dwell_on
                else:
                    this_dwell = dwell_off

                pixel_samps = max(1, int(this_dwell * rate))

                x_val = x_positions[col_idx]
                x_wave_list.append(np.full(pixel_samps, x_val))

                y_wave_list.append(np.full(pixel_samps, row_y))

                pixel_map[row_idx, col_idx] = pixel_samps
        x_wave = np.concatenate(x_wave_list)
        y_wave = np.concatenate(y_wave_list)
        return x_wave, y_wave, pixel_map
