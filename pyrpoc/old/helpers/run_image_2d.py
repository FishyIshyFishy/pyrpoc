import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping
from nidaqmx.errors import DaqWarning
import numpy as np
from pyrpoc.old.helpers.galvo_funcs import Galvo
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw, ImageOps
import warnings
warnings.filterwarnings("ignore", category=DaqWarning, message=".*200011.*")

def run_scan(ai_channels, galvo, modulate=False, mod_do_chans=None, mod_masks=None):
    if isinstance(ai_channels, str):
        ai_channels = [ai_channels]

    has_mods = modulate and mod_do_chans and mod_masks and (len(mod_do_chans) == len(mod_masks))
    
    composite_wave = galvo.waveform.copy()
    total_samps = galvo.total_samples

    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
        for chan in galvo.ao_chans:
            ao_task.ao_channels.add_ao_voltage_chan(f"{galvo.device}/{chan}")
        for ch in ai_channels:
            ai_task.ai_channels.add_ai_voltage_chan(ch)

        ao_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samps
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            source=f"/{galvo.device}/ao/SampleClock",
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samps
        )

        if has_mods:
            ttl_signals = []
            for m in mod_masks:
                m_arr = np.array(m) if isinstance(m, Image.Image) else m
                m_arr = m_arr > 0 # pretty sure that its a unint8, could be 0 to 1 as well so to be safe just > 0
                padded = []
                for row in range(galvo.numsteps_y):
                    padded_row = np.concatenate((
                        np.zeros(galvo.extrasteps_left, dtype=bool),
                        m_arr[row, :],
                        np.zeros(galvo.extrasteps_right, dtype=bool)
                    ))
                    padded.append(padded_row)
                flat = np.repeat(np.array(padded).ravel(), galvo.pixel_samples).astype(bool)
                ttl_signals.append(flat)
            
            if len(mod_do_chans) == 1:
                line = mod_do_chans[0]
                do_task.do_channels.add_do_chan(f"{galvo.device}/{line}")
                do_task.timing.cfg_samp_clk_timing(
                    rate=galvo.rate,
                    source=f"/{galvo.device}/ao/SampleClock",
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samps
                )
                do_task.write(ttl_signals[0].tolist(), auto_start=False)
            else:
                for chan in mod_do_chans:
                    do_task.do_channels.add_do_chan(f"{galvo.device}/{chan}")
                do_task.timing.cfg_samp_clk_timing(
                    rate=galvo.rate,
                    source=f"/{galvo.device}/ao/SampleClock",
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samps
                )
                data_to_write = [sig.tolist() for sig in ttl_signals]
                do_task.write(data_to_write, auto_start=False)

        ao_task.write(composite_wave, auto_start=False)
        ai_task.start()
        if has_mods:
            do_task.start()
        ao_task.start()

        ao_task.wait_until_done(timeout=total_samps / galvo.rate + 5)
        ai_task.wait_until_done(timeout=total_samps / galvo.rate + 5)
        if has_mods:
            do_task.wait_until_done(timeout=total_samps / galvo.rate + 5)

        acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samps))

    results = []
    for i in range(len(ai_channels)):
        channel_data = acq_data if len(ai_channels) == 1 else acq_data[i]
        reshaped = channel_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
        pixel_values = np.mean(reshaped, axis=2)
        cropped = pixel_values[:, galvo.extrasteps_left:galvo.extrasteps_left + galvo.numsteps_x]
        results.append(cropped)

    return results


def interpret_DAQ_output(ai_data_1d, mask, pixel_map, galvo):
    num_y, total_x = pixel_map.shape
    pixel_values_2d = np.zeros((num_y, total_x), dtype=float)
    cursor = 0
    for row_idx in range(num_y):
        for col_idx in range(total_x):
            samps = pixel_map[row_idx, col_idx]
            pixel_block = ai_data_1d[cursor:cursor + samps]
            cursor += samps
            pixel_values_2d[row_idx, col_idx] = np.mean(pixel_block)
    return pixel_values_2d