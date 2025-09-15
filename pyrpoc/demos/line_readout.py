import numpy as np
import time
import nidaqmx as nx
from nidaqmx.constants import Edge, AcquisitionType, Level, CounterFrequencyMethod
from nidaqmx.stream_readers import CounterReader

device = 'Dev1'
numX = 128
numY = 128
pixel_rate = 100000
line_rate = 500

line_term = f'{device}/PFI12'
pixel_term = f'{device}/PFI13'

with nx.Task() as line_co, nx.Task() as pix_co, nx.Task() as ci_task:
    line_co.co_channels.add_co_pulse_chan_freq( # creates a pulse with the defined duty cycle and frequency
        f'{device}', freq=line_rate, duty_cycle=0.5,
        idle_state=Level.LOW, initial_delay=0.0
    )
    line_co.timing.cfg_implicit_timing(samps_per_chan=numY) # time it by a given number of samples
    line_co.export_signals.ctr_out_event_output_term = line_term

    pix_co.co_channels.add_co_pulse_chan_freq(
        f'{device}/ctr1', freq=pixel_rate, duty_cycle=0.5,
        idle_state=Level.LOW, initial_delay=0.0
    )
    pix_co.timing.cfg_implicit_timing(samps_per_chan=numX)
    pix_co.triggers.start_trigger.cfg_dig_edge_start_trig(line_term)
    pix_co.triggers.start_trigger.retriggerable = True
    pix_co.export_signals.ctr_out_event_output_term = pixel_term

    ci_task.ci_channels.add_ci_count_edges_chan(f'{device}/ctr2')
    ci_task.timing.cfg_samp_clk_timing(
        rate=pixel_rate, source=pixel_term,
        active_edge=Edge.RISING, sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=numX*4
    )
    reader = CounterReader(ci_task.in_stream)

    line_buffer = np.zeros(numX, dtype=np.uint32)

    ci_task.start()
    pix_co.start()
    line_co.start()

    lines_read = 0
    try:
        while lines_read < numY:
            if ci_task.in_stream.avail_samp_per_chan >= numX:
                reader.read_many_sample_uint32(line_buffer, number_of_samples_per_channel=numX, timeout=1.0)
                lines_read += 1
            else:
                time.sleep(0.0001)
    finally:
        pass

