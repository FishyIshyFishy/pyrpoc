import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt

DEVICE      = 'Dev1'
DO_LINE     = 'port0/line0'     # hardware-timed DO line
AI_CHANNELS = ['ai0']           # list of channels to read
RATE_HZ     = 1_000_000         # sample rate
N_SAMPLES   = 50                # number of samples high (and low)
REPS        = 20                # number of high/low periods to repeat

one_cycle = np.concatenate([np.ones(N_SAMPLES, dtype=np.uint8),
                            np.zeros(N_SAMPLES, dtype=np.uint8)])
ttl = np.tile(one_cycle, REPS)
total_samps = len(ttl)

print(f"Total samples: {total_samps}, Total time: {total_samps / RATE_HZ * 1e6:.1f} µs")

with nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
    for ch in AI_CHANNELS:
        ai_task.ai_channels.add_ai_voltage_chan(f"{DEVICE}/{ch}")
    ai_task.timing.cfg_samp_clk_timing(rate=RATE_HZ,
                                       sample_mode=AcquisitionType.FINITE,
                                       samps_per_chan=total_samps)

    do_task.do_channels.add_do_chan(f"{DEVICE}/{DO_LINE}")
    do_task.timing.cfg_samp_clk_timing(rate=RATE_HZ,
                                       source=f"/{DEVICE}/ai/SampleClock",
                                       sample_mode=AcquisitionType.FINITE,
                                       samps_per_chan=total_samps)
    do_task.triggers.start_trigger.cfg_dig_edge_start_trig(f"/{DEVICE}/ai/StartTrigger")

    do_task.write(ttl.tolist())

    do_task.start()
    ai_task.start()
    ai_data = ai_task.read()

ai_data = np.atleast_2d(np.array(ai_data))
t = np.arange(total_samps) / RATE_HZ * 1e6

plt.figure()
plt.plot(t, ttl * np.max(np.abs(ai_data)), 'k--', label='TTL (scaled)')
for i, ch in enumerate(AI_CHANNELS):
    plt.plot(t, ai_data[i], label=f"AI {ch}")
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.tight_layout()
plt.show()
