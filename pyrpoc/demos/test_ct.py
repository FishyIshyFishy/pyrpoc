import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt

DEVICE      = 'Dev1'
DO_LINE     = 'port0/line5'     # hardware-timed DO line
AI_CHANNELS = ['ai0', 'ai5']           # list of channels to read
RATE_HZ     = 1_000_000         # sample rate
REPS        = 1                # number of high/low periods to repeat

one_cycle = np.concatenate([np.ones(5, dtype=np.uint8),
                            np.zeros(10, dtype=np.uint8)])
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

    do_task.write(ttl.astype(bool).tolist(), auto_start=False)

    do_task.start()
    ai_task.start()
    ai_data = ai_task.read(number_of_samples_per_channel=total_samps)

# Normalize AI data shape to (num_channels, total_samps)
ai_data = np.array(ai_data)
if ai_data.ndim == 1:
    ai_data = np.atleast_2d(ai_data)
    ai_data = ai_data - np.min(ai_data)
elif ai_data.ndim == 2 and ai_data.shape[0] == total_samps and ai_data.shape[1] == len(AI_CHANNELS):
    ai_data = ai_data.T
    ai_data = ai_data - np.min(ai_data)
t = np.arange(total_samps) / RATE_HZ * 1e6

plt.figure(figsize=(3,3))
plt.plot(t, ttl * np.max(np.abs(ai_data)), marker='o', color='black', label='TTL (scaled)')

colors = ['r', 'g', 'b', 'm']
for i, ch in enumerate(AI_CHANNELS):
    plt.plot(t, ai_data[i],  marker='o', color=colors[i], label=f"{ch}")
plt.xlabel("Time (us)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.tight_layout()
plt.show()
