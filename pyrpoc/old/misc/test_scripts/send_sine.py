import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import time
import pyvisa
import re
import matplotlib.pyplot as plt

ao_channel = 'Dev1/ao1'
amp = 0.1
freq = 2000
duration = 0.2
rate = 10000
osc_resource = 'USB0::0x0699::0x03C7::C010691::INSTR'

t = np.linspace(0, duration, int(duration * rate), endpoint=False)
waveform = amp * np.sin(2 * np.pi * freq * t)

rm = pyvisa.ResourceManager()
scope = rm.open_resource(osc_resource)
scope.write('*CLS')
scope.write('ACQuire:STATE OFF')
scope.write('ACQuire:STOPAfter SEQ')
scope.write('TRIGger:A:MODe NORM')
scope.write('TRIGger:A:EDGE:SOURCE CH1')
scope.write('TRIGger:A:EDGE:SLOPe RIS')
scope.write(f'TRIGger:A:LEVel {amp/2}')
scope.write('HORIZONTAL:SCALE 1e-4')

for ch in range(1, 5):
    scope.write(f"CH{ch}:SCALE 50E-3")
    scope.write(f"CH{ch}:POSITION 0")

scope.write('ACQuire:STATE ON')

time.sleep(0.2)

with nidaqmx.Task() as ao_task:
    ao_task.ao_channels.add_ao_voltage_chan(ao_channel)
    ao_task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(waveform))
    ao_task.write(waveform, auto_start=False)
    ao_task.start()
    ao_task.wait_until_done()

time.sleep(0.3)

def parse_waveform(raw_str):
    nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", raw_str)
    return np.array([float(x) for x in nums])

waveforms = {}
time_axes = {}
for ch in range(1, 5):
    scope.write(f"DATa:SOUrce CH{ch}")
    scope.write("DATa:ENCdg ASCII")
    scope.write("DATa:WIDth 1")
    scope.write("DATa:STARt 1")
    scope.write("DATa:STOP 10000")
    raw = scope.query("CURVe?")
    x_incr = float(scope.query("WFMOutpre:XINcr?"))
    x_zero = float(scope.query("WFMOutpre:XZEro?"))
    wave = parse_waveform(raw)
    t_axis = x_zero + x_incr * np.arange(len(wave))
    waveforms[f"CH{ch}"] = wave
    time_axes[f"CH{ch}"] = t_axis

scope.write('ACQuire:STATE OFF')

plt.figure(figsize=(12, 8))
for ch in range(1, 5):
    plt.subplot(2, 2, ch)
    plt.plot(time_axes[f"CH{ch}"], waveforms[f"CH{ch}"], label=f'CH{ch}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage')
    plt.legend()
plt.tight_layout()
plt.show()
