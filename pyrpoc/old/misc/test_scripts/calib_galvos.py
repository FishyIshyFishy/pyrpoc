import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, TransferFunction, bode, step
from scipy.optimize import curve_fit
import pyvisa as pv
import time


config = {
    'ao_channel': 'Dev1/ao1',    
    'ai_channel': 'Dev1/ai3',   
    'rate': 1e6,                 
    'duration': 10,            
    'test_type': 'sine',
    'step_amplitude': 0.5,
    'ramp_amplitude': 0.5,   
    'sine_amplitude': 0.3,   
    'sine_frequency': 500,          
    'save_plots': False,           
}

def generate_waveform(test_type, duration, rate):
    t = np.linspace(0, duration, int(duration * rate), endpoint=False)
    if test_type == 'step':
        waveform = np.ones_like(t) * config['step_amplitude']
        waveform[:len(t)//2] = 0
    elif test_type == 'ramp':
        waveform = np.linspace(0, config['ramp_amplitude'], len(t))
    elif test_type == 'sine':
        waveform = config['sine_amplitude'] * np.sin(2 * np.pi * config['sine_frequency'] * t)
    else:
        raise ValueError('Invalid test_type. Choose "step", "ramp", or "sinesine".')
    return t, waveform

def acquire_response(ao_channel, ai_channel, waveform, rate):
    total_samples = len(waveform)
    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
        ao_task.ao_channels.add_ao_voltage_chan(ao_channel)

        ai_task.ai_channels.add_ai_voltage_chan(
            ai_channel,
            terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
            min_val=-1.0, max_val=1.0
        )

        ao_task.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samples
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=rate,
            source=f'/{config["ao_channel"].split("/")[0]}/ao/SampleClock',
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samples
        )

        ao_task.write(waveform, auto_start=False)
        ai_task.start()
        ao_task.start()
        ao_task.wait_until_done()
        ai_task.wait_until_done()
        response = ai_task.read(number_of_samples_per_channel=total_samples)
    return np.array(response)

def acquire_response_oscilloscope(
        ao_channel, 
        waveform, 
        rate, 
        scope_resource, 
        horizontal_scale=4e-4,
        trigger_chan=1, trigger_chan_scale=0.2,
        output_chan=2, output_chan_scale=0.05
    ):
    total_samples = len(waveform)
    rm = pv.ResourceManager()
    scope = rm.open_resource(scope_resource)
    scope.timeout = 5000 

    scope.write(":ACQUIRE:STATE STOP")
    scope.write(":CLEAR")
    scope.write(":ACQUIRE:STOPAFTER SEQ")
    scope.write(f":TRIGGER:MAIn:EDGE:SOURCE CH{trigger_chan}")
    scope.write(":TRIGGER:MAIn:EDGE:SLOpe RISing")
    scope.write(":DATA:ENCdg ASCII")
    scope.write(":DATA:WIDTH 1") 
    scope.write(f'CH{trigger_chan}:SCALE {trigger_chan_scale}')
    scope.write(f'CH{output_chan}:SCALE {output_chan_scale}')
    scope.write(":ACQUIRE:STATE RUN")
    scope.write(f'HORIZONTAL:SCALE {horizontal_scale}')
    
    time.sleep(0.2)

    with nidaqmx.Task() as ao_task:
        ao_task.ao_channels.add_ao_voltage_chan(ao_channel)
        ao_task.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samples
        )
        ao_task.write(waveform, auto_start=False)
        ao_task.start()
        ao_task.wait_until_done()

    time.sleep(0.5)

    scope.write(f":DATA:SOURCE CH{trigger_chan}")
    raw_data_ch1 = scope.query(":CURVe?")
    ch1_values = raw_data_ch1.strip().split(',')
    v_ch1 = np.array(ch1_values, dtype=float)

    dt_ch1 = float(scope.query(":WFMPRE:XINCR?"))
    t0_ch1 = float(scope.query(":WFMPRE:XZERO?"))
    n_ch1 = len(v_ch1)
    t_ch1 = t0_ch1 + dt_ch1 * np.arange(n_ch1)

    scope.write(f":DATA:SOURCE CH{output_chan}")
    raw_data_ch2 = scope.query(":CURVe?")
    ch2_values = raw_data_ch2.strip().split(',')
    v_ch2 = np.array(ch2_values, dtype=float)

    dt_ch2 = float(scope.query(":WFMPRE:XINCR?"))
    t0_ch2 = float(scope.query(":WFMPRE:XZERO?"))
    n_ch2 = len(v_ch2)
    t_ch2 = t0_ch2 + dt_ch2 * np.arange(n_ch2)

    return (t_ch1, v_ch1), (t_ch2, v_ch2)

def plot_raw(t, command, response): 
    plt.figure(figsize=(12, 8))
    plt.plot(t, command, '--k', label='Command Signal', alpha=0.3)
    plt.plot(t, response, 'b', label='Measured Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Galvo Response')
    plt.legend()

    plt.show()

# '''oscilloscope main'''
if __name__ == '__main__':
    t, command_wave = generate_waveform(config['test_type'], config['duration'], config['rate'])
    scope_resource_str = "USB0::0x0699::0x03C7::C010691::INSTR" 
    (t_ch1, v_ch1), (t_ch2, v_ch2) = acquire_response_oscilloscope(
        config['ao_channel'],
        command_wave,
        config['rate'],
        scope_resource=scope_resource_str,
        trigger_chan=3,
        trigger_chan_scale=0.2,
        output_chan=2,
        output_chan_scale=0.1,
        horizontal_scale=5e-4
    )

    plt.figure()
    plt.plot(t_ch1, v_ch1, color='black', label='CH1')
    plt.plot(t_ch2, v_ch2, color='red', label='CH2')
    plt.legend()
    plt.show()

'''DAQ main'''
# if __name__ == '__main__':
#     t, command_wave = generate_waveform(config['test_type'], config['duration'], config['rate'])
#     response_wave = acquire_response(config['ao_channel'], config['ai_channel'], command_wave, config['rate'])

#     plot_raw(t, command_wave, response_wave)
