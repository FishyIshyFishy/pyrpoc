import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import time
import pyvisa
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import lti, lsim
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

ao_channel = 'Dev1/ao0'
step_amplitude = 0.5
duration = 0.01
rate = 1000000
osc_resource = 'USB0::0x0699::0x03C7::C010691::INSTR'
vertical_scale = 2000E-3
horizontal_scale = 1e-4
num_repeats = 10

num_samples = int(duration * rate)
waveform = np.zeros(num_samples)
waveform[num_samples // 2:] = step_amplitude

def init_scope(scope):
    scope.write('*CLS')
    scope.write('ACQuire:STATE OFF')
    scope.write('ACQuire:STOPAfter SEQ')
    scope.write('TRIGger:A:MODe NORM')
    scope.write('TRIGger:A:EDGE:SOURCE CH1')
    scope.write('TRIGger:A:EDGE:SLOPe RIS')
    scope.write(f'TRIGger:A:LEVel {step_amplitude / 2}')
    scope.write(f'HORIZONTAL:SCALE {horizontal_scale}')
    for ch in range(1, 5):
        if ch == 1:
            scope.write(f'CH{ch}:SCALE {step_amplitude/2.5}')
        elif ch == 2:
            scope.write(f'CH{ch}:SCALE {step_amplitude*2}')
        scope.write(f'CH{ch}:POSITION 0')
        scope.write(f'SELect:CH{ch} ON')

def parse_waveform(raw_str):
    nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', raw_str)
    return np.array([float(x) for x in nums])

def capture_response(scope):
    scope.write('ACQuire:STATE ON')
    time.sleep(0.1)
    with nidaqmx.Task() as ao_task:
        ao_task.ao_channels.add_ao_voltage_chan(ao_channel)
        ao_task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(waveform))
        ao_task.write(waveform, auto_start=False)
        ao_task.start()
        ao_task.wait_until_done()
    time.sleep(0.3)
    waveforms = {}
    time_axes = {}
    for ch in range(1, 5):
        scope.write(f'DATa:SOUrce CH{ch}')
        scope.write('DATa:ENCdg ASCII')
        scope.write('DATa:WIDth 1')
        scope.write('DATa:STARt 1')
        scope.write('DATa:STOP 10000')
        raw = scope.query('CURVe?')
        x_incr = float(scope.query('WFMOutpre:XINcr?'))
        x_zero = float(scope.query('WFMOutpre:XZEro?'))
        wave = parse_waveform(raw)
        t_axis = x_zero + x_incr * np.arange(len(wave))
        waveforms[f'CH{ch}'] = wave
        time_axes[f'CH{ch}'] = t_axis
    scope.write('ACQuire:STATE OFF')
    return waveforms, time_axes

def get_params(time, signal, step_amplitude):
    theta_inf = np.mean(signal[-int(0.05 * len(signal)):])
    K = theta_inf / step_amplitude
    peak_idx = np.argmax(signal)
    theta_peak = signal[peak_idx]
    t_peak = time[peak_idx]
    M_p = (theta_peak - theta_inf) / theta_inf
    if M_p <= 0 or M_p >= 1:
        raise ValueError('Invalid overshoot; cannot compute damping ratio.')
    zeta = -np.log(M_p) / np.sqrt(np.pi**2 + (np.log(M_p))**2)
    omega_n = np.pi / (t_peak * np.sqrt(1 - zeta**2))
    return K, omega_n, zeta

def step_response(t, K, wn, zeta, t0):
    t = t - t0
    t = np.maximum(t, 0)
    wd = wn * np.sqrt(1 - zeta**2)
    return K * step_amplitude * (1 - np.exp(-zeta * wn * t) *
            (np.cos(wd * t) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * t)))

def third_order_step_response(t, K, wn, zeta, tau, t0):
    t_shifted = t - t0
    t_shifted[t_shifted < 0] = 0
    num = [K * wn ** 2]
    den = np.polymul([1, 2 * zeta * wn, wn ** 2], [tau, 1])
    system = lti(num, den)
    t_uniform = np.linspace(t_shifted[0], t_shifted[-1], len(t_shifted))
    _, y_uniform, _ = lsim(system, U=np.ones_like(t_uniform), T=t_uniform)
    interp_func = interp1d(t_uniform, y_uniform, bounds_error=False, fill_value=(y_uniform[0], y_uniform[-1]))
    return interp_func(t_shifted)

def fit_third_order_model(t, y, step_amplitude):
    K_guess = np.max(y) / step_amplitude
    t_peak = t[np.argmax(y)]
    wn_guess = np.pi / t_peak if t_peak > 0 else 1000
    zeta_guess = 0.8
    tau_guess = 1 / (10 * wn_guess)
    t0_guess = t[np.argmax(np.gradient(y))]
    p0 = [K_guess, wn_guess, zeta_guess, tau_guess, t0_guess]
    bounds = ([0, 0, 0, 0, t[0]], [np.inf, np.inf, 1, np.inf, t[-1]])
    popt, _ = curve_fit(
        lambda t, K, wn, zeta, tau, t0: third_order_step_response(t, K, wn, zeta, tau, t0),
        t, y, p0=p0, bounds=bounds, maxfev=20000
    )
    return popt

if __name__ == '__main__':
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(osc_resource)
    init_scope(scope)

    accumulated = {f'CH{ch}': None for ch in range(1, 5)}

    for _ in range(num_repeats):
        waveforms, time_axes = capture_response(scope)
        for ch in range(1, 5):
            if accumulated[f'CH{ch}'] is None:
                accumulated[f'CH{ch}'] = waveforms[f'CH{ch}']
            else:
                accumulated[f'CH{ch}'] += waveforms[f'CH{ch}']

    for ch in range(1, 5):
        accumulated[f'CH{ch}'] /= num_repeats

    plt.figure(figsize=(12, 8))
    for ch in range(1, 5):
        plt.subplot(2, 2, ch)
        plt.plot(time_axes[f'CH{ch}'], accumulated[f'CH{ch}'], label=f'CH{ch}')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage')
        plt.legend()
    plt.tight_layout()
    plt.show()

    '''3rd order'''
    ch2_data = accumulated['CH2']
    ch2_time = time_axes['CH2']

    try:
        popt = fit_third_order_model(ch2_time, ch2_data, step_amplitude)
        K_fit, wn_fit, zeta_fit, tau_fit, t0_fit = popt
        print('Estimated parameters from Channel 2:')
        print(f'K = {K_fit}')
        print(f'ωn = {wn_fit}')
        print(f'ζ = {zeta_fit}')
        print(f'τ = {tau_fit}')
        print(f't0 = {t0_fit}')
    except Exception as e:
        print('Parameter extraction failed:', e)

    plt.plot(ch2_time, ch2_data, label='Measured Response (CH2)', color='black')
    plt.plot(ch2_time, third_order_step_response(ch2_time, *popt), label='Fitted Third-Order Model', linestyle='--', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    '''2nd order'''
    # ch2_data = accumulated['CH2']
    # ch2_time = time_axes['CH2']

    # try:
    #     K_est, wn_est, zeta_est = get_params(ch2_time, ch2_data, step_amplitude)
    #     print('Estimated parameters from Channel 2:')
    #     print('K =', K_est)
    #     print('ωn =', wn_est)
    #     print('ζ =', zeta_est)
    # except Exception as e:
    #     print('Parameter extraction failed:', e)

    # ch = 2
    # ch_data = accumulated[f'CH{ch}']
    # ch_time = time_axes[f'CH{ch}']

    # theta_inf = np.mean(ch_data[-int(0.05 * len(ch_data)):])
    # K_guess = theta_inf / step_amplitude
    # t_peak = ch_time[np.argmax(ch_data)]
    # wn_guess = np.pi / t_peak if t_peak > 0 else 1000
    # zeta_guess = 0.8
    # p0 = [K_guess, wn_guess, zeta_guess]

    # ch = 2
    # ch_data = accumulated[f'CH{ch}']
    # ch_time = time_axes[f'CH{ch}']

    # K_guess = np.max(ch_data) / step_amplitude
    # t_peak = ch_time[np.argmax(ch_data)]
    # wn_guess = np.pi / t_peak if t_peak > 0 else 1000
    # zeta_guess = 0.8
    # t0_guess = ch_time[np.argmax(np.gradient(ch_data))]  # initial step location
    # p0 = [K_guess, wn_guess, zeta_guess, t0_guess]

    # popt, _ = curve_fit(step_response, ch_time, ch_data, p0=p0, maxfev=10000)
    # K_fit, wn_fit, zeta_fit, t0_fit = popt

    # print('Estimated parameters from Channel 2:')
    # print(f'K = {K_fit}')
    # print(f'ωn = {wn_fit}')
    # print(f'ζ = {zeta_fit}')
    # print(f't0 = {t0_fit}')

    # plt.plot(ch_time, ch_data, label='Measured Response (CH2)', color='black')
    # plt.plot(ch_time, step_response(ch_time, *popt), label='Fitted Model', linestyle='--', color='red')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Voltage')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
