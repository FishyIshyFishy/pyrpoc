"""
Measure the phase noise behavior
"""

import matplotlib.pylab as plt
import numpy as np

from Swabian import TimeTagger

# General parameters
USE_AVERAGING = True
CHANNEL = 1
SAMPLES_PER_OCTAVE = 512
FC_SAMPLE_RATE = 1e0
TARGET_FREQUENCY = 10e6
JITTER_INT_LIMITS = [[1, 10e3], [12e3, 1e6]]

########################
# Tagger configuration #
########################
if len(TimeTagger.scanTimeTagger()) > 0:
    tagger = TimeTagger.createTimeTagger()

    if tagger.getModel() == "Time Tagger 20" and TARGET_FREQUENCY > 5e6:
        print('Limiting the target frequency to 5 MHz because of the USB2 bandwidth.')
        TARGET_FREQUENCY = 5e6

    # Check if there is a signal on this input
    with TimeTagger.Countrate(tagger=tagger, channels=[CHANNEL]) as cnt:
        cnt.startFor(1e12)
        cnt.waitUntilFinished()
        if cnt.getData()[0] < 1e3:
            print(
                f'No signal on input {CHANNEL} found, using the test signal instead.')
            tagger.setTestSignal(CHANNEL, True)
            tagger.setTestSignalDivider(1)

    # Check if we can average the rising and falling edges
    if USE_AVERAGING and tagger.getModel() != 'Time Tagger 20':
        # Measure how much we have to delay the falling edge to match the next rising one
        with TimeTagger.StartStop(
                tagger=tagger, click_channel=CHANNEL, start_channel=-CHANNEL, binwidth=1) as start_stop:
            start_stop.startFor(1e12)
            start_stop.waitUntilFinished()

            tau, cnts = start_stop.getData().T
            delay = int(np.sum(tau * cnts) / np.sum(cnts))
            assert tau[-1] - tau[0] < 500, "The on-device average requires a stable duty cycle, please disable USE_AVERAGING"

            print(
                f'Enabling the average of rising and falling events with a delay of {delay} ps.')
            tagger.setInputDelay(channel=-CHANNEL, delay=delay)
            tagger.xtra_setAvgRisingFalling(channel=CHANNEL, enable=True)

################################
# Time Tagger Virtual fallback #
################################
else:
    print('No Time Tagger found, so let\'s use a simulated oscillator instead.')

    tagger = TimeTagger.createTimeTaggerVirtual()
    tagger.run(speed=1.0)
    osc = TimeTagger.Experimental.OscillatorSimulation(
        tagger,
        nominal_frequency=TARGET_FREQUENCY,
        coeff_phase_white=1.0e-12,  # White and flicker PM matches the Time Tagger X
        coeff_phase_flicker=75e-15,
        # coeff_freq_white=1e-12,
        coeff_freq_flicker=10e-12,
        # coeff_random_drift=10e-12,
        coeff_linear_drift=10e-12,
    )
    CHANNEL = osc.getChannel()

# Calibrate to roughly 10 MHz
for loop in range(2):
    with TimeTagger.Countrate(tagger=tagger, channels=[CHANNEL]) as cnt:
        cnt.startFor(1e12)
        cnt.waitUntilFinished()
        init_freq = cnt.getData()[0]
        print(
            f'Measuring on Channel {CHANNEL} with the initial frequency {init_freq * 1e-6:.3f} MHz.')

        if loop == 0:
            divider = int(np.round(init_freq / TARGET_FREQUENCY))
            if divider > 1:
                print(f'Set the Event Divider to {divider}.')
                tagger.setEventDivider(channel=CHANNEL, divider=divider)
            else:
                break

#######################
# Create measurements #
#######################

# The general case: Periodic phase measurement
freq_count = TimeTagger.FrequencyCounter(
    tagger=tagger,
    channels=[CHANNEL],
    sampling_interval=1e12/FC_SAMPLE_RATE,
    fitting_window=1e12/FC_SAMPLE_RATE,
)

# FFT based quasi-logarithmic phase noise measurement
phase_noise = TimeTagger.PhaseNoise(
    tagger=tagger,
    channel=CHANNEL,
    samples_per_octave=SAMPLES_PER_OCTAVE,
)

# Quasi-logarithmic Allan deviation measurements
# This yields a partially overlapping Allan deviation
# From the fundamental period up to 2**40 periods (one month for 10MHz).
fs_steps = (
    (2**0, np.unique(np.logspace(start=0, stop=4,
     num=4*4+1, base=2, dtype=np.int64))[:-1]),
    (2**2, np.unique(np.logspace(start=2, stop=8,
     num=4*6+1, base=2, dtype=np.int64))[:-1]),
    (2**8, np.unique(np.logspace(start=2, stop=12,
     num=4*10+1, base=2, dtype=np.int64))[:-1]),
    (2**16, np.unique(np.logspace(start=4, stop=24,
     num=4*20+1, base=2, dtype=np.int64))[:-1]),
)
frequency_stability = [TimeTagger.FrequencyStability(
    tagger=tagger, channel=CHANNEL, steps=steps, average=avg) for avg, steps in fs_steps]

######################
# Main plotting loop #
######################
fig = plt.figure()
while True:
    plt.pause(1)
    plt.clf()

    # Fetch data from measurements
    fc_obj = freq_count.getDataObject()
    pn_obj = phase_noise.getDataObject()
    fs_obj = [fs.getDataObject() for fs in frequency_stability]

    # Show runtime statistics within the window title
    if fc_obj.size > 1:
        fig.canvas.manager.set_window_title(
            f'Phase Noise Analyzer, Frequency: {fc_obj.getFrequency()[0][-1] * 1e-6:.3f} MHz, Runtime: {phase_noise.getCaptureDuration() * 1e-12:.1f} s')

    # Stop this example after 30 seconds, just remove this line if you want to make real tests
    if phase_noise.getCaptureDuration() > 30e12:
        break

    ###################
    # Plot everything #
    ###################

    # Plot phase noise
    plt.subplot(311)
    plt.semilogx(pn_obj.getOffset(),
                 pn_obj.getPhaseNoise(), label='PhaseNoise')
    plt.grid(True, which='major', ls='-')
    plt.grid(True, which='minor', ls=':')
    plt.xlabel('Frequency offset (Hz)')
    plt.ylabel('SSB phase noise (dBc/Hz)')
    plt.legend()
    plt.title(
        'Phase noise analysis: From mHz offsets up to Nyquist. With integrated RMS jitter estimator.')

    # With integrated RMS jitter estimator
    pn = pn_obj.getPhaseNoise()
    off = pn_obj.getOffset()
    low_stop = np.min(pn) - 25
    for limits in JITTER_INT_LIMITS:
        mask = (limits[0] < off) & (off <= limits[1])

        jitter = pn_obj.getIntegratedJitter(limits[0], limits[1]) * 1e12
        plt.fill_between(off[mask], pn[mask], low_stop, alpha=.3)
        plt.text(np.sqrt(limits[0] * limits[1]), 10 + low_stop,
                 f'Integrated jitter: {jitter:.3g} ps', horizontalalignment='center', verticalalignment='center_baseline')

    # Plot Time deviation
    plt.subplot(323)
    plt.loglog(np.concatenate([obj.getTau() for obj in fs_obj]), np.concatenate(
        [obj.getTDEV() for obj in fs_obj]), label='FrequencyStability: TDEV')
    plt.grid(True, which='major', ls='-')
    plt.grid(True, which='minor', ls=':')
    plt.xlabel('Tau (s)')
    plt.ylabel('Time deviation (s)')
    plt.legend()
    plt.title('Frequency stability analysis: Starting from fundamental period.')

    # Plot Allan deviation
    plt.subplot(324)
    plt.loglog(np.concatenate([obj.getTau() for obj in fs_obj]), np.concatenate(
        [obj.getMDEV() for obj in fs_obj]), label='FrequencyStability: MDEV')
    plt.grid(True, which='major', ls='-')
    plt.grid(True, which='minor', ls=':')
    plt.xlabel('Tau (s)')
    plt.ylabel('Modified Allan deviation (1)')
    plt.legend()
    plt.title('Frequency stability analysis: Analyzing up to weeks.')

    # Plot frequency trace
    plt.subplot(325)
    plt.plot(fc_obj.getTime() * 1e-12,
             fc_obj.getFrequency().T, label='getFrequency')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.title('Flexible usage: Frequency measured with FrequencyCounter')

    # Plot time trace
    plt.subplot(326)
    plt.plot(fc_obj.getTime() * 1e-12, fc_obj.getPhase(TARGET_FREQUENCY).T,
             label=f'getPhase({TARGET_FREQUENCY:.2g})')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (1)')
    plt.legend()
    plt.title('Flexible usage: Phase measured with FrequencyCounter')

    plt.tight_layout()

# Close the connection to the Time Tagger
TimeTagger.freeTimeTagger(tagger)
