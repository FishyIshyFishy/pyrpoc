import numpy as np
from Swabian import TimeTagger


def get_delay_and_jitter(x, y):
    mean = np.average(x, weights=y)
    std = np.sqrt(np.average((x-mean)**2, weights=y))
    return mean, std


tagger = TimeTagger.createTimeTagger()
tagger.setTestSignal(1, True)
tagger.setTestSignal(2, True)
tagger.setTriggerLevel(channel=1, voltage=0.5)
tagger.setTriggerLevel(channel=2, voltage=0.5)

correlation = TimeTagger.Correlation(tagger=tagger, channel_1=1, channel_2=2, binwidth=10, n_bins=5000)
correlation.startFor(capture_duration=int(5E12))
correlation.waitUntilFinished()
delay, jitter_rms = get_delay_and_jitter(correlation.getIndex(), correlation.getData())
print(f'measured delay is {delay:.1f} ps with an RMS of {jitter_rms:.1f} ps.')

tagger.setInputDelay(channel=2, delay=int(round(delay)))
print(f'input delay channel 2 set to been set to {int(round(delay))} to compensate the time offset of the test signal.')

correlation.startFor(capture_duration=int(5E12))
correlation.waitUntilFinished()
delay, jitter_rms = get_delay_and_jitter(correlation.getIndex(), correlation.getData())
print(f'new measured delay is {delay:.1f} ps with an RMS of {jitter_rms:.1f} ps.')

TimeTagger.freeTimeTagger(tagger)
