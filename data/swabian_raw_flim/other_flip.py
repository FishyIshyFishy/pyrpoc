import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

npz_path = r'C:\Users\ishaa\Documents\ZhangLab\python\new_pysrs\data\swabian_raw_flim\20260304_125152_flim_test_001_raw.npz'

data = np.load(npz_path, allow_pickle=True)
frames = data['frames']

print('frames shape:', frames.shape)

frame = frames[0]
ny, nx = frame.shape

# collect all photon delays
all_delays = []

for y in range(ny):
    for x in range(nx):
        d = frame[y, x]
        if d is not None and len(d) > 0:
            all_delays.append(d)

all_delays = np.concatenate(all_delays)

print('total photons:', all_delays.size)
print('min delay (ps):', all_delays.min())
print('max delay (ps):', all_delays.max())

# fold into one laser period
rep_period_ps = 12500   # 80 MHz laser
wrapped = all_delays % rep_period_ps

# histogram settings
bin_width = 50
bins = np.arange(0, rep_period_ps + bin_width, bin_width)

counts, edges = np.histogram(wrapped, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])

# find peak bin
imax = np.argmax(counts)
t0_ps = centers[imax]
print('estimated t0 from peak (ps):', t0_ps)
print('estimated t0 from peak (ns):', t0_ps / 1000)

# circularly shift histogram so the peak becomes t = 0
counts_shifted = np.roll(counts, -imax)

# recalculate times so they run from 0 to rep_period_ps after the shift
shifted_centers_ps = np.arange(len(counts_shifted)) * bin_width + bin_width / 2
shifted_centers_ns = shifted_centers_ps / 1000

# exponential model
def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

# choose fit region after the peak
# exclude the first few bins near t=0 if desired, since they may contain IRF distortion
fit_start_ns = 0.0
fit_end_ns = 10.0

fit_mask = (shifted_centers_ns >= fit_start_ns) & (shifted_centers_ns <= fit_end_ns) & (counts_shifted > 0)

x_fit = shifted_centers_ns[fit_mask]
y_fit = counts_shifted[fit_mask]

# initial guesses
A0 = y_fit.max()
tau0 = 1.0

popt, pcov = curve_fit(exp_decay, x_fit, y_fit, p0=[A0, tau0], maxfev=10000)
A_fit, tau_fit = popt

print('fit A:', A_fit)
print('fit tau (ns):', tau_fit)

y_model = exp_decay(x_fit, A_fit, tau_fit)

fig, ax = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

# linear scale
ax[0].bar(shifted_centers_ns, counts_shifted, width=bin_width / 1000, color='black', alpha=0.7, label='shifted data')
ax[0].plot(x_fit, y_model, 'r-', linewidth=2, label=f'fit: tau = {tau_fit:.3f} ns')
ax[0].set_title('linear shifted')
ax[0].legend()

# log scale
ax[1].semilogy(shifted_centers_ns, np.maximum(counts_shifted, 1), color='black', label='shifted data')
ax[1].semilogy(x_fit, y_model, 'r-', linewidth=2, label=f'fit: tau = {tau_fit:.3f} ns')
ax[1].set_title('log shifted')
ax[1].set_xlabel('time from "excitation" (ns)')
ax[1].legend()

plt.tight_layout()
plt.show()