import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc

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
rep_period_ps = 12500
rep_period_ns = rep_period_ps / 1000.0
wrapped = (all_delays % rep_period_ps) / 1000.0   # ns

# histogram
bin_width_ps = 50
bin_width_ns = bin_width_ps / 1000.0
bins_ns = np.arange(0, rep_period_ns + bin_width_ns, bin_width_ns)

counts, edges = np.histogram(wrapped, bins=bins_ns)
centers = 0.5 * (edges[:-1] + edges[1:])

print('histogram bins:', len(centers))

def exgauss_pdf(t, mu, sigma, tau):
    # gaussian convolved with 1-sided exponential
    # units of 1/ns if t, mu, sigma, tau are in ns
    z = (sigma**2 / tau - (t - mu)) / (np.sqrt(2) * sigma)
    return (1.0 / (2.0 * tau)) * np.exp((sigma**2) / (2.0 * tau**2) - (t - mu) / tau) * erfc(z)

def wrapped_exgauss_counts(t, A, mu, sigma, tau, C, T, n_wraps):
    # histogram counts model on [0, T)
    y = np.zeros_like(t, dtype=np.float64)
    for n in range(n_wraps + 1):
        y += exgauss_pdf(t + n * T, mu, sigma, tau)
    return A * y + C

def fit_model(t, A, mu, sigma, tau, C):
    return wrapped_exgauss_counts(
        t=t,
        A=A,
        mu=mu,
        sigma=sigma,
        tau=tau,
        C=C,
        T=rep_period_ns,
        n_wraps=8
    )

# initial guesses
imax = np.argmax(counts)
mu0 = centers[imax]
sigma0 = 0.15   # ns, initial IRF width guess
tau0 = 2.0      # ns, initial lifetime guess
C0 = np.percentile(counts, 10)

# rough amplitude guess
shape0 = wrapped_exgauss_counts(
    centers,
    A=1.0,
    mu=mu0,
    sigma=sigma0,
    tau=tau0,
    C=0.0,
    T=rep_period_ns,
    n_wraps=8
)
A0 = (counts.max() - C0) / max(shape0.max(), 1e-12)

p0 = [A0, mu0, sigma0, tau0, C0]

# bounds
lower = [0.0, 0.0, 0.01, 2, 0.0]
upper = [np.inf, rep_period_ns, 2.0, 6, np.inf]

# poisson-ish weighting
sigma_y = np.sqrt(np.maximum(counts, 1.0))

popt, pcov = curve_fit(
    fit_model,
    centers,
    counts,
    p0=p0,
    bounds=(lower, upper),
    sigma=sigma_y,
    absolute_sigma=False,
    maxfev=50000
)

A_fit, mu_fit, sigma_fit, tau_fit, C_fit = popt

fit_counts = fit_model(centers, *popt)

print()
print('fit results')
print('-----------')
print(f'A     = {A_fit:.6g}')
print(f'mu    = {mu_fit:.6f} ns')
print(f'sigma = {sigma_fit:.6f} ns')
print(f'tau   = {tau_fit:.6f} ns')
print(f'C     = {C_fit:.6f} counts/bin')

# optional: separate unwrapped one-period model for intuition
t_dense = np.linspace(0, rep_period_ns, 4000)
fit_dense = fit_model(t_dense, *popt)

fig, ax = plt.subplots(2, 1, figsize=(7, 8))

# linear
ax[0].bar(centers, counts, width=bin_width_ns, color='black', alpha=0.7, label='data')
ax[0].plot(t_dense, fit_dense, 'r-', linewidth=2, label=f't0={mu_fit:.2f}, sig={sigma_fit:.3f}, tau={tau_fit:.3f}')
ax[0].set_title('G*ME fit')
ax[0].set_xlabel('delay (ns)')
ax[0].set_ylabel('counts')
ax[0].legend()

# log
ax[1].semilogy(centers, np.maximum(counts, 1), 'k-', linewidth=1.2)
ax[1].semilogy(t_dense, np.maximum(fit_dense, 1e-12), 'r-', linewidth=2)
ax[1].set_title('same fit on log scale')
ax[1].set_xlabel('delay (ns)')
ax[1].set_ylabel('counts')

plt.tight_layout()
plt.show()