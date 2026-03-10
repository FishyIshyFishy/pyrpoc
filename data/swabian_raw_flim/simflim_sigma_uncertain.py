import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import minimize_scalar


def exgauss_pdf(t, tau, sigma, t0):
    z = (sigma**2 / tau - (t - t0)) / (sigma * np.sqrt(2))
    prefactor = 1 / (2 * tau)
    expo = np.exp(sigma**2 / (2 * tau**2) - (t - t0) / tau)
    return prefactor * expo * erfc(z)


def sample_exgauss(n, tau, sigma, t0, rng):
    g = rng.normal(loc=0.0, scale=sigma, size=n)
    e = rng.exponential(scale=tau, size=n)
    return t0 + g + e


def neg_log_likelihood_tau(tau, t, sigma, t0, eps=1e-300):
    if tau <= 0:
        return np.inf
    p = exgauss_pdf(t, tau, sigma, t0)
    p = np.clip(p, eps, None)
    return -np.sum(np.log(p))


def fit_tau_mle(t, sigma, t0, tau_bounds):
    res = minimize_scalar(
        neg_log_likelihood_tau,
        bounds=tau_bounds,
        args=(t, sigma, t0),
        method='bounded'
    )
    return res.x, res.fun, res.success


def draw_positive_normal(mean, std, rng, min_value=1e-6):
    x = rng.normal(mean, std)
    while x <= min_value:
        x = rng.normal(mean, std)
    return x


def monte_carlo_tau_scan_with_sigma_uncertainty(
    sigma_assumed,
    sigma_hyper_std,
    t0_assumed,
    n_photons,
    alphas,
    n_trials=1000,
    tau_bounds_factor=(0.01, 10.0),
    seed=0,
    verbose=True
):
    rng = np.random.default_rng(seed)

    out = {
        'alpha': [],
        'tau_true': [],
        'tau_mean': [],
        'tau_std': [],
        'tau_mean_minus_std': [],
        'tau_mean_plus_std': [],
        'fail_rate': [],
        'sigma_true_mean': [],
        'sigma_true_std': [],
        'n_success': [],
        'n_fail': []
    }

    for alpha in alphas:
        tau_true = alpha * sigma_assumed
        tau_hats = []
        sigma_trues = []
        n_fail = 0

        lower = tau_bounds_factor[0] * sigma_assumed
        upper = tau_bounds_factor[1] * sigma_assumed

        for _ in range(n_trials):
            sigma_true = draw_positive_normal(
                mean=sigma_assumed,
                std=sigma_hyper_std,
                rng=rng
            )

            sigma_trues.append(sigma_true)

            t = sample_exgauss(n_photons, tau_true, sigma_true, t0_assumed, rng)

            try:
                tau_hat, _, success = fit_tau_mle(
                    t,
                    sigma=sigma_assumed,
                    t0=t0_assumed,
                    tau_bounds=(lower, upper)
                )
                if success and np.isfinite(tau_hat):
                    tau_hats.append(tau_hat)
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1

        tau_hats = np.array(tau_hats, dtype=float)
        sigma_trues = np.array(sigma_trues, dtype=float)

        if tau_hats.size == 0:
            tau_mean = np.nan
            tau_std = np.nan
            tau_mean_minus_std = np.nan
            tau_mean_plus_std = np.nan
        else:
            tau_mean = np.mean(tau_hats)
            tau_std = np.std(tau_hats, ddof=1) if tau_hats.size > 1 else 0.0
            tau_mean_minus_std = tau_mean - tau_std
            tau_mean_plus_std = tau_mean + tau_std

        fail_rate = n_fail / n_trials

        out['alpha'].append(alpha)
        out['tau_true'].append(tau_true)
        out['tau_mean'].append(tau_mean)
        out['tau_std'].append(tau_std)
        out['tau_mean_minus_std'].append(tau_mean_minus_std)
        out['tau_mean_plus_std'].append(tau_mean_plus_std)
        out['fail_rate'].append(fail_rate)
        out['sigma_true_mean'].append(np.mean(sigma_trues))
        out['sigma_true_std'].append(np.std(sigma_trues, ddof=1))
        out['n_success'].append(tau_hats.size)
        out['n_fail'].append(n_fail)

        if verbose:
            rel_err = (tau_mean - tau_true) / tau_true if np.isfinite(tau_mean) else np.nan
            print(
                f'alpha={alpha:.5f}, tau_true={tau_true:.5f}, '
                f'tau_mean={tau_mean:.5f}, tau_std={tau_std:.5f}, '
                f'rel_bias={rel_err:.5f}, fail_rate={fail_rate:.4f}'
            )

    for k in out:
        out[k] = np.array(out[k], dtype=float)

    return out


if __name__ == '__main__':
    sigma_assumed = 0.20
    sigma_hyper_std = 0.05
    t0_assumed = 0.0
    n_photons = 1000

    alpha_min = 0.02
    alpha_max = 0.5
    n_alpha = 100
    n_trials = 1000

    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)

    results = monte_carlo_tau_scan_with_sigma_uncertainty(
        sigma_assumed=sigma_assumed,
        sigma_hyper_std=sigma_hyper_std,
        t0_assumed=t0_assumed,
        n_photons=n_photons,
        alphas=alphas,
        n_trials=n_trials,
        tau_bounds_factor=(0.01, 10.0),
        seed=1,
        verbose=True
    )

    lower_band = np.maximum(results['tau_mean_minus_std'], 0)

    rel_err = (np.asarray(results['tau_true']) - np.asarray(results['tau_mean']))/np.asarray(results['tau_true'])
    # rel_err = np.abs(rel_err)

    plt.plot(results['alpha'], rel_err, 'k-')
    # ax[0].plot(results['alpha'], results['tau_mean'], label='mean extracted tau')
    # ax[0].fill_between(
    #     results['alpha'],
    #     lower_band,
    #     results['tau_mean_plus_std'],
    #     alpha=0.3,
    #     label='mean ± std'
    # )
    # ax[0].set_xscale('log')
    plt.xlabel(r'$\alpha = \tau/\sigma$')
    plt.ylabel(r'rel err')
    plt.title(f'sigma = {sigma_assumed} +/- {sigma_hyper_std} ns, t0 = {t0_assumed} ns, photons = {n_photons}')
    plt.legend()
    plt.tight_layout()
    plt.show()