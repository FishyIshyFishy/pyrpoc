import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def iter_neighborhood_indices(y, x, ny, nx, half):
    y0 = max(0, y - half)
    y1 = min(ny, y + half + 1)
    x0 = max(0, x - half)
    x1 = min(nx, x + half + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            yield yy, xx


def aggregate_delays_in_window(frame, y, x, half):
    ny, nx = frame.shape
    chunks = []
    for yy, xx in iter_neighborhood_indices(y, x, ny, nx, half):
        d = frame[yy, xx]
        if d is None:
            continue
        if len(d):
            chunks.append(d)
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(chunks)


def collect_all_delays(frame):
    ny, nx = frame.shape
    chunks = []
    for y in range(ny):
        for x in range(nx):
            d = frame[y, x]
            if d is None:
                continue
            if len(d):
                chunks.append(d)
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(chunks)


def wrap_delays_to_period_ps(delays_ps, laser_frequency_hz):
    period_ps = 1e12 / float(laser_frequency_hz)
    period_ps_int = int(round(period_ps))
    if period_ps_int <= 0:
        raise ValueError('laser_frequency_hz gives invalid period')

    d = delays_ps.astype(np.int64, copy=False)
    d = d[d >= 0]
    d = d % period_ps_int
    return d, period_ps_int


def lifetime_mle_exponential_ps(microtimes_ps, min_photons=50, onset_percentile=5.0):
    if microtimes_ps.size < min_photons:
        return np.nan

    # robust clip (avoid extreme wrap/garbage)
    lo = np.percentile(microtimes_ps, 2.0)
    hi = np.percentile(microtimes_ps, 98.0)
    d = microtimes_ps[(microtimes_ps >= lo) & (microtimes_ps <= hi)]
    if d.size < min_photons:
        return np.nan

    # estimate onset t0, then tau = mean(t - t0) (shifted exponential MLE)
    t0 = np.percentile(d, onset_percentile)
    dt = d - t0
    dt = dt[dt >= 0]
    if dt.size < min_photons:
        return np.nan

    return float(np.mean(dt))


def flim_lifetime_map_from_raw(
    frames,
    laser_frequency_hz,
    window_n=11,
    min_photons=200,
    onset_percentile=5.0,
):
    if frames.ndim != 3:
        raise ValueError(f'expected frames to be (n_frames, ny, nx) object array, got {frames.shape}')

    frame = frames[0]
    ny, nx = frame.shape
    half = window_n // 2

    tau_ps_map = np.full((ny, nx), np.nan, dtype=np.float32)
    photons_map = np.zeros((ny, nx), dtype=np.int32)

    for y in trange(ny):
        for x in range(nx):
            delays = aggregate_delays_in_window(frame, y, x, half)
            photons_map[y, x] = int(delays.size)
            if delays.size < min_photons:
                continue

            micro, _ = wrap_delays_to_period_ps(delays, laser_frequency_hz)
            tau_ps_map[y, x] = lifetime_mle_exponential_ps(
                micro,
                min_photons=min_photons,
                onset_percentile=onset_percentile,
            )

    return tau_ps_map, photons_map


def plot_global_decay(frames, laser_frequency_hz, bin_width_ps=50):
    frame = frames[0]
    all_delays = collect_all_delays(frame)
    micro, period_ps = wrap_delays_to_period_ps(all_delays, laser_frequency_hz)

    bins = np.arange(0, period_ps + bin_width_ps, bin_width_ps)
    counts, edges = np.histogram(micro, bins=bins)
    print(np.sum(counts))
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(7, 5))
    plt.plot(centers / 1000.0, counts)
    plt.xlabel('delay (ns)')
    plt.ylabel('counts')
    plt.title('global FLIM decay (microtime, summed over image)')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.semilogy(centers / 1000.0, np.maximum(counts, 1))
    plt.xlabel('delay (ns)')
    plt.ylabel('counts (log)')
    plt.title('global FLIM decay (microtime, semilog)')
    plt.tight_layout()
    plt.show()

    return centers, counts, period_ps


if __name__ == '__main__':
    npz_path = r'C:\Users\ishaa\Documents\ZhangLab\python\new_pysrs\data\swabian_raw_flim\pixelwise_timetags.npz'

    data = np.load(npz_path, allow_pickle=True)
    frames = data['frames']  # (1, 512, 512) object array

    # ---- user knobs
    laser_frequency_hz = 80e6
    window_n = 1         # spatial aggregation window (odd)
    min_photons = 5      # per-window photon threshold
    onset_percentile = 5.0 # onset estimate for shifted exponential MLE
    bin_width_ps = 50      # for global decay plotting

    # ---- plot global decay first (useful sanity check)
    plot_global_decay(frames, laser_frequency_hz, bin_width_ps=bin_width_ps)

    # ---- compute lifetime map
    tau_ps, nphot = flim_lifetime_map_from_raw(
        frames,
        laser_frequency_hz=laser_frequency_hz,
        window_n=window_n,
        min_photons=min_photons,
        onset_percentile=onset_percentile,
    )

    fig, ax = plt.subplots(1, 2)

    # ---- show photon count map (after spatial aggregation)
    im = ax[0].imshow(nphot, origin='upper')
    ax[0].set_title(f'counts per pixel in window {window_n}x{window_n}')
    cb = fig.colorbar(im, ax=ax[0])
    cb.set_label('counts')


    # ---- show lifetime map (ns)
    tau_ns = tau_ps / 1000.0
    tau_ns_masked = np.ma.masked_invalid(tau_ns)
    cmap = plt.cm.cool.copy()
    cmap.set_bad('black')
    im2 = ax[1].imshow(
        tau_ns_masked,
        origin='upper',
        cmap=cmap,
        interpolation='nearest'
    )

    ax[1].set_title(f'lifetime map (ns), window {window_n}x{window_n}, min photons {min_photons}')
    cb2 = fig.colorbar(im2, ax=ax[1])
    cb2.set_label('tau (ns)')

    plt.tight_layout()
    plt.show()
