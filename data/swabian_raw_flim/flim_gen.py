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
    out = []
    for yy, xx in iter_neighborhood_indices(y, x, ny, nx, half):
        d = frame[yy, xx]
        if d is None:
            continue
        if len(d):
            out.append(d)
    if not out:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(out)


def robust_lifetime_from_delays_ps(delays_ps, gate_min_ps=None, gate_max_ps=None):
    if delays_ps.size == 0:
        return np.nan

    d = delays_ps.astype(np.float64, copy=False)

    if gate_min_ps is not None:
        d = d[d >= gate_min_ps]
    if gate_max_ps is not None:
        d = d[d <= gate_max_ps]

    if d.size < 20:
        return np.nan

    # robust outlier handling: clip to central bulk
    lo = np.percentile(d, 2.0)
    hi = np.percentile(d, 98.0)
    d = d[(d >= lo) & (d <= hi)]
    if d.size < 20:
        return np.nan

    # very simple 1-exp model under ideal conditions: tau ~= mean(t - t0)
    # estimate t0 as the onset (low percentile) after gating/clipping
    t0 = np.percentile(d, 5.0)
    dt = d - t0
    dt = dt[dt >= 0]
    if dt.size < 20:
        return np.nan

    # MLE for exponential with unknown amplitude and known t0 is mean(dt)
    tau_ps = float(np.mean(dt))
    return tau_ps


def flim_lifetime_map_from_raw(
    frames,
    window_n=5,
    gate_min_ps=0,
    gate_max_ps=None,
    min_photons=50,
):
    if frames.ndim != 3:
        raise ValueError(f"expected frames to be (n_frames, ny, nx) object array, got {frames.shape}")

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
            tau_ps_map[y, x] = robust_lifetime_from_delays_ps(
                delays,
                gate_min_ps=gate_min_ps,
                gate_max_ps=gate_max_ps,
            )

    return tau_ps_map, photons_map

def collect_all_delays(frame):
    ny, nx = frame.shape
    all_delays = []
    for y in range(ny):
        for x in range(nx):
            d = frame[y, x]
            if d is not None and len(d):
                all_delays.append(d)
    if not all_delays:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(all_delays)


if __name__ == "__main__":
    npz_path = r'C:\Users\ishaa\Documents\ZhangLab\python\new_pysrs\data\swabian_raw_flim\20260304_125152_flim_test_001_raw.npz'

    data = np.load(npz_path, allow_pickle=True)
    frames = data["frames"]  # (1, 512, 512) object array of int64 arrays (ps)

    # choose a median/spatial window size (odd), and some reasonable photon threshold
    window_n = 11  # NxN aggregation for robustness
    min_photons = 100

    # optional gates (ps). if you know your rep period is 12.5 ns at 80 MHz, you can cap at 12500 ps.
    gate_min_ps = 0
    gate_max_ps = 12_500

    tau_ps, nphot = flim_lifetime_map_from_raw(
        frames,
        window_n=window_n,
        gate_min_ps=gate_min_ps,
        gate_max_ps=gate_max_ps,
        min_photons=min_photons,
    )

    # display lifetime map in ns
    tau_ns = tau_ps / 1000.0

    plt.figure(figsize=(7, 6))
    im = plt.imshow(tau_ns, origin="upper")
    plt.title(f"lifetime map (ns), window {window_n}x{window_n}, min photons {min_photons}")
    plt.xlabel("x")
    plt.ylabel("y")
    cb = plt.colorbar(im)
    cb.set_label("tau (ns)")
    plt.tight_layout()
    plt.show()

    # optional: show photon counts used per pixel (after window aggregation)
    plt.figure(figsize=(7, 6))
    im2 = plt.imshow(nphot, origin="upper")
    plt.title(f"aggregated photon counts, window {window_n}x{window_n}")
    plt.xlabel("x")
    plt.ylabel("y")
    cb2 = plt.colorbar(im2)
    cb2.set_label("counts")
    plt.tight_layout()
    plt.show()

    frame = frames[0]

    all_delays_ps = collect_all_delays(frame)

    # optional gating
    gate_min_ps = 0
    gate_max_ps = 12_500  # one laser period for 80 MHz

    mask = all_delays_ps >= gate_min_ps
    if gate_max_ps is not None:
        mask &= all_delays_ps <= gate_max_ps

    all_delays_ps = all_delays_ps[mask]

    # histogram binning
    bin_width_ps = 50
    bins = np.arange(0, gate_max_ps + bin_width_ps, bin_width_ps)

    counts, edges = np.histogram(all_delays_ps, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(7,5))
    plt.plot(centers/1000.0, counts)
    plt.xlabel("delay (ns)")
    plt.ylabel("counts")
    plt.title("global FLIM decay (summed over image)")
    plt.tight_layout()
    plt.show()

    # optional semilog view (very common for FLIM)
    plt.figure(figsize=(7,5))
    plt.semilogy(centers/1000.0, counts)
    plt.xlabel("delay (ns)")
    plt.ylabel("counts (log)")
    plt.title("global FLIM decay (semilog)")
    plt.tight_layout()
    plt.show()
