import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, cpu_count


npz_path = r'C:\Users\ishaa\Documents\ZhangLab\python\new_pysrs\data\swabian_raw_flim\20260304_125152_flim_test_001_raw.npz'

rep_period_ps = 12500
bin_width_ps = 50
window_n = 9

fit_start_ns = 0.0
fit_end_ns = 10.0
min_total_photons = 500
min_nonzero_bins = 8

max_workers = max(1, cpu_count() - 1)


def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)


def build_per_pixel_histograms(frame, bins, rep_period_ps):
    ny, nx = frame.shape
    nbins = len(bins) - 1
    hist_cube = np.zeros((ny, nx, nbins), dtype=np.uint32)

    for y in range(ny):
        print(f'building per-pixel histograms: row {y+1}/{ny}')
        for x in range(nx):
            d = frame[y, x]
            if d is None or len(d) == 0:
                continue

            wrapped = d % rep_period_ps
            counts, _ = np.histogram(wrapped, bins=bins)
            hist_cube[y, x, :] = counts.astype(np.uint32)

    return hist_cube


def window_sum_histograms(hist_cube, window_n):
    half = window_n // 2
    ny, nx, nbins = hist_cube.shape

    prefix = np.zeros((ny + 1, nx + 1, nbins), dtype=np.uint32)
    prefix[1:, 1:, :] = hist_cube.cumsum(axis=0).cumsum(axis=1)

    y = np.arange(ny)
    x = np.arange(nx)

    y0 = np.clip(y - half, 0, ny)
    y1 = np.clip(y + half + 1, 0, ny)
    x0 = np.clip(x - half, 0, nx)
    x1 = np.clip(x + half + 1, 0, nx)

    summed = (
        prefix[y1[:, None], x1[None, :], :]
        - prefix[y0[:, None], x1[None, :], :]
        - prefix[y1[:, None], x0[None, :], :]
        + prefix[y0[:, None], x0[None, :], :]
    )
    print(np.max(summed))

    return summed


_worker_cfg = {}


def init_worker(
    shm_hist_name,
    hist_shape,
    hist_dtype_str,
    shm_counts_name,
    counts_shape,
    counts_dtype_str,
    shm_valid_name,
    valid_shape,
    valid_dtype_str,
    fit_start_ns,
    fit_end_ns,
    bin_width_ps,
    min_nonzero_bins
):
    global _worker_cfg

    shm_hist = shared_memory.SharedMemory(name=shm_hist_name)
    shm_counts = shared_memory.SharedMemory(name=shm_counts_name)
    shm_valid = shared_memory.SharedMemory(name=shm_valid_name)

    _worker_cfg['summed_hist'] = np.ndarray(
        hist_shape,
        dtype=np.dtype(hist_dtype_str),
        buffer=shm_hist.buf
    )
    _worker_cfg['counts_map'] = np.ndarray(
        counts_shape,
        dtype=np.dtype(counts_dtype_str),
        buffer=shm_counts.buf
    )
    _worker_cfg['valid_mask'] = np.ndarray(
        valid_shape,
        dtype=np.dtype(valid_dtype_str),
        buffer=shm_valid.buf
    )

    _worker_cfg['shm_hist'] = shm_hist
    _worker_cfg['shm_counts'] = shm_counts
    _worker_cfg['shm_valid'] = shm_valid

    _worker_cfg['fit_start_ns'] = float(fit_start_ns)
    _worker_cfg['fit_end_ns'] = float(fit_end_ns)
    _worker_cfg['bin_width_ps'] = float(bin_width_ps)
    _worker_cfg['min_nonzero_bins'] = int(min_nonzero_bins)


def fit_rows(row_range):
    y_start, y_end = row_range

    summed_hist = _worker_cfg['summed_hist']
    valid_mask = _worker_cfg['valid_mask']

    fit_start_ns = _worker_cfg['fit_start_ns']
    fit_end_ns = _worker_cfg['fit_end_ns']
    bin_width_ps = _worker_cfg['bin_width_ps']
    min_nonzero_bins = _worker_cfg['min_nonzero_bins']

    nrows = y_end - y_start
    nx = summed_hist.shape[1]
    nbins = summed_hist.shape[2]

    tau_chunk = np.full((nrows, nx), np.nan, dtype=np.float64)

    shifted_centers_ps = np.arange(nbins, dtype=np.float64) * bin_width_ps + bin_width_ps / 2.0
    shifted_centers_ns = shifted_centers_ps / 1000.0
    fit_time_mask = (shifted_centers_ns >= fit_start_ns) & (shifted_centers_ns <= fit_end_ns)

    for local_y, y in enumerate(range(y_start, y_end)):
        valid_x = np.flatnonzero(valid_mask[y])

        if valid_x.size == 0:
            continue

        print(f'worker fitting row {y+1}/{summed_hist.shape[0]} ({valid_x.size} valid pixels)')

        for x in valid_x:
            counts = summed_hist[y, x, :]

            imax = np.argmax(counts)
            counts_shifted = np.roll(counts, -imax).astype(np.float64)

            fit_mask = fit_time_mask & (counts_shifted > 0)
            x_fit = shifted_centers_ns[fit_mask].astype(np.float64)
            y_fit = counts_shifted[fit_mask].astype(np.float64)

            if x_fit.size < min_nonzero_bins:
                continue

            A0 = float(y_fit.max())
            tau0 = 1.0

            try:
                popt, _ = curve_fit(
                    exp_decay,
                    x_fit,
                    y_fit,
                    p0=[A0, tau0],
                    bounds=([0.0, 0.0], [np.inf, np.inf]),
                    maxfev=10000
                )
                tau_chunk[local_y, x] = float(popt[1])
            except Exception:
                continue

    return y_start, y_end, tau_chunk


if __name__ == '__main__':
    data = np.load(npz_path, allow_pickle=True)
    frames = data['frames']

    print('frames shape:', frames.shape)

    frame = frames[0]
    ny, nx = frame.shape

    bins = np.arange(0, rep_period_ps + bin_width_ps, bin_width_ps)

    hist_cube = build_per_pixel_histograms(frame, bins, rep_period_ps)
    print('per-pixel histogram cube shape:', hist_cube.shape)

    summed_hist = window_sum_histograms(hist_cube, window_n)
    print('window-summed histogram cube shape:', summed_hist.shape)

    counts_map = summed_hist.sum(axis=2).astype(np.uint32)

    # only pixels meeting the threshold will be fit at all
    valid_mask = counts_map >= min_total_photons
    print(f'valid pixels to fit: {valid_mask.sum()} / {valid_mask.size}')

    shm_hist = shared_memory.SharedMemory(create=True, size=summed_hist.nbytes)
    shm_counts = shared_memory.SharedMemory(create=True, size=counts_map.nbytes)
    shm_valid = shared_memory.SharedMemory(create=True, size=valid_mask.nbytes)

    try:
        summed_hist_shared = np.ndarray(summed_hist.shape, dtype=summed_hist.dtype, buffer=shm_hist.buf)
        counts_map_shared = np.ndarray(counts_map.shape, dtype=counts_map.dtype, buffer=shm_counts.buf)
        valid_mask_shared = np.ndarray(valid_mask.shape, dtype=valid_mask.dtype, buffer=shm_valid.buf)

        summed_hist_shared[:] = summed_hist
        counts_map_shared[:] = counts_map
        valid_mask_shared[:] = valid_mask

        del summed_hist
        del hist_cube
        del valid_mask

        n_workers = min(max_workers, ny)
        chunk_size = int(np.ceil(ny / n_workers))
        row_ranges = []

        for y0 in range(0, ny, chunk_size):
            y1 = min(ny, y0 + chunk_size)
            row_ranges.append((y0, y1))

        tau_map = np.full((ny, nx), np.nan, dtype=np.float64)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(
                shm_hist.name,
                summed_hist_shared.shape,
                summed_hist_shared.dtype.str,
                shm_counts.name,
                counts_map_shared.shape,
                counts_map_shared.dtype.str,
                shm_valid.name,
                valid_mask_shared.shape,
                valid_mask_shared.dtype.str,
                fit_start_ns,
                fit_end_ns,
                bin_width_ps,
                min_nonzero_bins
            )
        ) as ex:
            for y0, y1, tau_chunk in ex.map(fit_rows, row_ranges):
                tau_map[y0:y1, :] = tau_chunk

        counts_map_final = np.array(counts_map_shared, dtype=np.uint32)

    finally:
        shm_hist.close()
        shm_hist.unlink()
        shm_counts.close()
        shm_counts.unlink()
        shm_valid.close()
        shm_valid.unlink()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].imshow(counts_map_final, origin='upper', cmap='gray')
    ax[0].set_title(f'photons in {window_n}x{window_n} window')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    cb0 = plt.colorbar(im0, ax=ax[0])
    cb0.set_label('photons')

    tau_masked = np.ma.masked_invalid(tau_map)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='black')

    im1 = ax[1].imshow(tau_masked, origin='upper', cmap=cmap, vmin=2, vmax=6)
    ax[1].set_title(f'extracted lifetime (ns), {window_n}x{window_n} window')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    cb1 = plt.colorbar(im1, ax=ax[1])
    cb1.set_label('tau (ns)')

    plt.tight_layout()
    plt.show()