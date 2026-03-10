<<<<<<< HEAD
from __future__ import annotations

import cv2
import numpy as np


def _to_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("segment expects a 2D image")
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _apply_morph(mask: np.ndarray, open_ksize: int = 3, close_ksize: int = 9) -> np.ndarray:
    result = mask.astype(np.uint8)
    if open_ksize > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k_open)
    if close_ksize > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k_close)
    return result


def _components_to_labels(mask: np.ndarray, min_area: int = 120) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        filtered[labels == idx] = next_id
        next_id += 1
    return filtered


def _threshold_components(
    image: np.ndarray,
    low: float = 0.2,
    high: float = 0.8,
    min_area: int = 120,
    open_ksize: int = 3,
    close_ksize: int = 9,
) -> np.ndarray:
    img_u8 = _to_u8(image)
    lo = int(max(0, min(255, round(low * 255.0))))
    hi = int(max(0, min(255, round(high * 255.0))))
    if hi < lo:
        lo, hi = hi, lo
    mask = cv2.inRange(img_u8, lo, hi)
    mask = _apply_morph(mask, open_ksize=open_ksize, close_ksize=close_ksize)
    return _components_to_labels(mask, min_area=min_area)


def _watershed(
    image: np.ndarray,
    low: float = 0.2,
    high: float = 0.8,
    min_area: int = 120,
    min_distance: int = 10,
    open_ksize: int = 3,
    close_ksize: int = 9,
) -> np.ndarray:
    base = _threshold_components(
        image,
        low=low,
        high=high,
        min_area=max(1, min_area // 2),
        open_ksize=open_ksize,
        close_ksize=close_ksize,
    )
    sure_fg = np.uint8(base > 0) * 255
    if np.count_nonzero(sure_fg) == 0:
        return np.zeros(base.shape, dtype=np.int32)

    dist = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, float(min_distance), 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(peaks)
    if n_markers <= 1:
        return _components_to_labels(sure_fg, min_area=min_area)

    markers = markers + 1
    markers[sure_fg == 0] = 0
    image_u8 = _to_u8(image)
    image_rgb = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(image_rgb, markers)
    labels = np.where(markers > 1, markers - 1, 0).astype(np.int32)

    filtered = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        region = labels == label_id
        if int(np.count_nonzero(region)) < min_area:
            continue
        filtered[region] = next_id
        next_id += 1
    return filtered


def _adaptive_threshold(
    image: np.ndarray,
    block_size: int = 61,
    C: int = -5,
    min_area: int = 120,
    open_ksize: int = 3,
    close_ksize: int = 9,
) -> np.ndarray:
    img_u8 = _to_u8(image)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    mask = cv2.adaptiveThreshold(
        img_u8,
=======
# pyrpoc/acquisitions/segmentation_methods.py
import numpy as np
import cv2

# these methods return a label image:
#  - dtype int32
#  - 0 = background
#  - 1..N = objects


def _to_float01(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    mx = float(np.max(img)) if img.size else 0.0
    if mx <= 1.0:
        return np.clip(img, 0, 1)
    return np.clip(img / mx, 0, 1)


def _to_uint8(img01):
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def _morph_cleanup(binary, open_ksize=3, close_ksize=5):
    binary = binary.astype(np.uint8) * 255

    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    return binary > 0


def labels_to_contours(labels, min_area=50):
    labels = np.asarray(labels)
    contours_out = []

    max_lab = int(labels.max()) if labels.size else 0
    for lab in range(1, max_lab + 1):
        mask = (labels == lab).astype(np.uint8) * 255
        if mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        contour = contour.reshape(-1, 2)
        contours_out.append(contour)

    return contours_out


def segment_threshold_cc(
    img,
    low=0.2,
    high=1.0,
    min_area=80,
    open_ksize=3,
    close_ksize=7,
):
    img01 = _to_float01(img)
    binary = (img01 >= low) & (img01 <= high)
    binary = _morph_cleanup(binary, open_ksize=open_ksize, close_ksize=close_ksize)

    num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    labels = labels.astype(np.int32)

    if num <= 1:
        return labels

    # filter small components
    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[0] = True
    keep[counts >= min_area] = True
    labels[~keep[labels]] = 0

    # relabel to 1..N
    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]
    out = np.zeros_like(labels, dtype=np.int32)
    for i, lab in enumerate(uniq, start=1):
        out[labels == lab] = i
    return out


def segment_adaptive_cc(
    img,
    block_size=51,
    C=-5,
    min_area=80,
    open_ksize=3,
    close_ksize=7,
):
    img01 = _to_float01(img)
    u8 = _to_uint8(img01)

    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, int(block_size))

    binary = cv2.adaptiveThreshold(
        u8,
>>>>>>> origin/main
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
<<<<<<< HEAD
        C,
    )
    mask = _apply_morph(mask, open_ksize=open_ksize, close_ksize=close_ksize)
    return _components_to_labels(mask, min_area=min_area)


def segment(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    name = method.strip().lower()
    if name == "threshold + components":
        return _threshold_components(image, **kwargs)
    if name == "watershed":
        return _watershed(image, **kwargs)
    if name == "adaptive threshold":
        return _adaptive_threshold(image, **kwargs)
    raise ValueError(f"unknown segmentation method '{method}'")


def labels_to_contours(labels: np.ndarray, min_area: int = 120) -> list[np.ndarray]:
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError("labels_to_contours expects a 2D label map")

    contours: list[np.ndarray] = []
    for label_id in np.unique(arr):
        if int(label_id) <= 0:
            continue
        mask = np.uint8(arr == label_id) * 255
        if int(np.count_nonzero(mask)) < min_area:
            continue
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in found:
            if contour.shape[0] < 3:
                continue
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            contours.append(contour.reshape(-1, 2))
    return contours
=======
        int(C),
    ) > 0

    binary = _morph_cleanup(binary, open_ksize=open_ksize, close_ksize=close_ksize)
    num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    labels = labels.astype(np.int32)

    if num <= 1:
        return labels

    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[0] = True
    keep[counts >= min_area] = True
    labels[~keep[labels]] = 0

    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]
    out = np.zeros_like(labels, dtype=np.int32)
    for i, lab in enumerate(uniq, start=1):
        out[labels == lab] = i
    return out


def segment_watershed(
    img,
    low=0.2,
    high=1.0,
    min_area=80,
    min_distance=8,
    open_ksize=3,
    close_ksize=7,
):
    # lazy imports so the module still works without scipy/skimage
    try:
        from scipy import ndimage
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except Exception:
        # fall back to threshold cc
        return segment_threshold_cc(
            img, low=low, high=high, min_area=min_area, open_ksize=open_ksize, close_ksize=close_ksize
        )

    img01 = _to_float01(img)
    binary = (img01 >= low) & (img01 <= high)
    binary = _morph_cleanup(binary, open_ksize=open_ksize, close_ksize=close_ksize)

    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.int32)

    dist = ndimage.distance_transform_edt(binary)

    coords = peak_local_max(dist, min_distance=int(min_distance), labels=binary)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        # no peaks -> fall back
        return segment_threshold_cc(
            img, low=low, high=high, min_area=min_area, open_ksize=open_ksize, close_ksize=close_ksize
        )

    labels = watershed(-dist, markers, mask=binary).astype(np.int32)

    # filter small
    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[0] = True
    keep[counts >= min_area] = True
    labels[~keep[labels]] = 0

    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]
    out = np.zeros_like(labels, dtype=np.int32)
    for i, lab in enumerate(uniq, start=1):
        out[labels == lab] = i
    return out


METHODS = {
    'threshold + components': segment_threshold_cc,
    'watershed': segment_watershed,
    'adaptive threshold': segment_adaptive_cc,
}


def segment(img, method, **kwargs):
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(f'unknown segmentation method: {method!r}')
    return fn(img, **kwargs)
>>>>>>> origin/main
