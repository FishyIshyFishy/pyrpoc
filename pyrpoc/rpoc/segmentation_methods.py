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
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
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
