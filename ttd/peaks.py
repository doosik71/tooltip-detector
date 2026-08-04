"""Heatmap peak detection shared by scripts/eval-model.py, scripts/tooltip-detector.py
and scripts/tooltip-tracker.py.
"""

import numpy as np
from scipy.ndimage import label as ndi_label


def find_peaks(
    heatmap: np.ndarray,
    threshold: float,
    min_distance: int,
) -> list[tuple[int, int, float]]:
    """Return (x, y, value) peaks in *heatmap* above *threshold*.

    Algorithm
    ---------
    1. Threshold the map → binary mask.
    2. Label connected components.
    3. Take the maximum-value pixel within each component as the peak.
    4. Apply NMS: discard any peak within *min_distance* pixels of a
       higher-valued peak already retained.

    Returns peaks sorted by value descending.
    """
    binary = heatmap >= threshold
    if not binary.any():
        return []

    labeled, n_components = ndi_label(binary)
    peaks: list[tuple[int, int, float]] = []
    for i in range(1, n_components + 1):
        masked_vals = np.where(labeled == i, heatmap, 0.0)
        flat_idx = int(np.argmax(masked_vals))
        y, x = divmod(flat_idx, heatmap.shape[1])
        peaks.append((x, y, float(heatmap[y, x])))

    peaks.sort(key=lambda p: -p[2])

    # NMS
    kept: list[tuple[int, int, float]] = []
    for p in peaks:
        if all(np.hypot(p[0] - k[0], p[1] - k[1]) >= min_distance for k in kept):
            kept.append(p)

    return kept
