"""Heatmap peak detection shared by scripts/eval-model.py, scripts/tooltip-detector.py
and scripts/tooltip-tracker.py.
"""

import cv2
import numpy as np
from scipy.ndimage import label as ndi_label


def find_peaks(
    heatmap: np.ndarray,
    threshold: float,
    min_distance: int,
    method: str = "connected-components",
) -> list[tuple[int, int, float]]:
    """Return (x, y, value) peaks in *heatmap* above *threshold*.

    ``connected-components`` preserves the historical behavior: label the
    threshold mask and take one maximum per component. ``watershed`` first
    splits a component with distance-transform markers, then takes one heatmap
    maximum from each watershed basin. All methods apply NMS afterward.
    """
    if method not in {"connected-components", "watershed"}:
        raise ValueError(f"Unknown peak method: {method}")
    binary = heatmap >= threshold
    if not binary.any():
        return []

    labeled, n_components = ndi_label(binary)
    if method == "connected-components":
        peak_masks = [labeled == i for i in range(1, n_components + 1)]
    else:
        peak_masks = _watershed_masks(heatmap, binary, labeled, n_components)

    peaks: list[tuple[int, int, float]] = []
    for peak_mask in peak_masks:
        masked_vals = np.where(peak_mask, heatmap, 0.0)
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


def _watershed_masks(
    heatmap: np.ndarray, binary: np.ndarray, labeled: np.ndarray, n_components: int,
) -> list[np.ndarray]:
    """Split threshold components using distance-transform watershed markers."""
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    masks: list[np.ndarray] = []
    image = cv2.cvtColor(np.clip(heatmap * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for component in range(1, n_components + 1):
        component_mask = labeled == component
        ys, xs = np.nonzero(component_mask)
        y0, y1 = max(0, ys.min() - 1), min(binary.shape[0], ys.max() + 2)
        x0, x1 = max(0, xs.min() - 1), min(binary.shape[1], xs.max() + 2)
        local_mask = component_mask[y0:y1, x0:x1]
        local_distance = distance[y0:y1, x0:x1]
        max_distance = float(local_distance[local_mask].max())
        if max_distance == 0:
            continue
        cores, n_cores = ndi_label(local_mask & (local_distance >= max_distance * 0.5))
        if n_cores <= 1:
            masks.append(component_mask)
            continue
        markers = np.zeros(local_mask.shape, dtype=np.int32)
        markers[~local_mask] = 1
        markers[cores > 0] = cores[cores > 0] + 1
        local_basins = cv2.watershed(image[y0:y1, x0:x1], markers)
        for marker in range(2, n_cores + 2):
            basin = local_basins == marker
            if basin.any():
                full_basin = np.zeros(binary.shape, dtype=bool)
                full_basin[y0:y1, x0:x1] = basin
                masks.append(full_basin)
    return masks
