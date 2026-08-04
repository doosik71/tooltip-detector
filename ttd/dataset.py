import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import label as ndi_label
from torch.utils.data import Dataset

# Target-generation methods (selected via `target_mode`):
#   "gradient-seg"  — needs segmentation masks. Per-tool distance gradient over
#                      the masked tool area (tip pixel = 1.0, farthest mask
#                      pixel in the same connected-component group = 0.0).
#   "gaussian-tip"  — needs only tip coordinates, no segmentation masks. A
#                      Gaussian centered on each tip (peak 1.0, std `gaussian_sigma`
#                      px), combined across tools by taking the per-pixel max.
TARGET_MODES = ("gradient-seg", "gaussian-tip")
DEFAULT_TARGET_MODE = "gradient-seg"
DEFAULT_GAUSSIAN_SIGMA = 15.0


class SurgicalToolDataset(Dataset):
    """Surgical tool dataset with distance-based tip heatmap targets.

    For each frame the target map is a float32 array in [0, 1], built by one of
    two interchangeable methods (see `target_mode`):

    ``"gradient-seg"`` (default, needs segmentation masks)
      Each segmentation mask pixel is assigned to one tool via connected-component
      analysis, then the heatmap is normalised independently per tool:
        - pixels outside the segmentation mask → 0.0
        - the tip pixel of each tool           → 1.0
        - pixels within a tool's region scale linearly from 1.0 (at that tool's
          tip) down to 0.0 (at the farthest pixel of the same component group)

    ``"gaussian-tip"`` (needs only tip coordinates, no segmentation masks)
      A 2-D Gaussian centered on each tool's tip pixel (peak 1.0, standard
      deviation `gaussian_sigma` px). Where two tools' Gaussians overlap, the
      per-pixel max is kept. Cheaper to label than `"gradient-seg"` since it
      does not require a segmentation mask, only the tip point.

    Args:
        root:           path to data/dataset/
        split:          one of "train", "val", "test"
        transform:      albumentations transform applied to both image and target mask
        target_mode:    one of TARGET_MODES (default: "gradient-seg")
        gaussian_sigma: standard deviation in px for "gaussian-tip" targets
                        (ignored for "gradient-seg")
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        target_mode: str = DEFAULT_TARGET_MODE,
        gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    ):
        if target_mode not in TARGET_MODES:
            raise ValueError(
                f"Unknown target_mode '{target_mode}'. Choose from {TARGET_MODES}"
            )
        self.ann_dir = os.path.join(root, "annotation", split)
        self.img_dir = os.path.join(root, "images", split)
        self.seg_dir = os.path.join(root, "segmentation", split)
        self.samples = sorted(glob.glob(os.path.join(self.ann_dir, "*.json")))
        self.transform = transform
        self.target_mode = target_mode
        self.gaussian_sigma = gaussian_sigma

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _build_target_gradient_seg(seg: np.ndarray, annotations: list[dict]) -> np.ndarray:
        """Build a per-tool normalised float32 heatmap.

        Algorithm
        ---------
        1. Label connected components of the binary segmentation mask.
        2. Match each component to the annotation whose bbox contains the most
           component pixels.  Components with no bbox overlap fall back to the
           annotation with the nearest tip (by component centroid).
        3. For each annotation, gather all pixels from its matched components
           and normalise distances to that annotation's tip independently:
           tip pixel → 1.0, farthest pixel in the group → 0.0.

        Args:
            seg:         H×W uint8 array (0 = background, >0 = tool)
            annotations: list of annotation dicts with keys ``"tip"`` and ``"bbox"``
                         as stored in the JSON files

        Returns:
            H×W float32 array in [0, 1]
        """
        target = np.zeros(seg.shape, dtype=np.float32)

        if not annotations or not (seg > 0).any():
            return target

        # ── Step 1: connected components ─────────────────────────────────
        labeled, n_components = ndi_label(seg > 0)
        if n_components == 0:
            return target

        # ── Step 2: match each component to an annotation ────────────────
        comp_to_ann: dict[int, int] = {}

        for comp_id in range(1, n_components + 1):
            comp_ys, comp_xs = np.where(labeled == comp_id)

            best_ann_idx = -1
            best_overlap = 0

            for ann_idx, ann in enumerate(annotations):
                b = ann["bbox"]
                bx0, by0 = b["x"], b["y"]
                bx1, by1 = bx0 + b["width"], by0 + b["height"]
                overlap = int(
                    ((comp_xs >= bx0) & (comp_xs < bx1) &
                     (comp_ys >= by0) & (comp_ys < by1)).sum()
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ann_idx = ann_idx

            if best_ann_idx == -1:
                # Fallback: assign to annotation whose tip is nearest the centroid
                cx, cy = comp_xs.mean(), comp_ys.mean()
                dists = [
                    np.hypot(cx - ann["tip"]["x"], cy - ann["tip"]["y"])
                    for ann in annotations
                ]
                best_ann_idx = int(np.argmin(dists))

            comp_to_ann[comp_id] = best_ann_idx

        # ── Step 3: per-annotation independent normalisation ─────────────
        ann_comp_ids: dict[int, list[int]] = defaultdict(list)
        for comp_id, ann_idx in comp_to_ann.items():
            ann_comp_ids[ann_idx].append(comp_id)

        for ann_idx, comp_ids in ann_comp_ids.items():
            tip_x = annotations[ann_idx]["tip"]["x"]
            tip_y = annotations[ann_idx]["tip"]["y"]

            # Collect all pixels belonging to this annotation's components
            xs_parts, ys_parts = [], []
            for cid in comp_ids:
                ys, xs = np.where(labeled == cid)
                xs_parts.append(xs)
                ys_parts.append(ys)
            all_xs = np.concatenate(xs_parts).astype(np.float32)
            all_ys = np.concatenate(ys_parts).astype(np.float32)

            dists = np.hypot(all_xs - tip_x, all_ys - tip_y)
            max_dist = dists.max()
            values = 1.0 - dists / max_dist if max_dist > 0 else np.ones_like(dists)

            target[all_ys.astype(int), all_xs.astype(int)] = values

        return target

    @staticmethod
    def _build_target_gaussian_tip(
        shape: tuple[int, int],
        annotations: list[dict],
        sigma: float,
    ) -> np.ndarray:
        """Build a per-tool Gaussian heatmap centered on each tip.

        Only the tip coordinate is used — no segmentation mask required.
        Overlapping tools' Gaussians are combined by taking the per-pixel max.

        Args:
            shape:       (H, W) of the target map
            annotations: list of annotation dicts with a ``"tip"`` key
            sigma:       Gaussian standard deviation in px

        Returns:
            H×W float32 array in [0, 1]
        """
        target = np.zeros(shape, dtype=np.float32)
        if not annotations:
            return target

        ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        for ann in annotations:
            tip_x, tip_y = ann["tip"]["x"], ann["tip"]["y"]
            dist_sq = (xs - tip_x) ** 2 + (ys - tip_y) ** 2
            gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
            np.maximum(target, gaussian, out=target)

        return target

    def _load_sample(self, idx: int):
        ann_path = self.samples[idx]
        stem = os.path.splitext(os.path.basename(ann_path))[0]

        with open(ann_path) as f:
            ann = json.load(f)

        image = np.array(Image.open(os.path.join(self.img_dir, stem + ".png")))

        if self.target_mode == "gradient-seg":
            seg = np.array(Image.open(os.path.join(self.seg_dir, stem + ".png")))
            target = self._build_target_gradient_seg(seg, ann["annotations"])
        else:  # "gaussian-tip"
            target = self._build_target_gaussian_tip(
                image.shape[:2], ann["annotations"], self.gaussian_sigma)

        if self.transform is not None:
            out = self.transform(image=image, mask=target)
            image, target = out["image"], out["mask"]

        return image, target

    def __getitem__(self, idx: int):
        # A handful of frames/annotations in this dataset are known to be
        # corrupt (e.g. truncated PNGs). Skip them by falling back to a
        # random other sample instead of letting one bad file kill an
        # unattended multi-hour training run.
        n = len(self.samples)
        for attempt in range(n):
            sample_idx = idx if attempt == 0 else random.randrange(n)
            try:
                return self._load_sample(sample_idx)
            except Exception as e:
                stem = os.path.splitext(os.path.basename(self.samples[sample_idx]))[0]
                print(f"Warning: {stem}.png - {e} (skipping corrupt sample)")

        raise RuntimeError(
            f"SurgicalToolDataset: all {n} samples in '{self.ann_dir}' failed to load")