import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import label as ndi_label
from torch.utils.data import Dataset


class SurgicalToolDataset(Dataset):
    """Surgical tool segmentation dataset with distance-based tip heatmap targets.

    For each frame the target map is a float32 array in [0, 1].
    Each segmentation mask pixel is assigned to one tool via connected-component
    analysis, then the heatmap is normalised independently per tool:
      - pixels outside the segmentation mask → 0.0
      - the tip pixel of each tool           → 1.0
      - pixels within a tool's region scale linearly from 1.0 (at that tool's tip)
        down to 0.0 (at the farthest pixel of the same component group)

    Args:
        root:      path to data/dataset/
        split:     one of "train", "val", "test"
        transform: albumentations transform applied to both image and target mask
    """

    def __init__(self, root: str, split: str, transform=None):
        self.ann_dir = os.path.join(root, "annotation", split)
        self.img_dir = os.path.join(root, "images", split)
        self.seg_dir = os.path.join(root, "segmentation", split)
        self.samples = sorted(glob.glob(os.path.join(self.ann_dir, "*.json")))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _build_target(seg: np.ndarray, annotations: list[dict]) -> np.ndarray:
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

    def __getitem__(self, idx: int):
        ann_path = self.samples[idx]
        stem = os.path.splitext(os.path.basename(ann_path))[0]

        try:
            with open(ann_path) as f:
                ann = json.load(f)

            image = np.array(Image.open(os.path.join(self.img_dir, stem + ".png")))
            seg = np.array(Image.open(os.path.join(self.seg_dir, stem + ".png")))

            target = self._build_target(seg, ann["annotations"])

            if self.transform is not None:
                out = self.transform(image=image, mask=target)
                image, target = out["image"], out["mask"]

            return image, target
        except Exception as e:
            print(f"Error: {stem}.png - {e}")
            raise