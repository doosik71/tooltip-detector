"""This repository's annotations, turned into a two-class detection dataset.

`data/dataset/<name>/` carries, per frame, an RGB PNG and a JSON file listing
one entry per visible instrument:

    {"annotations": [{"bbox": {"x", "y", "width", "height"},
                      "tip":  {"x", "y"}}, ...], "width": 736, "height": 480}

There are no instrument-class labels, so the detector is trained on two
classes derived from that same annotation:

    0  tool   the annotated bounding box
    1  tip    a square box of side `tip_box_size` centred on the tip

The tip box side is a real hyper-parameter, not a formality: it sets how many
anchor points fall inside a tip and therefore how many positives the assigner
can pick for it. This baseline defaults to 32 px where the CLAD-Net baseline
uses 10 px, so the two are *not* interchangeable without saying which size was
used -- `--tip-box-size` records it in the checkpoint.

Training uses mosaic augmentation; validation and test use a plain letterbox
so metrics are measured on undistorted frames.
"""

import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .boxes import letterbox

CLASS_NAMES = ("tool", "tip")
TOOL_CLASS, TIP_CLASS = 0, 1

# The tip is a point annotation; it becomes a square box of this side length,
# centred on the point, so it can be learned by a box detector. YOLOv8 assigns
# labels dynamically from anchor points *inside* the box, so a larger box gives
# the tip class more candidate positives than CLAD-Net's 10 px box does.
DEFAULT_TIP_BOX_SIZE = 32.0

SPLITS = ("train", "val", "test")

# Minimum share of a box that must survive a mosaic crop for it to be kept.
# A clipped tool box is still a usable tool box, but a clipped *tip* box no
# longer has the tip at its centre, so tips are held to a much stricter bar.
_MIN_AREA_KEPT = {TOOL_CLASS: 0.2, TIP_CLASS: 0.8}

_MOSAIC_SCALE_RANGE = (0.5, 1.5)
_HSV_GAINS = (0.015, 0.7, 0.4)      # h, s, v -- YOLOv5's defaults


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))       # baseline/yolov8sclone/common
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def default_data_root() -> str:
    return os.path.join(repo_root(), "data", "dataset")


def available_datasets(data_root: str | None = None) -> list[str]:
    data_root = data_root or default_data_root()
    if not os.path.isdir(data_root):
        return []
    return sorted(name for name in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, name, "images")))


def load_annotation(path: str, width: float, height: float,
                    tip_box_size: float = DEFAULT_TIP_BOX_SIZE) -> np.ndarray:
    """Read one annotation JSON as (n, 5) normalised [class, cx, cy, w, h].

    A frame with no visible instrument yields an empty array -- those frames
    are kept, as pure-background examples.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for item in payload.get("annotations", []):
        box = item.get("bbox")
        if box and box.get("width", 0) > 0 and box.get("height", 0) > 0:
            rows.append((TOOL_CLASS,
                         (box["x"] + box["width"] / 2) / width,
                         (box["y"] + box["height"] / 2) / height,
                         box["width"] / width,
                         box["height"] / height))
        tip = item.get("tip")
        if tip is not None:
            rows.append((TIP_CLASS, tip["x"] / width, tip["y"] / height,
                         tip_box_size / width, tip_box_size / height))
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


class SurgicalDetectionDataset(Dataset):
    """One split of one dataset, as (image tensor, targets) pairs."""

    def __init__(self, dataset: str, split: str, image_size: int = 640,
                 augment: bool = False, data_root: str | None = None,
                 frame_stride: int = 1, limit: int | None = None,
                 tip_box_size: float = DEFAULT_TIP_BOX_SIZE):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.dataset = dataset
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.tip_box_size = tip_box_size

        data_root = data_root or default_data_root()
        images_dir = os.path.join(data_root, dataset, "images", split)
        self.annotation_dir = os.path.join(data_root, dataset, "annotation", split)
        paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        if not paths:
            raise OSError(f"no frames under {images_dir}")
        # Consecutive video frames are near-duplicates; a stride > 1 trades
        # redundancy for epoch time without changing what the split contains.
        paths = paths[::max(1, frame_stride)]
        self.image_paths = paths[:limit] if limit else paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def frame_name(self, index: int) -> str:
        return os.path.basename(self.image_paths[index])

    def session_id(self, index: int) -> str:
        """Source video of a frame -- `video41_00000000.png` -> `video41`."""
        return os.path.splitext(self.frame_name(index))[0].rsplit("_", 1)[0]

    def _annotation_path(self, index: int) -> str:
        stem = os.path.splitext(self.frame_name(index))[0]
        return os.path.join(self.annotation_dir, stem + ".json")

    def read_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Original-resolution RGB frame plus normalised labels."""
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"could not read {self.image_paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        return image, load_annotation(self._annotation_path(index), w, h,
                                      self.tip_box_size)

    # ── Augmentation ──────────────────────────────────────────────────────

    def _load_scaled(self, index: int, jitter: bool) -> tuple[np.ndarray, np.ndarray]:
        """Frame resized so its long side is (jittered) `image_size`."""
        image, labels = self.read_frame(index)
        h, w = image.shape[:2]
        scale = self.image_size / max(h, w)
        if jitter:
            scale *= random.uniform(*_MOSAIC_SCALE_RANGE)
        new_w, new_h = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return image, labels

    def _mosaic(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Stitch four frames into a 2s x 2s canvas, then crop the centre s x s.

        This is YOLOv5's mosaic minus the affine warp: four frames at random
        scales are pasted around a random centre, and the crop back to s x s
        is what supplies the random translation.
        """
        s = self.image_size
        centre_x = int(random.uniform(s * 0.5, s * 1.5))
        centre_y = int(random.uniform(s * 0.5, s * 1.5))
        canvas = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)

        indices = [index] + [random.randrange(len(self)) for _ in range(3)]
        collected = []
        for corner, source in enumerate(indices):
            image, labels = self._load_scaled(source, jitter=True)
            h, w = image.shape[:2]

            if corner == 0:      # top-left
                x1a, y1a, x2a, y2a = max(centre_x - w, 0), max(centre_y - h, 0), centre_x, centre_y
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif corner == 1:    # top-right
                x1a, y1a, x2a, y2a = centre_x, max(centre_y - h, 0), min(centre_x + w, s * 2), centre_y
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif corner == 2:    # bottom-left
                x1a, y1a, x2a, y2a = max(centre_x - w, 0), centre_y, centre_x, min(s * 2, centre_y + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:                # bottom-right
                x1a, y1a, x2a, y2a = centre_x, centre_y, min(centre_x + w, s * 2), min(s * 2, centre_y + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            canvas[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            if len(labels):
                pixels = labels.copy()
                pixels[:, 1] = labels[:, 1] * w + (x1a - x1b)
                pixels[:, 2] = labels[:, 2] * h + (y1a - y1b)
                pixels[:, 3] = labels[:, 3] * w
                pixels[:, 4] = labels[:, 4] * h
                collected.append(pixels)

        labels = np.concatenate(collected, 0) if collected else np.zeros((0, 5), np.float32)
        crop = s // 2
        image = canvas[crop:crop + s, crop:crop + s]
        labels = _shift_and_clip(labels, -crop, -crop, s, s)
        return image, labels

    @staticmethod
    def _augment_colour(image: np.ndarray) -> np.ndarray:
        gains = np.random.uniform(-1, 1, 3) * _HSV_GAINS + 1
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] * gains[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # ── Item ──────────────────────────────────────────────────────────────

    def __getitem__(self, index: int):
        if self.augment:
            image, labels = self._mosaic(index)
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                if len(labels):
                    labels[:, 1] = self.image_size - labels[:, 1]
            image = self._augment_colour(image)
        else:
            frame, normalised = self.read_frame(index)
            h, w = frame.shape[:2]
            image, scale, pad_x, pad_y = letterbox(frame, self.image_size)
            labels = normalised.copy()
            if len(labels):
                labels[:, 1] = normalised[:, 1] * w * scale + pad_x
                labels[:, 2] = normalised[:, 2] * h * scale + pad_y
                labels[:, 3] = normalised[:, 3] * w * scale
                labels[:, 4] = normalised[:, 4] * h * scale

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        targets = torch.zeros((len(labels), 6), dtype=torch.float32)
        if len(labels):
            targets[:, 1] = torch.from_numpy(labels[:, 0])
            targets[:, 2:] = torch.from_numpy(labels[:, 1:5]) / self.image_size
        return tensor, targets, index


def _shift_and_clip(labels: np.ndarray, dx: int, dy: int, width: int, height: int) -> np.ndarray:
    """Translate pixel-space labels, clip them to the canvas, drop what is gone."""
    if not len(labels):
        return labels
    x1 = labels[:, 1] - labels[:, 3] / 2 + dx
    y1 = labels[:, 2] - labels[:, 4] / 2 + dy
    x2 = labels[:, 1] + labels[:, 3] / 2 + dx
    y2 = labels[:, 2] + labels[:, 4] / 2 + dy
    original_area = np.maximum((x2 - x1) * (y2 - y1), 1e-6)

    x1, x2 = x1.clip(0, width), x2.clip(0, width)
    y1, y2 = y1.clip(0, height), y2.clip(0, height)
    w, h = x2 - x1, y2 - y1

    min_area = np.array([_MIN_AREA_KEPT[int(c)] for c in labels[:, 0]], dtype=np.float32)
    keep = (w > 1) & (h > 1) & ((w * h / original_area) > min_area)

    out = labels[keep].copy()
    out[:, 1] = (x1[keep] + x2[keep]) / 2
    out[:, 2] = (y1[keep] + y2[keep]) / 2
    out[:, 3] = w[keep]
    out[:, 4] = h[keep]
    return out


def collate(batch):
    """Stack images and tag every target row with its index inside the batch."""
    images, targets, indices = zip(*batch)
    tagged = []
    for i, item in enumerate(targets):
        if len(item):
            item = item.clone()
            item[:, 0] = i
            tagged.append(item)
    stacked = torch.cat(tagged, 0) if tagged else torch.zeros((0, 6))
    return torch.stack(images, 0), stacked, list(indices)
