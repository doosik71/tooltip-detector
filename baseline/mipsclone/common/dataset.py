"""This repository's annotations, as tool boxes that each carry one tip.

`data/dataset/<name>/` carries, per frame, an RGB PNG and a JSON file listing
one entry per visible instrument:

    {"annotations": [{"bbox": {"x", "y", "width", "height"},
                      "tip":  {"x", "y"}}, ...], "width": 736, "height": 480}

The two-class baselines next door turn the tip into a second box class. This
model keeps the pairing instead: one instance is one tool box *and* the tip of
that tool, the top-down pose formulation with a single keypoint. A sample is

    image    (3, s, s) uint8, letterboxed or mosaicked to s = image_size
    targets  (n, 8) float  [batch index, x1, y1, x2, y2, tip x, tip y, tip valid]

in input pixels. `tip valid` is 0 when augmentation pushed the tip off the
canvas while enough of the tool box survived to keep it as a detection target:
the box is still learned, its keypoint is not.

Training uses mosaic (the same four-frame stitch as the YOLO clones) until the
trainer switches to a plain random resize-and-crop for the last epochs, as
RTMDet does; validation and test use a letterbox.
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

SPLITS = ("train", "val", "test")

# A tool box clipped by a crop is still a usable box down to this share of its
# original area (the YOLO clones use the same value for their tool class).
_MIN_AREA_KEPT = 0.2
_SCALE_RANGE = (0.5, 1.5)
_HSV_GAINS = (0.015, 0.7, 0.4)      # h, s, v


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))       # baseline/mipsclone/common
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def default_data_root() -> str:
    return os.path.join(repo_root(), "data", "dataset")


def available_datasets(data_root: str | None = None) -> list[str]:
    data_root = data_root or default_data_root()
    if not os.path.isdir(data_root):
        return []
    return sorted(name for name in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, name, "images")))


def load_annotation(path: str) -> np.ndarray:
    """One annotation JSON as (n, 7) pixel rows [x1, y1, x2, y2, tx, ty, valid].

    A frame with no visible instrument yields an empty array; such frames are
    kept as pure-background examples. A tool without a tip (none in cholec80,
    but the format allows it) keeps its box with `valid` 0.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return np.zeros((0, 7), dtype=np.float32)

    rows = []
    for item in payload.get("annotations", []):
        box = item.get("bbox")
        if not box or box.get("width", 0) <= 0 or box.get("height", 0) <= 0:
            continue
        tip = item.get("tip")
        tx, ty, valid = (tip["x"], tip["y"], 1.0) if tip is not None else (0.0, 0.0, 0.0)
        rows.append((box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"],
                     tx, ty, valid))
    if not rows:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


class ToolPoseDataset(Dataset):
    """One split of one dataset, as (image tensor, targets, index) triples."""

    def __init__(self, dataset: str, split: str, image_size: int = 640,
                 augment: bool = False, mosaic: bool = True,
                 data_root: str | None = None, frame_stride: int = 1,
                 limit: int | None = None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.dataset = dataset
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.mosaic = mosaic

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

    def read_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Original-resolution RGB frame plus its (n, 7) pixel annotation."""
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"could not read {self.image_paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        stem = os.path.splitext(self.frame_name(index))[0]
        return image, load_annotation(os.path.join(self.annotation_dir, stem + ".json"))

    # ── Augmentation ──────────────────────────────────────────────────────

    def _load_scaled(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Frame resized so its long side is a jittered `image_size`."""
        image, labels = self.read_frame(index)
        h, w = image.shape[:2]
        scale = self.image_size / max(h, w) * random.uniform(*_SCALE_RANGE)
        new_w, new_h = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        labels = labels.copy()
        labels[:, :6] *= scale
        return image, labels

    def _mosaic(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Stitch four frames into a 2s x 2s canvas, then crop the centre s x s."""
        s = self.image_size
        centre_x = int(random.uniform(s * 0.5, s * 1.5))
        centre_y = int(random.uniform(s * 0.5, s * 1.5))
        canvas = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)

        indices = [index] + [random.randrange(len(self)) for _ in range(3)]
        collected = []
        for corner, source in enumerate(indices):
            image, labels = self._load_scaled(source)
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
                # Clip to the pasted region first: the part of a frame that did
                # not make it onto the canvas must not leave boxes behind.
                region = _clip(labels, x1b, y1b, x2b, y2b)
                collected.append(_shift(region, x1a - x1b, y1a - y1b))

        labels = np.concatenate(collected, 0) if collected else np.zeros((0, 7), np.float32)
        crop = s // 2
        image = canvas[crop:crop + s, crop:crop + s]
        labels = _clip(_shift(labels, -crop, -crop), 0, 0, s, s)
        return image, labels

    def _resize_crop(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """One frame at a jittered scale, pasted or cropped at a random offset."""
        s = self.image_size
        image, labels = self._load_scaled(index)
        h, w = image.shape[:2]
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        # Offset of the frame on the canvas; negative when the frame is larger.
        off_x = random.randint(min(0, s - w), max(0, s - w))
        off_y = random.randint(min(0, s - h), max(0, s - h))
        x1a, y1a = max(off_x, 0), max(off_y, 0)
        x2a, y2a = min(off_x + w, s), min(off_y + h, s)
        canvas[y1a:y2a, x1a:x2a] = image[y1a - off_y:y2a - off_y, x1a - off_x:x2a - off_x]
        return canvas, _clip(_shift(labels, off_x, off_y), 0, 0, s, s)

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
            image, labels = self._mosaic(index) if self.mosaic else self._resize_crop(index)
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                if len(labels):
                    s = self.image_size
                    x1 = labels[:, 0].copy()
                    labels[:, 0] = s - labels[:, 2]
                    labels[:, 2] = s - x1
                    labels[:, 4] = s - labels[:, 4]
            image = self._augment_colour(image)
        else:
            frame, labels = self.read_frame(index)
            image, scale, pad_x, pad_y = letterbox(frame, self.image_size)
            labels = labels.copy()
            if len(labels):
                labels[:, [0, 2, 4]] = labels[:, [0, 2, 4]] * scale + pad_x
                labels[:, [1, 3, 5]] = labels[:, [1, 3, 5]] * scale + pad_y

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        targets = torch.zeros((len(labels), 8), dtype=torch.float32)
        if len(labels):
            targets[:, 1:] = torch.from_numpy(labels.astype(np.float32))
        return tensor, targets, index


def _shift(labels: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = labels.copy()
    if len(out):
        out[:, [0, 2, 4]] += dx
        out[:, [1, 3, 5]] += dy
    return out


def _clip(labels: np.ndarray, x_min: float, y_min: float, x_max: float,
          y_max: float) -> np.ndarray:
    """Clip boxes to a window, drop those mostly cut away, and invalidate tips
    that left the window."""
    if not len(labels):
        return labels
    area = np.maximum((labels[:, 2] - labels[:, 0]) * (labels[:, 3] - labels[:, 1]), 1e-6)
    out = labels.copy()
    out[:, [0, 2]] = out[:, [0, 2]].clip(x_min, x_max)
    out[:, [1, 3]] = out[:, [1, 3]].clip(y_min, y_max)
    w, h = out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]
    keep = (w > 1) & (h > 1) & (w * h / area > _MIN_AREA_KEPT)
    inside = ((out[:, 4] >= x_min) & (out[:, 4] < x_max) &
              (out[:, 5] >= y_min) & (out[:, 5] < y_max))
    out[:, 6] = np.where(inside, out[:, 6], 0.0)
    return out[keep]


def collate(batch):
    """Stack images and tag every target row with its index inside the batch."""
    images, targets, indices = zip(*batch)
    tagged = []
    for i, item in enumerate(targets):
        if len(item):
            item = item.clone()
            item[:, 0] = i
            tagged.append(item)
    stacked = torch.cat(tagged, 0) if tagged else torch.zeros((0, 8))
    return torch.stack(images, 0), stacked, list(indices)
