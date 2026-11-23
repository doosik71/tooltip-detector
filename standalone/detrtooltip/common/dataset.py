"""Frames and labels of data/dataset/<name>/, as detrtooltip trains on them.

Label source per frame, first one that has a file wins:

    1. annotation-manual/<split>/   hand-made              trusted
    2. annotation-opus/<split>/     Claude Opus labeller   trusted
    3. annotation/<split>/          final labels; mostly yolov8stooltip output

"Trusted" means the label also says where a second tip is *absent*. Labels from
annotation/ mostly come from a detector that can only give one tip, so a
missing `tip2` there says nothing, and the loss does not use it (see loss.py).

A label entry is {"bbox": {x, y, width, height}, "tip": {x, y}, "tip2": {x, y}
(optional), ...}. Entries without a tip or with an empty box are skipped.

Frames are identified by their video, `video41_00000123.png` -> `video41`, so
whole videos can be held out to pick checkpoints with.
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import IMAGE_HEIGHT, IMAGE_WIDTH, PIXEL_MEAN

SPLITS = ("train", "val", "test")
SOURCES = ("manual", "opus", "annotation")          # priority order
SOURCE_DIRS = {"manual": "annotation-manual", "opus": "annotation-opus", "annotation": "annotation"}
TRUSTED_SOURCES = ("manual", "opus")

_SCALE_RANGE = (0.75, 1.25)
_MIN_AREA_KEPT = 0.2
_HSV_GAINS = (0.015, 0.7, 0.4)                      # h, s, v -- YOLOv5's defaults
_FILL = tuple(int(round(c * 255)) for c in PIXEL_MEAN)


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))       # standalone/detrtooltip/common
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def default_data_root() -> str:
    return os.path.join(repo_root(), "data", "dataset")


def available_datasets(data_root: str | None = None) -> list[str]:
    data_root = data_root or default_data_root()
    if not os.path.isdir(data_root):
        return []
    return sorted(name for name in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, name, "images")))


def video_id(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0].rsplit("_", 1)[0]


@dataclass
class Frame:
    image: str
    label: str | None       # None: no label file in any source
    source: str | None
    two_tips: bool          # some tool in the label has a second tip


def parse_label(path: str | None) -> list[dict]:
    """Label file -> [{"box": (x1, y1, x2, y2), "tips": [(x, y), ...]}] in pixels."""
    if path is None:
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    tools = []
    for item in payload.get("annotations", []):
        box, tip = item.get("bbox"), item.get("tip")
        if not box or not tip or box.get("width", 0) <= 0 or box.get("height", 0) <= 0:
            continue
        tips = [(float(tip["x"]), float(tip["y"]))]
        tip2 = item.get("tip2")
        if isinstance(tip2, dict):
            tips.append((float(tip2["x"]), float(tip2["y"])))
        x, y = float(box["x"]), float(box["y"])
        tools.append({"box": (x, y, x + float(box["width"]), y + float(box["height"])),
                      "tips": tips})
    return tools


def _has_two_tips(path: str) -> bool:
    return any(len(tool["tips"]) == 2 for tool in parse_label(path))


def list_frames(dataset: str, data_root: str | None = None, splits=SPLITS,
                videos: set[str] | None = None, exclude_videos: set[str] | None = None,
                frame_stride: int = 1, workers: int = 16) -> list[Frame]:
    """Every frame of `splits`, optionally limited to or excluding some
    videos, with its label source resolved. Trusted labels are read once to
    find the frames that have a second tip."""
    data_root = data_root or default_data_root()
    root = os.path.join(data_root, dataset)
    frames: list[Frame] = []
    for split in splits:
        image_dir = os.path.join(root, "images", split)
        if not os.path.isdir(image_dir):
            continue
        names = sorted(name for name in os.listdir(image_dir) if name.endswith(".png"))
        if videos is not None:
            names = [n for n in names if video_id(n) in videos]
        if exclude_videos:
            names = [n for n in names if video_id(n) not in exclude_videos]
        names = names[::max(1, frame_stride)]
        available = {}
        for source in SOURCES:
            directory = os.path.join(root, SOURCE_DIRS[source], split)
            available[source] = (set(os.listdir(directory)) if os.path.isdir(directory)
                                 else set())
        for name in names:
            label_name = os.path.splitext(name)[0] + ".json"
            source = next((s for s in SOURCES if label_name in available[s]), None)
            label = (os.path.join(root, SOURCE_DIRS[source], split, label_name)
                     if source else None)
            frames.append(Frame(os.path.join(image_dir, name), label, source, False))

    trusted = [f for f in frames if f.source in TRUSTED_SOURCES]
    with ThreadPoolExecutor(workers) as pool:
        for frame, two in zip(trusted, pool.map(lambda f: _has_two_tips(f.label), trusted)):
            frame.two_tips = two
    return frames


def sample_weights(frames: list[Frame], two_tip_weight: float,
                   trusted_weight: float) -> list[float]:
    return [two_tip_weight if f.two_tips else
            trusted_weight if f.source in TRUSTED_SOURCES else 1.0 for f in frames]


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return image


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def random_scale_crop(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Resize by a random factor and crop or pad back to the model size.
    Returns (image, scale, dx, dy): a pixel (x, y) moves to (x*scale+dx, y*scale+dy)."""
    scale = random.uniform(*_SCALE_RANGE)
    h, w = image.shape[:2]
    new_w, new_h = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    canvas[:] = _FILL
    dx = random.randint(min(0, IMAGE_WIDTH - new_w), max(0, IMAGE_WIDTH - new_w))
    dy = random.randint(min(0, IMAGE_HEIGHT - new_h), max(0, IMAGE_HEIGHT - new_h))
    sx, sy = max(0, -dx), max(0, -dy)
    tx, ty = max(0, dx), max(0, dy)
    cw = min(new_w - sx, IMAGE_WIDTH - tx)
    ch = min(new_h - sy, IMAGE_HEIGHT - ty)
    canvas[ty:ty + ch, tx:tx + cw] = resized[sy:sy + ch, sx:sx + cw]
    return canvas, new_w / w, dx, dy


def augment_colour(image: np.ndarray) -> np.ndarray:
    gains = np.random.uniform(-1, 1, 3) * _HSV_GAINS + 1
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] * gains[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """Stage 1: images only, uint8 (3, H, W)."""

    def __init__(self, frames: list[Frame], augment: bool = True):
        self.frames = frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = read_rgb(self.frames[index].image)
        if self.augment:
            image, _, _, _ = random_scale_crop(image)
            if random.random() < 0.5:
                image = image[:, ::-1]
            image = augment_colour(np.ascontiguousarray(image))
        return to_tensor(image)


def build_target(tools: list[dict], trusted: bool) -> dict:
    """Pixel tools -> the normalised target dict loss.SetCriterion expects."""
    n = len(tools)
    boxes = np.zeros((n, 4), np.float32)
    tips = np.zeros((n, 2, 2), np.float32)
    valid = np.zeros((n, 2), bool)
    two = np.zeros(n, bool)
    for i, tool in enumerate(tools):
        x1, y1, x2, y2 = tool["box"]
        boxes[i] = ((x1 + x2) / 2 / IMAGE_WIDTH, (y1 + y2) / 2 / IMAGE_HEIGHT,
                    (x2 - x1) / IMAGE_WIDTH, (y2 - y1) / IMAGE_HEIGHT)
        two[i] = len(tool["tips"]) == 2
        for j, (x, y) in enumerate(tool["tips"][:2]):
            tips[i, j] = (x / IMAGE_WIDTH, y / IMAGE_HEIGHT)
            valid[i, j] = tool["valid"][j]
    return {"boxes": torch.from_numpy(boxes), "tips": torch.from_numpy(tips),
            "tip_valid": torch.from_numpy(valid), "two_tips": torch.from_numpy(two),
            "trusted": trusted}


def transform_tools(tools: list[dict], scale: float, dx: int, dy: int, flip: bool) -> list[dict]:
    """Move pixel tools with an augmentation; drop boxes mostly cut off and
    mark tips that left the image as invalid."""
    out = []
    for tool in tools:
        x1, y1, x2, y2 = (v * scale for v in tool["box"])
        x1, x2, y1, y2 = x1 + dx, x2 + dx, y1 + dy, y2 + dy
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        cx1, cx2 = min(max(x1, 0), IMAGE_WIDTH), min(max(x2, 0), IMAGE_WIDTH)
        cy1, cy2 = min(max(y1, 0), IMAGE_HEIGHT), min(max(y2, 0), IMAGE_HEIGHT)
        if cx2 - cx1 <= 1 or cy2 - cy1 <= 1 or (cx2 - cx1) * (cy2 - cy1) / area <= _MIN_AREA_KEPT:
            continue
        tips = [(x * scale + dx, y * scale + dy) for x, y in tool["tips"]]
        if flip:
            cx1, cx2 = IMAGE_WIDTH - cx2, IMAGE_WIDTH - cx1
            tips = [(IMAGE_WIDTH - x, y) for x, y in tips]
        valid = [0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT for x, y in tips]
        out.append({"box": (cx1, cy1, cx2, cy2), "tips": tips, "valid": valid + [False]})
    return out


class DetectionDataset(Dataset):
    """Stage 2: (uint8 image (3, H, W), target dict, index)."""

    def __init__(self, frames: list[Frame], augment: bool = False):
        self.frames = frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int):
        frame = self.frames[index]
        image = read_rgb(frame.image)
        tools = parse_label(frame.label)
        if self.augment:
            image, scale, dx, dy = random_scale_crop(image)
            flip = random.random() < 0.5
            if flip:
                image = image[:, ::-1]
            image = augment_colour(np.ascontiguousarray(image))
            tools = transform_tools(tools, scale, dx, dy, flip)
        else:
            tools = transform_tools(tools, 1.0, 0, 0, False)
        target = build_target(tools, frame.source in TRUSTED_SOURCES)
        return to_tensor(image), target, index


def collate(batch):
    images, targets, indices = zip(*batch)
    return torch.stack(images, 0), list(targets), list(indices)
