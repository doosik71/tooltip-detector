"""Checkpoint format and single-frame inference.

A stage-2 checkpoint holds the weights and everything needed to rebuild them:

    {"family": "detrtooltip", "model": state_dict, "arch": {...},
     "image_size": [736, 480], "class_names": ["tool"], "dataset": "cholec80",
     "stage": "train", "epoch": 12, "metrics": {...}}

`family` is what the interactive trainer reads to pick its runner. A stage-1
checkpoint ("stage": "pretrain") stores the `Pretrainer` instead; only its
backbone is carried over into stage 2.
"""

import os

import numpy as np
import torch

from .dataset import read_rgb
from .model import CLASS_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH, build, postprocess

FAMILY = "detrtooltip"
DEFAULT_CONF = 0.5


def data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))     # standalone/detrtooltip/common
    return os.path.join(os.path.dirname(here), "data")


def model_dir(dataset: str) -> str:
    return os.path.join(data_dir(), "model", dataset)


def results_dir(dataset: str) -> str:
    return os.path.join(data_dir(), "results", dataset)


def save_checkpoint(path: str, model, arch: dict, dataset: str, stage: str, epoch: int,
                    metrics: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    torch.save({
        "family": FAMILY,
        "model": model.state_dict(),
        "arch": arch,
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "class_names": list(CLASS_NAMES),
        "dataset": dataset,
        "stage": stage,
        "epoch": epoch,
        "metrics": metrics or {},
    }, temporary)
    os.replace(temporary, path)


def load_model(path: str, device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("family") != FAMILY or checkpoint.get("stage") == "pretrain":
        raise ValueError(f"{path} is not a detrtooltip detector checkpoint")
    model = build(**checkpoint.get("arch", {}))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def frame_tensor(frame_rgb: np.ndarray, device) -> torch.Tensor:
    if frame_rgb.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"expected a {IMAGE_WIDTH} x {IMAGE_HEIGHT} frame, "
                         f"got {frame_rgb.shape[1]} x {frame_rgb.shape[0]}")
    tensor = torch.from_numpy(np.ascontiguousarray(frame_rgb.transpose(2, 0, 1)))[None]
    return tensor.to(device).float().div_(255.0)


class Detector:
    """Loads a checkpoint and runs it on one RGB frame at a time."""

    def __init__(self, weights: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, checkpoint = load_model(weights, self.device)
        self.dataset = checkpoint.get("dataset")
        self.epoch = checkpoint.get("epoch")
        self.stage = checkpoint.get("stage")

    @torch.no_grad()
    def predict(self, frame_rgb: np.ndarray, conf: float = DEFAULT_CONF) -> list[dict]:
        """Annotations in frame pixels, best first: {"bbox", "tip", optional
        "tip2", "score", "tip2_score"}. Frames of another size are resized."""
        height, width = frame_rgb.shape[:2]
        if (height, width) != (IMAGE_HEIGHT, IMAGE_WIDTH):
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            outputs = self.model(frame_tensor(frame_rgb, self.device))
        return postprocess(outputs, conf, width, height)[0]

    def predict_file(self, path: str, conf: float = DEFAULT_CONF) -> list[dict]:
        return self.predict(read_rgb(path), conf)
