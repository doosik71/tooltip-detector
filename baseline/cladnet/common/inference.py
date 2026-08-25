"""Checkpoint format and single-frame inference, shared by eval-model and demo.

A checkpoint stores the weights *and* the architecture arguments that produced
them, so a demo never has to be told how the model was built:

    {"model": state_dict, "arch": {...}, "class_names": (...),
     "image_size": 640, "dataset": "cholec80", "epoch": 30, "metrics": {...}}
"""

import glob
import os
import time

import numpy as np
import torch

from .boxes import letterbox, non_max_suppression, undo_letterbox
from .model import CLASS_NAMES, build, decode

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45


def data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))          # baseline/cladnet/common
    return os.path.join(os.path.dirname(here), "data")


def model_dir(dataset: str) -> str:
    """Where one dataset's checkpoints live: data/model/<dataset>/."""
    return os.path.join(data_dir(), "model", dataset)


def results_dir(dataset: str) -> str:
    """Where one dataset's evaluation output lives: data/results/<dataset>/."""
    return os.path.join(data_dir(), "results", dataset)


def default_model_path(dataset: str | None = None) -> str:
    """Checkpoint of one dataset, or -- for the demo, which has no --dataset --
    the first trained one under data/model/, in alphabetical order."""
    if dataset:
        return os.path.join(model_dir(dataset), "model.pt")
    for path in sorted(glob.glob(os.path.join(data_dir(), "model", "*", "model.pt"))):
        return path
    return os.path.join(data_dir(), "model", "<dataset>", "model.pt")


def save_checkpoint(path: str, model, arch: dict, image_size: int, dataset: str,
                    epoch: int, metrics: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "arch": arch,
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "dataset": dataset,
        "epoch": epoch,
        "metrics": metrics or {},
    }, path)


class Detector:
    """Loads a checkpoint and runs it on one RGB frame at a time."""

    def __init__(self, weights: str, device: str | None = None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights = weights

        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.arch = checkpoint.get("arch", {})
        self.class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
        self.image_size = int(checkpoint.get("image_size", 640))
        self.dataset = checkpoint.get("dataset")
        self.epoch = checkpoint.get("epoch")
        self.metrics = checkpoint.get("metrics", {})

        self.model = build(num_classes=len(self.class_names), **self.arch)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray, conf: float = DEFAULT_CONF,
               iou: float = DEFAULT_IOU) -> tuple[np.ndarray, float]:
        """Return ((n, 6) [x1,y1,x2,y2,score,class] in frame pixels, elapsed ms)."""
        height, width = frame_rgb.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(frame_rgb, self.image_size)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1))[None]
        tensor = tensor.to(self.device).float().div_(255.0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = self.model(tensor)
        predictions = decode(outputs, self.model.anchors, self.model.strides)
        detections = non_max_suppression(predictions, conf, iou)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections = detections.cpu().numpy()
        if len(detections):
            detections[:, :4] = undo_letterbox(detections[:, :4], scale, pad_x, pad_y,
                                               width, height)
        return detections, elapsed_ms
