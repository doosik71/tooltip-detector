"""Checkpoint format and single-frame inference, shared by the scripts.

A checkpoint stores the weights *and* everything needed to rebuild and
interpret them, so a caller never has to be told how the model was trained:

    {"model": state_dict, "arch": {"scale": "s"}, "class_names": ["tool"],
     "kpt_shape": [1, 2], "image_size": 640, "kpt_sigma": 0.05,
     "dataset": "cholec80", "epoch": 30, "metrics": {...}}
"""

import glob
import os
import time

import numpy as np
import torch

from .boxes import letterbox, non_max_suppression, undo_letterbox, undo_letterbox_points
from .model import CLASS_NAMES, KPT_SHAPE, build, decode, fuse

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45


def data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))     # standalone/yolov8stooltip/common
    return os.path.join(os.path.dirname(here), "data")


def model_dir(dataset: str) -> str:
    """Where one run's checkpoints live: data/model/<dataset>/."""
    return os.path.join(data_dir(), "model", dataset)


def results_dir(dataset: str) -> str:
    """One run's evaluation output: data/results/<dataset>/."""
    return os.path.join(data_dir(), "results", dataset)


def trained_datasets() -> list[str]:
    paths = glob.glob(os.path.join(data_dir(), "model", "*", "model.pt"))
    return sorted(path.split(os.sep)[-2] for path in paths)


def default_model_path(dataset: str | None = None) -> str:
    """Checkpoint of one dataset, or -- when no dataset is given -- the first
    trained one, in alphabetical order."""
    if not dataset:
        trained = trained_datasets()
        dataset = trained[0] if trained else "<dataset>"
    return os.path.join(model_dir(dataset), "model.pt")


def save_checkpoint(path: str, model, arch: dict, image_size: int, kpt_sigma: float,
                    dataset: str, epoch: int, metrics: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "arch": arch,
        "class_names": list(CLASS_NAMES),
        "kpt_shape": list(model.kpt_shape),
        "image_size": image_size,
        "kpt_sigma": kpt_sigma,
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
        self.kpt_shape = tuple(checkpoint.get("kpt_shape", KPT_SHAPE))
        self.image_size = int(checkpoint.get("image_size", 640))
        self.dataset = checkpoint.get("dataset")
        self.epoch = checkpoint.get("epoch")
        self.metrics = checkpoint.get("metrics", {})

        self.model = build(num_classes=len(self.class_names), kpt_shape=self.kpt_shape,
                           **self.arch)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        # BatchNorm folded into the convolutions after loading, as the
        # baselines do; the checkpoint on disk is untouched.
        fuse(self.model)

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray, conf: float = DEFAULT_CONF,
               iou: float = DEFAULT_IOU) -> tuple[np.ndarray, float]:
        """Return ((n, 6 + nk) [x1,y1,x2,y2,score,class,kx,ky] in frame pixels,
        elapsed ms). Each row's keypoint is that tool's tip."""
        height, width = frame_rgb.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(frame_rgb, self.image_size)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1))[None]
        tensor = tensor.to(self.device).float().div_(255.0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = decode(self.model(tensor), self.model)
        detections = non_max_suppression(predictions, len(self.class_names), conf, iou)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detections = detections.cpu().numpy()
        if len(detections):
            detections[:, :4] = undo_letterbox(detections[:, :4], scale, pad_x, pad_y,
                                               width, height)
            detections[:, 6:] = undo_letterbox_points(detections[:, 6:], scale, pad_x, pad_y,
                                                      width, height)
        return detections, elapsed_ms
