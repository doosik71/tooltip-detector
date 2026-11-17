"""Checkpoint format and single-frame inference, shared by the scripts.

A checkpoint stores the weights and everything needed to rebuild them:

    {"model": state_dict, "arch": {"scale": "s", "num_keypoints": 1, ...},
     "image_size": 640, "dataset": "cholec80", "epoch": 30, "metrics": {...}}
"""

import os
import time

import numpy as np
import torch

from .boxes import letterbox, to_frame
from .model import build

DEFAULT_CONF = 0.25
DEFAULT_MAX_DET = 100
DEFAULT_NMS_IOU = 0.65


def data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))     # baseline/mipsclone/common
    return os.path.join(os.path.dirname(here), "data")


def model_dir(dataset: str) -> str:
    """Where one run's checkpoints live: data/model/<dataset>/."""
    return os.path.join(data_dir(), "model", dataset)


def results_dir(dataset: str) -> str:
    """Evaluation output of one dataset: data/results/<dataset>/."""
    return os.path.join(data_dir(), "results", dataset)


def default_model_path(dataset: str) -> str:
    return os.path.join(model_dir(dataset), "model.pt")


def save_checkpoint(path: str, model, image_size: int, dataset: str, epoch: int,
                    metrics: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model": model.state_dict(), "arch": model.arch(), "image_size": image_size,
                "dataset": dataset, "epoch": epoch, "metrics": metrics or {}}, path)


class Predictor:
    """Loads a checkpoint and runs it on one RGB frame at a time."""

    def __init__(self, weights: str, device: str | None = None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.arch = checkpoint["arch"]
        self.image_size = int(checkpoint.get("image_size", 640))
        self.dataset = checkpoint.get("dataset")
        self.epoch = checkpoint.get("epoch")
        self.model = build(**self.arch)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()

    def _tensor(self, frame_rgb: np.ndarray):
        padded, scale, pad_x, pad_y = letterbox(frame_rgb, self.image_size)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1))[None].to(self.device).float()
        return tensor, scale, pad_x, pad_y

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def predict(self, frame_rgb: np.ndarray, conf: float = DEFAULT_CONF,
                max_det: int = DEFAULT_MAX_DET, nms_iou: float = DEFAULT_NMS_IOU,
                pose_conf: float | None = None) -> tuple[dict, float]:
        """Tool boxes and tips in frame pixels, plus the forward time in ms.

        Returns {"boxes": (n, 4), "scores": (n,), "tips": (n, 2),
        "tip_scores": (n,)}; tips are NaN for boxes below `pose_conf`.
        """
        tensor, scale, pad_x, pad_y = self._tensor(frame_rgb)
        self._sync()
        started = time.perf_counter()
        result = self.model.predict(tensor, conf, max_det, nms_iou, pose_conf=pose_conf)[0]
        self._sync()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return self._to_frame(result, scale, pad_x, pad_y), elapsed_ms

    @torch.no_grad()
    def tips_for_boxes(self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Run only the pose branch on given frame-pixel boxes (e.g. ground
        truth) -- the pose branch's upper bound with a perfect detector."""
        if not len(boxes_xyxy):
            return np.zeros((0, 2), dtype=np.float32)
        tensor, scale, pad_x, pad_y = self._tensor(frame_rgb)
        feats, _, _ = self.model(tensor)
        boxes = torch.as_tensor(np.asarray(boxes_xyxy, dtype=np.float32) * scale, device=self.device)
        boxes[:, [0, 2]] += pad_x
        boxes[:, [1, 3]] += pad_y
        rois = self.model.rois(boxes)
        sx, sy = self.model.pose(feats[0], torch.zeros(len(boxes), device=self.device), rois)
        kpts, _ = self.model.decode_keypoints(sx, sy, rois)
        return to_frame(kpts[:, 0].cpu().numpy(), scale, pad_x, pad_y)

    @staticmethod
    def _to_frame(result: dict, scale, pad_x, pad_y) -> dict:
        boxes = result["boxes"].cpu().numpy()
        return {"boxes": to_frame(boxes, scale, pad_x, pad_y) if len(boxes) else boxes,
                "scores": result["scores"].cpu().numpy(),
                "tips": to_frame(result["keypoints"][:, 0].cpu().numpy(), scale, pad_x, pad_y),
                "tip_scores": result["kpt_scores"][:, 0].cpu().numpy()}
