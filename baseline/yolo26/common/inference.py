"""Loading a trained YOLO26 checkpoint and running it on one frame.

The checkpoint itself is an Ultralytics pickle, written by Ultralytics'
trainer, and this baseline does not rewrite it. What Ultralytics does not
record -- the tip box size the labels were built with, which dataset the run
used, whether it started from COCO weights -- is kept beside it:

    data/model/<dataset>/model.pt          best checkpoint (Ultralytics format)
    data/model/<dataset>/model-last.pt     last epoch, resumable
    data/model/<dataset>/model-info.json   the sidecar this module reads

`tip_box_size` matters at read time: it is the side of the box the model was
taught to draw around a tool tip, and a model trained at 32 px must not be
scored against 10 px labels. A checkpoint moved away from its sidecar still
loads, and falls back to the defaults with the values marked unknown.

YOLO26 is end-to-end: its head emits one box per object, so there is no NMS
and no IoU threshold to tune. `detect()` therefore takes a confidence
threshold and a cap on the number of boxes, and nothing else.
"""

import glob
import json
import os
import time

import numpy as np

from .dataset import CLASS_NAMES, DEFAULT_TIP_BOX_SIZE, data_dir, model_dir

DEFAULT_CONF = 0.25
DEFAULT_MAX_DET = 300

MODEL_FILENAME = "model.pt"
LAST_FILENAME = "model-last.pt"
INFO_FILENAME = "model-info.json"


def trained_datasets() -> list[str]:
    """Dataset names that already have a checkpoint under data/model/<dataset>/."""
    paths = glob.glob(os.path.join(data_dir(), "model", "*", MODEL_FILENAME))
    return sorted(os.path.basename(os.path.dirname(path)) for path in paths)


def default_model_path(dataset: str | None = None) -> str:
    """Checkpoint of one dataset, or -- when no dataset is given -- the first
    trained one under data/model/, in alphabetical order."""
    if not dataset:
        trained = trained_datasets()
        dataset = trained[0] if trained else "<dataset>"
    return os.path.join(model_dir(dataset), MODEL_FILENAME)


def info_path(weights: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(weights)), INFO_FILENAME)


def write_model_info(weights: str, info: dict) -> None:
    with open(info_path(weights), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def read_model_info(weights: str) -> dict:
    try:
        with open(info_path(weights), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


class Detector:
    """Loads a checkpoint once and runs it on one RGB frame at a time."""

    def __init__(self, weights: str, device: str | None = None):
        # Imported here, not at module import time: `ultralytics` pulls in
        # torch and takes seconds, and a caller that only wants the path
        # helpers above should not pay for it.
        from ultralytics import YOLO
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.weights = weights

        self.model = YOLO(weights)
        self.model.to(device)
        self.names: dict[int, str] = dict(self.model.names)
        self.class_names = tuple(self.names[i] for i in sorted(self.names))
        self.end2end = bool(getattr(self.model.model, "end2end", False))

        info = read_model_info(weights)
        self.info = info
        self.tip_box_size = float(info.get("tip_box_size", DEFAULT_TIP_BOX_SIZE))
        self.image_size = int(info.get("image_size", 640))
        self.dataset = info.get("dataset")
        self.epoch = info.get("epoch")
        self.metrics = info.get("metrics", {})
        self.has_info = bool(info)

    def detect(self, frame_rgb: np.ndarray, conf: float = DEFAULT_CONF,
               max_det: int = DEFAULT_MAX_DET) -> tuple[np.ndarray, float]:
        """Return ((n, 6) [x1, y1, x2, y2, score, class] in frame pixels, elapsed ms).

        Ultralytics letterboxes to `image_size` internally and maps the boxes
        back, so the returned coordinates are already in the pixel space of the
        array handed in -- no rescaling by the caller.
        """
        # Ultralytics reads a numpy array as BGR (it flips the channels itself
        # on the way into the network, and its training loader feeds cv2's BGR
        # frames). This project passes RGB everywhere, so the flip happens here
        # -- without it the model would see its channels swapped at inference
        # but not during training.
        frame_bgr = np.ascontiguousarray(frame_rgb[..., ::-1])

        started = time.perf_counter()
        result = self.model.predict(frame_bgr, conf=conf, max_det=max_det,
                                    imgsz=self.image_size, device=self.device,
                                    verbose=False)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32), elapsed_ms
        detections = np.concatenate([
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy()[:, None],
            boxes.cls.cpu().numpy()[:, None],
        ], axis=1).astype(np.float32)
        return detections[np.argsort(-detections[:, 4])], elapsed_ms


def class_index(class_names: tuple[str, ...], name: str) -> int:
    """Index of a class by name, falling back to this baseline's own order.

    The trained checkpoint carries its own name list; reading the index from it
    rather than assuming 0/1 keeps evaluation correct if a checkpoint is ever
    trained with the classes in another order.
    """
    names = class_names or CLASS_NAMES
    return names.index(name) if name in names else CLASS_NAMES.index(name)
