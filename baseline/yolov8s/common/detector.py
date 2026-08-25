"""Thin wrapper around the Ultralytics YOLOv8s Cholec80 checkpoint.

The checkpoint (`data/yolov8s_cholec80.pt`, see scripts/download-model.py) is
an Ultralytics pickle, so it can only be unpickled with `ultralytics`
installed -- that is the reason this baseline has its own virtualenv.

The class list is read from the checkpoint itself rather than hard-coded, so a
re-trained or extended checkpoint keeps working. For the published weights it
is: Bag, Bipolar, Clipper, Grasper, Hook, Irrigator, Scissors.

Ultralytics letterboxes each frame to 640x640 internally and maps the boxes
back, so `detect()` returns coordinates in the pixel space of the array it was
handed -- no rescaling is needed by the caller.
"""

import time
from typing import NamedTuple

import numpy as np

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70


class Detection(NamedTuple):
    """One predicted box, in the coordinate space of the input frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    label: str


class YoloDetector:
    """Loads the checkpoint once and runs it frame by frame."""

    def __init__(self, weights: str, device: str | None = None):
        # Imported here, not at module import time: `ultralytics` pulls in
        # torch and takes seconds, and a caller that only wants the path
        # helpers in `sources` should not pay for it.
        from ultralytics import YOLO
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.weights = weights
        self._model = YOLO(weights)
        self._model.to(device)
        self.names: dict[int, str] = dict(self._model.names)

    def detect(
        self,
        frame_rgb: np.ndarray,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> tuple[list[Detection], float]:
        """Run the detector on one RGB frame.

        Returns (detections sorted by descending confidence, elapsed ms). The
        elapsed time covers the whole Ultralytics call -- letterboxing, forward
        pass and NMS -- because that is what a demo frame actually costs.
        """
        t0 = time.perf_counter()
        result = self._model.predict(
            frame_rgb, conf=conf, iou=iou, device=self.device, verbose=False
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        boxes = result.boxes
        detections = [
            Detection(
                float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]),
                float(score), int(class_id),
                self.names.get(int(class_id), str(int(class_id))),
            )
            for xyxy, score, class_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            )
        ]
        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections, elapsed_ms
