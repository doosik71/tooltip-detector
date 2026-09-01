"""Checkpoint format and single-frame inference, shared by eval-model and demo.

A checkpoint stores the weights *and* everything needed to rebuild and
interpret them, so a demo never has to be told how the model was trained:

    {"model": state_dict, "arch": {"scale": "s"}, "class_names": (...),
     "image_size": 640, "tip_box_size": 32.0, "dataset": "cholec80",
     "epoch": 30, "metrics": {...}}

`tip_box_size` matters at read time: it is the side of the box the model was
taught to draw around a tool tip, and a checkpoint trained at 32 px is not
comparable with one trained at 10 px.

There is no NMS threshold here. YOLO26's head is end-to-end: the one-to-one
branch is trained to leave one box per object, so `detect()` takes a
confidence threshold and a cap on the number of boxes, and nothing else.
"""

import glob
import os
import time

import numpy as np
import torch

from .boxes import letterbox, undo_letterbox
from .dataset import DEFAULT_LABEL_SET, DEFAULT_TIP_BOX_SIZE
from .model import CLASS_NAMES, MAX_DET, build, decode, fuse, postprocess

DEFAULT_CONF = 0.25
DEFAULT_MAX_DET = MAX_DET


def data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))     # baseline/yolo26clone/common
    return os.path.join(os.path.dirname(here), "data")


# Checkpoints and results are split by dataset and then by label set, so a
# `tiponly` run never writes where a `tooltip` run has already written.
# The caller passes the mode, not the path: the directory name *is* the
# mode name, so there is nothing else to keep in step.
def model_dir(dataset: str, label_set: str = DEFAULT_LABEL_SET) -> str:
    """Where one run's checkpoints live: data/model/<dataset>/<label-set>/."""
    return os.path.join(data_dir(), "model", dataset, label_set)


def results_dir(dataset: str, label_set: str = DEFAULT_LABEL_SET) -> str:
    """One run's evaluation output: data/results/<dataset>/<label-set>/."""
    return os.path.join(data_dir(), "results", dataset, label_set)


def trained_datasets(label_set: str = DEFAULT_LABEL_SET) -> list[str]:
    """Dataset names already trained in one label set."""
    paths = glob.glob(os.path.join(data_dir(), "model", "*", label_set, "model.pt"))
    return sorted(path.split(os.sep)[-3] for path in paths)


def default_model_path(dataset: str | None = None,
                       label_set: str = DEFAULT_LABEL_SET) -> str:
    """Checkpoint of one dataset, or -- when no dataset is given -- the first
    trained one in that label set, in alphabetical order."""
    if not dataset:
        trained = trained_datasets(label_set)
        dataset = trained[0] if trained else "<dataset>"
    return os.path.join(model_dir(dataset, label_set), "model.pt")


def save_checkpoint(path: str, model, arch: dict, image_size: int, tip_box_size: float,
                    dataset: str, epoch: int, metrics: dict | None = None,
                    class_names: tuple[str, ...] | None = None,
                    label_set: str = DEFAULT_LABEL_SET) -> None:
    """Write a checkpoint that describes itself.

    `class_names` must be the names this model was *built* with, not the
    module default: a `tiponly` model has one class, and recording two would
    make `Detector` rebuild it at the wrong width and fail to load its own
    weights.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "arch": arch,
        "class_names": list(class_names if class_names is not None else CLASS_NAMES),
        "label_set": label_set,
        "image_size": image_size,
        "tip_box_size": tip_box_size,
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
        self.tip_box_size = float(checkpoint.get("tip_box_size", DEFAULT_TIP_BOX_SIZE))
        self.dataset = checkpoint.get("dataset")
        self.epoch = checkpoint.get("epoch")
        self.metrics = checkpoint.get("metrics", {})

        self.model = build(num_classes=len(self.class_names), **self.arch)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        # Two inference-only shortcuts, both applied after `load_state_dict`
        # so the checkpoint format is untouched. Together they take about a
        # third off the frame.
        #
        # `detect` reads the one-to-one branch and nothing else, so let the
        # head skip the other one. Detections are bit-identical: it is the
        # same tensors, just not computing the ones nobody reads. A plain
        # attribute, so `load_state_dict` never carries it.
        self.model.detect.one2one_only = True
        # Folding BatchNorm into the preceding convolution rewrites
        # `state_dict` keys, so it has to come last. The reference fuses the
        # same way when `AutoBackend` loads a checkpoint.
        #
        # This one is exact in real arithmetic but not on this hardware: TF32
        # convolutions keep ~10 mantissa bits, and folding changes the weight
        # magnitudes the rounding sees. Measured over 60 cholec80 frames the
        # scores move by up to 1.2e-2 and one anchor in 278 crosses conf=0.25;
        # with TF32 off the same comparison is 7.2e-6 and no crossings. That
        # is the size of the noise TF32 already puts on the unfused path, so
        # it is accepted here rather than paid for with fp32 convolutions.
        fuse(self.model)

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray, conf: float = DEFAULT_CONF,
               max_det: int = DEFAULT_MAX_DET) -> tuple[np.ndarray, float]:
        """Return ((n, 6) [x1,y1,x2,y2,score,class] in frame pixels, elapsed ms)."""
        height, width = frame_rgb.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(frame_rgb, self.image_size)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1))[None]
        tensor = tensor.to(self.device).float().div_(255.0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        # Only the one-to-one branch is read: it is the branch trained to
        # leave a single box per object, which is what removes the NMS step.
        predictions = decode(self.model(tensor)["one2one"])
        detections = postprocess(predictions, max_det)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detections = detections[detections[:, 4] >= conf].cpu().numpy()
        if len(detections):
            detections[:, :4] = undo_letterbox(detections[:, :4], scale, pad_x, pad_y,
                                               width, height)
        return detections, elapsed_ms
