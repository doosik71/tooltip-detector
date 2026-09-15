"""TensorRT engine loading and single-frame inference.

Loads what `scripts/convert-model.py` writes -- `model.engine` plus its
`model-info.json` sidecar (the engine format has no room for arbitrary
metadata, so the checkpoint's `dataset`/`tip_box_size`/`metrics`/... travel
next to it instead, the same way `baseline/yolov8s/common/inference.py` keeps
a sidecar beside its Ultralytics `.pt`):

    data/model/<dataset>/<label-set>/model.engine
    data/model/<dataset>/<label-set>/model-info.json

`Detector.detect()` returns the same `(n, 6)` `[x1, y1, x2, y2, score, class]`
array in frame pixels that `yolov8sclone.common.inference.Detector.detect()`
does, decoded and NMS'd the same way (`common/boxes.py`, copied verbatim from
that project) -- the engine itself only replaces the model forward pass plus
`common.model.decode`, both folded into the graph by `scripts/convert-model.py`.
"""

import glob
import json
import os
import time

import numpy as np
import tensorrt as trt
import torch

from .boxes import letterbox, non_max_suppression, undo_letterbox

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_LABEL_SET = "tooltip"
DEFAULT_TIP_BOX_SIZE = 32.0

ENGINE_FILENAME = "model.engine"
INFO_FILENAME = "model-info.json"

_TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


def data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))     # .../yolov8s/common
    return os.path.join(os.path.dirname(here), "data")


def model_dir(dataset: str, label_set: str = DEFAULT_LABEL_SET) -> str:
    """Where one converted engine lives: data/model/<dataset>/<label-set>/."""
    return os.path.join(data_dir(), "model", dataset, label_set)


def trained_datasets(label_set: str = DEFAULT_LABEL_SET) -> list[str]:
    """Dataset names already converted in one label set."""
    paths = glob.glob(os.path.join(data_dir(), "model", "*", label_set, ENGINE_FILENAME))
    return sorted(path.split(os.sep)[-3] for path in paths)


def default_model_path(dataset: str | None = None,
                       label_set: str = DEFAULT_LABEL_SET) -> str:
    """Engine of one dataset, or -- when no dataset is given -- the first
    converted one in that label set, in alphabetical order."""
    if not dataset:
        trained = trained_datasets(label_set)
        dataset = trained[0] if trained else "<dataset>"
    return os.path.join(model_dir(dataset, label_set), ENGINE_FILENAME)


def _read_info(engine_path: str) -> dict:
    info_path = os.path.join(os.path.dirname(os.path.abspath(engine_path)), INFO_FILENAME)
    try:
        with open(info_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


class Detector:
    """Loads a TensorRT engine once and runs it on one RGB frame at a time.

    TensorRT engines only run on the GPU they were built on, so unlike the
    other baselines' `Detector`, this one has no CPU fallback -- see
    README.md for why an engine cannot be moved between devices at all.
    """

    def __init__(self, engine_path: str, device: str | None = None):
        device = device or "cuda:0"
        if not device.startswith("cuda"):
            raise ValueError(
                f"TensorRT engines only run on CUDA, got device={device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible to torch")
        self.device = torch.device(device)
        self.engine_path = engine_path

        info = _read_info(engine_path)
        self.info = info
        self.has_info = bool(info)
        self.class_names = tuple(info.get("class_names", ("tool", "tip")))
        self.label_set = info.get("label_set", DEFAULT_LABEL_SET)
        self.image_size = int(info.get("image_size") or 640)
        self.tip_box_size = float(info.get("tip_box_size") or DEFAULT_TIP_BOX_SIZE)
        self.dataset = info.get("dataset")
        self.epoch = info.get("epoch")
        self.metrics = info.get("metrics", {})
        self.precision = info.get("precision")

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=self.device)

        # One input, one output -- true of every engine `scripts/convert-model.py`
        # produces, since it exports a single `images -> predictions` graph.
        self._input_name = self._output_name = None
        self._input = self._output = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH_DTYPE[self.engine.get_tensor_dtype(name)]
            tensor = torch.zeros(shape, dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name, self._input = name, tensor
            else:
                self._output_name, self._output = name, tensor
        if self._input is None or self._output is None:
            raise RuntimeError(
                f"expected one input and one output tensor, engine has "
                f"{self.engine.num_io_tensors}: {engine_path}")

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray, conf: float = DEFAULT_CONF,
              iou: float = DEFAULT_IOU) -> tuple[np.ndarray, float]:
        """Return ((n, 6) [x1, y1, x2, y2, score, class] in frame pixels, elapsed ms)."""
        height, width = frame_rgb.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(frame_rgb, self.image_size)
        frame = torch.from_numpy(padded.transpose(2, 0, 1)).to(
            self.device, dtype=self._input.dtype)
        self._input.copy_(frame.div(255.0).unsqueeze_(0))

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        detections = non_max_suppression(self._output.float(), conf, iou)[0]
        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detections = detections.cpu().numpy()
        if len(detections):
            detections[:, :4] = undo_letterbox(detections[:, :4], scale, pad_x, pad_y,
                                               width, height)
        return detections, elapsed_ms
