"""One checkpoint path in, tip coordinates out.

`scripts/tooltip-tracker.py` needs exactly one thing from a model: given an RGB
frame, where are the tool tips? Everything downstream of that -- the arrow
smoothing, the false-positive guards, the overlay -- reads nothing but a list
of `(x, y, score)` points in the 736 x 480 frame coordinate space. This module
is that contract, so the tracker does not have to know which family of model it
is driving.

Two families implement it:

  heatmap    this project's own models (`monai`, `monai_mini`). The network
             emits a distance heatmap; peaks above a threshold are the tips.
  detector   the reimplementation baselines under `baseline/` (`yolov8sclone`,
             `cladnet`, `yolo26clone`). The network emits `tool` and `tip`
             boxes; the centre of a `tip` box is a tip.

The Ultralytics-based baselines (`baseline/yolov8s`, `baseline/yolo26`) are
deliberately *not* supported: they need `ultralytics`, which requires
`opencv-python` and would clobber the root environment's
`opencv-python-headless`. Their architectures are covered by the
reimplementation baselines, which are ultralytics-free by design.

Which family a checkpoint belongs to is read off its path, because both layouts
are already fixed conventions (see `ttd.checkpoints` and each baseline's
README):

    data/models/<dataset>/<target-mode>/<model-type>/best.pt        -> heatmap
    baseline/<name>/data/model/<dataset>/model.pt                   -> detector

That is why the tracker takes a path and nothing else. It also means only one
baseline is ever imported per process, which matters: all three baselines name
their package `common`, so two of them cannot be on `sys.path` at once.
"""

import os
import sys
import time

import numpy as np
import torch

# Baselines whose checkpoints this module can drive. The Ultralytics-based
# ones are absent on purpose; see the module docstring.
SUPPORTED_BASELINES = ("yolov8sclone", "cladnet", "yolo26clone")

_TIP_CLASS_NAME = "tip"


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# What a path says about a checkpoint
# ---------------------------------------------------------------------------

class CheckpointSpec:
    """What the layout of a checkpoint path says about it.

    `kind` is "heatmap" or "detector"; the remaining fields are whatever that
    layout encodes, and are used only for the status line the GUI shows.
    """

    def __init__(self, path: str, kind: str, label: str, dataset: str | None = None,
                 baseline: str | None = None, model_type: str | None = None,
                 target_mode: str | None = None):
        self.path = path
        self.kind = kind
        self.label = label
        self.dataset = dataset
        self.baseline = baseline
        self.model_type = model_type
        self.target_mode = target_mode


def describe_checkpoint(path: str) -> CheckpointSpec:
    """Classify a checkpoint by where it sits, without opening it.

    Raises SystemExit with the two accepted layouts spelled out when the path
    matches neither -- a wrong path is the most likely way to get here, and a
    KeyError three frames deep would not say so.
    """
    absolute = os.path.abspath(path)
    if not os.path.exists(absolute):
        raise SystemExit(f"checkpoint not found: {path}")

    parts = absolute.split(os.sep)

    # baseline/<name>/data/model/<dataset>/<file>.pt
    if "baseline" in parts:
        index = len(parts) - 1 - parts[::-1].index("baseline")
        tail = parts[index + 1:]
        if len(tail) >= 5 and tail[1] == "data" and tail[2] == "model":
            name, dataset = tail[0], tail[3]
            if name not in SUPPORTED_BASELINES:
                raise SystemExit(
                    f"baseline '{name}' is not supported by tooltip-tracker.\n"
                    f"supported: {', '.join(SUPPORTED_BASELINES)}\n"
                    "The Ultralytics-based baselines need a separate virtualenv "
                    "(opencv-python vs opencv-python-headless), so they cannot be "
                    "loaded in this process. Their architectures are covered by the "
                    "reimplementation baselines listed above.")
            return CheckpointSpec(absolute, "detector",
                                  label=f"{name}/{dataset}",
                                  dataset=dataset, baseline=name)

    # data/models/<dataset>/<target-mode>/<model-type>/<file>.pt
    if "models" in parts:
        index = len(parts) - 1 - parts[::-1].index("models")
        tail = parts[index + 1:]
        if len(tail) >= 4:
            dataset, target_mode, model_type = tail[0], tail[1], tail[2]
            return CheckpointSpec(absolute, "heatmap",
                                  label=f"{dataset}/{target_mode}/{model_type}",
                                  dataset=dataset, model_type=model_type,
                                  target_mode=target_mode)

    raise SystemExit(
        f"cannot tell what kind of model this is from its path:\n  {path}\n\n"
        "tooltip-tracker reads the model's identity from the directory layout, "
        "so the path has to be one of:\n"
        "  data/models/<dataset>/<target-mode>/<model-type>/best.pt\n"
        "  baseline/<name>/data/model/<dataset>/model.pt   "
        f"(<name>: {', '.join(SUPPORTED_BASELINES)})")


# ---------------------------------------------------------------------------
# The two implementations
# ---------------------------------------------------------------------------

class HeatmapTipSource:
    """This project's own models: peaks of a distance heatmap are the tips."""

    #: The NMS radius slider suppresses neighbouring peaks, so it applies here.
    supports_nms_radius = True

    def __init__(self, spec: CheckpointSpec, device: torch.device):
        from ttd.model import build as build_model
        from ttd.peaks import find_peaks
        from ttd.transforms import _eval_transform

        self.spec = spec
        self.device = device
        self._find_peaks = find_peaks
        self._transform = _eval_transform()

        model = build_model(spec.model_type, num_classes=2).to(device)
        state = torch.load(spec.path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        self._model = model

    def peaks(self, frame_rgb: np.ndarray, threshold: float, nms_radius: int
              ) -> tuple[list[tuple[float, float, float]], float]:
        tensor = self._transform(image=frame_rgb)["image"]
        tensor = tensor.unsqueeze(0).to(self.device, dtype=torch.float32)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            prediction = self._model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        heatmap = torch.sigmoid(prediction[0, 1]).cpu().numpy()
        return self._find_peaks(heatmap, threshold, nms_radius), elapsed_ms


class DetectorTipSource:
    """A reimplementation baseline: the centre of a `tip` box is a tip.

    The baseline's own `common.inference.Detector` is reused as-is rather than
    reimplemented here, so the tracker sees exactly what that project's
    `eval-model` scores. It maps its boxes back out of the letterbox itself,
    so the coordinates arrive in the frame's own pixel space already.
    """

    #: The detector's own NMS (or, for yolo26clone, its end-to-end head) has
    #: already reduced each object to one box. Merging tip centres again by
    #: radius would risk collapsing the two tips a single scissors or grasper
    #: legitimately shows, so the slider is left inactive for this family.
    supports_nms_radius = False

    def __init__(self, spec: CheckpointSpec, device: torch.device):
        self.spec = spec
        self.device = device

        # Only ever one baseline per process -- the checkpoint is fixed at
        # launch -- so putting its directory on sys.path is safe even though
        # all three baselines call their package `common`.
        baseline_dir = os.path.join(repo_root(), "baseline", spec.baseline)
        already = sys.modules.get("common")
        if already is not None and not getattr(
                already, "__file__", "").startswith(baseline_dir + os.sep):
            # A second baseline in one process would silently reuse the first
            # one's `common.model`, which fails much later with an unrelated
            # TypeError about architecture keywords.
            raise SystemExit(
                f"another baseline is already loaded in this process, so "
                f"'{spec.baseline}' cannot be: all baselines name their package "
                "`common`.\nRun one tracker process per model.")
        if baseline_dir not in sys.path:
            sys.path.insert(0, baseline_dir)
        from common.inference import Detector

        self._detector = Detector(spec.path, str(device))
        names = self._detector.class_names
        if _TIP_CLASS_NAME not in names:
            raise SystemExit(
                f"this checkpoint has no `{_TIP_CLASS_NAME}` class "
                f"(classes: {', '.join(names)}); there is nothing for the "
                "tracker to follow.")
        self._tip_class = names.index(_TIP_CLASS_NAME)

    def peaks(self, frame_rgb: np.ndarray, threshold: float, nms_radius: int
              ) -> tuple[list[tuple[float, float, float]], float]:
        # `threshold` is the same slider the heatmap models use as a peak
        # cut-off; for a detector the equivalent decision is the confidence
        # threshold. `nms_radius` is ignored -- see supports_nms_radius.
        detections, elapsed_ms = self._detector.detect(frame_rgb, threshold)
        if not len(detections):
            return [], elapsed_ms
        tips = detections[detections[:, 5] == self._tip_class]
        peaks = [(float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2),
                  float(box[4])) for box in tips]
        return peaks, elapsed_ms


# ---------------------------------------------------------------------------
# What is available to load
# ---------------------------------------------------------------------------

def available_checkpoints(root: str | None = None) -> list[CheckpointSpec]:
    """Every checkpoint on disk that this module can drive, in a stable order.

    Used to answer "what can I run?" when the tracker is started without a
    path. Nothing is opened: the layouts alone say what each file is.
    """
    import glob

    root = root or repo_root()
    found: list[CheckpointSpec] = []

    # This project's own models: data/models/<dataset>/<target>/<type>/best.pt
    for path in sorted(glob.glob(os.path.join(
            root, "data", "models", "*", "*", "*", "best.pt"))):
        dataset, target_mode, model_type = path.split(os.sep)[-4:-1]
        found.append(CheckpointSpec(path, "heatmap",
                                    label=f"{dataset}/{target_mode}/{model_type}",
                                    dataset=dataset, model_type=model_type,
                                    target_mode=target_mode))

    # Baselines: baseline/<name>/data/model/<dataset>/model.pt
    for name in SUPPORTED_BASELINES:
        for path in sorted(glob.glob(os.path.join(
                root, "baseline", name, "data", "model", "*", "model.pt"))):
            dataset = path.split(os.sep)[-2]
            found.append(CheckpointSpec(path, "detector", label=f"{name}/{dataset}",
                                        dataset=dataset, baseline=name))
    return found


def format_available(root: str | None = None) -> str:
    """The checkpoint list as the tracker prints it when given no path."""
    root = root or repo_root()
    specs = available_checkpoints(root)
    if not specs:
        return ("No usable checkpoint found under:\n"
                f"  {os.path.join(root, 'data', 'models')}\n"
                f"  {os.path.join(root, 'baseline', '<name>', 'data', 'model')}\n\n"
                "Train one first: see docs/train-guide.md, or a baseline's "
                "docs/commands.md.")

    lines = ["Available models (pass one of these paths):", ""]
    for kind, heading in (("heatmap", "this project (heatmap)"),
                          ("detector", "baseline detectors")):
        group = [s for s in specs if s.kind == kind]
        if not group:
            continue
        lines.append(f"  {heading}")
        for spec in group:
            lines.append(f"    {os.path.relpath(spec.path, root)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def open_tip_source(path: str, device: torch.device):
    """Load whatever model `path` points at, ready to hand back tip peaks."""
    spec = describe_checkpoint(path)
    if spec.kind == "detector":
        return DetectorTipSource(spec, device)
    return HeatmapTipSource(spec, device)
