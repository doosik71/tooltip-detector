"""This repository's annotations, turned into an Ultralytics YOLO dataset.

`data/dataset/<name>/` carries, per frame, an RGB PNG and a JSON file listing
one entry per visible instrument:

    {"annotations": [{"bbox": {"x", "y", "width", "height"},
                      "tip":  {"x", "y"}}, ...], "width": 736, "height": 480}

There are no instrument-class labels, so the detector is trained on two classes
derived from that same annotation -- the same two the CLAD-Net and YOLOv8s-clone
baselines use, so the three are comparable:

    0  tool   the annotated bounding box
    1  tip    a square box of side `tip_box_size` centred on the tip

Ultralytics reads a dataset from its own directory layout, and `data/dataset/`
is a read-only mount, so `scripts/prepare-dataset.py` builds one next to this
package:

    data/yolo/<dataset>/
        dataset.yaml         what Ultralytics is pointed at
        prepare-status.json  tip box size, frame counts, how the lists were cut
        images/<split>       symlink to <repo>/data/dataset/<dataset>/images/<split>
        labels/<split>/*.txt one YOLO label file per listed frame
        <split>.txt          the frames of that split this dataset uses

The image directory is a symlink rather than a copy -- 200k PNGs are not worth
duplicating -- and only the frames named in `<split>.txt` get a label file, so
subsampling with `--frame-stride` also cuts the preparation cost.

The tip box side is a real hyper-parameter, not a formality: it decides how many
anchor points fall inside a tip and therefore how much learning signal the tip
class gets. It is recorded in prepare-status.json and copied into the trained
model's sidecar, so no evaluation ever scores a model against labels of a size
it never saw.
"""

import glob
import json
import os

import cv2
import numpy as np

CLASS_NAMES = ("tool", "tip")
TOOL_CLASS, TIP_CLASS = 0, 1

# Matches the YOLOv8s-clone baseline. See its README for the measurement: at
# 640 x 640 a 32 px box is the first size whose anchor count clears the
# assigner's topk, and the CLAD-Net baseline's 10 px box does not.
DEFAULT_TIP_BOX_SIZE = 32.0

SPLITS = ("train", "val", "test")

FRAME_EXTENSION = ".png"

# What every frame in these datasets measures. Only used as a last resort, for
# an annotation that does not record its own frame size.
FRAME_SIZE = (736, 480)


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))        # baseline/yolo26/common
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def project_root() -> str:
    """This sub-project's own directory, baseline/yolo26/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_data_root() -> str:
    """The source dataset the whole repository shares."""
    return os.path.join(repo_root(), "data", "dataset")


def data_dir() -> str:
    """This baseline's own outputs, all under baseline/yolo26/data/."""
    return os.path.join(project_root(), "data")


def yolo_dir(dataset: str) -> str:
    """The prepared Ultralytics dataset: data/yolo/<dataset>/."""
    return os.path.join(data_dir(), "yolo", dataset)


# The stage below <dataset> names which classes a run learned. The clone
# baselines can also train `tiponly` (the tip box with no tool class); this
# baseline only ever trains the two-class form, so the stage is a constant
# here. The layout is shared so one summary script and one dashboard rule
# cover every baseline.
LABEL_SET = "tooltip"


def model_dir(dataset: str) -> str:
    """One run's checkpoints: data/model/<dataset>/tooltip/."""
    return os.path.join(data_dir(), "model", dataset, LABEL_SET)


def results_dir(dataset: str) -> str:
    """One run's evaluation output: data/results/<dataset>/tooltip/."""
    return os.path.join(data_dir(), "results", dataset, LABEL_SET)


def available_datasets(data_root: str | None = None) -> list[str]:
    data_root = data_root or default_data_root()
    if not os.path.isdir(data_root):
        return []
    return sorted(name for name in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, name, "images")))


def prepared_datasets() -> list[str]:
    """Dataset names that already have a prepared Ultralytics dataset."""
    paths = glob.glob(os.path.join(data_dir(), "yolo", "*", "dataset.yaml"))
    return sorted(os.path.basename(os.path.dirname(path)) for path in paths)


# ---------------------------------------------------------------------------
# The prepared dataset
# ---------------------------------------------------------------------------

def dataset_yaml_path(dataset: str) -> str:
    return os.path.join(yolo_dir(dataset), "dataset.yaml")


def prepare_status_path(dataset: str) -> str:
    return os.path.join(yolo_dir(dataset), "prepare-status.json")


def read_prepare_status(dataset: str) -> dict | None:
    """What scripts/prepare-dataset.py recorded, or None if it never ran."""
    try:
        with open(prepare_status_path(dataset), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def require_prepared(dataset: str) -> dict:
    status = read_prepare_status(dataset)
    if status is None or not os.path.exists(dataset_yaml_path(dataset)):
        raise SystemExit(
            f"dataset '{dataset}' has not been prepared for Ultralytics yet.\n"
            f"run:  ./baseline/yolo26/run prepare-dataset --dataset {dataset}")
    return status


# ---------------------------------------------------------------------------
# Reading the source annotations
# ---------------------------------------------------------------------------

def images_dir(dataset: str, split: str, data_root: str | None = None) -> str:
    return os.path.join(data_root or default_data_root(), dataset, "images", split)


def annotation_dir(dataset: str, split: str, data_root: str | None = None) -> str:
    return os.path.join(data_root or default_data_root(), dataset, "annotation", split)


def frame_paths(dataset: str, split: str, data_root: str | None = None,
                frame_stride: int = 1, limit: int | None = None) -> list[str]:
    """Sorted frames of one split, optionally thinned.

    Filename order groups frames by source video and keeps each video in time
    order, so a stride > 1 drops near-duplicate neighbours evenly across every
    session instead of cutting whole videos.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    paths = sorted(glob.glob(os.path.join(images_dir(dataset, split, data_root),
                                          "*" + FRAME_EXTENSION)))
    if not paths:
        raise OSError(f"no frames under {images_dir(dataset, split, data_root)}")
    paths = paths[::max(1, frame_stride)]
    return paths[:limit] if limit else paths


def session_id(frame_name: str) -> str:
    """Source video of a frame -- `video41_00000000.png` -> `video41`."""
    return os.path.splitext(os.path.basename(frame_name))[0].rsplit("_", 1)[0]


def load_annotation(path: str, width: float | None = None, height: float | None = None,
                    tip_box_size: float = DEFAULT_TIP_BOX_SIZE) -> np.ndarray:
    """Read one annotation JSON as (n, 5) normalised [class, cx, cy, w, h].

    The coordinates are normalised by `width` x `height`. Leave them out and
    the frame size recorded in the annotation itself is used, which is what
    label conversion wants: it never opens the PNG, so it has nothing else to
    normalise by. Evaluation passes the size of the frame it actually read.

    A frame with no visible instrument yields an empty array -- those frames
    are kept, as pure-background examples.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return np.zeros((0, 5), dtype=np.float32)

    width = width or payload.get("width", FRAME_SIZE[0])
    height = height or payload.get("height", FRAME_SIZE[1])

    rows = []
    for item in payload.get("annotations", []):
        box = item.get("bbox")
        if box and box.get("width", 0) > 0 and box.get("height", 0) > 0:
            rows.append((TOOL_CLASS,
                         (box["x"] + box["width"] / 2) / width,
                         (box["y"] + box["height"] / 2) / height,
                         box["width"] / width,
                         box["height"] / height))
        tip = item.get("tip")
        if tip is not None:
            rows.append((TIP_CLASS, tip["x"] / width, tip["y"] / height,
                         tip_box_size / width, tip_box_size / height))
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def label_text(labels: np.ndarray) -> str:
    """(n, 5) normalised rows -> the contents of one YOLO .txt label file."""
    lines = [f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}"
             for row in labels if 0 < row[3] and 0 < row[4]]
    return "\n".join(lines) + ("\n" if lines else "")


class SplitFrames:
    """One split of one dataset, read straight from the source tree.

    Only evaluation and the summary scripts use this -- training reads through
    Ultralytics' own loader. There is no augmentation here and no resizing:
    `read_frame` hands back the original 736 x 480 frame, which is the pixel
    space every reported distance is measured in.
    """

    def __init__(self, dataset: str, split: str, data_root: str | None = None,
                 frame_stride: int = 1, limit: int | None = None,
                 tip_box_size: float = DEFAULT_TIP_BOX_SIZE):
        self.dataset = dataset
        self.split = split
        self.tip_box_size = tip_box_size
        self.annotation_dir = annotation_dir(dataset, split, data_root)
        self.image_paths = frame_paths(dataset, split, data_root, frame_stride, limit)

    def __len__(self) -> int:
        return len(self.image_paths)

    def frame_name(self, index: int) -> str:
        return os.path.basename(self.image_paths[index])

    def session_id(self, index: int) -> str:
        return session_id(self.frame_name(index))

    def annotation_path(self, index: int) -> str:
        stem = os.path.splitext(self.frame_name(index))[0]
        return os.path.join(self.annotation_dir, stem + ".json")

    def read_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Original-resolution RGB frame plus normalised labels."""
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"could not read {self.image_paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        return image, load_annotation(self.annotation_path(index), width, height,
                                      self.tip_box_size)
