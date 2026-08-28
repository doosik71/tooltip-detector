"""Frame sources for the demo: raw videos and extracted frame directories.

A copy of baseline/yolov8s/common/sources.py. The baselines are deliberately
independent sub-projects, so each carries its own copy rather than importing
across the boundary.

Two places hold laparoscopic footage for this project, and the demo reads
from both behind one interface:

  <tooltip-annotator>/data/dataset-src/<dataset>/*.mp4
      The original videos. No labels.

  data/dataset/<dataset>/images/<split>/*.png
      The 736x480 frames the tooltip-detector project trains on, with tip and
      bounding-box ground truth in the sibling annotation/<split>/*.json.

`data/dataset` is itself a symlink into the tooltip-annotator project, so
`dataset-src` is located by resolving that symlink and looking at its sibling
rather than by hard-coding the annotator's path.

Every source hands back RGB frames resized to 736x480 (FRAME_W x FRAME_H), the
frame geometry the rest of the project works in, so a video frame and a dataset
frame draw identically.
"""

import glob
import json
import os
from typing import NamedTuple

import cv2
import numpy as np

FRAME_W, FRAME_H = 736, 480

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")


# ---------------------------------------------------------------------------
# Locating the two roots
# ---------------------------------------------------------------------------

def repo_root() -> str:
    """Absolute path of the tooltip-detector repository root."""
    here = os.path.dirname(os.path.abspath(__file__))          # baseline/yolo26clone/common
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def default_frames_root() -> str:
    return os.path.join(repo_root(), "data", "dataset")


def default_videos_root() -> str:
    """`dataset-src`, resolved as the sibling of the real `data/dataset`.

    Falls back to a plain sibling of the (unresolved) path when data/dataset is
    missing, so the argument parser still has something concrete to show.
    """
    frames_root = default_frames_root()
    resolved = os.path.realpath(frames_root)
    return os.path.join(os.path.dirname(resolved), "dataset-src")


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

class SourceSpec(NamedTuple):
    """One selectable entry in the demo's Source dropdown."""

    label: str      # what the dropdown shows, e.g. "cholec80/test (frames)"
    kind: str       # "video" | "frames"
    path: str


def discover_sources(videos_root: str, frames_root: str) -> list[SourceSpec]:
    """List every video file and every extracted frame directory available.

    Missing roots are simply skipped -- a machine with only one of the two
    mounted still gets a usable dropdown.
    """
    specs: list[SourceSpec] = []

    if os.path.isdir(frames_root):
        for dataset in sorted(os.listdir(frames_root)):
            images_dir = os.path.join(frames_root, dataset, "images")
            if not os.path.isdir(images_dir):
                continue
            for split in sorted(os.listdir(images_dir)):
                split_dir = os.path.join(images_dir, split)
                if os.path.isdir(split_dir):
                    specs.append(SourceSpec(
                        f"{dataset}/{split} (frames)", "frames", split_dir))

    if os.path.isdir(videos_root):
        for dataset in sorted(os.listdir(videos_root)):
            dataset_dir = os.path.join(videos_root, dataset)
            if not os.path.isdir(dataset_dir):
                continue
            for name in sorted(os.listdir(dataset_dir)):
                if name.lower().endswith(VIDEO_EXTENSIONS):
                    specs.append(SourceSpec(
                        f"{dataset}/{name} (video)", "video",
                        os.path.join(dataset_dir, name)))

    return specs


def open_source(spec: SourceSpec) -> "FrameSource":
    if spec.kind == "frames":
        return FrameDirSource(spec.path, spec.label)
    return VideoSource(spec.path, spec.label)


# ---------------------------------------------------------------------------
# The sources themselves
# ---------------------------------------------------------------------------

def _fit(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.shape[1] != FRAME_W or frame_rgb.shape[0] != FRAME_H:
        frame_rgb = cv2.resize(frame_rgb, (FRAME_W, FRAME_H),
                               interpolation=cv2.INTER_LINEAR)
    return frame_rgb


class FrameSource:
    """Random-access sequence of RGB frames, optionally with ground truth."""

    label: str
    frame_count: int
    fps: float

    def read(self, index: int) -> np.ndarray | None:
        raise NotImplementedError

    def frame_name(self, index: int) -> str:
        return str(index)

    def ground_truth(self, index: int) -> list[dict] | None:
        """Annotations for this frame, or None when the source has no labels.

        Each entry is one tool: {"bbox": {x, y, width, height}, "tip": {x, y}}
        in 736x480 pixel coordinates, straight out of the annotation JSON.
        """
        return None

    def close(self) -> None:
        pass


class VideoSource(FrameSource):
    """A video file read through OpenCV.

    Seeking backwards in a long H.264 file is expensive, so `read()` only
    issues a seek when the requested frame is not the one that comes next --
    sequential playback stays a plain decode.
    """

    def __init__(self, path: str, label: str | None = None):
        self.path = path
        self.label = label or os.path.basename(path)
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise OSError(f"could not open video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.frame_count = max(1, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def read(self, index: int) -> np.ndarray | None:
        index = max(0, min(index, self.frame_count - 1))
        if index != int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        return _fit(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def frame_name(self, index: int) -> str:
        return f"{os.path.basename(self.path)}  #{index}"

    def close(self) -> None:
        self._cap.release()


class FrameDirSource(FrameSource):
    """A directory of extracted PNG frames, with the annotations beside it.

    Frames are listed in filename order, which groups them by source video
    (`video41_00000000.png`, `video41_00000025.png`, ...) and keeps each video's
    frames in time order -- so playing straight through walks one video at a
    time. The frames are sampled, not consecutive, so playback is not
    real-time footage; FPS is fixed at 25 only to give the seek/play controls a
    sensible pace.
    """

    PLAYBACK_FPS = 25.0

    def __init__(self, images_dir: str, label: str | None = None):
        self.images_dir = images_dir
        self.label = label or images_dir
        self.fps = self.PLAYBACK_FPS
        self._paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        if not self._paths:
            raise OSError(f"no PNG frames in: {images_dir}")
        self.frame_count = len(self._paths)
        # .../<dataset>/images/<split> -> .../<dataset>/annotation/<split>
        dataset_dir = os.path.dirname(os.path.dirname(images_dir))
        self._annotation_dir = os.path.join(
            dataset_dir, "annotation", os.path.basename(images_dir))

    def read(self, index: int) -> np.ndarray | None:
        index = max(0, min(index, self.frame_count - 1))
        frame_bgr = cv2.imread(self._paths[index], cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None
        return _fit(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def frame_name(self, index: int) -> str:
        index = max(0, min(index, self.frame_count - 1))
        return os.path.basename(self._paths[index])

    def ground_truth(self, index: int) -> list[dict] | None:
        index = max(0, min(index, self.frame_count - 1))
        stem = os.path.splitext(os.path.basename(self._paths[index]))[0]
        path = os.path.join(self._annotation_dir, stem + ".json")
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle).get("annotations", [])
        except (OSError, json.JSONDecodeError):
            # A frame without a readable annotation file is reported as
            # "unlabelled" rather than as "no tools in this frame" -- the two
            # look identical on screen otherwise.
            return None
