"""Interactive trainer: correct labels and fine-tune the detector in one loop.

The dataset labels were produced by a weak auto-labeller, and a detector trained
on them inherits that weakness. Fixing the labels first and retraining later
costs a lot on both sides, so this GUI runs the two together:

  1. Draw `--initial-files` (32) frames at random from the chosen split, run
     the detector on each, and score how well its prediction agrees with the
     frame's current label (see `agreement_score`). The list shows the frames
     lowest score first, and the lowest one opens first.
  2. The user edits the label as in tooltip-annotator's `annotation_editor`, or
     presses Update (Ctrl+U / F4) to start from the detector's prediction.
     The label and the prediction are overlaid in different colours.
  3. Save (Ctrl+S / F2) writes the label to annotation-manual/ and mirrors it
     into annotation/, exactly like `annotation_editor`, and opens the next
     frame of the list at once, in the order the list already has. The model
     trains in the background on that frame -- plus a few replayed frames from
     annotation-manual/ so it does not forget everything else -- until its
     prediction on the saved frame scores `--target-score` or it has taken
     `--max-steps` steps, and overwrites the checkpoint.
  4. After training, the frames in the list and `--new-files` (4) new ones
     are re-scored with the updated model, so the list grows by that many per
     training; past `--max-files` (100) only the lowest-scoring are kept. The
     refreshed list keeps the frame being edited open, so it may no longer be
     first: the next frame to edit comes from the previous ranking, which is
     the price of never waiting for the model.

The model is the YOLOv8s clone baseline (`baseline/yolov8sclone`), trained with
the `tooltip` label set: it predicts `tool` boxes and `tip` boxes, and the
centre of a tip box is the tip. Every model call -- loading, scoring, training
-- runs on one background thread, so the GUI never waits for the GPU and the
model is never touched by two threads at once.

Frames already in annotation-manual/ are never drawn for the list: a person
has already checked them. They serve as the replay set instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont, messagebox, ttk

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "baseline" / "yolov8sclone"

# The baseline's package is called `common`, the same convention
# ttd/tip_source.py relies on; this process loads no other baseline.
if True:
    sys.path.insert(0, str(BASELINE_DIR))

    from common.boxes import letterbox, non_max_suppression, undo_letterbox
    from common.dataset import SurgicalDetectionDataset, available_datasets
    from common.loss import DetectionLoss
    from common.model import build, decode


DATA_ROOT = REPO_ROOT / "data" / "dataset"
DEFAULT_MODEL_PATH = REPO_ROOT / "data" / "itrain-model" / "model.pt"
SPLIT_NAMES = ("train", "val", "test")
ANNOTATION_SUFFIX = ".json"

DEFAULT_INITIAL_FILES = 32
DEFAULT_NEW_FILES = 4
DEFAULT_MAX_FILES = 100
DEFAULT_TARGET_SCORE = 0.9
DEFAULT_MAX_STEPS = 20
DEFAULT_REPLAY = 3
DEFAULT_LR = 1e-4
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
GRAD_CLIP_NORM = 10.0       # matches the baseline's trainer

# A predicted tip pairs with a predicted tool box only if it lies inside the box
# grown by this many pixels on each side; tip boxes are centred on the tip, so
# a tip on the box edge can land just outside it.
PAIR_MARGIN = 16
# In the agreement score, a tip this far from its label contributes nothing.
# It is the side of the tip box the baseline is trained with.
TIP_DISTANCE_SCALE = 32.0

HANDLE_SIZE = 7.5
TIP_RADIUS = 6
SEGMENTATION_ALPHA = 0.35
# Add mode dims the image to this brightness, except inside the box being drawn.
ADD_MODE_BRIGHTNESS = 0.5
MIN_BOX_SIZE = 4
# Pointer shape and fill colour for the tip or resize handle under the pointer.
HANDLE_CURSOR = "crosshair"
HANDLE_HOVER_FILL = "#ffffff"

# The label keeps annotation_editor's colours (cyan box, red tip); the model's
# prediction is drawn dashed in magenta beneath it.
PREDICTION_COLOR = "#ff4dff"
PREDICTION_COLOR_RGB = (255, 77, 255)
PREDICTION_TIP_RADIUS = 7
# Canvas items cannot be translucent, so the prediction is drawn into the
# displayed image instead and blended in at this opacity.
PREDICTION_ALPHA = 0.5

# Frames with a hand-made annotation (annotation-manual/) are highlighted in
# the list, in the colours annotation_editor uses for them.
MANUAL_ITEM_BACKGROUND = "#ffe9a8"
MANUAL_ITEM_FOREGROUND = "#1a1a1a"
MANUAL_ITEM_SELECT_BACKGROUND = "#e8a33d"
MANUAL_ITEM_SELECT_FOREGROUND = "#1a1a1a"
# The frame on screen is shown in sky blue, over the manual colours, so it
# stays easy to find after a background refresh reorders the list.
CURRENT_ITEM_BACKGROUND = "#87ceeb"
CURRENT_ITEM_FOREGROUND = "#1a1a1a"

# The tool number drawn on each bbox, as a multiple of the canvas default font.
ANNOTATION_LABEL_FONT_SCALE = 2

# The frame list, as a multiple of the default fixed-width font.
LIST_FONT_SCALE = 1.2

# The tool count in the status bar stands out from the rest of the readout.
TOOL_COUNT_FONT_SCALE = 1.5
TOOL_COUNT_FOREGROUND = "#1565c0"

BUTTON_STYLES = {
    "Add.TButton": "#1565c0",     # blue
    "Delete.TButton": "#c62828",  # red
    "Save.TButton": "#e65100",    # orange
    "Reload.TButton": "#6a1b9a",  # purple
    "Update.TButton": "#2e7d32",  # green
}
ADD_ACTIVE_STYLE = "AddActive.TButton"
ADD_ACTIVE_BACKGROUND = "#ffe680"
ADD_ACTIVE_HOVER_BACKGROUND = "#ffd84d"
BUTTON_DISABLED_FOREGROUND = "#9e9e9e"

ADD_BUTTON_LABEL = "Add Tool [INS/Wheel]"
ADD_BUTTON_ACTIVE_LABEL = "Adding... (Esc/Wheel to cancel)"
DELETE_BUTTON_LABEL = "Delete Tool [DEL]"
SAVE_BUTTON_LABEL = "Save [F2]"
RELOAD_BUTTON_LABEL = "Reload [F3]"
UPDATE_BUTTON_LABEL = "Update [F4]"


# ---------------------------------------------------------------------------
# Annotation files
# ---------------------------------------------------------------------------

def images_dir(dataset: str, split: str) -> Path:
    return DATA_ROOT / dataset / "images" / split


def annotation_path(dataset: str, split: str, stem: str) -> Path:
    return DATA_ROOT / dataset / "annotation" / split / f"{stem}{ANNOTATION_SUFFIX}"


def manual_dir(dataset: str, split: str) -> Path:
    return DATA_ROOT / dataset / "annotation-manual" / split


def segmentation_path(dataset: str, split: str, stem: str) -> Path:
    return DATA_ROOT / dataset / "segmentation" / split / f"{stem}.png"


def read_annotations(path: Path) -> list[dict[str, dict[str, int]]]:
    """One annotation JSON as a list of {"bbox", "tip"}; [] if absent or broken."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    annotations = []
    for item in payload.get("annotations", []):
        bbox = item.get("bbox", {})
        tip = item.get("tip", {})
        try:
            annotations.append({
                "bbox": {
                    "x": int(bbox.get("x", 0)),
                    "y": int(bbox.get("y", 0)),
                    "width": max(int(bbox.get("width", 1)), 1),
                    "height": max(int(bbox.get("height", 1)), 1),
                },
                "tip": {"x": int(tip.get("x", 0)), "y": int(tip.get("y", 0))},
            })
        except (TypeError, ValueError):
            continue
    return annotations


def write_manual_annotation(path: Path, image_name: str, width: int, height: int,
                            annotations: list[dict[str, dict[str, int]]]) -> None:
    """Write the same payload tooltip-annotator's `build_payload` makes for a
    manual edit, so generate_annotation and annotation_editor read it as theirs."""
    payload = {
        "image": image_name,
        "width": int(width),
        "height": int(height),
        "source": "manual",
        "annotations": annotations,
        "edited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def draw_dashed_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int],
                     color: tuple[int, int, int], thickness: int, dash: int, gap: int) -> None:
    """cv2.line in `dash`-pixel segments separated by `gap` pixels."""
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length == 0:
        return
    ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        cv2.line(image,
                 (round(start[0] + ux * position), round(start[1] + uy * position)),
                 (round(start[0] + ux * stop), round(start[1] + uy * stop)),
                 color, thickness, cv2.LINE_AA)
        position += dash + gap


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Prediction <-> label
# ---------------------------------------------------------------------------

def pair_detections(detections: np.ndarray, tool_class: int, tip_class: int,
                    width: int, height: int) -> list[dict]:
    """Turn detector boxes into annotations: one tool box with one tip each.

    The detector predicts tool boxes and tip boxes independently. A tip may
    pair only with a tool box it lies in (grown by PAIR_MARGIN), and among the
    feasible pairings the one with the highest total tip confidence wins. A
    tool box left without a tip gets the box corner nearest the image centre,
    since tools enter from the edge of the frame; a tip left without a box is
    dropped, because an annotation needs both.
    """
    tools = detections[detections[:, 5] == tool_class]
    tips = detections[detections[:, 5] == tip_class]
    tip_points = (tips[:, :2] + tips[:, 2:4]) / 2

    tip_of: dict[int, int] = {}
    if len(tools) and len(tips):
        inside = (
            (tip_points[None, :, 0] >= tools[:, None, 0] - PAIR_MARGIN)
            & (tip_points[None, :, 0] <= tools[:, None, 2] + PAIR_MARGIN)
            & (tip_points[None, :, 1] >= tools[:, None, 1] - PAIR_MARGIN)
            & (tip_points[None, :, 1] <= tools[:, None, 3] + PAIR_MARGIN)
        )
        gain = np.where(inside, tips[None, :, 4], 0.0)
        for row, col in zip(*linear_sum_assignment(-gain)):
            if inside[row, col]:
                tip_of[int(row)] = int(col)

    annotations = []
    for index, tool in enumerate(tools):
        x1, y1, x2, y2 = tool[:4]
        if index in tip_of:
            tip_x, tip_y = tip_points[tip_of[index]]
        else:
            tip_x = x1 if abs(x1 - width / 2) < abs(x2 - width / 2) else x2
            tip_y = y1 if abs(y1 - height / 2) < abs(y2 - height / 2) else y2
        tip_x = min(max(int(round(tip_x)), 0), width - 1)
        tip_y = min(max(int(round(tip_y)), 0), height - 1)
        # Grow the box to hold the tip pixel, as annotation_editor does when a
        # tip is dragged outside its box.
        left = max(min(int(math.floor(x1)), tip_x), 0)
        top = max(min(int(math.floor(y1)), tip_y), 0)
        right = min(max(int(math.ceil(x2)), tip_x + 1), width)
        bottom = min(max(int(math.ceil(y2)), tip_y + 1), height)
        annotations.append({
            "bbox": {"x": left, "y": top,
                     "width": max(right - left, 1), "height": max(bottom - top, 1)},
            "tip": {"x": tip_x, "y": tip_y},
            "score": float(tool[4]),
        })
    annotations.sort(key=lambda item: (item["bbox"]["x"], item["bbox"]["y"]))
    return annotations


def _box_iou(a: dict[str, int], b: dict[str, int]) -> float:
    inter_w = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    inter_h = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
    inter = max(inter_w, 0) * max(inter_h, 0)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def agreement_score(predicted: list[dict], labelled: list[dict]) -> float:
    """How well a prediction agrees with a label, from 0 (not at all) to 1.

    Each (prediction, label) pair is worth the mean of its box IoU and its tip
    closeness, 1 - distance / TIP_DISTANCE_SCALE clipped at 0. Pairs are
    matched one-to-one to maximise the total, and the total is divided by the
    larger of the two counts, so a missed or an extra tool costs as much as a
    completely wrong one. Two empty frames agree perfectly.
    """
    if not predicted and not labelled:
        return 1.0
    if not predicted or not labelled:
        return 0.0
    quality = np.zeros((len(predicted), len(labelled)))
    for i, p in enumerate(predicted):
        for j, l in enumerate(labelled):
            distance = math.hypot(p["tip"]["x"] - l["tip"]["x"], p["tip"]["y"] - l["tip"]["y"])
            closeness = max(0.0, 1.0 - distance / TIP_DISTANCE_SCALE)
            quality[i, j] = 0.5 * _box_iou(p["bbox"], l["bbox"]) + 0.5 * closeness
    rows, cols = linear_sum_assignment(-quality)
    return float(quality[rows, cols].sum()) / max(len(predicted), len(labelled))


# ---------------------------------------------------------------------------
# The model, owned by the worker thread
# ---------------------------------------------------------------------------

class ModelRunner:
    """The detector checkpoint, kept unfused so it can keep training.

    `common.inference.Detector` folds BatchNorm into the convolutions, which
    makes the model faster but no longer trainable, so the model is built
    here directly from the same checkpoint format.
    """

    def __init__(self, path: Path, device: torch.device, args: argparse.Namespace):
        self.path = path
        self.device = device
        self.args = args

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.class_names = tuple(checkpoint.get("class_names", ()))
        if "tool" not in self.class_names or "tip" not in self.class_names:
            raise RuntimeError(
                f"{path} predicts {self.class_names}; the interactive trainer needs a "
                "`tooltip` checkpoint that predicts both `tool` and `tip`.")
        self.tool_class = self.class_names.index("tool")
        self.tip_class = self.class_names.index("tip")
        self.image_size = int(checkpoint.get("image_size", 640))
        self.tip_box_size = float(checkpoint["tip_box_size"])
        # Everything but the weights is written back unchanged on save, so the
        # baseline's own tools can still open the checkpoint.
        self.metadata = {key: value for key, value in checkpoint.items() if key != "model"}

        self.model = build(num_classes=len(self.class_names), **checkpoint.get("arch", {}))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        self.criterion = DetectionLoss(self.model)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=args.lr)

    def describe(self) -> str:
        rounds = self.metadata.get("itrain_rounds", 0)
        return (f"{self.path.name} ({self.metadata.get('dataset')}, "
                f"epoch {self.metadata.get('epoch')}, {rounds} rounds) on {self.device}")

    @torch.no_grad()
    def predict(self, frame_rgb: np.ndarray) -> list[dict]:
        self.model.eval()
        height, width = frame_rgb.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(frame_rgb, self.image_size)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1))[None]
        tensor = tensor.to(self.device).float().div_(255.0)
        predictions = decode(self.model(tensor), self.model)
        detections = non_max_suppression(predictions, self.args.conf, self.args.iou)[0]
        detections = detections.cpu().numpy()
        if len(detections):
            detections[:, :4] = undo_letterbox(detections[:, :4], scale, pad_x, pad_y,
                                               width, height)
        return pair_detections(detections, self.tool_class, self.tip_class, width, height)

    def train(self, frame_rgb: np.ndarray, annotations: list[dict],
              replay: list[tuple[Path, Path]], report) -> tuple[int, float, float]:
        """Train until the saved frame scores `--target-score` or `--max-steps`.

        Each step's batch is the saved frame plus up to `--replay` frames drawn
        afresh from `replay`. Only the saved frame decides when to stop.
        Returns (steps, score before, score after).
        """
        before = score = agreement_score(self.predict(frame_rgb), annotations)
        steps = 0
        while score < self.args.target_score and steps < self.args.max_steps:
            batch = [(frame_rgb, annotations)]
            for image_path, label_path in random.sample(replay, min(self.args.replay, len(replay))):
                batch.append((read_rgb(image_path), read_annotations(label_path)))
            images, targets = self._training_batch(batch)

            self.model.train()
            # A batch of a few frames gives BatchNorm useless statistics, so its
            # running averages stay as the full training run left them.
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
            loss, _ = self.criterion(self.model(images), targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP_NORM)
            self.optimizer.step()

            steps += 1
            score = agreement_score(self.predict(frame_rgb), annotations)
            report(steps, score)
        if steps:
            self._save()
        return steps, before, score

    def _training_batch(self, batch: list[tuple[np.ndarray, list[dict]]]
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Letterbox, random horizontal flip and colour jitter, in the
        baseline's target format: (n, 6) [image, class, cx, cy, w, h] normalised."""
        size = self.image_size
        images, targets = [], []
        for index, (frame, annotations) in enumerate(batch):
            padded, scale, pad_x, pad_y = letterbox(frame, size)
            rows = []
            for annotation in annotations:
                bbox, tip = annotation["bbox"], annotation["tip"]
                rows.append((self.tool_class, bbox["x"] + bbox["width"] / 2,
                             bbox["y"] + bbox["height"] / 2, bbox["width"], bbox["height"]))
                rows.append((self.tip_class, tip["x"], tip["y"],
                             self.tip_box_size, self.tip_box_size))
            labels = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            labels[:, 1] = labels[:, 1] * scale + pad_x
            labels[:, 2] = labels[:, 2] * scale + pad_y
            labels[:, 3:5] *= scale
            if random.random() < 0.5:
                padded = np.ascontiguousarray(padded[:, ::-1])
                labels[:, 1] = size - labels[:, 1]
            padded = SurgicalDetectionDataset._augment_colour(padded)

            images.append(torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))))
            target = torch.zeros((len(labels), 6), dtype=torch.float32)
            target[:, 0] = index
            target[:, 1] = torch.from_numpy(labels[:, 0])
            target[:, 2:] = torch.from_numpy(labels[:, 1:5]) / size
            targets.append(target)
        images = torch.stack(images).to(self.device).float().div_(255.0)
        return images, torch.cat(targets)

    def _save(self) -> None:
        """Overwrite the checkpoint, through a temporary file so a crash or a
        closed window mid-write never leaves a truncated model behind."""
        self.metadata["itrain_rounds"] = self.metadata.get("itrain_rounds", 0) + 1
        payload = dict(self.metadata)
        payload["model"] = self.model.state_dict()
        temporary = self.path.with_name(self.path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, self.path)


class ModelWorker(threading.Thread):
    """Runs every model call in order on one thread, reporting through `events`.

    Jobs (put on `jobs`):
      ("score", generation, [(image path, label path), ...])
      ("train", image path, annotations, [(image path, label path), ...])
    Events (read by the GUI):
      ("ready", text) ("error", text)
      ("score-progress", generation, done, total)
      ("scored", generation, [{"image", "prediction", "score"}, ...])
      ("train-progress", image path, step, score)
      ("trained", image path, steps, score before, score after)
    """

    def __init__(self, args: argparse.Namespace, device: torch.device, events: queue.Queue):
        super().__init__(daemon=True)
        self.args = args
        self.device = device
        self.events = events
        self.jobs: queue.Queue = queue.Queue()

    def run(self) -> None:
        try:
            runner = ModelRunner(self.args.model, self.device, self.args)
        except Exception as exc:  # reported in the GUI rather than lost on stderr
            traceback.print_exc()
            self.events.put(("error", f"Failed to load model: {exc}"))
            return
        self.events.put(("ready", runner.describe()))

        while True:
            job = self.jobs.get()
            try:
                if job[0] == "score":
                    self._score(runner, *job[1:])
                elif job[0] == "train":
                    self._train(runner, *job[1:])
            except Exception as exc:
                traceback.print_exc()
                self.events.put(("error", f"{job[0]} failed: {exc}"))

    def _score(self, runner: ModelRunner, generation: int,
               items: list[tuple[Path, Path]]) -> None:
        results = []
        for done, (image_path, label_path) in enumerate(items, start=1):
            prediction = runner.predict(read_rgb(image_path))
            score = agreement_score(prediction, read_annotations(label_path))
            results.append({"image": image_path, "prediction": prediction, "score": score})
            self.events.put(("score-progress", generation, done, len(items)))
        self.events.put(("scored", generation, results))

    def _train(self, runner: ModelRunner, image_path: Path, annotations: list[dict],
               replay: list[tuple[Path, Path]]) -> None:
        steps, before, after = runner.train(
            read_rgb(image_path), annotations, replay,
            lambda step, score: self.events.put(("train-progress", image_path, step, score)))
        self.events.put(("trained", image_path, steps, before, after))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class InteractiveTrainer:
    """annotation_editor's canvas editing, driven by a model-ranked frame list."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.datasets = available_datasets(str(DATA_ROOT))
        self.current_dataset = args.dataset
        self.current_split = args.split

        self.root = tk.Tk()
        self.root.title("Interactive Trainer")
        self.root.geometry("1600x960")
        self.root.minsize(1100, 700)

        canvas_font = tkfont.nametofont("TkDefaultFont")
        self.annotation_label_font = tkfont.Font(
            family=canvas_font.cget("family"),
            size=canvas_font.cget("size") * ANNOTATION_LABEL_FONT_SCALE,
            weight=canvas_font.cget("weight"),
        )
        self.tool_count_font = tkfont.Font(
            family=canvas_font.cget("family"),
            size=round(canvas_font.cget("size") * TOOL_COUNT_FONT_SCALE),
            weight="bold",
        )
        fixed_font = tkfont.nametofont("TkFixedFont")
        self.list_font = tkfont.Font(
            family=fixed_font.cget("family"),
            size=round(fixed_font.cget("size") * LIST_FONT_SCALE),
            weight=fixed_font.cget("weight"),
        )

        self.dataset_var = tk.StringVar(value=self.current_dataset)
        self.split_var = tk.StringVar(value=self.current_split)
        # Off by default: the labels being corrected need not come from a mask.
        self.show_segmentation_var = tk.BooleanVar(value=False)
        self.show_annotation_var = tk.BooleanVar(value=True)
        self.show_prediction_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.model_status_var = tk.StringVar(value="Model: loading...")
        self.tool_count_var = tk.StringVar(value="0")
        self.images_label_var = tk.StringVar(value="Images (0 / 0)")

        # The split's frames, and the frames a person has already checked
        # (annotation-manual/), which are never drawn for the list.
        self.split_images: list[Path] = []
        self.manual_stems: set[str] = set()
        # The ranked list: {"image": Path, "prediction": list, "score": float}.
        self.entries: list[dict] = []
        self.saved_images: set[Path] = set()
        # Bumped on every dataset/split switch, so scores computed for the
        # previous split are recognised and dropped when they arrive.
        self.generation = 0
        self.pending_trains = 0
        self.model_ready = False

        self.current_index: int | None = None
        self.current_image_path: Path | None = None
        self.current_image_rgb: np.ndarray | None = None
        self.current_segmentation_mask: np.ndarray | None = None
        self.current_annotations: list[dict[str, dict[str, int]]] = []
        self.image_width = 0
        self.image_height = 0

        self.selected_annotation_index: int | None = None
        self.mode = "select"
        self.drag_state: dict[str, object] | None = None
        self.pointer_position: tuple[float, float] | None = None
        self.hovered_handle: tuple[str, int, str | None] | None = None
        self.preview_box: tuple[float, float, float, float] | None = None
        self.dirty = False
        self.photo_image: tk.PhotoImage | None = None
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0.0
        self.canvas_offset_y = 0.0
        self.suppress_listbox_event = False

        self._build_ui()
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue = queue.Queue()
        self.worker = ModelWorker(args, device, self.events)
        self.worker.start()
        self._load_split()
        self.root.after(100, self._poll_events)

    # ── UI ────────────────────────────────────────────────────────────────

    def _configure_button_styles(self) -> None:
        style = ttk.Style()
        for style_name, foreground in BUTTON_STYLES.items():
            style.configure(style_name, foreground=foreground)
            style.map(style_name, foreground=[("disabled", BUTTON_DISABLED_FOREGROUND)])
        style.configure(ADD_ACTIVE_STYLE, foreground=BUTTON_STYLES["Add.TButton"],
                        background=ADD_ACTIVE_BACKGROUND)
        style.map(ADD_ACTIVE_STYLE, foreground=[("disabled", BUTTON_DISABLED_FOREGROUND)],
                  background=[("active", ADD_ACTIVE_HOVER_BACKGROUND)])

    def _build_ui(self) -> None:
        self._configure_button_styles()

        top_bar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_bar, text="Dataset").pack(side=tk.LEFT)
        self.dataset_combobox = ttk.Combobox(
            top_bar, textvariable=self.dataset_var, values=self.datasets,
            state="readonly", width=14)
        self.dataset_combobox.pack(side=tk.LEFT, padx=(8, 18))

        ttk.Label(top_bar, text="Split").pack(side=tk.LEFT)
        self.split_combobox = ttk.Combobox(
            top_bar, textvariable=self.split_var, values=SPLIT_NAMES,
            state="readonly", width=8)
        self.split_combobox.pack(side=tk.LEFT, padx=(8, 18))

        self.add_button = ttk.Button(top_bar, text=ADD_BUTTON_LABEL, style="Add.TButton",
                                     command=self._on_add_button)
        self.add_button.pack(side=tk.LEFT)
        self.delete_button = ttk.Button(top_bar, text=DELETE_BUTTON_LABEL, style="Delete.TButton",
                                        command=self._delete_selected)
        self.delete_button.pack(side=tk.LEFT, padx=(8, 0))
        self.save_button = ttk.Button(top_bar, text=SAVE_BUTTON_LABEL, style="Save.TButton",
                                      command=self._save_current_annotation)
        self.save_button.pack(side=tk.LEFT, padx=(8, 0))
        self.reload_button = ttk.Button(top_bar, text=RELOAD_BUTTON_LABEL, style="Reload.TButton",
                                        command=self._reload_current_annotation)
        self.reload_button.pack(side=tk.LEFT, padx=(8, 0))
        self.update_button = ttk.Button(top_bar, text=UPDATE_BUTTON_LABEL, style="Update.TButton",
                                        command=self._update_from_prediction)
        self.update_button.pack(side=tk.LEFT, padx=(8, 0))

        for text, variable, padding in (
            ("Show Segmentation [F5]", self.show_segmentation_var, (18, 0)),
            ("Show Annotation [F6]", self.show_annotation_var, (12, 0)),
            ("Show Prediction [F7]", self.show_prediction_var, (12, 0)),
        ):
            ttk.Checkbutton(top_bar, text=text, variable=variable,
                            command=self._render_scene).pack(side=tk.LEFT, padx=padding)
        self._build_legend(top_bar).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left_panel = ttk.Frame(body, padding=(0, 0, 10, 0))
        body.add(left_panel, weight=0)
        ttk.Label(left_panel, textvariable=self.images_label_var).pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(left_panel, textvariable=self.model_status_var, wraplength=330,
                  justify=tk.LEFT).pack(side=tk.BOTTOM, anchor=tk.W, pady=(6, 0))

        list_frame = ttk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.image_listbox = tk.Listbox(list_frame, width=46, exportselection=False,
                                        font=self.list_font)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                       command=self.image_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.configure(yscrollcommand=list_scrollbar.set)

        right_panel = ttk.Frame(body)
        body.add(right_panel, weight=1)
        self.canvas = tk.Canvas(right_panel, background="#101417", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        instruction = (
            "Drag bbox to move, handles to resize, red tip to move the tip. "
            "Arrow keys or mouse wheel move between images. Update = Ctrl+U / F4."
        )
        bottom_bar = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bottom_bar, text=instruction).pack(side=tk.LEFT)
        status_frame = ttk.Frame(bottom_bar)
        status_frame.pack(side=tk.RIGHT)
        ttk.Label(status_frame, text="tools").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.tool_count_var, font=self.tool_count_font,
                  foreground=TOOL_COUNT_FOREGROUND).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(status_frame, text="|").pack(side=tk.LEFT, padx=6)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

        self._update_buttons()

    def _build_legend(self, parent: tk.Misc) -> tk.Canvas:
        """Swatches of the label and prediction overlays, drawn with the same
        outline, dash and tip marker the image canvas uses."""
        background = ttk.Style().lookup("TFrame", "background") or parent.winfo_toplevel().cget("background")
        legend = tk.Canvas(parent, width=230, height=26, background=background, highlightthickness=0)
        # Label: solid cyan box with a filled red tip.
        legend.create_rectangle(4, 5, 30, 21, outline="#3bd7ff", width=2)
        legend.create_oval(30 - TIP_RADIUS, 13 - TIP_RADIUS, 30 + TIP_RADIUS, 13 + TIP_RADIUS,
                           outline="#7a0000", fill="#ff4d4d", width=2)
        legend.create_text(42, 13, text="Label", anchor=tk.W)
        # Prediction: dashed magenta box with a hollow diamond tip.
        r = PREDICTION_TIP_RADIUS
        legend.create_rectangle(104, 5, 130, 21, outline=PREDICTION_COLOR, width=2, dash=(8, 4))
        legend.create_polygon(130, 13 - r, 130 + r, 13, 130, 13 + r, 130 - r, 13,
                              outline=PREDICTION_COLOR, fill="", width=2)
        legend.create_text(142, 13, text="Prediction", anchor=tk.W)
        return legend

    def _bind_events(self) -> None:
        self.dataset_combobox.bind("<<ComboboxSelected>>", self._on_dataset_changed)
        self.split_combobox.bind("<<ComboboxSelected>>", self._on_split_changed)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_selected)
        # The Listbox class bindings move the selection on Up/Down by
        # themselves; handling them here and returning "break" keeps it to the
        # one step the toplevel binding below takes.
        self.image_listbox.bind("<Up>", self._on_previous_image_shortcut)
        self.image_listbox.bind("<Down>", self._on_next_image_shortcut)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<Configure>", lambda _event: self._render_scene())
        self.canvas.bind("<ButtonPress-2>", self._on_canvas_wheel_click)
        self.canvas.bind("<MouseWheel>", self._on_canvas_wheel)
        self.canvas.bind("<Button-4>", self._on_previous_image_shortcut)
        self.canvas.bind("<Button-5>", self._on_next_image_shortcut)
        # Bare letter keys are deliberately not bound: a stray "d" must not
        # delete a tool.
        for sequence, handler in (
            ("<Insert>", self._enter_add_mode),
            ("<Delete>", self._delete_selected),
            ("<Control-s>", self._save_current_annotation),
            ("<Control-S>", self._save_current_annotation),
            ("<F2>", self._save_current_annotation),
            ("<F3>", self._reload_current_annotation),
            ("<Control-u>", self._update_from_prediction),
            ("<Control-U>", self._update_from_prediction),
            ("<F4>", self._update_from_prediction),
        ):
            self.root.bind(sequence, lambda _event, handler=handler: (handler(), "break")[1])
        for sequence, variable in (
            ("<F5>", self.show_segmentation_var),
            ("<F6>", self.show_annotation_var),
            ("<F7>", self.show_prediction_var),
        ):
            self.root.bind(sequence, lambda _event, variable=variable: self._toggle_overlay(variable))
        self.root.bind("<Up>", self._on_previous_image_shortcut)
        self.root.bind("<Left>", self._on_previous_image_shortcut)
        self.root.bind("<Down>", self._on_next_image_shortcut)
        self.root.bind("<Right>", self._on_next_image_shortcut)
        self.root.bind("<Escape>", self._on_escape)

    def _toggle_overlay(self, variable: tk.BooleanVar) -> str:
        """Flip a Show ... checkbox from its shortcut, as a click on it would."""
        variable.set(not variable.get())
        self._render_scene()
        return "break"

    # ── Model events ──────────────────────────────────────────────────────

    def _poll_events(self) -> None:
        """Apply what the worker reported since the last poll (Tk is not
        thread-safe, so the worker never touches a widget itself)."""
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "ready":
                    self.model_ready = True
                    self.model_status_var.set(f"Model: {event[1]}")
                elif kind == "error":
                    self.model_status_var.set(f"Model error: {event[1]}")
                    messagebox.showerror("Interactive Trainer", event[1])
                elif kind == "score-progress":
                    if event[1] == self.generation:
                        self.model_status_var.set(f"Scoring {event[2]} / {event[3]} frames...")
                elif kind == "scored":
                    if event[1] == self.generation:
                        self._apply_scores(event[2])
                        self.model_status_var.set(self._idle_status("Scored"))
                elif kind == "train-progress":
                    self.model_status_var.set(
                        f"Training on {event[1].name}: step {event[2]} / "
                        f"{self.args.max_steps}, score {event[3]:.3f} "
                        f"(target {self.args.target_score:.2f})")
                elif kind == "trained":
                    self._on_trained(*event[1:])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _idle_status(self, what: str) -> str:
        return f"{what}. Model idle; {len(self.saved_images)} frames saved this session."

    def _on_trained(self, image_path: Path, steps: int, before: float, after: float) -> None:
        self.pending_trains -= 1
        self.status_var.set(
            f"Trained on {image_path.name}: {steps} steps, score {before:.3f} -> {after:.3f}")
        # Saves made while the model trains queue more trainings; re-rank once,
        # after the last, rather than once per training with a model about to
        # change.
        if self.pending_trains == 0 and self.entries:
            self._request_scores(self.args.new_files)

    def _request_scores(self, new_count: int) -> None:
        """Re-score the listed frames plus `new_count` freshly drawn ones."""
        listed = [entry["image"] for entry in self.entries]
        excluded = set(listed)
        candidates = [path for path in self.split_images
                      if path not in excluded and path.stem not in self.manual_stems]
        drawn = random.sample(candidates, min(new_count, len(candidates)))
        items = [(path, annotation_path(self.current_dataset, self.current_split, path.stem))
                 for path in listed + drawn]
        if not items:
            self.model_status_var.set("No unchecked frames left in this split.")
            return
        self.model_status_var.set(f"Scoring {len(items)} frames...")
        self.worker.jobs.put(("score", self.generation, items))

    def _apply_scores(self, results: list[dict]) -> None:
        """List the frames worst first, at most the lowest `--max-files` of them.

        This arrives in the background while the user works, so the frame on
        screen stays open, and stays listed even when it no longer ranks; only
        the very first list of a split opens its worst frame.
        """
        ranked = sorted(results, key=lambda result: result["score"])
        kept = ranked[:self.args.max_files]
        current = self.current_image_path
        if current is not None and all(result["image"] != current for result in kept):
            kept += [result for result in ranked if result["image"] == current]
            kept.sort(key=lambda result: result["score"])
        self.entries = kept
        # Re-read from disk, so frames corrected meanwhile in another tool
        # (annotation_editor) are highlighted, and no longer drawn, too.
        self.manual_stems = self._list_manual_stems()

        self.suppress_listbox_event = True
        self.image_listbox.delete(0, tk.END)
        for entry in self.entries:
            self.image_listbox.insert(tk.END, self._listbox_label(entry))
        self.suppress_listbox_event = False
        self._style_listbox_rows()

        if current is None:
            if self.entries:
                self._select_listbox_row(0)
                self._open_image_at_index(0)
            return
        self.current_index = next(
            (index for index, entry in enumerate(self.entries) if entry["image"] == current), None)
        if self.current_index is not None:
            self._select_listbox_row(self.current_index)
        self._update_images_label()
        self._update_status()
        self._render_scene()

    def _listbox_label(self, entry: dict) -> str:
        marker = "* " if entry["image"].stem in self.manual_stems else "  "
        return f"{marker}{entry['score']:.3f}  {entry['image'].name}"

    def _style_listbox_rows(self) -> None:
        """Colour every row: sky blue for the frame on screen, yellow for a
        hand-annotated one, default otherwise. The list holds at most
        `--max-files` rows, so repainting all of them is cheap."""
        for index, entry in enumerate(self.entries):
            if entry["image"] == self.current_image_path:
                self.image_listbox.itemconfig(
                    index, background=CURRENT_ITEM_BACKGROUND, foreground=CURRENT_ITEM_FOREGROUND,
                    selectbackground=CURRENT_ITEM_BACKGROUND,
                    selectforeground=CURRENT_ITEM_FOREGROUND)
            else:
                self._apply_listbox_style(index, entry["image"].stem in self.manual_stems)

    def _apply_listbox_style(self, index: int, is_manual: bool) -> None:
        if is_manual:
            self.image_listbox.itemconfig(
                index, background=MANUAL_ITEM_BACKGROUND, foreground=MANUAL_ITEM_FOREGROUND,
                selectbackground=MANUAL_ITEM_SELECT_BACKGROUND,
                selectforeground=MANUAL_ITEM_SELECT_FOREGROUND)
        else:
            self.image_listbox.itemconfig(index, background="", foreground="",
                                          selectbackground="", selectforeground="")

    def _select_listbox_row(self, index: int) -> None:
        self.suppress_listbox_event = True
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(index)
        self.image_listbox.activate(index)
        self.image_listbox.see(index)
        self.suppress_listbox_event = False

    # ── Dataset, split and frame selection ───────────────────────────────

    def _on_close(self) -> None:
        if not self._confirm_leave_current_image():
            return
        if self.pending_trains and not messagebox.askyesno(
            "Training in progress",
            "The model is still training. Quit anyway? The unfinished training "
            "is lost; the checkpoint on disk stays as the last finished one.",
            default=messagebox.NO,
        ):
            return
        self.root.destroy()

    def _on_dataset_changed(self, _event: tk.Event) -> None:
        if self.dataset_var.get() == self.current_dataset:
            return
        if not self._confirm_leave_current_image():
            self.dataset_var.set(self.current_dataset)
            return
        self.current_dataset = self.dataset_var.get()
        self._load_split()

    def _on_split_changed(self, _event: tk.Event) -> None:
        if self.split_var.get() == self.current_split:
            return
        if not self._confirm_leave_current_image():
            self.split_var.set(self.current_split)
            return
        self.current_split = self.split_var.get()
        self._load_split()

    def _load_split(self) -> None:
        """Forget the list and draw a fresh one from the selected split."""
        self.generation += 1
        self.split_images = sorted(images_dir(self.current_dataset, self.current_split).glob("*.png"))
        self.manual_stems = self._list_manual_stems()
        self.entries = []
        self.image_listbox.delete(0, tk.END)
        self.current_index = None
        self.current_image_path = None
        self.current_image_rgb = None
        self.current_segmentation_mask = None
        self.current_annotations = []
        self.selected_annotation_index = None
        self.preview_box = None
        self.drag_state = None
        self.mode = "select"
        self.dirty = False
        self._update_images_label()
        self._update_buttons()
        self._render_scene()
        if not self.split_images:
            self._update_status("No images found in the selected split.")
            return
        self._update_status(f"{len(self.split_images):,} frames, "
                            f"{len(self.manual_stems):,} already checked by hand.")
        self._request_scores(self.args.initial_files)

    def _list_manual_stems(self) -> set[str]:
        manual = manual_dir(self.current_dataset, self.current_split)
        if not manual.is_dir():
            return set()
        return {path.stem for path in manual.glob(f"*{ANNOTATION_SUFFIX}")}

    def _on_image_selected(self, _event: tk.Event) -> None:
        if self.suppress_listbox_event:
            return
        selection = self.image_listbox.curselection()
        if not selection:
            return
        new_index = int(selection[0])
        if new_index == self.current_index:
            return
        if not self._confirm_leave_current_image():
            self._restore_listbox_selection()
            return
        self._open_image_at_index(new_index)

    def _open_image_at_index(self, index: int) -> None:
        self.current_index = index
        self.current_image_path = self.entries[index]["image"]
        self.current_image_rgb = read_rgb(self.current_image_path)
        self.image_height, self.image_width = self.current_image_rgb.shape[:2]
        self.current_segmentation_mask = self._load_segmentation_mask(self.current_image_path)
        self.current_annotations = self._load_annotation(self.current_image_path)
        self.selected_annotation_index = 0 if self.current_annotations else None
        self.preview_box = None
        self.drag_state = None
        self.mode = "select"
        self.dirty = False
        self._style_listbox_rows()
        self._update_images_label()
        self._update_buttons()
        self._update_status()
        self._render_scene()

    def _load_segmentation_mask(self, image_path: Path) -> np.ndarray | None:
        mask_path = segmentation_path(self.current_dataset, self.current_split, image_path.stem)
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _load_annotation(self, image_path: Path) -> list[dict[str, dict[str, int]]]:
        path = annotation_path(self.current_dataset, self.current_split, image_path.stem)
        return [{"bbox": self._normalize_bbox(item["bbox"]), "tip": self._normalize_tip(item["tip"])}
                for item in read_annotations(path)]

    def _current_entry(self) -> dict | None:
        if self.current_image_path is None:
            return None
        return next((entry for entry in self.entries
                     if entry["image"] == self.current_image_path), None)

    # ── Save / Reload / Update ───────────────────────────────────────────

    def _save_current_annotation(self) -> None:
        """Write the label as annotation_editor does, then train on it.

        Save stays available on an unchanged label too: confirming that the
        label is right and the model wrong is as useful a lesson as a fix.

        Nothing here waits for the model: the next frame of the list, in its
        current order, opens at once, and the training and the re-ranked list
        that follows it arrive in the background.
        """
        if self.current_image_path is None:
            return
        annotations = [{"bbox": dict(item["bbox"]), "tip": dict(item["tip"])}
                       for item in self.current_annotations]
        stem = self.current_image_path.stem
        manual = manual_dir(self.current_dataset, self.current_split)
        manual_path = manual / f"{stem}{ANNOTATION_SUFFIX}"
        write_manual_annotation(manual_path, self.current_image_path.name,
                                self.image_width, self.image_height, annotations)
        mirror = annotation_path(self.current_dataset, self.current_split, stem)
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_bytes(manual_path.read_bytes())

        self.saved_images.add(self.current_image_path)
        self.manual_stems.add(stem)
        if self.current_index is not None:
            entry = self.entries[self.current_index]
            self.image_listbox.delete(self.current_index)
            self.image_listbox.insert(self.current_index, self._listbox_label(entry))
            self._select_listbox_row(self.current_index)
        self.dirty = False
        self._update_buttons()

        images = images_dir(self.current_dataset, self.current_split)
        replay = [(images / f"{other}.png", manual / f"{other}{ANNOTATION_SUFFIX}")
                  for other in self.manual_stems if other != stem]
        replay = [pair for pair in replay if pair[0].exists()]
        self.pending_trains += 1
        self.worker.jobs.put(("train", self.current_image_path, annotations, replay))

        saved_name = self.current_image_path.name
        if self.current_index is not None and self.current_index + 1 < len(self.entries):
            self._select_listbox_row(self.current_index + 1)
            self._open_image_at_index(self.current_index + 1)
            ending = ""
        else:
            self._style_listbox_rows()
            ending = " This is the last frame in the list."
        self._update_status(f"Saved {saved_name}; training queued "
                            f"({self.pending_trains} pending).{ending}")

    def _reload_current_annotation(self) -> None:
        if self.current_image_path is None:
            return
        self.current_annotations = self._load_annotation(self.current_image_path)
        self.selected_annotation_index = 0 if self.current_annotations else None
        self.preview_box = None
        self.drag_state = None
        self.mode = "select"
        self.dirty = False
        self._update_buttons()
        self._update_status("Reloaded annotation from file.")
        self._render_scene()

    def _update_from_prediction(self) -> None:
        """Replace the label being edited with the model's prediction (unsaved)."""
        entry = self._current_entry()
        if entry is None:
            return
        self.current_annotations = [
            {"bbox": self._normalize_bbox(item["bbox"]), "tip": self._normalize_tip(item["tip"])}
            for item in entry["prediction"]
        ]
        self.selected_annotation_index = 0 if self.current_annotations else None
        self.mode = "select"
        self.preview_box = None
        self.drag_state = None
        self._mark_dirty(render=False)
        self._update_status("Label replaced with the prediction. Edit, then Save.")
        self._render_scene()

    # ── Shortcuts ────────────────────────────────────────────────────────

    def _on_previous_image_shortcut(self, _event: tk.Event) -> str:
        self._navigate_images(-1)
        return "break"

    def _on_next_image_shortcut(self, _event: tk.Event) -> str:
        self._navigate_images(1)
        return "break"

    def _on_add_button(self) -> None:
        if self.mode == "add":
            self._on_escape(None)
        else:
            self._enter_add_mode()

    def _on_canvas_wheel_click(self, event: tk.Event) -> str:
        if self.mode == "add":
            return self._on_escape(event)
        self._enter_add_mode()
        return "break"

    def _on_canvas_wheel(self, event: tk.Event) -> str:
        if event.delta > 0:
            self._navigate_images(-1)
        elif event.delta < 0:
            self._navigate_images(1)
        return "break"

    def _on_escape(self, _event: tk.Event | None) -> str:
        if self.mode == "add":
            self.mode = "select"
            self.preview_box = None
            self.drag_state = None
            self._update_status()
            self._update_buttons()
            self._render_scene()
        return "break"

    # ── Rendering ────────────────────────────────────────────────────────

    def _render_scene(self) -> None:
        self.canvas.delete("all")
        if self.current_image_rgb is None:
            self.hovered_handle = None
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                text="No image loaded" if self.model_ready or not self.split_images
                else "Loading model...",
                fill="#c7d0d9", font=("TkDefaultFont", 16))
            return

        composed = self._compose_display_image()
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        scale = max(min(canvas_width / self.image_width, canvas_height / self.image_height), 0.01)
        display_width = max(1, int(round(self.image_width * scale)))
        display_height = max(1, int(round(self.image_height * scale)))
        self.canvas_scale = scale
        self.canvas_offset_x = (canvas_width - display_width) / 2.0
        self.canvas_offset_y = (canvas_height - display_height) / 2.0

        resized = cv2.resize(composed, (display_width, display_height), interpolation=cv2.INTER_AREA)
        if self.mode == "add":
            resized = self._dim_for_add_mode(resized)
        entry = self._current_entry()
        if self.show_prediction_var.get() and entry is not None and entry["prediction"]:
            resized = self._blend_predictions(resized, entry["prediction"])
        self.photo_image = self._rgb_to_photoimage(resized)
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y,
                                 anchor=tk.NW, image=self.photo_image)

        self.hovered_handle = self._handle_under_pointer()
        if self.show_annotation_var.get():
            for index, annotation in enumerate(self.current_annotations):
                self._draw_annotation_box(index, annotation)
            for index, annotation in enumerate(self.current_annotations):
                self._draw_annotation_tip(index, annotation)

        if self.preview_box is not None:
            x1, y1, x2, y2 = self.preview_box
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#f8e45c", width=2, dash=(6, 3))

    def _blend_predictions(self, display_rgb: np.ndarray, predictions: list[dict]) -> np.ndarray:
        """Predicted tools -- dashed box, confidence, diamond tip -- drawn into
        the display-sized image at PREDICTION_ALPHA opacity.

        The shapes go onto a copy, and the copy is blended with the original:
        pixels nothing was drawn on are equal in both, so only the shapes fade.
        """
        overlay = display_rgb.copy()
        scale = self.canvas_scale
        color = PREDICTION_COLOR_RGB
        for annotation in predictions:
            bbox, tip = annotation["bbox"], annotation["tip"]
            x1, y1 = round(bbox["x"] * scale), round(bbox["y"] * scale)
            x2 = round((bbox["x"] + bbox["width"]) * scale)
            y2 = round((bbox["y"] + bbox["height"]) * scale)
            tip_x, tip_y = round(tip["x"] * scale), round(tip["y"] * scale)
            for start, end in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                               ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
                draw_dashed_line(overlay, start, end, color, 2, dash=8, gap=4)
            draw_dashed_line(overlay, ((x1 + x2) // 2, (y1 + y2) // 2), (tip_x, tip_y),
                             color, 1, dash=3, gap=3)
            r = PREDICTION_TIP_RADIUS
            diamond = np.array([(tip_x, tip_y - r), (tip_x + r, tip_y),
                                (tip_x, tip_y + r), (tip_x - r, tip_y)], dtype=np.int32)
            cv2.polylines(overlay, [diamond], True, color, 2, cv2.LINE_AA)
            cv2.putText(overlay, f"{annotation['score']:.2f}", (x1 + 4, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return cv2.addWeighted(overlay, PREDICTION_ALPHA, display_rgb, 1.0 - PREDICTION_ALPHA, 0.0)

    def _dim_for_add_mode(self, display_rgb: np.ndarray) -> np.ndarray:
        dimmed = cv2.convertScaleAbs(display_rgb, alpha=ADD_MODE_BRIGHTNESS)
        if self.preview_box is not None:
            x1, y1, x2, y2 = self.preview_box
            left = int(min(x1, x2) * self.canvas_scale)
            top = int(min(y1, y2) * self.canvas_scale)
            right = int(math.ceil(max(x1, x2) * self.canvas_scale))
            bottom = int(math.ceil(max(y1, y2) * self.canvas_scale))
            dimmed[top:bottom, left:right] = display_rgb[top:bottom, left:right]
        return dimmed

    def _compose_display_image(self) -> np.ndarray:
        composed = self.current_image_rgb.copy()
        if self.show_segmentation_var.get() and self.current_segmentation_mask is not None:
            mask = self.current_segmentation_mask > 0
            overlay = composed.copy()
            overlay[mask] = np.array([30, 220, 80], dtype=np.uint8)
            composed = cv2.addWeighted(overlay, SEGMENTATION_ALPHA, composed,
                                       1.0 - SEGMENTATION_ALPHA, 0.0)
        return composed

    def _draw_annotation_box(self, index: int, annotation: dict[str, dict[str, int]]) -> None:
        bbox = annotation["bbox"]
        cx1, cy1 = self._image_to_canvas(bbox["x"], bbox["y"])
        cx2, cy2 = self._image_to_canvas(bbox["x"] + bbox["width"], bbox["y"] + bbox["height"])

        is_selected = index == self.selected_annotation_index
        outline = "#ffd34d" if is_selected else "#3bd7ff"
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=outline,
                                     width=3 if is_selected else 2)
        self.canvas.create_text(cx1 + 6, cy1 + 6, text=str(index + 1), fill=outline,
                                anchor=tk.NW, font=self.annotation_label_font)

        if is_selected:
            mid_x = (cx1 + cx2) / 2
            mid_y = (cy1 + cy2) / 2
            handle_points = {
                "nw": (cx1, cy1), "ne": (cx2, cy1), "sw": (cx1, cy2), "se": (cx2, cy2),
                "n": (mid_x, cy1), "s": (mid_x, cy2), "w": (cx1, mid_y), "e": (cx2, mid_y),
            }
            for handle_name, (handle_x, handle_y) in handle_points.items():
                hovered = self.hovered_handle == ("resize", index, handle_name)
                self.canvas.create_rectangle(
                    handle_x - HANDLE_SIZE, handle_y - HANDLE_SIZE,
                    handle_x + HANDLE_SIZE, handle_y + HANDLE_SIZE,
                    outline="#111111", fill=HANDLE_HOVER_FILL if hovered else "#ffd34d", width=1)

    def _draw_annotation_tip(self, index: int, annotation: dict[str, dict[str, int]]) -> None:
        bbox = annotation["bbox"]
        cx1, cy1 = self._image_to_canvas(bbox["x"], bbox["y"])
        cx2, cy2 = self._image_to_canvas(bbox["x"] + bbox["width"], bbox["y"] + bbox["height"])
        tip_x, tip_y = self._image_to_canvas(annotation["tip"]["x"], annotation["tip"]["y"])
        self.canvas.create_line((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0, tip_x, tip_y,
                                fill="#ff8080", width=2, dash=(3, 2))
        self.canvas.create_oval(
            tip_x - TIP_RADIUS, tip_y - TIP_RADIUS, tip_x + TIP_RADIUS, tip_y + TIP_RADIUS,
            outline="#7a0000",
            fill=HANDLE_HOVER_FILL if self.hovered_handle == ("tip", index, None) else "#ff4d4d",
            width=2)

    def _rgb_to_photoimage(self, rgb: np.ndarray) -> tk.PhotoImage:
        height, width = rgb.shape[:2]
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return (self.canvas_offset_x + x * self.canvas_scale,
                self.canvas_offset_y + y * self.canvas_scale)

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.canvas_offset_x) / self.canvas_scale,
                (y - self.canvas_offset_y) / self.canvas_scale)

    def _point_inside_image(self, x: float, y: float) -> bool:
        return 0 <= x <= self.image_width and 0 <= y <= self.image_height

    # ── Editing (as in annotation_editor) ────────────────────────────────

    def _enter_add_mode(self) -> None:
        if self.current_image_path is None:
            return
        self.mode = "add"
        self.preview_box = None
        self.drag_state = None
        self._update_buttons()
        self._update_status("Add mode: drag on the image to create a new bbox.")
        self._render_scene()

    def _delete_selected(self) -> None:
        if self.selected_annotation_index is None:
            return
        del self.current_annotations[self.selected_annotation_index]
        if self.current_annotations:
            self.selected_annotation_index = min(self.selected_annotation_index,
                                                 len(self.current_annotations) - 1)
        else:
            self.selected_annotation_index = None
        self._mark_dirty()

    def _navigate_images(self, step: int) -> None:
        if self.current_index is None or not self.entries:
            return
        new_index = self.current_index + step
        if new_index < 0 or new_index >= len(self.entries):
            return
        if not self._confirm_leave_current_image():
            self._restore_listbox_selection()
            return
        self._select_listbox_row(new_index)
        self._open_image_at_index(new_index)

    def _on_canvas_press(self, event: tk.Event) -> None:
        if self.current_image_rgb is None:
            return

        image_x, image_y = self._canvas_to_image(event.x, event.y)
        if self.mode == "add":
            if not self._point_inside_image(image_x, image_y):
                return
            self.drag_state = {"kind": "add",
                               "start": (self._clamp_x(image_x), self._clamp_y(image_y))}
            self.preview_box = (image_x, image_y, image_x, image_y)
            self._render_scene()
            return

        hit = self._hit_test(image_x, image_y) if self.show_annotation_var.get() else None
        if hit is None:
            self.selected_annotation_index = None
            self.drag_state = None
            self._update_buttons()
            self._render_scene()
            return

        self.selected_annotation_index = hit["index"]
        annotation = self.current_annotations[self.selected_annotation_index]
        self.drag_state = {
            "kind": hit["kind"],
            "index": hit["index"],
            "start_point": (image_x, image_y),
            "start_bbox": annotation["bbox"].copy(),
            "start_tip": annotation["tip"].copy(),
            "handle": hit.get("handle"),
        }
        self._update_buttons()
        self._render_scene()

    def _on_canvas_motion(self, event: tk.Event) -> None:
        """Show a crosshair and highlight the tip or resize handle under the pointer."""
        self.pointer_position = (event.x, event.y)
        handle = self._handle_under_pointer()
        cursor = HANDLE_CURSOR if handle is not None else ""
        if self.canvas.cget("cursor") != cursor:
            self.canvas.configure(cursor=cursor)
        if handle != self.hovered_handle:
            self._render_scene()

    def _on_canvas_leave(self, _event: tk.Event) -> None:
        self.pointer_position = None
        if self.hovered_handle is not None:
            self._render_scene()

    def _handle_under_pointer(self) -> tuple[str, int, str | None] | None:
        if self.drag_state is not None and self.drag_state["kind"] in ("tip", "resize"):
            return (
                str(self.drag_state["kind"]),
                int(self.drag_state["index"]),
                self.drag_state["handle"] if self.drag_state["kind"] == "resize" else None,
            )
        if (
            self.pointer_position is None
            or self.current_image_rgb is None
            or self.mode != "select"
            or not self.show_annotation_var.get()
        ):
            return None
        hit = self._hit_test(*self._canvas_to_image(*self.pointer_position))
        if hit is None or hit["kind"] not in ("tip", "resize"):
            return None
        return (hit["kind"], hit["index"], hit.get("handle"))

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.current_image_rgb is None or self.drag_state is None:
            return

        image_x, image_y = self._canvas_to_image(event.x, event.y)
        image_x = self._clamp_x(image_x)
        image_y = self._clamp_y(image_y)
        kind = self.drag_state["kind"]

        if kind == "add":
            start_x, start_y = self.drag_state["start"]
            self.preview_box = (start_x, start_y, image_x, image_y)
            self._render_scene()
            return

        annotation = self.current_annotations[int(self.drag_state["index"])]
        if kind == "tip":
            tip = {"x": int(round(image_x)), "y": int(round(image_y))}
            annotation["tip"] = tip
            # Grow the box from its drag-start extent so it always contains the
            # tip pixel; dragging the tip back inside restores the original box.
            start_bbox = self.drag_state["start_bbox"]
            annotation["bbox"] = self._normalize_box_from_points(
                min(start_bbox["x"], tip["x"]),
                min(start_bbox["y"], tip["y"]),
                max(start_bbox["x"] + start_bbox["width"], tip["x"] + 1),
                max(start_bbox["y"] + start_bbox["height"], tip["y"] + 1),
            )
        elif kind == "move":
            start_x, start_y = self.drag_state["start_point"]
            start_bbox = self.drag_state["start_bbox"]
            start_tip = self.drag_state["start_tip"]
            new_x = self._clamp_int(start_bbox["x"] + int(round(image_x - start_x)), 0,
                                    self.image_width - start_bbox["width"])
            new_y = self._clamp_int(start_bbox["y"] + int(round(image_y - start_y)), 0,
                                    self.image_height - start_bbox["height"])
            annotation["bbox"] = {"x": new_x, "y": new_y,
                                  "width": start_bbox["width"], "height": start_bbox["height"]}
            annotation["tip"] = self._normalize_tip({
                "x": start_tip["x"] + new_x - start_bbox["x"],
                "y": start_tip["y"] + new_y - start_bbox["y"],
            })
        elif kind == "resize":
            start_bbox = self.drag_state["start_bbox"]
            x1 = start_bbox["x"]
            y1 = start_bbox["y"]
            x2 = start_bbox["x"] + start_bbox["width"]
            y2 = start_bbox["y"] + start_bbox["height"]
            handle = self.drag_state["handle"]
            # Corner handles move both coordinates; edge handles only their own side.
            if "n" in handle:
                y1 = image_y
            if "s" in handle:
                y2 = image_y
            if "w" in handle:
                x1 = image_x
            if "e" in handle:
                x2 = image_x
            normalized = self._normalize_box_from_points(x1, y1, x2, y2)
            annotation["bbox"] = normalized
            # Pull the tip from its drag-start position onto the nearest pixel
            # inside the box, so shrinking never leaves it outside.
            start_tip = self.drag_state["start_tip"]
            annotation["tip"] = {
                "x": self._clamp_int(start_tip["x"], normalized["x"],
                                     normalized["x"] + normalized["width"] - 1),
                "y": self._clamp_int(start_tip["y"], normalized["y"],
                                     normalized["y"] + normalized["height"] - 1),
            }
        self._mark_dirty(render=False)
        self._render_scene()

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self.drag_state is None:
            return

        if self.drag_state["kind"] == "add":
            image_x = self._clamp_x(self._canvas_to_image(event.x, event.y)[0])
            image_y = self._clamp_y(self._canvas_to_image(event.x, event.y)[1])
            start_x, start_y = self.drag_state["start"]
            bbox = self._normalize_box_from_points(start_x, start_y, image_x, image_y)
            self.preview_box = None
            if bbox["width"] >= MIN_BOX_SIZE and bbox["height"] >= MIN_BOX_SIZE:
                self.current_annotations.append({
                    "bbox": bbox,
                    "tip": self._normalize_tip({"x": bbox["x"] + bbox["width"] // 2,
                                                "y": bbox["y"] + bbox["height"] // 2}),
                })
                self.selected_annotation_index = len(self.current_annotations) - 1
                self._mark_dirty(render=False)
            self.mode = "select"
            self._update_buttons()
            self._update_status()
            self._render_scene()

        self.drag_state = None

    def _hit_test(self, image_x: float, image_y: float) -> dict[str, object] | None:
        """Resolve a click to a tip handle, a resize handle, or a bbox.

        Handles are tested before box interiors, and the smallest box wins among
        the ones under the cursor, so a tool nested inside a larger one stays
        reachable instead of being swallowed by it.
        """
        threshold = max(8.0 / self.canvas_scale, 3.0)
        handle_threshold = max(12.0 / self.canvas_scale, 3.0)
        tip_threshold_sq = (threshold * 1.5) ** 2

        tip_hit: tuple[tuple[float, bool], int] | None = None
        for index, annotation in enumerate(self.current_annotations):
            tip = annotation["tip"]
            distance_sq = (tip["x"] - image_x) ** 2 + (tip["y"] - image_y) ** 2
            if distance_sq > tip_threshold_sq:
                continue
            key = (distance_sq, index != self.selected_annotation_index)
            if tip_hit is None or key < tip_hit[0]:
                tip_hit = (key, index)
        if tip_hit is not None:
            return {"index": tip_hit[1], "kind": "tip"}

        if self.selected_annotation_index is not None:
            bbox = self.current_annotations[self.selected_annotation_index]["bbox"]
            x1, y1 = bbox["x"], bbox["y"]
            x2, y2 = x1 + bbox["width"], y1 + bbox["height"]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            handles = {
                "nw": (x1, y1), "ne": (x2, y1), "sw": (x1, y2), "se": (x2, y2),
                "n": (mid_x, y1), "s": (mid_x, y2), "w": (x1, mid_y), "e": (x2, mid_y),
            }
            for handle_name, (handle_x, handle_y) in handles.items():
                if abs(handle_x - image_x) <= handle_threshold and abs(handle_y - image_y) <= handle_threshold:
                    return {"index": self.selected_annotation_index, "kind": "resize",
                            "handle": handle_name}

        box_hit: tuple[tuple[int, bool], int] | None = None
        for index, annotation in enumerate(self.current_annotations):
            bbox = annotation["bbox"]
            x1, y1 = bbox["x"], bbox["y"]
            x2, y2 = x1 + bbox["width"], y1 + bbox["height"]
            if not (x1 <= image_x <= x2 and y1 <= image_y <= y2):
                continue
            key = (bbox["width"] * bbox["height"], index != self.selected_annotation_index)
            if box_hit is None or key < box_hit[0]:
                box_hit = (key, index)
        if box_hit is not None:
            return {"index": box_hit[1], "kind": "move"}
        return None

    def _mark_dirty(self, render: bool = True) -> None:
        self.dirty = True
        self._update_buttons()
        self._update_status()
        if render:
            self._render_scene()

    def _update_buttons(self) -> None:
        has_image = self.current_image_path is not None
        self.save_button.configure(state=tk.NORMAL if has_image else tk.DISABLED)
        self.reload_button.configure(state=tk.NORMAL if has_image else tk.DISABLED)
        self.update_button.configure(state=tk.NORMAL if has_image else tk.DISABLED)
        self.delete_button.configure(
            state=tk.NORMAL if has_image and self.selected_annotation_index is not None
            else tk.DISABLED)
        adding = self.mode == "add"
        self.add_button.configure(
            text=ADD_BUTTON_ACTIVE_LABEL if adding else ADD_BUTTON_LABEL,
            style=ADD_ACTIVE_STYLE if adding else "Add.TButton",
            state=tk.NORMAL if has_image else tk.DISABLED,
        )

    def _update_images_label(self) -> None:
        total = len(self.entries)
        position = self.current_index + 1 if self.current_index is not None and total else 0
        self.images_label_var.set(f"Images ({position} / {total}), lowest score first")

    def _update_status(self, message: str | None = None) -> None:
        self.tool_count_var.set(str(len(self.current_annotations)))
        if message is not None:
            self.status_var.set(message)
            return
        entry = self._current_entry()
        score = f"{entry['score']:.3f}" if entry is not None else "-"
        edited = ", edited" if self.dirty else ""
        self.status_var.set(f"mode={self.mode}, score={score}{edited}")

    def _confirm_leave_current_image(self) -> bool:
        if not self.dirty:
            return True
        response = messagebox.askyesnocancel(
            "Unsaved changes",
            "Save changes (and train on them) before leaving this image?",
            default=messagebox.YES,
        )
        if response is None:
            return False
        if response:
            self._save_current_annotation()
            return not self.dirty
        return True

    def _restore_listbox_selection(self) -> None:
        if self.current_index is not None:
            self._select_listbox_row(self.current_index)

    def _normalize_bbox(self, bbox: dict[str, int]) -> dict[str, int]:
        x = self._clamp_int(int(bbox["x"]), 0, max(self.image_width - 1, 0))
        y = self._clamp_int(int(bbox["y"]), 0, max(self.image_height - 1, 0))
        width = min(max(int(bbox["width"]), 1), max(self.image_width - x, 1))
        height = min(max(int(bbox["height"]), 1), max(self.image_height - y, 1))
        return {"x": x, "y": y, "width": width, "height": height}

    def _normalize_box_from_points(self, x1: float, y1: float, x2: float, y2: float) -> dict[str, int]:
        left = self._clamp_int(int(round(min(x1, x2))), 0, max(self.image_width - 1, 0))
        top = self._clamp_int(int(round(min(y1, y2))), 0, max(self.image_height - 1, 0))
        right = self._clamp_int(int(round(max(x1, x2))), 0, max(self.image_width, 1))
        bottom = self._clamp_int(int(round(max(y1, y2))), 0, max(self.image_height, 1))
        return self._normalize_bbox({"x": left, "y": top,
                                     "width": max(right - left, 1), "height": max(bottom - top, 1)})

    def _normalize_tip(self, tip: dict[str, int]) -> dict[str, int]:
        return {
            "x": self._clamp_int(int(tip["x"]), 0, max(self.image_width - 1, 0)),
            "y": self._clamp_int(int(tip["y"]), 0, max(self.image_height - 1, 0)),
        }

    def _clamp_int(self, value: int, low: int, high: int) -> int:
        if high < low:
            return low
        return max(low, min(value, high))

    def _clamp_x(self, value: float) -> float:
        return max(0.0, min(float(value), max(float(self.image_width - 1), 0.0)))

    def _clamp_y(self, value: float) -> float:
        return max(0.0, min(float(value), max(float(self.image_height - 1), 0.0)))

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct labels and fine-tune the detector together, worst frames first.")
    parser.add_argument("--dataset", default=None,
                        help="Initial dataset (default: the first under data/dataset). "
                             "Switchable in the GUI.")
    parser.add_argument("--split", default="train", choices=SPLIT_NAMES,
                        help="Initial split. Switchable in the GUI.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="yolov8sclone `tooltip` checkpoint. It is overwritten "
                             "after every training round.")
    parser.add_argument("--device", default="",
                        help="Torch device, e.g. cuda:1 (default: cuda if available, else cpu).")
    parser.add_argument("--initial-files", type=int, default=DEFAULT_INITIAL_FILES,
                        help="Frames drawn when a split is opened.")
    parser.add_argument("--new-files", type=int, default=DEFAULT_NEW_FILES,
                        help="Frames newly drawn and added to the list after each training.")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                        help="Most frames the list holds; past it the lowest-scoring are kept.")
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE,
                        help="Stop training once the saved frame scores this.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Stop training after this many steps regardless.")
    parser.add_argument("--replay", type=int, default=DEFAULT_REPLAY,
                        help="Hand-checked frames added to every training step.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="AdamW learning rate.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    datasets = available_datasets(str(DATA_ROOT))
    if not datasets:
        raise SystemExit(f"no datasets under {DATA_ROOT}")
    if args.dataset is None:
        args.dataset = datasets[0]
    elif args.dataset not in datasets:
        raise SystemExit(f"unknown dataset {args.dataset!r}; available: {', '.join(datasets)}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    InteractiveTrainer(args, device).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
