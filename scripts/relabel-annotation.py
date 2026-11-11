"""Relabel a dataset's automatic annotations with a tool-tip detector.

Every frame in data/dataset/<dataset>/annotation/<split>/ that is not a
manual label is overwritten with the detector's prediction, and its `source`
becomes the checkpoint path relative to the repository root, e.g.

    "source": "standalone/yolov8stooltip/data/model/erop/model.pt"

so a label always says which model produced it. Frames whose annotation says
`"source": "manual"`, or that have a file in annotation-manual/, are left
alone.

The model is loaded and run exactly as interactive-trainer does it
(`ModelRunner`, same confidence and NMS thresholds), so either family works:
yolov8stooltip (the default) gives each tool its own tip, and a yolov8sclone
`tooltip` checkpoint has its tool and tip boxes paired by `pair_detections`.
The labels are written with tooltip-annotator's payload helpers so the JSON
layout is unchanged.

Frames already carrying this checkpoint's `source` are skipped, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tooltip-annotator"))

from tooltip.annotation import SOURCE_MANUAL, build_payload, write_annotation  # noqa: E402


def load_interactive_trainer():
    """interactive-trainer.py as a module, for its model loading and prediction."""
    path = REPO_ROOT / "scripts" / "interactive-trainer.py"
    spec = importlib.util.spec_from_file_location("interactive_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


itrainer = load_interactive_trainer()

SPLIT_NAMES = itrainer.SPLIT_NAMES
DATA_ROOT = itrainer.DATA_ROOT


def default_model(dataset: str) -> Path:
    return REPO_ROOT / "standalone" / "yolov8stooltip" / "data" / "model" / dataset / "model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset under data/dataset/ (erop / cholec80)")
    parser.add_argument("--model", type=Path, default=None,
                        help="yolov8stooltip or yolov8sclone `tooltip` checkpoint; defaults "
                             "to standalone/yolov8stooltip/data/model/<dataset>/model.pt")
    parser.add_argument("--split", choices=SPLIT_NAMES, nargs="+", default=list(SPLIT_NAMES))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--conf", type=float, default=itrainer.DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=itrainer.DEFAULT_IOU)
    parser.add_argument("--workers", type=int, default=8, help="threads reading images")
    return parser.parse_args()


def read_source(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("source", "auto")
    except (OSError, ValueError):
        return None


def relabel_split(runner, dataset: str, split: str, source: str, args) -> dict[str, int]:
    image_dir = DATA_ROOT / dataset / "images" / split
    annotation_dir = DATA_ROOT / dataset / "annotation" / split
    manual_dir = DATA_ROOT / dataset / "annotation-manual" / split

    counts = {"relabelled": 0, "manual": 0, "done": 0, "unreadable": 0}
    todo = []
    for image_path in sorted(image_dir.glob("*.png")):
        stem = image_path.stem
        current = read_source(annotation_dir / f"{stem}.json")
        if current == SOURCE_MANUAL or (manual_dir / f"{stem}.json").exists():
            counts["manual"] += 1
        elif current == source:
            counts["done"] += 1
        else:
            todo.append(image_path)

    def read_ahead(pool):
        # Executor.map would read every image up front; keep only a few in flight.
        pending = deque()
        for image_path in todo:
            pending.append((image_path, pool.submit(cv2.imread, str(image_path), cv2.IMREAD_COLOR)))
            if len(pending) >= 4 * args.workers:
                path, future = pending.popleft()
                yield path, future.result()
        while pending:
            path, future = pending.popleft()
            yield path, future.result()

    with ThreadPoolExecutor(args.workers) as pool:
        for image_path, image_bgr in tqdm(read_ahead(pool), total=len(todo),
                                          desc=f"{dataset}/{split}",
                                          unit="image", ascii=True, ncols=100):
            # A person may save this frame (interactive-trainer, annotation_editor)
            # after the list above was built; their label must win.
            if (manual_dir / f"{image_path.stem}.json").exists():
                counts["manual"] += 1
                continue
            if image_bgr is None:
                tqdm.write(f"unreadable image, label kept: {image_path}")
                counts["unreadable"] += 1
                continue
            height, width = image_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # Drop the detection score that predict() adds to each tool.
            annotations = [{"bbox": item["bbox"], "tip": item["tip"]}
                           for item in runner.predict(frame_rgb)]
            payload = build_payload(image_path.name, width, height, annotations, source)
            write_annotation(annotation_dir / f"{image_path.stem}.json", payload)
            counts["relabelled"] += 1
    return counts


def main() -> int:
    args = parse_args()
    model = (args.model or default_model(args.dataset)).resolve()
    source = model.relative_to(REPO_ROOT).as_posix()
    args.lr = itrainer.DEFAULT_LR  # ModelRunner builds an optimizer; it is never stepped here
    runner = itrainer.ModelRunner.open(model, torch.device(args.device), args)
    print(f"source: {source} ({runner.describe()})")

    for split in args.split:
        counts = relabel_split(runner, args.dataset, split, source, args)
        print(f"{split}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
