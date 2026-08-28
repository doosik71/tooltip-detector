#!/usr/bin/env python3
"""Build the Ultralytics YOLO dataset this baseline trains on.

Ultralytics reads images and labels from its own directory layout, and this
repository's `data/dataset/` is a read-only mount holding JSON annotations, so
one conversion step stands between the two. It runs once per dataset:

    data/yolo/<dataset>/
        dataset.yaml         what train-model points Ultralytics at
        prepare-status.json  tip box size, frame counts, how the lists were cut
        images/<split>       symlink to <repo>/data/dataset/<dataset>/images/<split>
        labels/<split>/*.txt one YOLO label file per listed frame
        <split>.txt          the frames of that split this dataset uses

Two classes come out of one annotation file (see common/dataset.py): `tool`,
the annotated bounding box, and `tip`, a square box of --tip-box-size px
centred on the annotated tool tip. Both are learned by one detector, so the
centre of a predicted `tip` box is the tip coordinate the root project's
metrics are defined on.

Consecutive video frames are near-duplicates, so --frame-stride keeps every Nth
training frame; --val-frames caps the per-epoch validation set the same way.
Both are recorded, and only the frames actually listed get a label file -- the
subsampling cuts the preparation cost too.

Re-running with the same arguments is a no-op. Anything else (a different tip
box size, a different stride) rewrites the split from scratch, because the
prepared dataset must always match what prepare-status.json claims it is.

Usage:
    ./baseline/yolo26/run prepare-dataset --dataset cholec80
    ./baseline/yolo26/run prepare-dataset --dataset erop --frame-stride 1 --splits train val test
"""

import argparse
import json
import os
import shutil
import time

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import (CLASS_NAMES, DEFAULT_TIP_BOX_SIZE, SPLITS, TIP_CLASS,
                                TOOL_CLASS, available_datasets, dataset_yaml_path,
                                default_data_root, frame_paths, images_dir, label_text,
                                load_annotation, prepare_status_path, read_prepare_status,
                                yolo_dir)
    from common.progress import progress

DEFAULT_FRAME_STRIDE = 5
DEFAULT_VAL_FRAMES = 2000

def link_images(prepared_root: str, dataset: str, split: str, data_root: str) -> str:
    """Point data/yolo/<dataset>/images/<split> at the real frame directory.

    A symlink, not a copy: the frames are 200k PNGs and they never change.
    Ultralytics derives a label path by swapping `/images/` for `/labels/` in
    the image path, so the link has to sit under the prepared tree even though
    the frames themselves do not.
    """
    source = images_dir(dataset, split, data_root)
    if not os.path.isdir(source):
        raise SystemExit(f"no such split directory: {source}")

    link = os.path.join(prepared_root, "images", split)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(source):
            return link
        os.unlink(link)
    elif os.path.isdir(link):
        raise SystemExit(f"{link} exists and is a real directory, not a link; "
                         "remove it and re-run")
    os.symlink(os.path.realpath(source), link)
    return link


def write_split(prepared_root: str, dataset: str, split: str, data_root: str,
                paths: list[str], tip_box_size: float) -> dict:
    """Write one split's label files and its image list; return the counts."""
    link = link_images(prepared_root, dataset, split, data_root)
    labels_root = os.path.join(prepared_root, "labels", split)
    if os.path.isdir(labels_root):
        shutil.rmtree(labels_root)
    os.makedirs(labels_root, exist_ok=True)

    # Ultralytics caches the scanned labels next to them; a stale cache would
    # describe the labels this run has just replaced.
    cache = os.path.join(prepared_root, "labels", split + ".cache")
    if os.path.exists(cache):
        os.remove(cache)

    annotation_root = os.path.join(data_root, dataset, "annotation", split)
    listed, n_tool, n_tip, n_empty = [], 0, 0, 0

    for path in progress(paths, desc=f"  {split}"):
        stem = os.path.splitext(os.path.basename(path))[0]
        # No frame size is passed: each annotation records the size it was
        # made against, and the PNGs are never opened here.
        labels = load_annotation(os.path.join(annotation_root, stem + ".json"),
                                 tip_box_size=tip_box_size)
        n_tool += int((labels[:, 0] == TOOL_CLASS).sum())
        n_tip += int((labels[:, 0] == TIP_CLASS).sum())
        n_empty += int(not len(labels))
        with open(os.path.join(labels_root, stem + ".txt"), "w", encoding="utf-8") as handle:
            handle.write(label_text(labels))
        listed.append(os.path.join(link, os.path.basename(path)))

    list_path = os.path.join(prepared_root, split + ".txt")
    with open(list_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(listed) + "\n")

    return {"frames": len(listed), "tool_boxes": n_tool, "tip_boxes": n_tip,
            "frames_without_tools": n_empty, "list": os.path.basename(list_path)}


def write_yaml(prepared_root: str, splits: dict) -> str:
    """The dataset.yaml Ultralytics is pointed at.

    Written by hand rather than through pyyaml: it is four keys and a class
    list, and keeping it plain text makes it readable in a diff.
    """
    path = os.path.join(prepared_root, "dataset.yaml")
    lines = [
        "# Generated by baseline/yolo26/scripts/prepare-dataset.py -- do not edit.",
        "# Re-run that script to change anything here.",
        f"path: {prepared_root}",
    ]
    for split in ("train", "val", "test"):
        if split in splits:
            lines.append(f"{split}: {splits[split]['list']}")
    lines += ["names:"]
    lines += [f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def up_to_date(status: dict | None, request: dict, prepared_root: str) -> bool:
    """True when a previous run already produced exactly what was asked for."""
    if status is None:
        return False
    for key in ("tip_box_size", "frame_stride", "val_frames", "data_root"):
        if status.get(key) != request[key]:
            return False
    if sorted(status.get("splits", {})) != sorted(request["splits"]):
        return False
    for split, summary in status.get("splits", {}).items():
        labels_root = os.path.join(prepared_root, "labels", split)
        if not os.path.isdir(labels_root):
            return False
        if len(os.listdir(labels_root)) != summary.get("frames"):
            return False
    return os.path.exists(os.path.join(prepared_root, "dataset.yaml"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a tooltip-detector dataset into an Ultralytics YOLO dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None,
                        help="dataset directory under data/dataset (e.g. cholec80)")
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=SPLITS,
                        help="splits to prepare (default: train val; evaluation reads the "
                             "source annotations directly and needs no prepared test split)")
    parser.add_argument("--tip-box-size", type=float, default=DEFAULT_TIP_BOX_SIZE,
                        help="side of the square box drawn around each annotated tip, in "
                             f"original-frame px (default: {DEFAULT_TIP_BOX_SIZE:g})")
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE,
                        help="keep every Nth train frame; consecutive video frames are "
                             f"near-duplicates (default: {DEFAULT_FRAME_STRIDE})")
    parser.add_argument("--val-frames", type=int, default=DEFAULT_VAL_FRAMES,
                        help="cap on the validation frames Ultralytics scores each epoch, "
                             f"spread evenly over the split (0 = all, default: {DEFAULT_VAL_FRAMES})")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when the prepared dataset already matches")
    args = parser.parse_args()

    prepared_root = yolo_dir(args.dataset)
    request = {"tip_box_size": args.tip_box_size, "frame_stride": args.frame_stride,
               "val_frames": args.val_frames, "data_root": os.path.abspath(args.data_root),
               "splits": list(args.splits)}
    previous = read_prepare_status(args.dataset)

    if not args.force and up_to_date(previous, request, prepared_root):
        print(f"already prepared: {prepared_root}")
        for split, summary in previous["splits"].items():
            print(f"  {split:5s} {summary['frames']:,} frames  "
                  f"{summary['tool_boxes']:,} tool  {summary['tip_boxes']:,} tip")
        print("re-run with --force to rebuild it anyway")
        return

    os.makedirs(prepared_root, exist_ok=True)
    print(f"dataset : {args.dataset}  ->  {prepared_root}")
    print(f"labels  : tool = annotated bbox, tip = {args.tip_box_size:g} px box on the tip")

    started = time.perf_counter()
    summaries = {}
    for split in args.splits:
        # The stride thins training frames; validation is capped instead, so a
        # long split still gets scored on frames spread across every session.
        if split == "train":
            paths = frame_paths(args.dataset, split, args.data_root,
                                frame_stride=args.frame_stride)
        else:
            paths = frame_paths(args.dataset, split, args.data_root)
            if args.val_frames and len(paths) > args.val_frames:
                stride = -(-len(paths) // args.val_frames)      # ceil
                paths = paths[::stride][:args.val_frames]
        summaries[split] = write_split(prepared_root, args.dataset, split,
                                       args.data_root, paths, args.tip_box_size)

    yaml_path = write_yaml(prepared_root, summaries)
    status = {**request, "dataset": args.dataset, "class_names": list(CLASS_NAMES),
              "splits": summaries, "yaml": yaml_path,
              "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(prepare_status_path(args.dataset), "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)

    print()
    for split, summary in summaries.items():
        print(f"  {split:5s} {summary['frames']:,} frames  "
              f"{summary['tool_boxes']:,} tool boxes  {summary['tip_boxes']:,} tip boxes  "
              f"({summary['frames_without_tools']:,} with no tool)")
    print(f"written {dataset_yaml_path(args.dataset)}  ({time.perf_counter() - started:.1f} s)")
    print(f"next    ./baseline/yolo26/run train-model --dataset {args.dataset}")


if __name__ == "__main__":
    main()
