#!/usr/bin/env python3
"""Train YOLO26 on this repository's surgical-tool annotations.

The model, the recipe and the training loop are all Ultralytics' -- this
script is the wrapper that points them at the dataset
`scripts/prepare-dataset.py` produced, and mirrors what comes out into the same
file layout the other baselines use, so one summary script can read all of them:

    data/model/<dataset>/
        model.pt            best checkpoint by Ultralytics' fitness
        model-last.pt       last epoch, resumable
        model-info.json     tip box size, image size, scale, dataset, epoch
        train-status.json   epochs completed, best fitness, every run argument
        metric.csv          one row per epoch (losses, mAP, per-class AP, lr, time)
        ultralytics/        Ultralytics' own run directory, untouched
                            (weights/, args.yaml, results.csv, plots)

Two classes are learned from one annotation file: `tool` (the annotated
bounding box) and `tip` (a square box centred on the annotated tool tip), so
one detector produces both the instrument box and the tip coordinate the root
project's metrics are defined on. The tip box size is fixed when the dataset is
prepared, not here, and is copied into model-info.json.

What is YOLO26-specific and not a choice made here: the head is end-to-end, so
there is no NMS at inference and no IoU threshold to tune; DFL is gone, so the
third loss term is an L1 on the box rather than a distribution loss; and
`--optimizer auto` resolves to MuSGD on a run this long, which is YOLO26's own
optimizer.

Starting point: `yolo26<scale>.pt`, the COCO-pretrained weights, downloaded
once into data/pretrained/. `--no-pretrained` starts from random initialisation
instead, which is what the CLAD-Net and YOLOv8s-clone baselines do -- use it
when the comparison against those two has to be like-for-like.

Usage:
    ./baseline/yolo26/run train-model --dataset cholec80
    ./baseline/yolo26/run train-model --dataset erop --device cuda:1 --no-pretrained
"""

import argparse
import csv
import json
import os
import pathlib
import shutil

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import (CLASS_NAMES, available_datasets, data_dir,
                                dataset_yaml_path, model_dir, require_prepared)
    from common.inference import (LAST_FILENAME, MODEL_FILENAME, write_model_info)

SCALES = ("n", "s", "m", "l", "x")
DEFAULT_SCALE = "s"
RUN_DIRNAME = "ultralytics"

# Ultralytics names its metrics with a task suffix; these are the detection
# ones this script mirrors into metric.csv.
MAP50_KEY = "metrics/mAP50(B)"
MAP50_95_KEY = "metrics/mAP50-95(B)"
PRECISION_KEY = "metrics/precision(B)"
RECALL_KEY = "metrics/recall(B)"


def use_local_weights_dir() -> str:
    """Keep Ultralytics' own downloads inside this sub-project.

    Ultralytics' AMP check downloads yolo26n.pt into the global WEIGHTS_DIR
    from its settings file, which points at whatever directory it was first
    run from -- for this machine, the repository root. Rebinding the module
    attribute redirects that one download next to the weights this baseline
    downloads itself, and affects only this process: the user's settings file
    is left alone.
    """
    import ultralytics.utils

    target = os.path.join(data_dir(), "pretrained")
    os.makedirs(target, exist_ok=True)
    ultralytics.utils.WEIGHTS_DIR = pathlib.Path(target)
    return target


def pretrained_weights(scale: str) -> str:
    """COCO weights for one scale, downloaded once into data/pretrained/."""
    from ultralytics.utils.downloads import attempt_download_asset

    target = os.path.join(data_dir(), "pretrained", f"yolo26{scale}.pt")
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"downloading yolo26{scale}.pt -> {target}")
    return attempt_download_asset(target)


def per_class_ap50(trainer) -> dict[str, float]:
    """AP@0.5 per class name, read off the validator that just ran.

    Ultralytics only reports classes it saw ground truth for, so a class with
    no instances in the validation split is simply absent here rather than
    reported as zero.
    """
    metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    if metrics is None or not len(getattr(metrics, "ap_class_index", [])):
        return {}
    names = trainer.data.get("names", dict(enumerate(CLASS_NAMES)))
    result = {}
    for i, class_index in enumerate(metrics.ap_class_index):
        try:
            _, _, ap50, _ = metrics.class_result(i)
        except (TypeError, ValueError):
            continue
        result[str(names.get(int(class_index), class_index))] = round(float(ap50), 4)
    return result


def metric_row(trainer) -> dict:
    """One epoch of Ultralytics' state, in this project's column names.

    The loss names are read from the trainer rather than hard-coded: YOLO26
    dropped DFL, so its third term is `l1_loss` where YOLOv8's is `dfl_loss`,
    and a future variant may differ again.
    """
    row = {"epoch": trainer.epoch + 1}
    losses = {**trainer.label_loss_items(trainer.tloss, prefix="train"),
              **{k: v for k, v in trainer.metrics.items() if k.startswith("val/")}}
    for key, value in losses.items():
        prefix, _, name = key.partition("/")
        row[f"{prefix}_{name}"] = round(float(value), 5)

    val_losses = [v for k, v in row.items() if k.startswith("val_")]
    row["val_loss"] = round(sum(val_losses), 5) if val_losses else None

    metrics = trainer.metrics
    row["map50"] = _round(metrics.get(MAP50_KEY))
    row["map50_95"] = _round(metrics.get(MAP50_95_KEY))
    row["precision"] = _round(metrics.get(PRECISION_KEY))
    row["recall"] = _round(metrics.get(RECALL_KEY))
    for name, ap50 in per_class_ap50(trainer).items():
        row[f"{name}_ap50"] = ap50
    row["lr"] = round(float(list(trainer.lr.values())[0]), 6) if trainer.lr else None
    row["seconds"] = round(float(getattr(trainer, "epoch_time", 0.0)), 1)
    return row


def append_metric_row(path: str, row: dict) -> None:
    """Append one epoch, keeping the header of an existing file authoritative.

    A resumed run must not change the columns half-way through the file, so an
    existing header wins: new keys are dropped and missing ones left blank.
    """
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as handle:
            fieldnames = next(csv.reader(handle), list(row))
        row = {key: row.get(key, "") for key in fieldnames}
        header = False
    else:
        fieldnames, header = list(row), True
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerow(row)


def mirror_outputs(trainer, output_dir: str, info: dict, args, metric_path: str,
                   status_path: str) -> None:
    """Copy this epoch's checkpoints and numbers out of the Ultralytics run.

    Called after every epoch, from on_fit_epoch_end -- Ultralytics has already
    written last.pt and (when it improved) best.pt by then. Mirroring per epoch
    rather than at the end means an interrupted run still leaves a usable
    model.pt and a complete metric.csv.

    The same callback fires once more when training is over, for the final
    validation of best.pt. That pass carries no training losses and would
    append a phantom epoch past the run's length, so it is skipped here.
    """
    if trainer.epoch + 1 > trainer.epochs:
        return

    row = metric_row(trainer)
    append_metric_row(metric_path, row)

    epoch = trainer.epoch + 1
    best_fitness = float(trainer.best_fitness) if trainer.best_fitness is not None else None
    improved = best_fitness is not None and float(trainer.fitness or -1) >= best_fitness

    if os.path.exists(trainer.last):
        shutil.copy2(trainer.last, os.path.join(output_dir, LAST_FILENAME))
    if improved and os.path.exists(trainer.best):
        shutil.copy2(trainer.best, os.path.join(output_dir, MODEL_FILENAME))
        write_model_info(os.path.join(output_dir, MODEL_FILENAME),
                         {**info, "epoch": epoch, "best_fitness": best_fitness,
                          "metrics": row})

    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump({"dataset": args.dataset, "epochs_completed": epoch,
                   "epochs_total": int(trainer.epochs), "best_fitness": best_fitness,
                   "last_metrics": row, "args": vars(args), "info": info},
                  handle, indent=2)


def finished_run(status_path: str) -> tuple[int, int] | None:
    """(completed, total) when train-status.json says the run is already done.

    Ultralytics refuses to resume a finished run with an assertion; catching
    that here turns a traceback into one line of advice. On resume it also
    takes the epoch budget from the checkpoint, so a longer run really does
    have to start over.
    """
    try:
        with open(status_path, encoding="utf-8") as handle:
            status = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    completed, total = status.get("epochs_completed", 0), status.get("epochs_total", 0)
    return (completed, total) if total and completed >= total else None


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26 on a tooltip-detector dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None,
                        help="dataset directory under data/dataset (e.g. cholec80); it must "
                             "have been prepared with scripts/prepare-dataset.py first")
    parser.add_argument("--scale", default=DEFAULT_SCALE, choices=SCALES,
                        help=f"YOLO26 depth/width scale (default: {DEFAULT_SCALE}, the size "
                             "the YOLOv8s baselines use)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="initial learning rate; ignored while --optimizer is auto, "
                             "which picks the rate along with the optimizer")
    parser.add_argument("--optimizer", default="auto",
                        help="auto (default; MuSGD on a run this long), SGD, MuSGD, AdamW, ...")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None,
                        help="e.g. cuda:1 or cpu (default: cuda:0 if available)")
    parser.add_argument("--patience", type=int, default=0,
                        help="early-stopping patience in epochs (0 = never stop early)")
    parser.add_argument("--output-dir", default=None,
                        help="where the checkpoints go (default: data/model/<dataset>)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="start from random weights instead of the COCO checkpoint")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision (also skips Ultralytics' AMP check, "
                             "which downloads yolo26n.pt the first time)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing model-last.pt and start over")
    args = parser.parse_args()

    from ultralytics import YOLO

    use_local_weights_dir()
    status = require_prepared(args.dataset)
    # Absolute on purpose. Ultralytics treats a *relative* `project` as being
    # under the runs directory in its global settings, which would scatter the
    # run somewhere else entirely while this script mirrored its outputs to a
    # path relative to the working directory.
    args.output_dir = os.path.abspath(args.output_dir or model_dir(args.dataset))
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join(args.output_dir, RUN_DIRNAME)
    last_run = os.path.join(run_dir, "weights", "last.pt")
    metric_path = os.path.join(args.output_dir, "metric.csv")
    status_path = os.path.join(args.output_dir, "train-status.json")

    resume = not args.no_resume and os.path.exists(last_run)
    if resume:
        done = finished_run(status_path)
        if done is not None:
            print(f"nothing to resume: {done[0]} / {done[1]} epochs already completed")
            print(f"  results   {args.output_dir}")
            print("  train longer or with other settings by re-running with --no-resume")
            return
    if args.no_resume and os.path.exists(metric_path):
        os.remove(metric_path)      # the run it described is being discarded

    if resume:
        model = YOLO(last_run)
        print(f"resuming {last_run}")
    elif args.no_pretrained:
        model = YOLO(f"yolo26{args.scale}.yaml")
        print(f"starting yolo26{args.scale} from random weights")
    else:
        weights = pretrained_weights(args.scale)
        model = YOLO(weights)
        print(f"starting from {weights}")

    info = {"dataset": args.dataset, "scale": args.scale,
            "image_size": args.image_size,
            "tip_box_size": status["tip_box_size"],
            "class_names": list(CLASS_NAMES),
            "pretrained": not args.no_pretrained,
            "frame_stride": status["frame_stride"],
            "weights": "yolo26" + args.scale}

    print(f"dataset : {dataset_yaml_path(args.dataset)}")
    print(f"labels  : tip box {status['tip_box_size']:g} px, "
          f"train frames {status['splits']['train']['frames']:,}")
    print(f"output  : {args.output_dir}")

    model.add_callback("on_fit_epoch_end",
                       lambda trainer: mirror_outputs(trainer, args.output_dir, info,
                                                      args, metric_path, status_path))

    train_kwargs = dict(
        data=dataset_yaml_path(args.dataset),
        project=args.output_dir, name=RUN_DIRNAME, exist_ok=True,
        imgsz=args.image_size, batch=args.batch_size, workers=args.workers,
        device=args.device, amp=not args.no_amp,
        # 0 means "never stop early" to Ultralytics too, but only via a large
        # number; patience=0 there disables the counter entirely in some
        # versions, so the epoch budget is made explicit instead.
        patience=args.patience or args.epochs + 1,
        plots=True, seed=0,
    )
    if resume:
        # On resume Ultralytics restores every other argument from the
        # checkpoint; only these may be overridden.
        model.train(resume=last_run, **{k: v for k, v in train_kwargs.items()
                                        if k in {"device", "workers", "batch", "imgsz",
                                                 "patience", "plots", "project", "name",
                                                 "exist_ok", "data"}})
    else:
        model.train(epochs=args.epochs, lr0=args.lr, optimizer=args.optimizer,
                    pretrained=not args.no_pretrained, **train_kwargs)

    best = os.path.join(args.output_dir, MODEL_FILENAME)
    print()
    print(f"done. best checkpoint -> {best}")
    print(f"next  ./baseline/yolo26/run eval-model --dataset {args.dataset}")


def _round(value, digits: int = 5):
    return None if value is None else round(float(value), digits)


if __name__ == "__main__":
    main()
