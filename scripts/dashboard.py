#!/usr/bin/env python3
"""Experiment dashboard: train/eval completion status per experiment row.

Two tabs, one per project, because the two have different experiment axes:

  This project   one row per dataset x target-mode x model-type
  Baselines      one row per baseline x dataset (a detector has no target
                 mode: it regresses `tool` and `tip` boxes directly)

Either way the Train and Eval columns say how far that row has progressed,
and selecting a cell shows the exact command needed to carry out that task,
ready to copy. The detail and command panels below the tabs always describe
the tab in front.

Status is derived purely from files on disk, so the dashboard never has to
load a model or a dataset:

  Train  data/models/<dataset>/<target-mode>/<model-type>/
           best.pt / last.pt          — checkpoints written by train-model
           train-status.json          — completed_epochs / epochs, best val loss
           metric.csv                 — one row per completed epoch
  Eval   data/results/<dataset>/<target-mode>/<model-type>/
           summary.json               — metrics + run parameters
           per_tip.csv                — one row per GT tip

  Train  baseline/<name>/data/model/<dataset>/
           model.pt / model-last.pt   — best / last checkpoint
           train-status.json          — epochs_completed / epochs_total, best mAP
           metric.csv                 — one row per completed epoch
  Eval   baseline/<name>/data/results/<dataset>/test/
           summary.json               — detection + tip metrics, run parameters
           per_tip.csv                — one row per GT tip

Usage:
    uv run python scripts/dashboard.py [--dataset erop]

Navigation:
    click a Train/Eval cell   select that task
    up / down                 change row
    left / right              switch between Train and Eval
    F5                        refresh now (also auto-refreshes every 5 s)
"""
import argparse
import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttd.checkpoints import default_results_dir, model_dir
from ttd.dataset import DATASETS, DEFAULT_GAUSSIAN_SIGMA, TARGET_MODES
from ttd.model import REGISTRY as MODEL_REGISTRY

# `run` dispatches to scripts/<name>.py; Windows uses the .bat twin. The path
# is qualified with ./ (.\ on Windows) because the command shown here is meant
# to be pasted straight into a shell: the project root is not on PATH, so a
# bare `run` fails with "command not found" (and PowerShell rejects a bare
# `run.bat` for the same reason). Prose in README/docs keeps the shorter form.
RUNNER = r".\run.bat" if os.name == "nt" else "./run"

# The baselines under baseline/, paired clone-next-to-original. Each is its
# own sub-project with its own runner, so the command shown for a baseline row
# is `./baseline/<name>/run ...`, not the root `./run`.
#
# `train_flags` are the flags that baseline's docs/commands.md uses to
# reproduce its reported numbers. They are not the script defaults
# (--frame-stride defaults to 1 and --val-frames to 2000), so omitting them
# would print a command that trains on a different set of frames than the
# checkpoint already on disk. yolov8s and yolo26 take their frame stride when
# their dataset is converted for Ultralytics instead, so nothing about the
# training set is left to their train-model call.
BASELINES = (
    {"name": "yolov8s", "train_flags": "--workers 12"},
    {"name": "yolov8sclone",
     "train_flags": "--frame-stride 5 --val-frames 1500 --workers 12"},
    {"name": "yolo26", "train_flags": "--workers 12"},
    {"name": "yolo26clone",
     "train_flags": "--frame-stride 5 --val-frames 1500 --workers 12"},
    {"name": "cladnet",
     "train_flags": "--frame-stride 5 --val-frames 1500 --workers 12"},
)


def baseline_runner(name: str) -> str:
    """The runner of one baseline sub-project, as a pasteable path."""
    if os.name == "nt":
        return rf".\baseline\{name}\run.bat"
    return f"./baseline/{name}/run"


# Re-scanning is cheap (a handful of stat() calls per row), so the table can
# follow a training run that is writing checkpoints in another terminal.
AUTO_REFRESH_MS = 5000


def task_at_column(tree, column_id: str) -> str | None:
    """Which task a clicked column belongs to, or None for the other columns.

    Train and Eval are always the last two columns of a status table, which is
    what lets one click handler serve both tabs even though the project table
    has five columns and the baseline table four.
    """
    try:
        index = int(column_id.lstrip("#"))
    except ValueError:
        return None
    count = len(tree["columns"])
    return {count - 1: "Train", count: "Eval"}.get(index)


# Cell labels stay ASCII: Tk's default widget font has no glyph for symbols
# such as U+2714, which would render as empty boxes. Row colour carries the
# at-a-glance signal instead.


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _read_json(path: str):
    """Parse a JSON file, or return None if it is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fmt_time(path: str) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def _fmt_size(path: str) -> str:
    mb = os.path.getsize(path) / (1024 * 1024)
    return f"{mb:.1f} MB" if mb >= 1.0 else f"{os.path.getsize(path) / 1024:.0f} KB"


def _fmt_file(path: str) -> str:
    """One-line description of a produced artefact file."""
    if not os.path.exists(path):
        return "missing"
    return f"{_fmt_size(path)}   {_fmt_time(path)}"


def _num(source: dict, key: str, fmt: str = "{:.2f}") -> str:
    """Format one recorded metric, or "n/a" when the file does not carry it.

    Read defensively: summary.json / train-status.json files written by older
    revisions of either project do not have every field.
    """
    value = source.get(key)
    return fmt.format(value) if isinstance(value, (int, float)) else "n/a"


def _count_epochs(metric_path: str) -> int | None:
    """Number of data rows in metric.csv (bounded by the epoch count)."""
    if not os.path.exists(metric_path):
        return None
    try:
        with open(metric_path, newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Status scanning
# ---------------------------------------------------------------------------

def scan_train(combo: dict, models_root: str) -> dict:
    """Inspect a combination's model directory.

    Returns a dict with:
        state  — "done" (all requested epochs finished) / "partial" (resumable
                 run in progress) / "weights" (checkpoint without
                 train-status.json, e.g. from before resume support) / "none"
        label  — short text for the table cell
        detail — list of (key, value) lines for the detail panel
        sigma  — gaussian_sigma recorded by the run, if any
    """
    mdir = model_dir(combo["model_type"], combo["dataset"],
                     combo["target_mode"], root=models_root)
    best_path = os.path.join(mdir, "best.pt")
    last_path = os.path.join(mdir, "last.pt")
    status = _read_json(os.path.join(mdir, "train-status.json"))
    metric_path = os.path.join(mdir, "metric.csv")

    detail = [("Model dir", mdir if os.path.isdir(mdir) else f"{mdir}  (not created yet)")]

    if status is not None:
        completed = status.get("completed_epochs", 0)
        total = status.get("epochs", completed)
        state = "done" if completed >= total else "partial"
        label = (f"done  {completed}/{total} epochs" if state == "done"
                 else f"{completed}/{total} epochs  (resumable)")
        best_val = status.get("best_val_loss")
        detail += [
            ("Epochs", f"{completed} / {total} completed"),
            ("Best val loss", f"{best_val:.6f}" if isinstance(best_val, (int, float)) else "n/a"),
            ("Last val loss", f"{status['val_loss']:.6f}" if isinstance(status.get("val_loss"), (int, float)) else "n/a"),
            ("Batch / lr", f"{status.get('batch_size', '?')}  /  {status.get('lr', '?')}"),
            ("Last epoch at", status.get("timestamp", "n/a")),
        ]
    elif os.path.exists(last_path) or os.path.exists(best_path):
        state, label = "weights", "done  (weights only)"
        detail.append(
            ("Epochs", "unknown - no train-status.json (weights-only checkpoint)"))
    else:
        state, label = "none", "not started"
        detail.append(("Epochs", "no checkpoint yet"))

    epochs_logged = _count_epochs(metric_path)
    detail += [
        ("best.pt", _fmt_file(best_path)),
        ("last.pt", _fmt_file(last_path)),
        ("metric.csv", f"{epochs_logged} epoch rows   {_fmt_time(metric_path)}"
                       if epochs_logged is not None else "missing"),
    ]

    sigma = status.get("gaussian_sigma") if status else None
    return {"state": state, "label": label, "detail": detail, "sigma": sigma,
            "best_path": best_path}


def scan_eval(combo: dict, results_root: str) -> dict:
    """Inspect a combination's results directory.

    Returns a dict with `state` ("done" / "none"), `label` and `detail` as in
    `scan_train`. Metric keys are read defensively: summary.json files written
    by older eval-model revisions do not carry every field.
    """
    rdir = default_results_dir(combo["model_type"], combo["dataset"],
                               combo["target_mode"], root=results_root)
    summary_path = os.path.join(rdir, "summary.json")
    csv_path = os.path.join(rdir, "per_tip.csv")
    summary = _read_json(summary_path)

    detail = [("Results dir", rdir if os.path.isdir(rdir) else f"{rdir}  (not created yet)")]

    if summary is None:
        broken = os.path.exists(summary_path)
        label = "! unreadable summary.json" if broken else "not run"
        detail.append(("summary.json", "unreadable" if broken else "missing"))
        return {"state": "none", "label": label, "detail": detail}

    def num(key, fmt="{:.2f}"):
        return _num(summary, key, fmt)

    hit20 = summary.get("hit_rate_20px_pct")
    label = (f"done  hit@20 {hit20:.2f}%" if isinstance(hit20, (int, float))
             else "done")
    detail += [
        ("Evaluated at", summary.get("timestamp", _fmt_time(summary_path))),
        ("Threshold / NMS", f"{summary.get('threshold', '?')}  /  "
                            f"{summary.get('nms_radius', '?')} px"),
        ("GT tips / frames", f"{num('n_gt_tips', '{:,}')} tips in "
                             f"{num('n_frames_with_tools', '{:,}')} frames"),
        ("Miss rate", f"{num('miss_rate_pct')} %"),
        ("Hit@10 / 20 / 50", f"{num('hit_rate_10px_pct')} / "
                             f"{num('hit_rate_20px_pct')} / "
                             f"{num('hit_rate_50px_pct')} %"),
        ("Mean / median dist", f"{num('mean_dist_px')} / {num('median_dist_px')} px"),
        ("P90 dist", f"{num('p90_dist_px')} px"),
        ("summary.json", _fmt_file(summary_path)),
        ("per_tip.csv", _fmt_file(csv_path)),
    ]
    return {"state": "done", "label": label, "detail": detail}


def baseline_model_dir(combo: dict, baseline_root: str) -> str:
    return os.path.join(baseline_root, combo["baseline"], "data", "model",
                        combo["dataset"])


def baseline_results_dir(combo: dict, baseline_root: str) -> str:
    # Baselines evaluate one split at a time and keep the split in the path;
    # only the test split is what the reports quote, so that is what is shown.
    return os.path.join(baseline_root, combo["baseline"], "data", "results",
                        combo["dataset"], "test")


def scan_baseline_train(combo: dict, baseline_root: str) -> dict:
    """Inspect a baseline row's model directory.

    Same return shape as `scan_train`, so the table, the detail panel and the
    notes are written once. Only the field names differ: a baseline records
    epochs_completed / epochs_total and a best mAP (the clones) or Ultralytics
    fitness (yolov8s, yolo26), where this project records completed_epochs /
    epochs and a best validation loss.
    """
    mdir = baseline_model_dir(combo, baseline_root)
    best_path = os.path.join(mdir, "model.pt")
    last_path = os.path.join(mdir, "model-last.pt")
    status = _read_json(os.path.join(mdir, "train-status.json"))
    metric_path = os.path.join(mdir, "metric.csv")

    detail = [("Model dir", mdir if os.path.isdir(mdir) else f"{mdir}  (not created yet)")]

    if status is not None:
        completed = status.get("epochs_completed", 0)
        total = status.get("epochs_total", completed)
        state = "done" if completed >= total else "partial"
        label = (f"done  {completed}/{total} epochs" if state == "done"
                 else f"{completed}/{total} epochs  (resumable)")
        # The clones select their best checkpoint on mAP@0.5:0.95; the two
        # Ultralytics runs on its own fitness score. Whichever is recorded is
        # the number that decided which epoch became model.pt.
        best_key = ("best_map50_95" if "best_map50_95" in status
                    else "best_fitness")
        best = status.get(best_key)
        last = status.get("last_metrics") or {}
        args = status.get("args") or {}
        detail += [
            ("Epochs", f"{completed} / {total} completed"),
            (f"Best {'mAP@0.5:0.95' if best_key == 'best_map50_95' else 'fitness'}",
             f"{best:.4f}" if isinstance(best, (int, float)) else "n/a"),
            ("Last val loss",
             f"{last['val_loss']:.5f}" if isinstance(last.get("val_loss"), (int, float)) else "n/a"),
            ("Last mAP50 / 50:95",
             f"{_num(last, 'map50', '{:.4f}')} / {_num(last, 'map50_95', '{:.4f}')}"),
            ("Batch / lr", f"{args.get('batch_size', '?')}  /  {args.get('lr', '?')}"),
            ("Device", args.get("device", "n/a")),
        ]
    elif os.path.exists(last_path) or os.path.exists(best_path):
        state, label = "weights", "done  (weights only)"
        detail.append(
            ("Epochs", "unknown - no train-status.json (weights-only checkpoint)"))
    else:
        state, label = "none", "not started"
        detail.append(("Epochs", "no checkpoint yet"))

    epochs_logged = _count_epochs(metric_path)
    detail += [
        ("model.pt", _fmt_file(best_path)),
        ("model-last.pt", _fmt_file(last_path)),
        ("metric.csv", f"{epochs_logged} epoch rows   {_fmt_time(metric_path)}"
                       if epochs_logged is not None else "missing"),
    ]

    return {"state": state, "label": label, "detail": detail,
            "best_path": best_path}


def scan_baseline_eval(combo: dict, baseline_root: str) -> dict:
    """Inspect a baseline row's results directory.

    A baseline summary.json nests its tip metrics under "tip" and adds
    detection metrics (mAP) and a measured frame rate, so the panel shows more
    than the project's eval does. The tip block itself carries the same keys,
    which is what makes the two tabs comparable at all.
    """
    rdir = baseline_results_dir(combo, baseline_root)
    summary_path = os.path.join(rdir, "summary.json")
    csv_path = os.path.join(rdir, "per_tip.csv")
    summary = _read_json(summary_path)

    detail = [("Results dir", rdir if os.path.isdir(rdir) else f"{rdir}  (not created yet)")]

    if summary is None:
        broken = os.path.exists(summary_path)
        label = "! unreadable summary.json" if broken else "not run"
        detail.append(("summary.json", "unreadable" if broken else "missing"))
        return {"state": "none", "label": label, "detail": detail}

    tip = summary.get("tip") or {}
    detection = summary.get("detection") or {}

    hit20 = tip.get("hit_rate_20px_pct")
    label = (f"done  hit@20 {hit20:.2f}%" if isinstance(hit20, (int, float))
             else "done")

    # yolo26 and yolo26clone have an end-to-end head and so no NMS IoU at all;
    # they cap detections with --max-det instead. Showing whichever is
    # recorded keeps the panel honest about what produced these numbers.
    suppression = (f"NMS IoU {summary['iou_threshold']}"
                   if "iou_threshold" in summary else
                   f"max det {summary.get('max_det', '?')}  (no NMS)")
    detail += [
        ("Evaluated at", _fmt_time(summary_path)),
        ("Epoch / split", f"epoch {summary.get('trained_epoch', '?')}  /  "
                          f"{summary.get('split', '?')} split"),
        ("Conf / suppression", f"{summary.get('conf_threshold', '?')}  /  {suppression}"),
        ("GT tips / frames", f"{_num(tip, 'n_gt_tips', '{:,}')} tips in "
                             f"{_num(summary, 'n_frames', '{:,}')} frames"),
        ("Miss rate", f"{_num(tip, 'miss_rate_pct')} %"),
        ("Hit@10 / 20 / 50", f"{_num(tip, 'hit_rate_10px_pct')} / "
                             f"{_num(tip, 'hit_rate_20px_pct')} / "
                             f"{_num(tip, 'hit_rate_50px_pct')} %"),
        ("Mean / median dist", f"{_num(tip, 'mean_dist_px')} / "
                               f"{_num(tip, 'median_dist_px')} px"),
        ("P90 dist", f"{_num(tip, 'p90_dist_px')} px"),
        ("mAP50 / 50:95", f"{_num(detection, 'map50', '{:.4f}')} / "
                          f"{_num(detection, 'map50_95', '{:.4f}')}"),
        ("Speed", f"{_num(summary, 'ms_per_frame')} ms/frame   "
                  f"{_num(summary, 'fps', '{:.1f}')} fps   "
                  f"on {summary.get('device', 'n/a')}"),
        ("summary.json", _fmt_file(summary_path)),
        ("per_tip.csv", _fmt_file(csv_path)),
    ]
    return {"state": "done", "label": label, "detail": detail}


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def train_command(combo: dict, train: dict) -> str:
    """The command that trains (or resumes) this combination.

    Resuming is train-model's default, so an interrupted run and a fresh run
    take the same command. --gaussian-sigma is emitted only when the recorded
    run used a non-default value, so resuming keeps its targets identical.
    """
    cmd = (f"{RUNNER} train-model --dataset {combo['dataset']}"
           f" --target-mode {combo['target_mode']}"
           f" --model-type {combo['model_type']}")
    sigma = train.get("sigma")
    if (combo["target_mode"] == "gaussian-tip"
            and isinstance(sigma, (int, float)) and sigma != DEFAULT_GAUSSIAN_SIGMA):
        cmd += f" --gaussian-sigma {sigma}"
    return cmd


def eval_command(combo: dict) -> str:
    """The command that evaluates this combination's best.pt on the test set."""
    return (f"{RUNNER} eval-model --dataset {combo['dataset']}"
            f" --target-mode {combo['target_mode']}"
            f" --model-type {combo['model_type']}")


def task_note(task: str, combo: dict, train: dict, evaluation: dict) -> str:
    """One-line guidance about running this task in its current state."""
    if task == "Train":
        if train["state"] == "done":
            return ("Already complete. The same command exits with "
                    "\"Nothing to do\": pass a larger --epochs to keep "
                    "training, or --no-resume to start over.")
        if train["state"] == "partial":
            return ("Resumes from last.pt at the next epoch (resuming is the "
                    "default); pass --no-resume to train from scratch.")
        if train["state"] == "weights":
            return ("last.pt exists without train-status.json, so the epoch "
                    "count restarts at 1 while the weights are kept.")
        return "Trains from scratch and creates the model directory."

    if not os.path.exists(train["best_path"]):
        return ("best.pt does not exist yet: train this combination first, "
                "or pass --model to point at other weights.")
    if evaluation["state"] == "done":
        return ("summary.json / per_tip.csv already exist and are overwritten "
                "by a re-run (e.g. after further training).")
    return "Evaluates best.pt on the test set and writes the results directory."


def baseline_train_command(combo: dict) -> str:
    """The command that trains (or resumes) this baseline on this dataset."""
    flags = next(b["train_flags"] for b in BASELINES
                 if b["name"] == combo["baseline"])
    return (f"{baseline_runner(combo['baseline'])} train-model"
            f" --dataset {combo['dataset']} {flags}")


def baseline_eval_command(combo: dict) -> str:
    """The command that evaluates this baseline's model.pt on the test split."""
    return (f"{baseline_runner(combo['baseline'])} eval-model"
            f" --dataset {combo['dataset']} --split test")


def baseline_task_note(task: str, train: dict, evaluation: dict) -> str:
    """One-line guidance about running a baseline task in its current state."""
    if task == "Train":
        if train["state"] == "done":
            return ("Already complete. The same command runs no further "
                    "epochs: pass a larger --epochs to keep training, or "
                    "--no-resume to start over.")
        if train["state"] == "partial":
            return ("Resumes from model-last.pt at the next epoch (resuming "
                    "is the default); pass --no-resume to train from scratch.")
        if train["state"] == "weights":
            return ("The checkpoint has no train-status.json, so no epoch "
                    "count can be shown; the run still resumes from the epoch "
                    "model-last.pt itself records.")
        return "Trains from scratch and creates the model directory."

    if not os.path.exists(train["best_path"]):
        return ("model.pt does not exist yet: train this baseline on this "
                "dataset first.")
    if evaluation["state"] == "done":
        return ("summary.json / per_tip.csv already exist and are overwritten "
                "by a re-run (e.g. after further training).")
    return ("Evaluates model.pt on the full test split and writes the results "
            "directory.")


def row_label(row: dict) -> str:
    """How a table row is named in the detail panel's title."""
    if "baseline" in row:
        return f"{row['baseline']} / {row['dataset']}"
    return f"{row['dataset']} / {row['target_mode']} / {row['model_type']}"


def row_tag(scan: dict) -> str:
    """Row colour: green once both tasks are done, grey while neither is."""
    train, evaluation = scan["train"], scan["eval"]
    if train["state"] in ("done", "weights") and evaluation["state"] == "done":
        return "complete"
    if train["state"] == "none" and evaluation["state"] == "none":
        return "none"
    return "partial"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class Dashboard(tk.Tk):
    def __init__(self, models_root: str, results_root: str, baseline_root: str,
                 dataset: str | None):
        super().__init__()
        self.title("Dashboard - train / eval status")
        self.minsize(940, 700)

        self._models_root = models_root
        self._results_root = results_root
        self._baseline_root = baseline_root

        # One row per experiment combination, dataset-major.
        self._combos = [
            {"dataset": ds, "target_mode": tm, "model_type": mt}
            for ds in DATASETS for tm in TARGET_MODES for mt in MODEL_REGISTRY
        ]
        # Baselines have no target-mode axis, so a baseline row is just the
        # pair (baseline, dataset). Baseline-major, to keep each baseline's
        # two datasets side by side.
        self._baseline_combos = [
            {"baseline": b["name"], "dataset": ds}
            for b in BASELINES for ds in DATASETS
        ]
        self._scans: list[dict] = []
        self._baseline_scans: list[dict] = []
        self._task = "Train"
        # Wall-clock deadline until which the status label keeps showing the
        # copy confirmation instead of the scan timestamp.
        self._copy_feedback_until = 0.0

        self._build_ui()
        self.refresh()

        # Both tabs start on a row, so switching tabs always has something to
        # describe; --dataset picks the first row of that dataset in each.
        for tree, combos in ((self._tree, self._combos),
                             (self._baseline_tree, self._baseline_combos)):
            initial = 0
            if dataset is not None:
                initial = next((i for i, c in enumerate(combos)
                                if c["dataset"] == dataset), 0)
            self._select(initial, tree)

        self.bind("<F5>", lambda _: self.refresh())
        self.bind("<Left>", lambda _: self._set_task("Train"))
        self.bind("<Right>", lambda _: self._set_task("Eval"))
        self.after(AUTO_REFRESH_MS, self._auto_refresh)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        ctrl = tk.Frame(self, pady=6, padx=8)
        ctrl.pack(fill=tk.X)

        tk.Button(ctrl, text="Refresh", command=self.refresh).pack(side=tk.LEFT)
        self._refreshed_var = tk.StringVar()
        tk.Label(ctrl, textvariable=self._refreshed_var,
                 fg="gray").pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(ctrl, text="click a Train/Eval cell    Left/Right : task    "
                            "Up/Down : row    F5 : refresh",
                 fg="gray").pack(side=tk.RIGHT)

        # ── Status tables ────────────────────────────────────────────────
        # One tab per project: this repository's heatmap models and the
        # detector baselines. They are separate tables rather than one because
        # their axes differ (a baseline has no target mode) and because each
        # is driven by a different runner.
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.X, padx=8)

        self._tree = self._add_table("This project", (
            ("dataset", "Dataset", 100),
            ("target", "Target mode", 120),
            ("model", "Model type", 110),
            ("train", "Train", 190),
            ("eval", "Eval", 190),
        ), len(self._combos))

        self._baseline_tree = self._add_table("Baselines", (
            ("baseline", "Baseline", 130),
            ("dataset", "Dataset", 100),
            ("train", "Train", 195),
            ("eval", "Eval", 195),
        ), len(self._baseline_combos))

        # Tk moves the selection into the newly shown table, but not before
        # this binding runs, so the panels are rendered once it has.
        self._notebook.bind("<<NotebookTabChanged>>",
                            lambda _: self.after_idle(self._render_detail))

        # ── Selected cell detail ─────────────────────────────────────────
        self._detail_frame = tk.LabelFrame(self, text="Selected", padx=6, pady=4)
        self._detail_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        # Tall enough for the longest panel (a baseline evaluation, 13 lines),
        # and CHAR wrap so a long --models-root path is never clipped out of
        # sight.
        self._detail = tk.Text(self._detail_frame, height=13, wrap=tk.CHAR,
                               font=("Monospace", 9), bg="#f7f7f7",
                               relief=tk.FLAT, state=tk.DISABLED)
        self._detail.pack(fill=tk.BOTH, expand=True)

        # ── Command to run ───────────────────────────────────────────────
        cmd_frame = tk.LabelFrame(self, text="Command", padx=6, pady=6)
        cmd_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        entry_row = tk.Frame(cmd_frame)
        entry_row.pack(fill=tk.X)

        self._cmd_var = tk.StringVar()
        tk.Entry(entry_row, textvariable=self._cmd_var, font=("Monospace", 10),
                 state="readonly", readonlybackground="#ffffff").pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(entry_row, text="Copy", width=8,
                  command=self._copy_command).pack(side=tk.LEFT, padx=(6, 0))

        self._note_var = tk.StringVar()
        tk.Label(cmd_frame, textvariable=self._note_var, fg="#555555",
                 wraplength=880, justify=tk.LEFT, anchor="w").pack(
            fill=tk.X, pady=(4, 0))

        runner_hint = ("Run from the project root. On Linux/macOS use "
                       "`./run` instead of `.\\run.bat`." if os.name == "nt" else
                       "Run from the project root. On Windows use "
                       "`.\\run.bat` instead of `./run`.")
        tk.Label(cmd_frame, text=runner_hint, fg="gray", anchor="w").pack(
            fill=tk.X)

    def _add_table(self, title: str, columns, n_rows: int) -> ttk.Treeview:
        """Add one tab holding an empty status table; return its Treeview.

        The rows are filled in by `refresh`; only their number is needed here,
        to size the table so neither tab has to scroll.
        """
        frame = tk.Frame(self._notebook, padx=4, pady=4)
        self._notebook.add(frame, text=f"  {title}  ")

        tree = ttk.Treeview(frame, columns=tuple(name for name, _, _ in columns),
                            show="headings", height=n_rows, selectmode="browse")
        for name, text, width in columns:
            tree.heading(name, text=text)
            tree.column(name, width=width, anchor=tk.W,
                        stretch=(name in ("train", "eval")))
        tree.pack(fill=tk.X)

        # Row colour summarises the pair of statuses at a glance.
        tree.tag_configure("complete", foreground="#1a7f2e")
        tree.tag_configure("partial", foreground="#b35c00")
        tree.tag_configure("none", foreground="#808080")

        for _ in range(n_rows):
            tree.insert("", tk.END, values=("",) * len(columns))

        tree.bind("<Button-1>", self._on_click)
        tree.bind("<<TreeviewSelect>>", lambda _: self._render_detail())
        tree.bind("<Left>", self._on_tree_arrow)
        tree.bind("<Right>", self._on_tree_arrow)
        return tree

    # ── Selection ────────────────────────────────────────────────────────

    def _active(self) -> tuple[ttk.Treeview, list[dict], list[dict]]:
        """The table in front, with the rows and the scans that belong to it."""
        if self._notebook.index("current") == 0:
            return self._tree, self._combos, self._scans
        return self._baseline_tree, self._baseline_combos, self._baseline_scans

    def _rows(self, tree: ttk.Treeview) -> list[str]:
        return list(tree.get_children(""))

    def _selected_index(self) -> int | None:
        tree = self._active()[0]
        sel = tree.selection()
        return self._rows(tree).index(sel[0]) if sel else None

    def _select(self, index: int, tree: ttk.Treeview):
        rows = self._rows(tree)
        index = max(0, min(index, len(rows) - 1))
        tree.selection_set(rows[index])
        tree.focus(rows[index])

    def _set_task(self, task: str):
        self._task = task
        self._render_detail()

    def _on_tree_arrow(self, event):
        """Left/Right inside the table switch task instead of collapsing rows."""
        self._set_task("Train" if event.keysym == "Left" else "Eval")
        return "break"

    def _on_click(self, event):
        """Clicking a Train/Eval cell selects that task for the clicked row."""
        tree = event.widget
        if tree.identify("region", event.x, event.y) != "cell":
            return
        task = task_at_column(tree, tree.identify_column(event.x))
        if task is not None:
            self._task = task
        # Tk applies its own selection after this binding; render once it has.
        self.after_idle(self._render_detail)

    # ── Refresh & render ─────────────────────────────────────────────────

    def _auto_refresh(self):
        # Reschedule first: a scan that trips over a file being rewritten by a
        # running training job must not silently end the refresh chain.
        self.after(AUTO_REFRESH_MS, self._auto_refresh)
        self.refresh()

    def refresh(self):
        """Re-scan every row's directories and repaint both tables."""
        self._scans = [
            {
                "train": scan_train(combo, self._models_root),
                "eval": scan_eval(combo, self._results_root),
            }
            for combo in self._combos
        ]
        self._baseline_scans = [
            {
                "train": scan_baseline_train(combo, self._baseline_root),
                "eval": scan_baseline_eval(combo, self._baseline_root),
            }
            for combo in self._baseline_combos
        ]

        for row, combo, scan in zip(self._rows(self._tree), self._combos,
                                    self._scans):
            self._tree.item(row, tags=(row_tag(scan),), values=(
                combo["dataset"], combo["target_mode"], combo["model_type"],
                scan["train"]["label"], scan["eval"]["label"]))

        for row, combo, scan in zip(self._rows(self._baseline_tree),
                                    self._baseline_combos, self._baseline_scans):
            self._baseline_tree.item(row, tags=(row_tag(scan),), values=(
                combo["baseline"], combo["dataset"],
                scan["train"]["label"], scan["eval"]["label"]))

        if time.time() >= self._copy_feedback_until:
            self._refreshed_var.set(f"scanned {time.strftime('%H:%M:%S')}")
        self._render_detail()

    def _render_detail(self):
        _, combos, scans = self._active()
        index = self._selected_index()
        # A tab switch can land here before the first scan of that table.
        if index is None or index >= len(scans):
            return
        combo, scan = combos[index], scans[index]
        train, evaluation = scan["train"], scan["eval"]

        active = train if self._task == "Train" else evaluation
        self._detail_frame.config(
            text=f"  {self._task}  :  {row_label(combo)}   "
                 f"[{active['label']}]  ")

        lines = "\n".join(f"{key:<22}: {value}" for key, value in active["detail"])
        self._detail.config(state=tk.NORMAL)
        self._detail.delete("1.0", tk.END)
        self._detail.insert("1.0", lines)
        self._detail.config(state=tk.DISABLED)

        if "baseline" in combo:
            command = (baseline_train_command(combo) if self._task == "Train"
                       else baseline_eval_command(combo))
            note = baseline_task_note(self._task, train, evaluation)
        else:
            command = (train_command(combo, train) if self._task == "Train"
                       else eval_command(combo))
            note = task_note(self._task, combo, train, evaluation)
        self._cmd_var.set(command)
        self._note_var.set(note)

    # ── Clipboard ────────────────────────────────────────────────────────

    def _copy_command(self):
        self.clipboard_clear()
        self.clipboard_append(self._cmd_var.get())
        self._refreshed_var.set("copied to clipboard")
        self._copy_feedback_until = time.time() + 2.0
        self.after(2000, lambda: self._refreshed_var.set(
            f"scanned {time.strftime('%H:%M:%S')}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Show train/eval completion status for every "
                    "dataset x target-mode x model-type combination and for "
                    "every baseline x dataset pair"
    )
    parser.add_argument("--dataset", default=None, choices=list(DATASETS),
                        help="Preselect the first row of this dataset "
                             "(optional — every dataset is always listed)")
    parser.add_argument("--models-root", default="data/models",
                        help="Root of the checkpoint tree "
                             "(default: data/models)")
    parser.add_argument("--results-root", default="data/results",
                        help="Root of the evaluation-results tree "
                             "(default: data/results)")
    parser.add_argument("--baseline-root", default="baseline",
                        help="Root of the baseline sub-projects "
                             "(default: baseline)")
    args = parser.parse_args()

    app = Dashboard(models_root=args.models_root,
                    results_root=args.results_root,
                    baseline_root=args.baseline_root,
                    dataset=args.dataset)
    app.mainloop()


if __name__ == "__main__":
    main()
