#!/usr/bin/env python3
"""Experiment dashboard: train/eval completion status per experiment combination.

Every combination of dataset x target-mode x model-type is one row; the Train
and Eval columns show how far that combination has progressed. Selecting a
cell shows the exact command needed to carry out that task, ready to copy.

Status is derived purely from files on disk, so the dashboard never has to
load a model or a dataset:

  Train  data/models/<dataset>/<target-mode>/<model-type>/
           best.pt / last.pt          — checkpoints written by train-model
           train-status.json          — completed_epochs / epochs, best val loss
           metric.csv                 — one row per completed epoch
  Eval   data/results/<dataset>/<target-mode>/<model-type>/
           summary.json               — metrics + run parameters
           per_tip.csv                — one row per GT tip

Usage:
    uv run python scripts/dashboard.py [--dataset erop]

Navigation:
    click a Train/Eval cell   select that task
    up / down                 change combination
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

# Re-scanning is cheap (a handful of stat() calls per row), so the table can
# follow a training run that is writing checkpoints in another terminal.
AUTO_REFRESH_MS = 5000

# Column ids of the two task columns, in Treeview display order.
TASK_COLUMNS = {"#4": "Train", "#5": "Eval"}

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
        v = summary.get(key)
        return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

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


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class Dashboard(tk.Tk):
    def __init__(self, models_root: str, results_root: str, dataset: str | None):
        super().__init__()
        self.title("Dashboard - train / eval status")
        self.minsize(940, 620)

        self._models_root = models_root
        self._results_root = results_root

        # One row per experiment combination, dataset-major.
        self._combos = [
            {"dataset": ds, "target_mode": tm, "model_type": mt}
            for ds in DATASETS for tm in TARGET_MODES for mt in MODEL_REGISTRY
        ]
        self._scans: list[dict] = []
        self._task = "Train"
        # Wall-clock deadline until which the status label keeps showing the
        # copy confirmation instead of the scan timestamp.
        self._copy_feedback_until = 0.0

        self._build_ui()
        self.refresh()

        initial = 0
        if dataset is not None:
            initial = next((i for i, c in enumerate(self._combos)
                            if c["dataset"] == dataset), 0)
        self._select(initial)

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

        # ── Status table ─────────────────────────────────────────────────
        table = tk.Frame(self, padx=8)
        table.pack(fill=tk.X)

        columns = ("dataset", "target", "model", "train", "eval")
        self._tree = ttk.Treeview(table, columns=columns, show="headings",
                                  height=len(self._combos), selectmode="browse")
        for col, text, width, anchor in (
            ("dataset", "Dataset", 100, tk.W),
            ("target", "Target mode", 120, tk.W),
            ("model", "Model type", 110, tk.W),
            ("train", "Train", 190, tk.W),
            ("eval", "Eval", 190, tk.W),
        ):
            self._tree.heading(col, text=text)
            self._tree.column(col, width=width, anchor=anchor,
                              stretch=(col in ("train", "eval")))
        self._tree.pack(fill=tk.X)

        # Row colour summarises the pair of statuses at a glance.
        self._tree.tag_configure("complete", foreground="#1a7f2e")
        self._tree.tag_configure("partial", foreground="#b35c00")
        self._tree.tag_configure("none", foreground="#808080")

        for combo in self._combos:
            self._tree.insert("", tk.END, values=(
                combo["dataset"], combo["target_mode"], combo["model_type"], "", ""))

        self._tree.bind("<Button-1>", self._on_click)
        self._tree.bind("<<TreeviewSelect>>", lambda _: self._render_detail())
        self._tree.bind("<Left>", self._on_tree_arrow)
        self._tree.bind("<Right>", self._on_tree_arrow)

        # ── Selected cell detail ─────────────────────────────────────────
        self._detail_frame = tk.LabelFrame(self, text="Selected", padx=6, pady=4)
        self._detail_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        # CHAR wrap so a long --models-root path is never clipped out of sight.
        self._detail = tk.Text(self._detail_frame, height=10, wrap=tk.CHAR,
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

    # ── Selection ────────────────────────────────────────────────────────

    def _rows(self) -> list[str]:
        return list(self._tree.get_children(""))

    def _selected_index(self) -> int | None:
        sel = self._tree.selection()
        return self._rows().index(sel[0]) if sel else None

    def _select(self, index: int):
        rows = self._rows()
        index = max(0, min(index, len(rows) - 1))
        self._tree.selection_set(rows[index])
        self._tree.focus(rows[index])

    def _set_task(self, task: str):
        self._task = task
        self._render_detail()

    def _on_tree_arrow(self, event):
        """Left/Right inside the table switch task instead of collapsing rows."""
        self._set_task("Train" if event.keysym == "Left" else "Eval")
        return "break"

    def _on_click(self, event):
        """Clicking a Train/Eval cell selects that task for the clicked row."""
        if self._tree.identify("region", event.x, event.y) != "cell":
            return
        task = TASK_COLUMNS.get(self._tree.identify_column(event.x))
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
        """Re-scan every combination's directories and repaint the table."""
        self._scans = [
            {
                "train": scan_train(combo, self._models_root),
                "eval": scan_eval(combo, self._results_root),
            }
            for combo in self._combos
        ]

        for row, combo, scan in zip(self._rows(), self._combos, self._scans):
            train, evaluation = scan["train"], scan["eval"]
            if train["state"] in ("done", "weights") and evaluation["state"] == "done":
                tag = "complete"
            elif train["state"] == "none" and evaluation["state"] == "none":
                tag = "none"
            else:
                tag = "partial"
            self._tree.item(row, tags=(tag,), values=(
                combo["dataset"], combo["target_mode"], combo["model_type"],
                train["label"], evaluation["label"]))

        if time.time() >= self._copy_feedback_until:
            self._refreshed_var.set(f"scanned {time.strftime('%H:%M:%S')}")
        self._render_detail()

    def _render_detail(self):
        index = self._selected_index()
        if index is None:
            return
        combo, scan = self._combos[index], self._scans[index]
        train, evaluation = scan["train"], scan["eval"]

        active = train if self._task == "Train" else evaluation
        self._detail_frame.config(
            text=f"  {self._task}  :  {combo['dataset']} / "
                 f"{combo['target_mode']} / {combo['model_type']}   "
                 f"[{active['label']}]  ")

        lines = "\n".join(f"{key:<20}: {value}" for key, value in active["detail"])
        self._detail.config(state=tk.NORMAL)
        self._detail.delete("1.0", tk.END)
        self._detail.insert("1.0", lines)
        self._detail.config(state=tk.DISABLED)

        self._cmd_var.set(train_command(combo, train) if self._task == "Train"
                          else eval_command(combo))
        self._note_var.set(task_note(self._task, combo, train, evaluation))

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
                    "dataset x target-mode x model-type combination"
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
    args = parser.parse_args()

    app = Dashboard(models_root=args.models_root,
                    results_root=args.results_root,
                    dataset=args.dataset)
    app.mainloop()


if __name__ == "__main__":
    main()
