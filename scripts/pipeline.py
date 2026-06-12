"""Integrated pipeline GUI for tooltip-annotator."""
from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import scrolledtext, ttk


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')


def _pick_ui_font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    available = set(tkfont.families())
    for name in ("Ubuntu Sans", "Ubuntu", "DejaVu Sans", "Liberation Sans"):
        if name in available:
            return (name, size, weight)
    return ("TkDefaultFont", size, weight)


def _pick_symbol_font(size: int) -> tuple[str, int] | None:
    """Font with broad Unicode symbol coverage. Returns None if unavailable."""
    available = set(tkfont.families())
    for name in ("DejaVu Sans", "Noto Sans", "Liberation Sans"):
        if name in available:
            return (name, size)
    return None


def _pick_log_font(size: int) -> tuple[str, int]:
    available = set(tkfont.families())
    for name in ("Ubuntu Sans Mono", "Ubuntu Mono", "DejaVu Sans Mono", "Liberation Mono"):
        if name in available:
            return (name, size)
    return ("TkFixedFont", size)


ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
DATA = ROOT / "data"
TEMP = ROOT / "temp"

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI", ".MKV", ".MOV"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MASK_EXTS = {".png"}
JSON_EXTS = {".json"}
SPLITS = ("train", "val", "test")


def _count(directory: Path, extensions: set[str]) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix in extensions)


STEPS: list[dict] = [
    {
        "id": "generate_progressive",
        "label": "1. Generate Progressive",
        "desc": "Convert source videos to progressive format",
        "script": "generate_progressive",
        "editor": False,
    },
    {
        "id": "generate_dataset",
        "label": "2. Generate Dataset",
        "desc": "Extract video frames into train/val/test images",
        "script": "generate_dataset",
        "editor": False,
    },
    {
        "id": "download_model",
        "label": "3. Download Model",
        "desc": "Download MONAI segmentation model from HuggingFace",
        "script": "download_model",
        "editor": False,
    },
    {
        "id": "generate_segmentation",
        "label": "4. Generate Segmentation",
        "desc": "Generate binary segmentation mask per image",
        "script": "generate_segmentation",
        "editor": False,
    },
    {
        "id": "generate_annotation",
        "label": "5. Generate Annotation",
        "desc": "Generate bbox/tip annotation JSON from masks",
        "script": "generate_annotation",
        "editor": False,
    },
    {
        "id": "annotation_editor",
        "label": "6. Annotation Editor",
        "desc": "Manually refine annotations (separate window)",
        "script": "annotation_editor",
        "editor": True,
    },
]


def _step_progress(step_id: str) -> tuple[str, str]:
    """Return (status, detail). status ∈ done | partial | pending | waiting | ready."""
    if step_id == "generate_progressive":
        src = _count(DATA / "video", VIDEO_EXTS)
        dst = _count(DATA / "progressive", VIDEO_EXTS)
        if src == 0:
            return "waiting", "no source videos in data/video"
        if dst >= src:
            return "done", f"{dst}/{src} videos"
        if dst > 0:
            return "partial", f"{dst}/{src} videos"
        return "pending", f"0/{src} videos"

    if step_id == "generate_dataset":
        counts = [_count(DATA / "dataset" / "images" / s, IMAGE_EXTS) for s in SPLITS]
        if sum(counts):
            return "done", "  ".join(f"{s}:{n}" for s, n in zip(SPLITS, counts))
        return "pending", "no images"

    if step_id == "download_model":
        if (TEMP / "models" / "model.pt").exists():
            return "done", "model.pt"
        return "pending", "not downloaded"

    if step_id == "generate_segmentation":
        img = [_count(DATA / "dataset" / "images" / s, IMAGE_EXTS) for s in SPLITS]
        seg = [_count(DATA / "dataset" / "segmentation" / s, MASK_EXTS) for s in SPLITS]
        detail = "  ".join(f"{s}:{n}" for s, n in zip(SPLITS, seg))
        if sum(seg) == 0:
            return "pending", "no masks"
        if sum(img) > 0 and sum(seg) >= sum(img):
            return "done", detail
        return "partial", detail

    if step_id == "generate_annotation":
        seg = [_count(DATA / "dataset" / "segmentation" / s, MASK_EXTS) for s in SPLITS]
        ann = [_count(DATA / "dataset" / "annotation" / s, JSON_EXTS) for s in SPLITS]
        detail = "  ".join(f"{s}:{n}" for s, n in zip(SPLITS, ann))
        if sum(ann) == 0:
            return "pending", "no annotations"
        if sum(seg) > 0 and sum(ann) >= sum(seg):
            return "done", detail
        return "partial", detail

    if step_id == "annotation_editor":
        total = sum(_count(DATA / "dataset" / "annotation" / s, JSON_EXTS) for s in SPLITS)
        if total:
            return "ready", f"{total} annotations"
        return "waiting", "run step 5 first"

    return "unknown", ""


_STATUS_UNICODE: dict[str, tuple[str, str]] = {
    "done":    ("✓", "#2e7d32"),
    "partial": ("~", "#e65100"),
    "pending": ("○", "#757575"),
    "waiting": ("–", "#9e9e9e"),
    "ready":   ("✓", "#1565c0"),
    "running": ("▶", "#1976d2"),
    "unknown": ("?", "#616161"),
}

_STATUS_ASCII: dict[str, tuple[str, str]] = {
    "done":    ("+", "#2e7d32"),
    "partial": ("~", "#e65100"),
    "pending": (".", "#757575"),
    "waiting": ("-", "#9e9e9e"),
    "ready":   ("+", "#1565c0"),
    "running": (">", "#1976d2"),
    "unknown": ("?", "#616161"),
}


class PipelineApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("tooltip-annotator Pipeline")
        self.root.minsize(960, 620)

        ui_font = tkfont.Font(family=_pick_ui_font(12)[0], size=12)
        self.root.option_add("*Font", ui_font)

        sym = _pick_symbol_font(13)
        self._sym_font: tuple = sym if sym is not None else _pick_ui_font(13)
        self._status_style = _STATUS_UNICODE if sym is not None else _STATUS_ASCII

        self._log_queue: queue.Queue[str | None] = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._running_idx: int | None = None
        self._build_ui()
        self._refresh()
        self._poll()

    def _build_ui(self) -> None:
        steps_frame = ttk.LabelFrame(self.root, text="Pipeline Steps", padding=10)
        steps_frame.pack(fill="x", padx=12, pady=(10, 2))

        self._rows: list[dict] = []
        for i, step in enumerate(STEPS):
            if i > 0:
                ttk.Separator(steps_frame, orient="horizontal").pack(fill="x", pady=(2, 0))

            outer = ttk.Frame(steps_frame)
            outer.pack(fill="x", pady=(4, 2))

            # --- line 1: status icon | label | desc | Run button ---
            line1 = ttk.Frame(outer)
            line1.pack(fill="x")

            status_lbl = ttk.Label(line1, text=self._status_style["pending"][0],
                                   width=3, anchor="center", font=self._sym_font)
            status_lbl.pack(side="left", padx=(0, 6))

            ttk.Label(
                line1, text=step["label"], width=26, anchor="w",
                font=_pick_ui_font(12, "bold"),
            ).pack(side="left")

            detail_lbl = ttk.Label(line1, text="", anchor="w",
                                   font=_pick_ui_font(11), foreground="#1565c0")
            detail_lbl.pack(side="left", fill="x", expand=True)

            btn_text = "Launch" if step["editor"] else "Run"
            btn = ttk.Button(line1, text=btn_text, width=8,
                             command=lambda idx=i: self._run(idx))
            btn.pack(side="right", padx=4)

            # --- line 2: indented desc ---
            line2 = ttk.Frame(outer)
            line2.pack(fill="x")

            ttk.Label(line2, width=3).pack(side="left", padx=(0, 6))
            ttk.Label(line2, text=step["desc"], anchor="w",
                      font=_pick_ui_font(10), foreground="#555555").pack(
                side="left", fill="x", expand=True
            )

            self._rows.append({"status": status_lbl, "detail": detail_lbl, "btn": btn})

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=12, pady=6)
        ttk.Button(ctrl, text="Refresh", command=self._refresh).pack(side="left", padx=4)
        self._stop_btn = ttk.Button(ctrl, text="Stop", command=self._stop, state="disabled")
        self._stop_btn.pack(side="left", padx=4)
        ttk.Button(ctrl, text="Clear Log", command=self._clear_log).pack(side="left", padx=4)

        log_frame = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        log_font = _pick_log_font(11)
        self._log = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            font=log_font,
            state="disabled",
            bg="#1e1e1e",
            fg="#d4d4d4",
        )
        self._log.pack(fill="both", expand=True)

    def _refresh(self) -> None:
        if self._running_idx is not None:
            return
        for i, step in enumerate(STEPS):
            status, detail = _step_progress(step["id"])
            icon, color = self._status_style.get(status, ("?", "#616161"))
            self._rows[i]["status"].config(text=icon, foreground=color)
            self._rows[i]["detail"].config(text=detail)

    def _set_running(self, idx: int | None) -> None:
        self._running_idx = idx
        for row in self._rows:
            row["btn"].config(state="disabled" if idx is not None else "normal")
        self._stop_btn.config(state="normal" if idx is not None else "disabled")
        if idx is not None:
            icon, color = self._status_style["running"]
            self._rows[idx]["status"].config(text=icon, foreground=color)
            self._rows[idx]["detail"].config(text="running...")

    def _run(self, idx: int) -> None:
        step = STEPS[idx]
        script = BIN / step["script"]
        self._append_log(f"\n{'=' * 60}\n▶ {step['label']}\n{'=' * 60}\n")
        self._set_running(idx)

        if step["editor"]:
            try:
                subprocess.Popen([str(script)], cwd=str(ROOT))
                self._log_queue.put("annotation_editor launched\n")
            except Exception as exc:
                self._log_queue.put(f"✗ launch failed: {exc}\n")
            self._log_queue.put(None)
            return

        def _worker() -> None:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            try:
                proc = subprocess.Popen(
                    [str(script)],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                self._proc = proc
                for line in proc.stdout:
                    self._log_queue.put(line)
                proc.wait()
                code = proc.returncode
                outcome = "✓ done" if code == 0 else f"✗ error (exit {code})"
                self._log_queue.put(f"\n{'=' * 60}\n{outcome}\n{'=' * 60}\n")
            except Exception as exc:
                self._log_queue.put(f"\n✗ launch failed: {exc}\n")
            finally:
                self._proc = None
                self._log_queue.put(None)

        threading.Thread(target=_worker, daemon=True).start()

    def _stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._append_log("\n⚠ stop requested\n")

    def _poll(self) -> None:
        try:
            while True:
                item = self._log_queue.get_nowait()
                if item is None:
                    self._set_running(None)
                    self._refresh()
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _append_log(self, text: str) -> None:
        text = _ANSI_RE.sub('', text)
        self._log.config(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self) -> None:
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")


def main() -> None:
    root = tk.Tk()
    PipelineApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
