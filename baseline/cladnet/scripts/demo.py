#!/usr/bin/env python3
"""GUI demo for the CLAD-Net surgical-tool baseline.

Plays a laparoscopic video -- or a directory of extracted dataset frames --
through a trained CLAD-Net checkpoint (`baseline/cladnet/data/model/<dataset>/model.pt`,
the alphabetically first one by default) and draws what it predicts on every
frame:

  tool   the instrument's bounding box
  tip    a 10 x 10 px box on the instrument tip, marked with a cross at its
         centre, which is the coordinate the tip metrics are computed from

Sources
-------
The "Source" dropdown is filled by scanning the two places this repository
keeps footage (see common/sources.py):

  <tooltip-annotator>/data/dataset-src/<dataset>/*.mp4   original videos, no labels
  data/dataset/<dataset>/images/<split>/                 736x480 frames + annotations

"Open File..." takes any other video from disk. Frames from a dataset
directory also carry ground truth, so "Show GT" overlays the annotated boxes
and tips in white next to the coloured predictions.

This is a viewer, not an evaluation -- nothing here computes AP or hit-rate.
Use scripts/eval-model.py for that.

Controls
--------
  Source / Open File...  choose what to play
  Play / Pause           wall-clock paced playback (frames are dropped, never
                         reordered, when inference cannot keep up)
  <- ->                  step one frame
  Seek bar               jump anywhere in the source
  Conf / IoU             detection confidence threshold and NMS IoU threshold
  Show GT                overlay ground truth (dataset frame sources only)

Usage
-----
    ./baseline/cladnet/run demo
    ./baseline/cladnet/run demo --device cpu
"""

import argparse
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

# NOTE: deliberately not a top-level import. `common.*` needs baseline/cladnet
# on sys.path, which only exists after the insert() below runs. Nesting the
# imports under `if True:` keeps an editor's "organize imports" from hoisting
# them back above the sys.path fix-up.
if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.draw import CLASS_COLORS, draw_detections, draw_ground_truth
    from common.inference import (DEFAULT_CONF, DEFAULT_IOU, Detector,
                                  default_model_path)
    from common.sources import (FRAME_H, FRAME_W, SourceSpec, default_frames_root,
                                default_videos_root, discover_sources, open_source)

# On-screen panel size: the sources already hand back 736 x 480, so this is 1:1.
PANEL_W, PANEL_H = FRAME_W, FRAME_H


def _to_photo(frame_rgb: np.ndarray) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(Image.fromarray(frame_rgb))


class CladNetDemoApp(tk.Tk):
    def __init__(self, weights: str, device: str | None,
                 videos_root: str, frames_root: str):
        super().__init__()
        self.title("CLAD-Net — Surgical Tool & Tip Detection Demo")
        self.resizable(True, True)

        self._weights = weights
        self._detector: Detector | None = None
        self._source = None
        self._last_frame: np.ndarray | None = None
        self._last_index = 0
        self._photo: ImageTk.PhotoImage | None = None

        self._playing = False
        self._play_after_id: str | None = None
        self._play_start_wall = 0.0
        self._play_start_index = 0
        self._seek_after_id: str | None = None

        self._specs = discover_sources(videos_root, frames_root)
        self._detector = self._load_detector(weights, device)

        self._build_ui()

        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

        self.bind("<Left>", lambda _: self._step(-1))
        self.bind("<Right>", lambda _: self._step(+1))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if self._specs:
            self._open_spec(self._specs[0])
        else:
            self._set_info(
                f"No sources found under:\n  {videos_root}\n  {frames_root}\n"
                'Use "Open File..." to pick a video manually.')

    # ── Model ────────────────────────────────────────────────────────────

    def _load_detector(self, weights: str, device: str | None) -> Detector | None:
        if not os.path.exists(weights):
            print(f"WARNING: checkpoint not found: {weights}\n"
                  "         train one first with scripts/train-model.py")
            return None
        try:
            detector = Detector(weights, device=device)
            print(f"Model loaded: {weights}  [{detector.device}]  "
                  f"trained on {detector.dataset}, epoch {detector.epoch}")
            return detector
        except Exception as exc:
            print(f"WARNING: could not load {weights} — {exc}")
            return None

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        # ── Row 1: source + transport ──────────────────────────────────────
        ctrl = tk.Frame(self, pady=6, padx=8)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="Source:").pack(side=tk.LEFT)
        self._source_var = tk.StringVar()
        self._source_cb = ttk.Combobox(
            ctrl, textvariable=self._source_var, width=32, state="readonly",
            values=[spec.label for spec in self._specs])
        self._source_cb.pack(side=tk.LEFT, padx=(2, 8))
        self._source_cb.bind("<<ComboboxSelected>>", self._on_source_change)

        tk.Button(ctrl, text="Open File...",
                  command=self._open_file).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(ctrl, text="<-", width=3,
                  command=lambda: self._step(-1)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="->", width=3,
                  command=lambda: self._step(+1)).pack(side=tk.LEFT, padx=(0, 8))

        self._play_btn = tk.Button(ctrl, text="Play", width=6,
                                   command=self._play, state=tk.DISABLED)
        self._play_btn.pack(side=tk.LEFT, padx=(0, 2))
        self._pause_btn = tk.Button(ctrl, text="Pause", width=6,
                                    command=self._pause, state=tk.DISABLED)
        self._pause_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._frame_lbl = tk.Label(ctrl, text="— / —", width=16, anchor="w")
        self._frame_lbl.pack(side=tk.LEFT)

        tk.Label(ctrl, text="<- -> : step one frame", fg="gray").pack(side=tk.RIGHT)

        # ── Row 2: detection parameters ────────────────────────────────────
        param = tk.Frame(self, pady=4, padx=8)
        param.pack(fill=tk.X)

        tk.Label(param, text="Conf:").pack(side=tk.LEFT)
        self._conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        ttk.Scale(param, from_=0.05, to=0.95, orient=tk.HORIZONTAL,
                  variable=self._conf_var, length=120,
                  command=lambda _: self._on_param_change()
                  ).pack(side=tk.LEFT, padx=(4, 2))
        self._conf_lbl = tk.Label(param, text=f"{DEFAULT_CONF:.2f}", width=5)
        self._conf_lbl.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(param, text="IoU:").pack(side=tk.LEFT)
        self._iou_var = tk.DoubleVar(value=DEFAULT_IOU)
        ttk.Scale(param, from_=0.1, to=0.95, orient=tk.HORIZONTAL,
                  variable=self._iou_var, length=120,
                  command=lambda _: self._on_param_change()
                  ).pack(side=tk.LEFT, padx=(4, 2))
        self._iou_lbl = tk.Label(param, text=f"{DEFAULT_IOU:.2f}", width=5)
        self._iou_lbl.pack(side=tk.LEFT, padx=(0, 12))

        self._show_gt_var = tk.BooleanVar(value=True)
        tk.Checkbutton(param, text="Show GT", variable=self._show_gt_var,
                       command=self._on_param_change).pack(side=tk.LEFT, padx=(0, 12))

        self._status_var = tk.StringVar()
        self._status_lbl = tk.Label(param, textvariable=self._status_var,
                                    font=("Monospace", 9), anchor="w")
        self._status_lbl.pack(side=tk.LEFT)
        self._refresh_status()

        # ── Row 3: class legend ────────────────────────────────────────────
        # A Tk Label renders one colour, so the legend is one small Label per
        # class rather than a single line of text.
        legend = tk.Frame(self, padx=8)
        legend.pack(fill=tk.X)
        tk.Label(legend, text="classes:").pack(side=tk.LEFT, padx=(0, 4))
        for name, color in CLASS_COLORS.items():
            tk.Label(legend, text=f" {name} ", bg=color, fg="black",
                     font=("Monospace", 8)).pack(side=tk.LEFT, padx=1)
        tk.Label(legend, text="   white box + circle = ground truth (bbox / tip)",
                 fg="gray").pack(side=tk.LEFT)

        # ── Row 4: frame panel ─────────────────────────────────────────────
        panel = tk.LabelFrame(self, text="Prediction overlay", padx=4, pady=4)
        panel.pack(padx=8, pady=4)
        self._lbl_frame = tk.Label(panel, width=PANEL_W, height=PANEL_H, bg="#1a1a1a")
        self._lbl_frame.pack()

        # A Label sizes itself in characters/lines until an image is assigned;
        # without this placeholder the empty panel would balloon the window.
        self._photo = _to_photo(np.full((PANEL_H, PANEL_W, 3), 26, dtype=np.uint8))
        self._lbl_frame.config(image=self._photo)

        # ── Row 5: per-frame info ──────────────────────────────────────────
        info = tk.Frame(self, padx=8, pady=2)
        info.pack(fill=tk.BOTH, expand=True)
        self._info_text = tk.Text(info, height=5, wrap=tk.WORD,
                                  font=("Monospace", 9), relief=tk.FLAT,
                                  bg=self.cget("bg"), state=tk.DISABLED)
        self._info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Row 6: seek bar ────────────────────────────────────────────────
        seek = tk.Frame(self, padx=8, pady=2)
        seek.pack(fill=tk.X)
        tk.Label(seek, text="0", width=7, anchor="e").pack(side=tk.LEFT)
        self._seek_var = tk.DoubleVar(value=0)
        self._seekbar = ttk.Scale(seek, from_=0, to=1, orient=tk.HORIZONTAL,
                                  variable=self._seek_var,
                                  command=self._on_seek, state=tk.DISABLED)
        self._seekbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._seek_end_lbl = tk.Label(seek, text="—", width=7, anchor="w")
        self._seek_end_lbl.pack(side=tk.LEFT)

    def _set_info(self, text: str):
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert(tk.END, text)
        self._info_text.config(state=tk.DISABLED)

    def _refresh_status(self):
        if self._detector is None:
            self._status_var.set(f"model NOT loaded: {os.path.basename(self._weights)}")
            self._status_lbl.config(fg="#aa0000")
            return
        metrics = self._detector.metrics or {}
        suffix = ""
        if metrics.get("map50") is not None:
            suffix = f"  val mAP@0.5 {metrics['map50']:.3f}"
        self._status_var.set(
            f"{os.path.basename(self._weights)}  [{self._detector.device}]  "
            f"{self._detector.dataset} ep{self._detector.epoch}{suffix}")
        self._status_lbl.config(fg="#005500")

    # ── Source handling ──────────────────────────────────────────────────

    def _on_source_change(self, _=None):
        label = self._source_var.get()
        for spec in self._specs:
            if spec.label == label:
                self._open_spec(spec)
                return

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")])
        if path:
            self._open_spec(SourceSpec(os.path.basename(path), "video", path))

    def _open_spec(self, spec: SourceSpec):
        self._pause()
        try:
            source = open_source(spec)
        except OSError as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        if self._source is not None:
            self._source.close()
        self._source = source
        self._source_var.set(spec.label)

        self._seekbar.config(state=tk.NORMAL, to=max(1, source.frame_count - 1))
        self._seek_end_lbl.config(text=str(max(0, source.frame_count - 1)))
        self._play_btn.config(state=tk.NORMAL)
        self._goto(0)

    # ── Frame rendering ──────────────────────────────────────────────────

    def _goto(self, index: int):
        if self._source is None:
            return
        index = max(0, min(index, self._source.frame_count - 1))
        frame = self._source.read(index)
        if frame is None:
            return
        self._last_frame = frame
        self._last_index = index
        self._update_frame_label()
        self._render(frame, index)

    def _rerender(self):
        """Redraw the current frame after a parameter change, without moving."""
        if self._last_frame is not None:
            self._render(self._last_frame, self._last_index)

    def _render(self, frame: np.ndarray, index: int):
        if self._detector is None:
            self._photo = _to_photo(frame)
            self._lbl_frame.config(image=self._photo)
            self._set_info(
                f"{self._source.frame_name(index)}\n"
                "no checkpoint loaded — showing the raw frame.\n"
                "Train one with scripts/train-model.py, then restart the demo.")
            return

        detections, infer_ms = self._detector.detect(
            frame, conf=self._conf_var.get(), iou=self._iou_var.get())

        display = draw_detections(frame.copy(), detections, self._detector.class_names)

        annotations = self._source.ground_truth(index)
        if self._show_gt_var.get() and annotations:
            display = draw_ground_truth(display, annotations)

        self._photo = _to_photo(display)
        self._lbl_frame.config(image=self._photo)

        names = self._detector.class_names
        tools = [d for d in detections if names[int(d[5])] == "tool"]
        tips = [d for d in detections if names[int(d[5])] == "tip"]
        tip_text = ", ".join(f"({(d[0] + d[2]) / 2:.0f}, {(d[1] + d[3]) / 2:.0f}) {d[4]:.2f}"
                             for d in tips) or "(none)"
        if annotations is None:
            gt_line = "ground truth: none (unlabelled source)"
        else:
            gt_line = f"ground truth: {len(annotations)} tool(s) annotated"

        self._set_info(
            f"{self._source.frame_name(index)}\n"
            f"inference: {infer_ms:.1f} ms  ({1000.0 / max(infer_ms, 1e-6):.1f} FPS)\n"
            f"tool boxes: {len(tools)}   tip boxes: {len(tips)}\n"
            f"predicted tips: {tip_text}\n"
            f"{gt_line}")

    def _update_frame_label(self):
        total = max(0, self._source.frame_count - 1)
        self._frame_lbl.config(text=f"{self._last_index} / {total}")
        self._seek_var.set(self._last_index)

    # ── Controls ─────────────────────────────────────────────────────────

    def _on_param_change(self):
        self._conf_lbl.config(text=f"{self._conf_var.get():.2f}")
        self._iou_lbl.config(text=f"{self._iou_var.get():.2f}")
        if not self._playing:
            self._rerender()

    def _step(self, delta: int):
        if self._source is None:
            return
        self._pause()
        self._goto(self._last_index + delta)

    def _on_seek(self, _=None):
        if self._seek_after_id is not None:
            self.after_cancel(self._seek_after_id)
        self._seek_after_id = self.after(150, self._apply_seek)

    def _apply_seek(self):
        self._seek_after_id = None
        if self._source is None:
            return
        self._pause()
        self._goto(int(round(self._seek_var.get())))

    # ── Playback ─────────────────────────────────────────────────────────

    def _play(self):
        if self._source is None or self._playing:
            return
        self._playing = True
        self._play_btn.config(state=tk.DISABLED)
        self._pause_btn.config(state=tk.NORMAL)
        self._play_start_wall = time.perf_counter()
        self._play_start_index = self._last_index
        self._play_after_id = self.after(1, self._play_tick)

    def _pause(self):
        self._playing = False
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None
        self._play_btn.config(state=tk.NORMAL if self._source else tk.DISABLED)
        self._pause_btn.config(state=tk.DISABLED)

    def _play_tick(self):
        if not self._playing or self._source is None:
            return

        # Wall-clock pacing: the target frame comes from elapsed time, so when
        # inference is slower than the source's frame rate playback drops
        # frames instead of falling behind. It never goes backwards.
        elapsed = time.perf_counter() - self._play_start_wall
        target = self._play_start_index + int(elapsed * self._source.fps)
        target = max(target, self._last_index + 1)

        if target >= self._source.frame_count:
            self._pause()
            return

        self._goto(target)
        self._play_after_id = self.after(1, self._play_tick)

    def _on_close(self):
        self._pause()
        if self._source is not None:
            self._source.close()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="GUI demo for the CLAD-Net surgical tool + tip detection baseline")
    parser.add_argument("--weights", default=default_model_path(),
                        help="path to the checkpoint; with several datasets trained, "
                             "pass data/model/<dataset>/model.pt explicitly "
                             f"(default: {default_model_path()})")
    parser.add_argument("--device", default=None,
                        help="torch device, e.g. cuda:0 or cpu (default: cuda if available)")
    parser.add_argument("--videos-root", default=default_videos_root(),
                        help="directory of <dataset>/<video>.mp4 "
                             f"(default: {default_videos_root()})")
    parser.add_argument("--frames-root", default=default_frames_root(),
                        help="directory of <dataset>/images/<split>/ "
                             f"(default: {default_frames_root()})")
    args = parser.parse_args()

    app = CladNetDemoApp(weights=args.weights, device=args.device,
                         videos_root=args.videos_root, frames_root=args.frames_root)
    app.mainloop()


if __name__ == "__main__":
    main()
