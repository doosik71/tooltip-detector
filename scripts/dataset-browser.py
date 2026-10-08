#!/usr/bin/env python3
"""Interactive dataset browser for SurgicalToolDataset.

Shows one image panel whose content is chosen by the tabs above it:
the original image, the original with annotations, or the original with the
distance-based heatmap overlay.
Navigate with ← → arrow keys, buttons, or by clicking a file in the list on
the left (which also shows the labelled tool count per frame). The "Dataset"
dropdown switches between data/dataset/<dataset-name>/ directories at runtime.

Usage:
    uv run python scripts/dataset-browser.py [--dataset cholec80] [--split SPLIT]
"""
import argparse
import bisect
import json
import os
import sys

import numpy as np
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttd.dataset import DATASETS, SurgicalToolDataset


SPLITS = ["train", "val", "test"]
PANEL_W = 552   # 736 × 0.75
PANEL_H = 360   # 480 × 0.75
# View tabs, in display order
VIEWS = ["Original", "Original + Annotations", "Original + Distance Heatmap"]
# Tool-count filter choices for the file list
TOOL_FILTERS = ["All", "0", "1", "2", "3", "4", "5+"]
INFO_LINES = 3      # visible lines in the info box (more scroll)
INFO_FONT = ("Monospace", 9)
LIST_CHUNK = 2000   # file-list rows inserted per idle callback

# Colors cycled per tool index
_TOOL_COLORS = ["#00FF00", "#FF8800", "#00AAFF", "#FF00FF", "#FFFF00",
                "#FF4444", "#44FFFF", "#FF44FF"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _colorize(target: np.ndarray) -> np.ndarray:
    """Float32 [0,1] → RGB uint8 using a hot colormap.

    Gradient: 0 → black, 1/3 → red, 2/3 → yellow, 1 → white.
    """
    r = np.clip(target * 3.0,       0.0, 1.0)
    g = np.clip(target * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(target * 3.0 - 2.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)


def _blend_heatmap(image: np.ndarray, target: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Overlay colorized heatmap on the image where target > 0."""
    colored = _colorize(target)
    mask = (target > 0)[..., np.newaxis]
    return np.where(mask,
                    (image * (1.0 - alpha) + colored * alpha).astype(np.uint8),
                    image)


def _draw_annotations(image: np.ndarray, annotations: list) -> np.ndarray:
    """Draw bounding boxes and tip cross-hairs onto the image."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for i, ann in enumerate(annotations):
        color = _TOOL_COLORS[i % len(_TOOL_COLORS)]
        b = ann["bbox"]
        draw.rectangle(
            [b["x"], b["y"], b["x"] + b["width"] - 1, b["y"] + b["height"] - 1],
            outline=color, width=2,
        )
        tx, ty = ann["tip"]["x"], ann["tip"]["y"]
        r = 5
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=color, outline="white")
        draw.line([tx - 12, ty, tx + 12, ty], fill="white", width=1)
        draw.line([tx, ty - 12, tx, ty + 12], fill="white", width=1)
    return np.array(pil)


def _to_photo(arr: np.ndarray) -> ImageTk.PhotoImage:
    pil = Image.fromarray(arr).resize((PANEL_W, PANEL_H), Image.BILINEAR)
    return ImageTk.PhotoImage(pil)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class DatasetBrowser(tk.Tk):
    def __init__(self, data_root: str, dataset_name: str, split: str):
        super().__init__()
        self.title("Dataset Browser — SurgicalToolDataset")
        self.resizable(False, False)

        self._data_root = data_root
        self._dataset_name = dataset_name
        self._ds: SurgicalToolDataset | None = None
        self._idx = 0
        # Bumped on every split load so a stale file-list fill stops early
        self._list_gen = 0
        # Per-frame tool counts, filled as the file list reads annotations
        self._counts: list[int] = []
        # Indices currently listed (those passing the Tools filter), ascending
        self._visible: list[int] = []
        # Show the first listed frame once one appears (current one filtered out)
        self._jump_pending = False
        # Hold references so GC does not delete PhotoImages
        self._photo: ImageTk.PhotoImage | None = None
        # Current frame's (image, target, annotations), re-rendered on tab change
        self._frame: tuple[np.ndarray, np.ndarray, list] | None = None

        self._build_ui()
        self._load_split(split)

        self.bind("<Left>",  lambda _: self._navigate(-1))
        self.bind("<Right>", lambda _: self._navigate(+1))
        self.bind("<r>",     lambda _: self._random())

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        # ── Controls row ─────────────────────────────────────────────────
        ctrl = tk.Frame(self, pady=6, padx=8)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="Dataset:").pack(side=tk.LEFT)
        self._dataset_var = tk.StringVar(value=self._dataset_name)
        dataset_cb = ttk.Combobox(ctrl, textvariable=self._dataset_var,
                                  values=list(DATASETS), width=10, state="readonly")
        dataset_cb.pack(side=tk.LEFT, padx=(2, 12))
        dataset_cb.bind("<<ComboboxSelected>>", self._on_dataset_change)

        tk.Label(ctrl, text="Split:").pack(side=tk.LEFT)
        self._split_var = tk.StringVar()
        split_cb = ttk.Combobox(ctrl, textvariable=self._split_var,
                                values=SPLITS, width=6, state="readonly")
        split_cb.pack(side=tk.LEFT, padx=(2, 12))
        split_cb.bind("<<ComboboxSelected>>",
                      lambda _: self._load_split(self._split_var.get()))

        tk.Label(ctrl, text="Tools:").pack(side=tk.LEFT)
        self._tools_var = tk.StringVar(value="All")
        tools_cb = ttk.Combobox(ctrl, textvariable=self._tools_var,
                                values=TOOL_FILTERS, width=4, state="readonly")
        tools_cb.pack(side=tk.LEFT, padx=(2, 12))
        tools_cb.bind("<<ComboboxSelected>>", lambda _: self._fill_file_list())

        tk.Button(ctrl, text="◄", width=3,
                  command=lambda: self._navigate(-1)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="►", width=3,
                  command=lambda: self._navigate(+1)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Rand",
                  command=self._random).pack(side=tk.LEFT, padx=(4, 12))

        self._idx_var = tk.StringVar()
        idx_entry = tk.Entry(ctrl, textvariable=self._idx_var, width=8)
        idx_entry.pack(side=tk.LEFT)
        idx_entry.bind("<Return>", lambda _: self._jump())

        self._total_lbl = tk.Label(ctrl, text="/ —")
        self._total_lbl.pack(side=tk.LEFT, padx=(2, 0))

        tk.Label(ctrl, text="  ← → : navigate   R : random",
                 fg="gray").pack(side=tk.RIGHT)

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        # ── File list (left) ─────────────────────────────────────────────
        self._list_frame = list_frame = tk.LabelFrame(body, text="Files",
                                                      padx=4, pady=4)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0), pady=4)

        self._tree = ttk.Treeview(list_frame, columns=("file", "tools"),
                                  show="headings", selectmode="browse")
        self._tree.heading("file", text="File")
        self._tree.heading("tools", text="Tools")
        self._tree.column("file", width=230, anchor="w")
        self._tree.column("tools", width=50, anchor="e", stretch=False)
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                    command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.Y)
        tree_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self._tree.bind("<<TreeviewSelect>>", self._on_list_select)

        main = tk.Frame(body)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── View tabs + image panel ──────────────────────────────────────
        # The notebook is only a tab strip; its pages are empty and the one
        # shared image label below shows whichever view is selected.
        self._view_tabs = ttk.Notebook(main)
        for view in VIEWS:
            self._view_tabs.add(tk.Frame(self._view_tabs, height=0), text=view)
        self._view_tabs.pack(fill=tk.X, padx=8, pady=(4, 0))
        self._view_tabs.bind("<<NotebookTabChanged>>", lambda _: self._render())

        self._lbl_img = tk.Label(main, width=PANEL_W, height=PANEL_H,
                                 bg="#1a1a1a")
        self._lbl_img.pack(padx=8, pady=(0, 4))

        # ── Info bar ─────────────────────────────────────────────────────
        # Fixed-size box (panel width × INFO_LINES lines) so long tip lists
        # wrap and scroll instead of resizing the window.
        line_h = tkfont.Font(font=INFO_FONT).metrics("linespace")
        info_frame = tk.Frame(main, width=PANEL_W,
                              height=line_h * INFO_LINES + 8)
        info_frame.pack_propagate(False)
        info_frame.pack(padx=8, pady=4)
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL)
        self._info_text = tk.Text(info_frame, font=INFO_FONT, wrap=tk.WORD,
                                  padx=4, pady=2, relief=tk.FLAT,
                                  bg=self.cget("bg"),
                                  yscrollcommand=info_scroll.set)
        info_scroll.config(command=self._info_text.yview)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._info_text.config(state=tk.DISABLED)

        # ── Seek bar ──────────────────────────────────────────────────────
        seek_frame = tk.Frame(main, padx=8, pady=4)
        seek_frame.pack(fill=tk.X)

        self._seek_start_lbl = tk.Label(seek_frame, text="0", width=6, anchor="e")
        self._seek_start_lbl.pack(side=tk.LEFT)

        self._seek_var = tk.DoubleVar(value=0)
        self._seekbar = ttk.Scale(
            seek_frame,
            from_=0, to=1,
            orient=tk.HORIZONTAL,
            variable=self._seek_var,
            command=self._on_seek,
        )
        self._seekbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self._seek_end_lbl = tk.Label(seek_frame, text="—", width=6, anchor="w")
        self._seek_end_lbl.pack(side=tk.LEFT)

        self._seek_after_id: str | None = None

    # ── Dataset operations ───────────────────────────────────────────────

    def _on_dataset_change(self, _=None):
        self._dataset_name = self._dataset_var.get()
        self._load_split(self._split_var.get() or "train")

    def _load_split(self, split: str):
        self._split_var.set(split)
        dataset_root = os.path.join(self._data_root, self._dataset_name)
        self._ds = SurgicalToolDataset(dataset_root, split)
        n = len(self._ds)
        self._total_lbl.config(text=f"/ {n}")
        self._seekbar.config(to=max(1, n - 1))
        self._seek_end_lbl.config(text=str(n - 1))
        self._idx = 0
        self._counts = []
        self._show(0)
        self._fill_file_list()

    def _tool_count(self, idx: int) -> int:
        """Labelled tool count of frame idx, cached in self._counts."""
        if idx < len(self._counts):
            return self._counts[idx]
        with open(self._ds.samples[idx]) as f:
            return len(json.load(f)["annotations"])

    def _passes_filter(self, n_tools: int) -> bool:
        f = self._tools_var.get()
        if f == "All":
            return True
        if f == "5+":
            return n_tools >= 5
        return n_tools == int(f)

    def _fill_file_list(self):
        """Repopulate the file list, reading tool counts in idle-time chunks.

        Only frames passing the Tools filter are listed. If the frame on
        screen is filtered out, the first listed frame is shown instead.
        """
        if self._ds is None:
            return
        self._list_gen += 1
        self._tree.delete(*self._tree.get_children())
        self._visible = []
        self._jump_pending = not self._passes_filter(self._tool_count(self._idx))
        self._append_file_rows(self._list_gen, 0)

    def _append_file_rows(self, gen: int, start: int):
        if gen != self._list_gen or self._ds is None:
            return
        samples = self._ds.samples
        end = min(start + LIST_CHUNK, len(samples))
        for i in range(start, end):
            n_tools = self._tool_count(i)
            if i == len(self._counts):
                self._counts.append(n_tools)
            if not self._passes_filter(n_tools):
                continue
            stem = os.path.splitext(os.path.basename(samples[i]))[0]
            self._tree.insert("", tk.END, iid=str(i), values=(stem, n_tools))
            self._visible.append(i)
        if self._jump_pending and self._visible:
            self._jump_pending = False
            self._show(self._visible[0])
        # The current frame may have been shown before its row existed
        elif start <= self._idx < end:
            self._select_in_list(self._idx)
        done = "" if end == len(samples) else "+"
        self._list_frame.config(
            text=f"Files  ({len(self._visible)}{done} / {len(samples)})")
        if end < len(samples):
            self.after(1, self._append_file_rows, gen, end)

    def _select_in_list(self, idx: int):
        iid = str(idx)
        if self._tree.exists(iid) and self._tree.selection() != (iid,):
            self._tree.selection_set(iid)
            self._tree.see(iid)

    def _on_list_select(self, _=None):
        sel = self._tree.selection()
        if sel and self._ds:
            idx = int(sel[0])
            if idx != self._idx:
                self._show(idx)

    def _navigate(self, delta: int):
        """Step to the previous/next frame among those in the file list."""
        if not self._ds:
            return
        if self._tools_var.get() == "All":
            self._show((self._idx + delta) % len(self._ds))
            return
        vis = self._visible
        if not vis:
            return
        if delta > 0:
            pos = bisect.bisect_right(vis, self._idx)
        else:
            pos = bisect.bisect_left(vis, self._idx) - 1
        self._show(vis[pos % len(vis)])

    def _random(self):
        if not self._ds:
            return
        if self._tools_var.get() == "All":
            self._show(int(np.random.randint(0, len(self._ds))))
        elif self._visible:
            self._show(self._visible[np.random.randint(0, len(self._visible))])

    def _jump(self):
        if not self._ds:
            return
        try:
            idx = int(self._idx_var.get())
            self._show(max(0, min(idx, len(self._ds) - 1)))
        except ValueError:
            pass

    def _on_seek(self, _value=None):
        """Debounced seek-bar handler — defers _show by 150 ms."""
        if self._seek_after_id is not None:
            self.after_cancel(self._seek_after_id)
        self._seek_after_id = self.after(150, self._apply_seek)

    def _apply_seek(self):
        self._seek_after_id = None
        if not self._ds:
            return
        idx = int(round(self._seek_var.get()))
        idx = max(0, min(idx, len(self._ds) - 1))
        if idx != self._idx:
            self._show(idx)

    # ── Render ───────────────────────────────────────────────────────────

    def _show(self, idx: int):
        if self._ds is None:
            return
        self._idx = idx
        self._idx_var.set(str(idx))
        self._seek_var.set(idx)
        self._select_in_list(idx)

        image, target = self._ds[idx]       # ndarray uint8 (H,W,3), float32 (H,W)
        ann_path = self._ds.samples[idx]

        with open(ann_path) as f:
            ann_data = json.load(f)
        annotations = ann_data["annotations"]

        self._frame = (image, target, annotations)
        self._render()

        # Info bar
        stem = os.path.splitext(os.path.basename(ann_path))[0]
        n = len(annotations)
        tips = "  ".join(
            f"tip[{i}]=({a['tip']['x']}, {a['tip']['y']})"
            for i, a in enumerate(annotations)
        )
        heat_range = (f"heatmap: [{target.min():.3f}, {target.max():.3f}]"
                      if target.max() > 0 else "heatmap: all zero (no tool)")
        self._set_info(
            f"File: {stem}   Tools: {n}   {heat_range}\n"
            + (tips if tips else "(no tools in this frame)")
        )

    def _set_info(self, text: str):
        """Replace the read-only info box contents and scroll to the top."""
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert("1.0", text)
        self._info_text.config(state=tk.DISABLED)
        self._info_text.yview_moveto(0)

    def _render(self):
        """Draw the current frame in the view selected by the tabs."""
        if self._frame is None:
            return
        image, target, annotations = self._frame
        view = VIEWS[self._view_tabs.index("current")]
        if view == "Original + Annotations":
            image = _draw_annotations(image, annotations)
        elif view == "Original + Distance Heatmap":
            image = _blend_heatmap(image, target)
        # Keep a reference alive (required for tkinter PhotoImage)
        self._photo = _to_photo(image)
        self._lbl_img.config(image=self._photo)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Browse SurgicalToolDataset with distance-based heatmap overlay"
    )
    parser.add_argument("--dataset", default="cholec80",
                        choices=list(DATASETS),
                        help="Dataset name under --data-root (default: cholec80)")
    parser.add_argument("--data-root", default="data/dataset",
                        help="Root directory containing <dataset>/ subdirectories "
                             "(default: data/dataset)")
    parser.add_argument("--split", default="train", choices=SPLITS,
                        help="Dataset split to open initially")
    args = parser.parse_args()

    app = DatasetBrowser(data_root=args.data_root, dataset_name=args.dataset, split=args.split)
    app.mainloop()


if __name__ == "__main__":
    main()
