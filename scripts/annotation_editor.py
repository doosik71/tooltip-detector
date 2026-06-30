from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None


SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HANDLE_SIZE = 5
TIP_RADIUS = 6
SEGMENTATION_ALPHA = 0.35
MIN_BOX_SIZE = 4


class AnnotationEditor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.images_root = args.images
        self.segmentation_root = args.segmentation
        self.annotation_root = args.annotation

        self.root = tk.Tk()
        self.root.title("Tooltip Annotation Editor")
        self.root.geometry("1600x960")
        self.root.minsize(1100, 700)

        self.split_var = tk.StringVar(value=SPLIT_NAMES[0])
        self.show_segmentation_var = tk.BooleanVar(value=True)
        self.show_annotation_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.images_label_var = tk.StringVar(value="Images (0 / 0)")

        self.current_split = SPLIT_NAMES[0]
        self.image_paths: list[Path] = []
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
        self.preview_box: tuple[float, float, float, float] | None = None
        self.dirty = False
        self.photo_image: tk.PhotoImage | None = None
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0.0
        self.canvas_offset_y = 0.0
        self.canvas_image_width = 0
        self.canvas_image_height = 0
        self.suppress_listbox_event = False

        self._build_ui()
        self._bind_events()
        self._ensure_output_dirs()
        self._load_split(self.current_split)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        top_bar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_bar, text="Split").pack(side=tk.LEFT)
        self.split_combobox = ttk.Combobox(
            top_bar,
            textvariable=self.split_var,
            values=SPLIT_NAMES,
            state="readonly",
            width=8,
        )
        self.split_combobox.pack(side=tk.LEFT, padx=(8, 18))

        self.add_button = ttk.Button(top_bar, text="Add Tool", command=self._enter_add_mode)
        self.add_button.pack(side=tk.LEFT)

        self.delete_button = ttk.Button(top_bar, text="Delete Tool", command=self._delete_selected)
        self.delete_button.pack(side=tk.LEFT, padx=(8, 0))

        self.reload_button = ttk.Button(top_bar, text="Reload", command=self._reload_current_annotation)
        self.reload_button.pack(side=tk.LEFT, padx=(8, 0))

        self.save_button = ttk.Button(top_bar, text="Save", command=self._save_current_annotation)
        self.save_button.pack(side=tk.LEFT, padx=(18, 0))

        self.show_segmentation_checkbutton = ttk.Checkbutton(
            top_bar,
            text="Show Segmentation",
            variable=self.show_segmentation_var,
            command=self._render_scene,
        )
        self.show_segmentation_checkbutton.pack(side=tk.LEFT, padx=(18, 0))

        self.show_annotation_checkbutton = ttk.Checkbutton(
            top_bar,
            text="Show Annotation",
            variable=self.show_annotation_var,
            command=self._render_scene,
        )
        self.show_annotation_checkbutton.pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(top_bar, textvariable=self.status_var).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left_panel = ttk.Frame(body, padding=(0, 0, 10, 0))
        body.add(left_panel, weight=0)

        ttk.Label(left_panel, textvariable=self.images_label_var).pack(anchor=tk.W, pady=(0, 6))

        list_frame = ttk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.image_listbox = tk.Listbox(list_frame, width=38, exportselection=False)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.configure(yscrollcommand=list_scrollbar.set)

        right_panel = ttk.Frame(body)
        body.add(right_panel, weight=1)

        self.canvas = tk.Canvas(right_panel, background="#101417", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        instruction = (
            "Drag bbox interior to move. Drag corner handles to resize. Drag red tip handle to move the tip. "
            "A or Insert adds, D or Delete removes, R reloads, S or Ctrl+S saves, and arrow keys move between images."
        )
        ttk.Label(self.root, text=instruction, padding=(10, 0, 10, 10)).pack(side=tk.BOTTOM, fill=tk.X)

        self._update_buttons()

    def _bind_events(self) -> None:
        self.split_combobox.bind("<<ComboboxSelected>>", self._on_split_changed)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_selected)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", self._on_canvas_resized)
        self.root.bind("<KeyPress-a>", self._on_add_shortcut)
        self.root.bind("<KeyPress-A>", self._on_add_shortcut)
        self.root.bind("<Insert>", self._on_add_shortcut)
        self.root.bind("<KeyPress-d>", self._on_delete_shortcut)
        self.root.bind("<KeyPress-D>", self._on_delete_shortcut)
        self.root.bind("<Control-s>", self._on_save_shortcut)
        self.root.bind("<Control-S>", self._on_save_shortcut)
        self.root.bind("<KeyPress-s>", self._on_save_shortcut)
        self.root.bind("<KeyPress-S>", self._on_save_shortcut)
        self.root.bind("<KeyPress-r>", self._on_reload_shortcut)
        self.root.bind("<KeyPress-R>", self._on_reload_shortcut)
        self.root.bind("<Delete>", self._on_delete_shortcut)
        self.root.bind("<Up>", self._on_previous_image_shortcut)
        self.root.bind("<Left>", self._on_previous_image_shortcut)
        self.root.bind("<Down>", self._on_next_image_shortcut)
        self.root.bind("<Right>", self._on_next_image_shortcut)
        self.root.bind("<Escape>", self._on_escape)

    def _ensure_output_dirs(self) -> None:
        for split_name in SPLIT_NAMES:
            (self.annotation_root / split_name).mkdir(parents=True, exist_ok=True)

    def _on_close(self) -> None:
        if not self._confirm_leave_current_image():
            return
        self.root.destroy()

    def _on_split_changed(self, _event: tk.Event[tk.Misc]) -> None:
        requested_split = self.split_var.get()
        if requested_split == self.current_split:
            return
        if not self._confirm_leave_current_image():
            self.split_var.set(self.current_split)
            return
        self._load_split(requested_split)

    def _on_image_selected(self, _event: tk.Event[tk.Misc]) -> None:
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

    def _on_canvas_resized(self, _event: tk.Event[tk.Misc]) -> None:
        self._render_scene()

    def _on_add_shortcut(self, _event: tk.Event[tk.Misc]) -> str:
        self._enter_add_mode()
        return "break"

    def _on_save_shortcut(self, _event: tk.Event[tk.Misc]) -> str:
        self._save_current_annotation()
        return "break"

    def _on_delete_shortcut(self, _event: tk.Event[tk.Misc]) -> str:
        self._delete_selected()
        return "break"

    def _on_reload_shortcut(self, _event: tk.Event[tk.Misc]) -> str:
        self._reload_current_annotation()
        return "break"

    def _on_previous_image_shortcut(self, _event: tk.Event[tk.Misc]) -> str:
        self._navigate_images(-1)
        return "break"

    def _on_next_image_shortcut(self, _event: tk.Event[tk.Misc]) -> str:
        self._navigate_images(1)
        return "break"

    def _on_escape(self, _event: tk.Event[tk.Misc]) -> str:
        if self.mode == "add":
            self.mode = "select"
            self.preview_box = None
            self.drag_state = None
            self._update_status()
            self._update_buttons()
            self._render_scene()
        return "break"

    def _load_split(self, split_name: str) -> None:
        self.current_split = split_name
        self.split_var.set(split_name)
        split_dir = self.images_root / split_name
        self.image_paths = self._list_images(split_dir)
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

        self.suppress_listbox_event = True
        self.image_listbox.delete(0, tk.END)
        for image_path in self.image_paths:
            self.image_listbox.insert(tk.END, image_path.name)
        self.suppress_listbox_event = False

        if self.image_paths:
            self.image_listbox.selection_set(0)
            self.image_listbox.activate(0)
            self._open_image_at_index(0)
        else:
            self._update_images_label()
            self._render_scene()
            self._update_status("No images found in the selected split.")
        self._update_buttons()

    def _open_image_at_index(self, index: int) -> None:
        self.current_index = index
        self.current_image_path = self.image_paths[index]
        self.current_image_rgb = self._load_rgb_image(self.current_image_path)
        self.image_height, self.image_width = self.current_image_rgb.shape[:2]
        self.current_segmentation_mask = self._load_segmentation_mask(self.current_image_path)
        self.current_annotations = self._load_annotation(self.current_image_path)
        self._update_images_label()
        self.selected_annotation_index = 0 if self.current_annotations else None
        self.preview_box = None
        self.drag_state = None
        self.mode = "select"
        self.dirty = False
        self._update_buttons()
        self._update_status()
        self._render_scene()

    def _load_rgb_image(self, image_path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _load_segmentation_mask(self, image_path: Path) -> np.ndarray | None:
        mask_path = self.segmentation_root / self.current_split / f"{image_path.stem}.png"
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
        annotation_path = self._annotation_path_for_image(image_path)
        if not annotation_path.exists():
            return []

        try:
            payload = json.loads(annotation_path.read_text())
        except json.JSONDecodeError:
            messagebox.showwarning("Annotation", f"Invalid JSON: {annotation_path}")
            return []

        annotations = payload.get("annotations", [])
        normalized: list[dict[str, dict[str, int]]] = []
        for item in annotations:
            bbox = item.get("bbox", {})
            tip = item.get("tip", {})
            try:
                normalized.append(
                    {
                        "bbox": self._normalize_bbox(
                            {
                                "x": int(bbox.get("x", 0)),
                                "y": int(bbox.get("y", 0)),
                                "width": int(bbox.get("width", 1)),
                                "height": int(bbox.get("height", 1)),
                            }
                        ),
                        "tip": self._normalize_tip(
                            {
                                "x": int(tip.get("x", 0)),
                                "y": int(tip.get("y", 0)),
                            }
                        ),
                    }
                )
            except (TypeError, ValueError):
                continue
        return normalized

    def _save_current_annotation(self) -> None:
        if self.current_image_path is None:
            return

        payload = {
            "image": self.current_image_path.name,
            "width": int(self.image_width),
            "height": int(self.image_height),
            "annotations": [
                {
                    "bbox": annotation["bbox"],
                    "tip": annotation["tip"],
                }
                for annotation in self.current_annotations
            ],
        }
        annotation_path = self._annotation_path_for_image(self.current_image_path)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self.dirty = False
        self._update_buttons()
        self._update_status("Saved annotation.")

    def _annotation_path_for_image(self, image_path: Path) -> Path:
        return self.annotation_root / self.current_split / f"{image_path.stem}.json"

    def _render_scene(self) -> None:
        self.canvas.delete("all")
        if self.current_image_rgb is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="No image loaded",
                fill="#c7d0d9",
                font=("TkDefaultFont", 16),
            )
            return

        composed = self._compose_display_image()
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        scale = min(canvas_width / self.image_width, canvas_height / self.image_height)
        scale = max(scale, 0.01)

        display_width = max(1, int(round(self.image_width * scale)))
        display_height = max(1, int(round(self.image_height * scale)))
        offset_x = (canvas_width - display_width) / 2.0
        offset_y = (canvas_height - display_height) / 2.0

        self.canvas_scale = scale
        self.canvas_offset_x = offset_x
        self.canvas_offset_y = offset_y
        self.canvas_image_width = display_width
        self.canvas_image_height = display_height

        resized = cv2.resize(composed, (display_width, display_height), interpolation=cv2.INTER_AREA)
        self.photo_image = self._rgb_to_photoimage(resized)
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.photo_image)

        if self.show_annotation_var.get():
            for index, annotation in enumerate(self.current_annotations):
                self._draw_annotation_box(index, annotation)

            for index, annotation in enumerate(self.current_annotations):
                self._draw_annotation_tip(index, annotation)

        if self.preview_box is not None:
            x1, y1, x2, y2 = self.preview_box
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)
            self.canvas.create_rectangle(
                cx1,
                cy1,
                cx2,
                cy2,
                outline="#f8e45c",
                width=2,
                dash=(6, 3),
            )

    def _compose_display_image(self) -> np.ndarray:
        composed = self.current_image_rgb.copy()
        if self.show_segmentation_var.get() and self.current_segmentation_mask is not None:
            mask = self.current_segmentation_mask > 0
            overlay = composed.copy()
            overlay[mask] = np.array([30, 220, 80], dtype=np.uint8)
            composed = cv2.addWeighted(overlay, SEGMENTATION_ALPHA, composed, 1.0 - SEGMENTATION_ALPHA, 0.0)
        return composed

    def _draw_annotation_box(self, index: int, annotation: dict[str, dict[str, int]]) -> None:
        bbox = annotation["bbox"]
        x1 = bbox["x"]
        y1 = bbox["y"]
        x2 = bbox["x"] + bbox["width"]
        y2 = bbox["y"] + bbox["height"]
        cx1, cy1 = self._image_to_canvas(x1, y1)
        cx2, cy2 = self._image_to_canvas(x2, y2)

        is_selected = index == self.selected_annotation_index
        outline = "#ffd34d" if is_selected else "#3bd7ff"
        width = 3 if is_selected else 2

        self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=outline, width=width)
        self.canvas.create_text(cx1 + 6, cy1 + 6, text=str(index + 1), fill=outline, anchor=tk.NW)

        if is_selected:
            for handle_x, handle_y in ((cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)):
                self.canvas.create_rectangle(
                    handle_x - HANDLE_SIZE,
                    handle_y - HANDLE_SIZE,
                    handle_x + HANDLE_SIZE,
                    handle_y + HANDLE_SIZE,
                    outline="#111111",
                    fill="#ffd34d",
                    width=1,
                )

    def _draw_annotation_tip(self, index: int, annotation: dict[str, dict[str, int]]) -> None:
        bbox = annotation["bbox"]
        tip = annotation["tip"]
        x1 = bbox["x"]
        y1 = bbox["y"]
        x2 = bbox["x"] + bbox["width"]
        y2 = bbox["y"] + bbox["height"]
        cx1, cy1 = self._image_to_canvas(x1, y1)
        cx2, cy2 = self._image_to_canvas(x2, y2)
        tip_x, tip_y = self._image_to_canvas(tip["x"], tip["y"])

        self.canvas.create_line(
            (cx1 + cx2) / 2.0,
            (cy1 + cy2) / 2.0,
            tip_x,
            tip_y,
            fill="#ff8080",
            width=2,
            dash=(3, 2),
        )
        self.canvas.create_oval(
            tip_x - TIP_RADIUS,
            tip_y - TIP_RADIUS,
            tip_x + TIP_RADIUS,
            tip_y + TIP_RADIUS,
            outline="#7a0000",
            fill="#ff4d4d",
            width=2,
        )

    def _rgb_to_photoimage(self, rgb: np.ndarray) -> tk.PhotoImage:
        height, width = rgb.shape[:2]
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.canvas_offset_x + x * self.canvas_scale,
            self.canvas_offset_y + y * self.canvas_scale,
        )

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.canvas_offset_x) / self.canvas_scale,
            (y - self.canvas_offset_y) / self.canvas_scale,
        )

    def _point_inside_image(self, x: float, y: float) -> bool:
        return 0 <= x <= self.image_width and 0 <= y <= self.image_height

    def _enter_add_mode(self) -> None:
        if self.current_image_path is None:
            return
        self.mode = "add"
        self.preview_box = None
        self.drag_state = None
        self._update_buttons()
        self._update_status("Add mode: drag on the image to create a new bbox.")

    def _delete_selected(self) -> None:
        if self.selected_annotation_index is None:
            return
        del self.current_annotations[self.selected_annotation_index]
        if self.current_annotations:
            self.selected_annotation_index = min(self.selected_annotation_index, len(self.current_annotations) - 1)
        else:
            self.selected_annotation_index = None
        self._mark_dirty()
        self._render_scene()

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

    def _navigate_images(self, step: int) -> None:
        if self.current_index is None or not self.image_paths:
            return

        new_index = self.current_index + step
        if new_index < 0 or new_index >= len(self.image_paths):
            return
        if not self._confirm_leave_current_image():
            self._restore_listbox_selection()
            return

        self.suppress_listbox_event = True
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(new_index)
        self.image_listbox.activate(new_index)
        self.image_listbox.see(new_index)
        self.suppress_listbox_event = False
        self._open_image_at_index(new_index)

    def _on_canvas_press(self, event: tk.Event[tk.Misc]) -> None:
        if self.current_image_rgb is None:
            return

        image_x, image_y = self._canvas_to_image(event.x, event.y)
        if self.mode == "add":
            if not self._point_inside_image(image_x, image_y):
                return
            self.drag_state = {
                "kind": "add",
                "start": (self._clamp_x(image_x), self._clamp_y(image_y)),
            }
            self.preview_box = (image_x, image_y, image_x, image_y)
            self._render_scene()
            return

        if not self.show_annotation_var.get():
            self.selected_annotation_index = None
            self.drag_state = None
            self._update_buttons()
            self._render_scene()
            return

        hit = self._hit_test(image_x, image_y)
        if hit is None:
            self.selected_annotation_index = None
            self.drag_state = None
            self._update_buttons()
            self._render_scene()
            return

        self.selected_annotation_index = hit["index"]
        annotation = self.current_annotations[self.selected_annotation_index]
        bbox = annotation["bbox"].copy()
        tip = annotation["tip"].copy()
        self.drag_state = {
            "kind": hit["kind"],
            "index": hit["index"],
            "start_point": (image_x, image_y),
            "start_bbox": bbox,
            "start_tip": tip,
            "corner": hit.get("corner"),
        }
        self._update_buttons()
        self._render_scene()

    def _on_canvas_drag(self, event: tk.Event[tk.Misc]) -> None:
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
            annotation["tip"] = {
                "x": int(round(image_x)),
                "y": int(round(image_y)),
            }
            self._mark_dirty(render=False)
            self._render_scene()
            return

        if kind == "move":
            start_x, start_y = self.drag_state["start_point"]
            start_bbox = self.drag_state["start_bbox"]
            start_tip = self.drag_state["start_tip"]
            delta_x = int(round(image_x - start_x))
            delta_y = int(round(image_y - start_y))

            new_x = self._clamp_int(start_bbox["x"] + delta_x, 0, self.image_width - start_bbox["width"])
            new_y = self._clamp_int(start_bbox["y"] + delta_y, 0, self.image_height - start_bbox["height"])
            applied_delta_x = new_x - start_bbox["x"]
            applied_delta_y = new_y - start_bbox["y"]

            annotation["bbox"] = {
                "x": new_x,
                "y": new_y,
                "width": start_bbox["width"],
                "height": start_bbox["height"],
            }
            annotation["tip"] = self._normalize_tip(
                {
                    "x": start_tip["x"] + applied_delta_x,
                    "y": start_tip["y"] + applied_delta_y,
                }
            )
            self._mark_dirty(render=False)
            self._render_scene()
            return

        if kind == "resize":
            start_bbox = self.drag_state["start_bbox"]
            x1 = start_bbox["x"]
            y1 = start_bbox["y"]
            x2 = start_bbox["x"] + start_bbox["width"]
            y2 = start_bbox["y"] + start_bbox["height"]
            corner = self.drag_state["corner"]

            if corner == "nw":
                x1, y1 = image_x, image_y
            elif corner == "ne":
                x2, y1 = image_x, image_y
            elif corner == "sw":
                x1, y2 = image_x, image_y
            elif corner == "se":
                x2, y2 = image_x, image_y

            normalized = self._normalize_box_from_points(x1, y1, x2, y2)
            annotation["bbox"] = normalized
            annotation["tip"] = self._normalize_tip(annotation["tip"])
            self._mark_dirty(render=False)
            self._render_scene()

    def _on_canvas_release(self, event: tk.Event[tk.Misc]) -> None:
        if self.drag_state is None:
            return

        kind = self.drag_state["kind"]
        if kind == "add":
            image_x, image_y = self._canvas_to_image(event.x, event.y)
            image_x = self._clamp_x(image_x)
            image_y = self._clamp_y(image_y)
            start_x, start_y = self.drag_state["start"]
            bbox = self._normalize_box_from_points(start_x, start_y, image_x, image_y)
            self.preview_box = None
            if bbox["width"] >= MIN_BOX_SIZE and bbox["height"] >= MIN_BOX_SIZE:
                tip_x = bbox["x"] + bbox["width"] // 2
                tip_y = bbox["y"] + bbox["height"] // 2
                self.current_annotations.append(
                    {
                        "bbox": bbox,
                        "tip": self._normalize_tip({"x": tip_x, "y": tip_y}),
                    }
                )
                self.selected_annotation_index = len(self.current_annotations) - 1
                self._mark_dirty(render=False)
            self.mode = "select"
            self._update_buttons()
            self._update_status()
            self._render_scene()

        self.drag_state = None

    def _hit_test(self, image_x: float, image_y: float) -> dict[str, object] | None:
        threshold = max(8.0 / self.canvas_scale, 3.0)
        tip_threshold_sq = (threshold * 1.5) ** 2

        ordered_indices: list[int] = []
        if self.selected_annotation_index is not None:
            ordered_indices.append(self.selected_annotation_index)
        ordered_indices.extend(i for i in range(len(self.current_annotations)) if i not in ordered_indices)

        for index in ordered_indices:
            annotation = self.current_annotations[index]
            bbox = annotation["bbox"]
            tip = annotation["tip"]
            if (tip["x"] - image_x) ** 2 + (tip["y"] - image_y) ** 2 <= tip_threshold_sq:
                return {"index": index, "kind": "tip"}

            x1 = bbox["x"]
            y1 = bbox["y"]
            x2 = bbox["x"] + bbox["width"]
            y2 = bbox["y"] + bbox["height"]

            if index == self.selected_annotation_index:
                corners = {
                    "nw": (x1, y1),
                    "ne": (x2, y1),
                    "sw": (x1, y2),
                    "se": (x2, y2),
                }
                for corner_name, (corner_x, corner_y) in corners.items():
                    if abs(corner_x - image_x) <= threshold and abs(corner_y - image_y) <= threshold:
                        return {"index": index, "kind": "resize", "corner": corner_name}

            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                return {"index": index, "kind": "move"}

        return None

    def _mark_dirty(self, render: bool = True) -> None:
        self.dirty = True
        self._update_buttons()
        self._update_status()
        if render:
            self._render_scene()

    def _update_buttons(self) -> None:
        self.save_button.configure(state=tk.NORMAL if self.dirty else tk.DISABLED)
        self.delete_button.configure(state=tk.NORMAL if self.selected_annotation_index is not None else tk.DISABLED)
        self.add_button.configure(text="Adding... (Esc to cancel)" if self.mode == "add" else "Add Tool")

    def _update_images_label(self) -> None:
        total_count = len(self.image_paths)
        current_position = self.current_index + 1 if self.current_index is not None and total_count else 0
        self.images_label_var.set(f"Images ({current_position:,} / {total_count:,})")

    def _update_status(self, message: str | None = None) -> None:
        if message is not None:
            self.status_var.set(message)
            return

        filename = self.current_image_path.name if self.current_image_path else "<no image>"
        tool_count = len(self.current_annotations)
        selected = self.selected_annotation_index + 1 if self.selected_annotation_index is not None else 0
        dirty_text = "modified" if self.dirty else "saved"
        mode_text = "add" if self.mode == "add" else "select"
        self.status_var.set(
            f"{self.current_split} | {filename} | tools={tool_count} | selected={selected} | {dirty_text} | mode={mode_text}"
        )

    def _confirm_leave_current_image(self) -> bool:
        if not self.dirty:
            return True

        response = messagebox.askyesnocancel(
            "Unsaved changes",
            "Save changes before switching images or closing the editor?",
            default=messagebox.YES,
        )
        if response is None:
            return False
        if response:
            self._save_current_annotation()
            return not self.dirty
        return True

    def _restore_listbox_selection(self) -> None:
        if self.current_index is None:
            return
        self.suppress_listbox_event = True
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.activate(self.current_index)
        self.suppress_listbox_event = False

    def _list_images(self, image_dir: Path) -> list[Path]:
        if not image_dir.exists():
            return []
        return sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _normalize_bbox(self, bbox: dict[str, int]) -> dict[str, int]:
        x = self._clamp_int(int(bbox["x"]), 0, max(self.image_width - 1, 0))
        y = self._clamp_int(int(bbox["y"]), 0, max(self.image_height - 1, 0))
        width = max(int(bbox["width"]), 1)
        height = max(int(bbox["height"]), 1)
        width = min(width, max(self.image_width - x, 1))
        height = min(height, max(self.image_height - y, 1))
        return {"x": x, "y": y, "width": width, "height": height}

    def _normalize_box_from_points(self, x1: float, y1: float, x2: float, y2: float) -> dict[str, int]:
        left = self._clamp_int(int(round(min(x1, x2))), 0, max(self.image_width - 1, 0))
        top = self._clamp_int(int(round(min(y1, y2))), 0, max(self.image_height - 1, 0))
        right = self._clamp_int(int(round(max(x1, x2))), 0, max(self.image_width, 1))
        bottom = self._clamp_int(int(round(max(y1, y2))), 0, max(self.image_height, 1))
        width = max(right - left, 1)
        height = max(bottom - top, 1)
        return self._normalize_bbox({"x": left, "y": top, "width": width, "height": height})

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
        description="GUI editor for bbox/tip annotations generated from segmentation masks."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("./data/dataset/images"),
        help="Root directory containing train/val/test source images.",
    )
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=Path("./data/dataset/segmentation"),
        help="Root directory containing train/val/test segmentation masks.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("./data/dataset/annotation"),
        help="Root directory containing train/val/test annotation JSON files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if CV2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "opencv-python is required to display images and overlays. "
            "Install it and run the script again."
        ) from CV2_IMPORT_ERROR
    if NUMPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "numpy is required to compose overlays and edit annotations. "
            "Install it and run the script again."
        ) from NUMPY_IMPORT_ERROR

    for path, label in (
        (args.images, "images"),
        (args.segmentation, "segmentation"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"{label} path is not a directory: {path}")

    args.annotation.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_NAMES:
        images_split = args.images / split_name
        segmentation_split = args.segmentation / split_name
        if not images_split.exists():
            raise FileNotFoundError(f"Image split directory does not exist: {images_split}")
        if not segmentation_split.exists():
            raise FileNotFoundError(f"Segmentation split directory does not exist: {segmentation_split}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    editor = AnnotationEditor(args)
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
