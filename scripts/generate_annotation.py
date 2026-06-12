from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

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
ANNOTATION_SUFFIX = ".json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate surgical-tool tip annotations from binary segmentation masks."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/dataset/segmentation"),
        help="Root directory containing train/val/test segmentation masks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/dataset/annotation"),
        help="Root directory where train/val/test annotation JSON files will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files instead of skipping them.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if CV2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "opencv-python is required to read segmentation masks. "
            "Install it and run the script again."
        ) from CV2_IMPORT_ERROR
    if NUMPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "numpy is required to analyze segmentation contours. "
            "Install it and run the script again."
        ) from NUMPY_IMPORT_ERROR

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")
    if not args.input.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input}")

    for split_name in SPLIT_NAMES:
        split_dir = args.input / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
        if not split_dir.is_dir():
            raise NotADirectoryError(f"Split path is not a directory: {split_dir}")


def ensure_output_dirs(output_root: Path) -> None:
    for split_name in SPLIT_NAMES:
        (output_root / split_name).mkdir(parents=True, exist_ok=True)


def _all_outputs_exist(input_root: Path, output_root: Path) -> bool:
    for split_name in SPLIT_NAMES:
        for image_path in list_images(input_root / split_name):
            if not (output_root / split_name / f"{image_path.stem}{ANNOTATION_SUFFIX}").exists():
                return False
    return True


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read segmentation mask: {mask_path}")

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def contour_centroid(points: np.ndarray) -> np.ndarray:
    contour = points.reshape(-1, 1, 2).astype(np.int32)
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        return np.array(
            [
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            ],
            dtype=np.float32,
        )

    return points.mean(axis=0, dtype=np.float32)


def farthest_point(reference: np.ndarray, points: np.ndarray) -> np.ndarray:
    distances = np.sum((points - reference) ** 2, axis=1)
    return points[int(np.argmax(distances))]


def compute_tip(points: np.ndarray, image_center: np.ndarray) -> np.ndarray:
    center = contour_centroid(points)
    point_a = farthest_point(center, points)
    point_b = farthest_point(point_a.astype(np.float32), points)

    distance_a = np.sum((point_a.astype(np.float32) - image_center) ** 2)
    distance_b = np.sum((point_b.astype(np.float32) - image_center) ** 2)
    return point_a if distance_a <= distance_b else point_b


def extract_annotations(mask: np.ndarray) -> list[dict[str, object]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_height, image_width = mask.shape[:2]
    image_center = np.array([image_width / 2.0, image_height / 2.0], dtype=np.float32)

    contour_entries: list[tuple[tuple[int, int, int, int], dict[str, object]]] = []

    for contour in contours:
        if contour.size == 0:
            continue

        points = contour.reshape(-1, 2).astype(np.float32)
        tip = compute_tip(points, image_center)
        x, y, width, height = cv2.boundingRect(contour.astype(np.int32))

        annotation = {
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            },
            "tip": {
                "x": int(round(float(tip[0]))),
                "y": int(round(float(tip[1]))),
            },
        }
        contour_entries.append(((x, y, width, height), annotation))

    contour_entries.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]))
    return [entry[1] for entry in contour_entries]


def build_annotation_payload(mask_path: Path, mask: np.ndarray) -> dict[str, object]:
    height, width = mask.shape[:2]
    annotations = extract_annotations(mask)
    return {
        "image": mask_path.name,
        "width": int(width),
        "height": int(height),
        "annotations": annotations,
    }


def write_annotation(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def process_split(
    split_name: str,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
) -> tuple[int, int]:
    input_dir = input_root / split_name
    output_dir = output_root / split_name
    image_paths = list_images(input_dir)

    saved_count = 0
    skipped_count = 0

    for image_path in tqdm(image_paths, desc=split_name, unit="image"):
        output_path = output_dir / f"{image_path.stem}{ANNOTATION_SUFFIX}"
        if output_path.exists() and not overwrite:
            skipped_count += 1
            continue

        mask = load_binary_mask(image_path)
        payload = build_annotation_payload(image_path, mask)
        write_annotation(output_path, payload)
        saved_count += 1

    return saved_count, skipped_count


def main() -> int:
    args = parse_args()
    validate_args(args)
    ensure_output_dirs(args.output)

    if not args.overwrite and _all_outputs_exist(args.input, args.output):
        print("All annotation files already exist. Nothing to do.")
        return 0

    totals = {
        split_name: {"saved": 0, "skipped": 0}
        for split_name in SPLIT_NAMES
    }

    for split_name in SPLIT_NAMES:
        saved_count, skipped_count = process_split(
            split_name=split_name,
            input_root=args.input,
            output_root=args.output,
            overwrite=args.overwrite,
        )
        totals[split_name]["saved"] = saved_count
        totals[split_name]["skipped"] = skipped_count

    print("Annotation generation complete.")
    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: "
            f"saved={totals[split_name]['saved']}, "
            f"skipped={totals[split_name]['skipped']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
