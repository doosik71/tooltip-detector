from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None


IMAGE_SUFFIX = ".png"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv"}
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and split them into train/val/test datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/progressive"),
        help="Directory containing input video files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/dataset"),
        help="Directory where extracted frames will be saved.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=10,
        help="Extract one frame every N frames.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=736,
        help="Output image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output image height.",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=80,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val",
        type=int,
        default=10,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=10,
        help="Test split ratio.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if CV2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "opencv-python is required to read video files. "
            "Install it and run the script again."
        ) from CV2_IMPORT_ERROR

    if args.frame <= 0:
        raise ValueError("--frame must be a positive integer.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive integers.")

    ratios = (args.train, args.val, args.test)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("--train, --val, and --test must be non-negative integers.")
    if sum(ratios) <= 0:
        raise ValueError("The sum of --train, --val, and --test must be greater than 0.")

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")
    if not args.input.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input}")


def list_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def allocate_split_counts(total_items: int, ratios: tuple[int, int, int]) -> dict[str, int]:
    ratio_sum = sum(ratios)
    raw_counts = [total_items * ratio / ratio_sum for ratio in ratios]
    counts = [int(raw_count) for raw_count in raw_counts]
    remainder = total_items - sum(counts)

    fractional_parts = sorted(
        (
            (raw_count - int(raw_count), index)
            for index, raw_count in enumerate(raw_counts)
        ),
        reverse=True,
    )

    for _, index in fractional_parts[:remainder]:
        counts[index] += 1

    return dict(zip(SPLIT_NAMES, counts, strict=True))


def assign_splits(frame_numbers: list[int], ratios: tuple[int, int, int]) -> dict[int, str]:
    split_counts = allocate_split_counts(len(frame_numbers), ratios)
    assignments: dict[int, str] = {}
    cursor = 0

    for split_name in SPLIT_NAMES:
        split_count = split_counts[split_name]
        for frame_number in frame_numbers[cursor:cursor + split_count]:
            assignments[frame_number] = split_name
        cursor += split_count

    return assignments


def resize_with_letterbox(frame, width: int, height: int):
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)

    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized_frame = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    output_frame = cv2.copyMakeBorder(
        resized_frame,
        top=(height - resized_height) // 2,
        bottom=height - resized_height - (height - resized_height) // 2,
        left=(width - resized_width) // 2,
        right=width - resized_width - (width - resized_width) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return output_frame


def save_frames_for_video(
    video_path: Path,
    output_dir: Path,
    frame_step: int,
    ratios: tuple[int, int, int],
    output_width: int,
    output_height: int,
) -> dict[str, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 0:
            frame_count = 0

        frame_numbers = list(range(0, frame_count, frame_step))
        assignments = assign_splits(frame_numbers, ratios)
        saved_counts = {split_name: 0 for split_name in SPLIT_NAMES}

        progress = tqdm(
            total=frame_count,
            desc=video_path.name,
            unit="frame",
            leave=False,
        )

        frame_number = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            split_name = assignments.get(frame_number)
            if split_name is not None:
                output_frame = resize_with_letterbox(frame, output_width, output_height)
                output_path = (
                    output_dir
                    / split_name
                    / f"{video_path.stem}_{frame_number:08d}{IMAGE_SUFFIX}"
                )
                if not cv2.imwrite(str(output_path), output_frame):
                    raise RuntimeError(f"Failed to write image: {output_path}")
                saved_counts[split_name] += 1

            frame_number += 1
            progress.update(1)

        progress.close()
        return saved_counts
    finally:
        capture.release()


def ensure_output_dirs(output_dir: Path) -> None:
    for split_name in SPLIT_NAMES:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    validate_args(args)

    ensure_output_dirs(args.output)
    video_files = list_video_files(args.input)

    if not video_files:
        print(f"No video files found in {args.input}")
        return 0

    total_saved = {split_name: 0 for split_name in SPLIT_NAMES}
    ratios = (args.train, args.val, args.test)

    for video_path in tqdm(video_files, desc="Videos", unit="video"):
        saved_counts = save_frames_for_video(
            video_path=video_path,
            output_dir=args.output,
            frame_step=args.frame,
            ratios=ratios,
            output_width=args.width,
            output_height=args.height,
        )
        for split_name, count in saved_counts.items():
            total_saved[split_name] += count

    print("Extraction complete.")
    print(
        "Saved frames: "
        f"train={total_saved['train']}, "
        f"val={total_saved['val']}, "
        f"test={total_saved['test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
