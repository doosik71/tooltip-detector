from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import random

from tqdm import tqdm

from tooltip.dataset_paths import images_dir, resolve_path, resolve_video_input

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
        "--dataset",
        default=None,
        help=(
            "Dataset name, e.g. 'erop' or 'cholec80'. Used to resolve default "
            "--input/--output when not given explicitly."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Directory containing input video files. Defaults to "
            "data/dataset/<dataset>/progressive if it exists and is non-empty, "
            "otherwise data/dataset-src/<dataset>."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory where extracted frames will be saved. Defaults to data/dataset/<dataset>/images.",
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
        default=60,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val",
        type=int,
        default=20,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=20,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic split assignment within each video.",
    )
    parser.add_argument(
        "--verify",
        choices=("fast", "full"),
        default="fast",
        help=(
            "How existing images are validated on resume. "
            "'fast' (default) checks the PNG header/trailer only; "
            "'full' fully decodes each image to detect any corruption."
        ),
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


def create_split_rng(video_path: Path, seed: int) -> random.Random:
    seed_material = f"{video_path.resolve()}::{seed}".encode("utf-8")
    seed_digest = hashlib.blake2b(seed_material, digest_size=16).digest()
    return random.Random(int.from_bytes(seed_digest, byteorder="big"))


def assign_splits(
    frame_numbers: list[int],
    ratios: tuple[int, int, int],
    rng: random.Random,
) -> dict[int, str]:
    split_counts = allocate_split_counts(len(frame_numbers), ratios)
    assignments: dict[int, str] = {}
    shuffled_frame_numbers = frame_numbers[:]
    rng.shuffle(shuffled_frame_numbers)
    cursor = 0

    for split_name in SPLIT_NAMES:
        split_count = split_counts[split_name]
        for frame_number in shuffled_frame_numbers[cursor:cursor + split_count]:
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


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IEND = b"IEND\xaeB`\x82"


def _has_png_envelope(path: Path) -> bool:
    """Lightweight integrity check: PNG signature at the start and IEND at the end.

    A write interrupted by a forced termination leaves a truncated file that is
    missing its IEND trailer, so this catches the common resume-corruption case
    by reading only a few bytes. It does not detect corruption inside an
    otherwise complete envelope (use the full check for that).
    """
    try:
        with path.open("rb") as handle:
            if handle.read(8) != _PNG_SIGNATURE:
                return False
            handle.seek(-8, os.SEEK_END)
            return handle.read(8) == _PNG_IEND
    except OSError:
        return False


def is_valid_image(path: Path, full: bool = False) -> bool:
    """Return True only if the file exists and looks like a readable image.

    Existence alone is not a safe resume condition: a run killed mid-write can
    leave a truncated/corrupt PNG behind. Such a file must be re-extracted
    instead of skipped. ``full=True`` fully decodes the image (slow, thorough);
    otherwise a fast header/trailer check is used (default).
    """
    if not path.is_file():
        return False
    if full:
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None
    return _has_png_envelope(path)


def write_image_atomic(path: Path, image) -> None:
    """Encode and write an image so the final path never holds a partial file.

    The image is encoded in memory, written to a temporary sibling file, and
    then atomically moved into place with ``os.replace``. If the process is
    killed mid-write, only the temporary file can be left incomplete; the final
    path keeps either the previous file or nothing.
    """
    success, buffer = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")

    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_bytes(buffer.tobytes())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_frames_for_video(
    video_path: Path,
    output_dir: Path,
    frame_step: int,
    ratios: tuple[int, int, int],
    output_width: int,
    output_height: int,
    seed: int,
    full_verify: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 0:
            frame_count = 0

        frame_numbers = list(range(0, frame_count, frame_step))
        split_rng = create_split_rng(video_path, seed)
        assignments = assign_splits(frame_numbers, ratios, split_rng)

        # Build the full output-path map and separate pending from already-done.
        output_map: dict[int, tuple[str, Path]] = {
            fn: (
                sn,
                output_dir / sn / f"{video_path.stem}_{fn:08d}{IMAGE_SUFFIX}",
            )
            for fn, sn in assignments.items()
        }
        # Re-extract a frame when its image is missing OR exists but does not
        # decode (e.g. a truncated file left behind by a forced termination).
        pending: dict[int, tuple[str, Path]] = {}
        corrupt_count = 0
        for fn, (sn, p) in output_map.items():
            if is_valid_image(p, full=full_verify):
                continue
            if p.exists():
                corrupt_count += 1
            pending[fn] = (sn, p)

        if corrupt_count:
            tqdm.write(
                f"{video_path.name}: re-extracting {corrupt_count} "
                "unreadable image(s) left from a previous run"
            )

        saved_counts = {sn: 0 for sn in SPLIT_NAMES}
        skipped_counts = {sn: 0 for sn in SPLIT_NAMES}
        for fn, (sn, _) in output_map.items():
            if fn not in pending:
                skipped_counts[sn] += 1

        if not pending:
            return saved_counts, skipped_counts

        pending_set = set(pending.keys())

        progress = tqdm(
            total=frame_count,
            desc=video_path.name,
            unit="frame",
            leave=False,
            ascii=True,
            ncols=100,
        )

        frame_number = 0
        while True:
            if frame_number in pending_set:
                success, frame = capture.read()
                if not success:
                    break
                split_name, output_path = pending[frame_number]
                output_frame = resize_with_letterbox(frame, output_width, output_height)
                write_image_atomic(output_path, output_frame)
                saved_counts[split_name] += 1
            else:
                # Use grab() to advance without full decode.
                success = capture.grab()
                if not success:
                    break

            frame_number += 1
            progress.update(1)

        progress.close()
        return saved_counts, skipped_counts
    finally:
        capture.release()


def ensure_output_dirs(output_dir: Path) -> None:
    for split_name in SPLIT_NAMES:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    args.input = resolve_video_input(args.input, args.dataset)
    args.output = resolve_path(args.output, args.dataset, images_dir, "--output")
    validate_args(args)

    ensure_output_dirs(args.output)
    video_files = list_video_files(args.input)

    if not video_files:
        print(f"No video files found in {args.input}")
        return 0

    total_saved = {split_name: 0 for split_name in SPLIT_NAMES}
    total_skipped = {split_name: 0 for split_name in SPLIT_NAMES}
    ratios = (args.train, args.val, args.test)
    full_verify = args.verify == "full"

    for video_path in tqdm(video_files, desc="Videos", unit="video", ascii=True, ncols=100):
        saved_counts, skipped_counts = save_frames_for_video(
            video_path=video_path,
            output_dir=args.output,
            frame_step=args.frame,
            ratios=ratios,
            output_width=args.width,
            output_height=args.height,
            seed=args.seed,
            full_verify=full_verify,
        )
        for split_name in SPLIT_NAMES:
            total_saved[split_name] += saved_counts[split_name]
            total_skipped[split_name] += skipped_counts[split_name]

    print("Extraction complete.")
    print(
        "Saved frames: "
        f"train={total_saved['train']}, "
        f"val={total_saved['val']}, "
        f"test={total_saved['test']}"
    )
    print(
        "Skipped frames: "
        f"train={total_skipped['train']}, "
        f"val={total_skipped['val']}, "
        f"test={total_skipped['test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
