from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from tooltip.dataset_paths import dataset_src_dir, progressive_dir, resolve_path


VALID_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI", ".MKV", ".MOV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deinterlace and reencode source videos into a progressive format "
            "suitable for frame extraction."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset name under data/dataset-src, e.g. 'erop' or 'cholec80'. "
            "Used to resolve default --input/--output when not given explicitly."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Directory containing source videos. Defaults to data/dataset-src/<dataset>.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory where progressive videos will be saved. Defaults to data/dataset/<dataset>/progressive.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")
    if not args.input.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input}")


def convert_to_progressive(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        f for f in input_dir.iterdir() if f.is_file() and f.suffix in VALID_EXTENSIONS
    ]

    total_files = len(video_files)
    if total_files == 0:
        print(f"No video files found in '{input_dir}'.")
        return

    print(f"Found {total_files} video(s). Starting conversion...\n")

    for idx, video_file in enumerate(video_files, start=1):
        output_file = output_dir / video_file.name

        if output_file.exists():
            print(f"==> [{idx}/{total_files}] Skip: {video_file.name} (output already exists)")
            continue

        print(f"==> [{idx}/{total_files}] Converting: {video_file.name} ...")

        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(video_file),
            "-vf",
            "yadif",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-c:a",
            "copy",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"    Done: {output_file.name}\n")
        except subprocess.CalledProcessError as e:
            print(f"    Error ({video_file.name}): ffmpeg conversion failed.")
            print(f"    Exit code: {e}\n")
        except FileNotFoundError:
            print("ffmpeg is not installed. Run 'sudo apt install ffmpeg'.")
            return

    print("All conversions complete.")


def main() -> int:
    args = parse_args()
    args.input = resolve_path(args.input, args.dataset, dataset_src_dir, "--input")
    args.output = resolve_path(args.output, args.dataset, progressive_dir, "--output")
    validate_args(args)

    convert_to_progressive(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
