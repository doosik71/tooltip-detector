import os
import subprocess
from pathlib import Path


def convert_to_progressive(input_dir, output_dir):
    # 1. 경로 객체 생성 및 출력 폴더 자동 생성
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. 변환할 대상 동영상 확장자 정의
    valid_extensions = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI", ".MKV", ".MOV"}

    # 3. 입력 폴더에서 동영상 파일 목록 수집
    video_files = [
        f for f in input_path.iterdir() if f.is_file() and f.suffix in valid_extensions
    ]

    total_files = len(video_files)
    if total_files == 0:
        print(f"No video files found in '{input_dir}'.")
        return

    print(f"Found {total_files} video(s). Starting conversion...\n")

    for idx, video_file in enumerate(video_files, start=1):
        output_file = output_path / video_file.name

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


if __name__ == "__main__":
    INPUT_FOLDER = "./data/video"
    OUTPUT_FOLDER = "./data/progressive"

    convert_to_progressive(INPUT_FOLDER, OUTPUT_FOLDER)
