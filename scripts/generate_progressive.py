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
        print(f"❌ '{input_dir}' 폴더에 변환할 동영상 파일이 없습니다.")
        return

    print(f"🚀 총 {total_files}개의 동영상 검사 및 변환을 시작합니다...\n")

    # 4. 루프를 돌며 ffmpeg 명령어 실행
    for idx, video_file in enumerate(video_files, start=1):
        output_file = output_path / video_file.name

        # 🔍 [추가된 로직] 출력 파일이 이미 존재하는지 확인
        if output_file.exists():
            print(f"==> [{idx}/{total_files}] ⏩ 건너뛰기(Skip): {video_file.name} (이미 변환된 파일이 존재합니다.)")
            continue

        print(f"==> [{idx}/{total_files}] ⏳ 변환 중: {video_file.name} ...")

        # ffmpeg 명령어 구성
        # -y 옵션은 자동 skip을 위해 제거했습니다.
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(video_file),
            "-vf",
            "yadif",  # 디인터레이스 필터
            "-c:v",
            "libx264",  # H.264 코덱
            "-crf",
            "20",  # 화질 설정
            "-c:a",
            "copy",  # 오디오 복사
            str(output_file),
        ]

        try:
            # 명령어 실행 및 완료될 때까지 대기
            subprocess.run(cmd, check=True)
            print(f"    ✨ 완료: {output_file.name}\n")
        except subprocess.CalledProcessError as e:
            print(f"    💥 에러 발생 ({video_file.name}): ffmpeg 변환에 실패했습니다.")
            print(f"    상세 에러 코드: {e}\n")
        except FileNotFoundError:
            print(
                "❌ 시스템에 'ffmpeg'이 설치되어 있지 않습니다. 'sudo apt install ffmpeg'을 실행해 주세요."
            )
            return

    print("🎉 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    INPUT_FOLDER = "./data/video"
    OUTPUT_FOLDER = "./data/progressive"

    convert_to_progressive(INPUT_FOLDER, OUTPUT_FOLDER)
