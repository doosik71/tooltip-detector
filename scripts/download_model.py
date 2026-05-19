from __future__ import annotations

import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure simple console logging for setup scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def download_monai_model(
    repo_id: str,
    filename: str,
    output_dir: Path,
    token: str | None = None
) -> Path:
    """Download the MONAI model into the specified temp directory if needed."""

    # 1. temp/ 폴더 및 하위 경로 생성 정의
    local_model_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미 모델이 존재하는지 체크
    if local_model_path.exists():
        LOGGER.info("Model already exists: %s", local_model_path)
        return local_model_path

    LOGGER.info("Downloading model from %s to %s", repo_id, output_dir)

    # 2. huggingface_hub를 이용해 temp/ 폴더로 직접 다운로드
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

    resolved_path = Path(downloaded_path)
    LOGGER.info("Downloaded model: %s", resolved_path)
    return resolved_path


def main() -> int:
    """Download all required setup assets."""
    configure_logging()

    # ---------------------------------------------------------
    # [설정 항목] 여기에 실제 다운로드할 모델 정보를 적어주세요.
    # ---------------------------------------------------------
    REPO_ID = "MONAI/endoscopic_tool_segmentation"
    FILENAME = "models/model.pt"
    HF_TOKEN = None                         # 공개 저장소라면 None, 비공개면 "hf_xxx" 토큰 문자열 입력

    # 현재 작업 디렉토리 기준 temp/ 폴더 지정
    TEMP_DIR = Path("./temp")

    try:
        model_path = download_monai_model(
            repo_id=REPO_ID,
            filename=FILENAME,
            output_dir=TEMP_DIR,
            token=HF_TOKEN
        )
        LOGGER.info("Model ready: %s", model_path)
        return 0
    except Exception as e:
        LOGGER.error("Failed to download model: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
