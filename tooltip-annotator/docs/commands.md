# 실험 재현 명령 (Reproduction Commands)

`erop`, `cholec80` 두 데이터셋을 기준으로, 실행 순서대로 정리한 명령 목록이다. 다른 데이터셋은 `--dataset` 값만 바꾸면 된다.

```bash
# 0) 의존성 설치
uv sync

# 1) MONAI segmentation 모델 다운로드 (데이터셋 공용, 최초 1회만)
uv run python scripts/download_model.py

# 2) GPU 환경 점검 (GPU 사용 시, 선택)
uv run python -m scripts.check_cuda --device cuda:0

# --- erop ---

# 3) erop 원본 비디오를 progressive로 변환
uv run python scripts/generate_progressive.py --dataset erop

# 4) erop 프레임 추출(0.3초마다) 및 train/val/test 분할
uv run python -m scripts.generate_dataset --dataset erop

# 5) erop 분할 마스크 생성
uv run python -m scripts.generate_segmentation --dataset erop --device cuda:0

# 6) erop bbox/tip annotation 자동 생성
uv run python -m scripts.generate_annotation --dataset erop

# 7) erop annotation 수동 보정 (GUI)
uv run python -m scripts.annotation_editor --dataset erop

# --- cholec80 ---

# 8) cholec80 원본 비디오를 progressive로 변환
uv run python scripts/generate_progressive.py --dataset cholec80

# 9) cholec80 프레임 추출(1초마다) 및 train/val/test 분할
uv run python -m scripts.generate_dataset --dataset cholec80 --frame 25

# 10) cholec80 분할 마스크 생성
uv run python -m scripts.generate_segmentation --dataset cholec80 --device cuda:0

# 11) cholec80 bbox/tip annotation 자동 생성
uv run python -m scripts.generate_annotation --dataset cholec80

# 12) cholec80 annotation 수동 보정 (GUI)
uv run python -m scripts.annotation_editor --dataset cholec80

# (선택) 통합 파이프라인 GUI: 위 3)~12) 단계를 데이터셋 드롭다운으로 선택해 버튼으로 실행
uv run python -m scripts.pipeline
```
