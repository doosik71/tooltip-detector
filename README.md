# tooltip-annotator

수술 영상에서 프레임 데이터를 만들고, semantic segmentation과 tip annotation을 생성 및 수정하는 도구 모음이다.

현재 파이프라인은 다음 단계로 구성된다.

1. 원본 비디오를 progressive 형태로 변환
2. 비디오 프레임을 이미지 데이터셋으로 추출
3. MONAI segmentation 모델 다운로드
4. 이미지별 binary segmentation mask 생성
5. segmentation mask 기반 bbox/tip annotation JSON 생성
6. GUI 편집기로 annotation 수동 보정

## 요구 사항

- Python `3.12` 이상
- 프로젝트 의존성 설치
- `ffmpeg`
- GUI 편집기를 사용할 경우 `tkinter`
- GPU 추론을 사용할 경우 CUDA가 동작하는 PyTorch 환경

의존성 설치 예시:

```bash
uv sync
```

또는 가상환경에서 직접 설치 후 사용한다.

## 데이터 구조

여러 출처의 데이터셋을 비교 평가할 수 있도록 원본과 파이프라인 산출물을 분리한다.

```text
./data/
├── dataset-src/            # 읽기 전용 원본 데이터셋 소스 (심볼릭 링크 등)
│   ├── erop/                 # 예: 이롭 내시경 수술 동영상
│   └── cholec80/             # 예: cholec80 공개 데이터셋 동영상
└── dataset/                # 파이프라인이 생성하는 내부 데이터셋
    ├── erop/
    │   ├── progressive/        # (선택) 디인터레이스/재인코딩된 동영상
    │   ├── images/
    │   │   ├── train/
    │   │   ├── val/
    │   │   └── test/
    │   ├── segmentation/
    │   │   ├── train/
    │   │   ├── val/
    │   │   └── test/
    │   └── annotation/
    │       ├── train/
    │       ├── val/
    │       └── test/
    └── cholec80/
        └── ... (동일 구조)
```

각 스크립트는 `--dataset <name>` 옵션을 받아 위 컨벤션에 따른 기본 입출력 경로를 자동 계산한다. 예를 들어 `--dataset cholec80`을 주면 입력/출력 경로가 `data/dataset-src/cholec80`, `data/dataset/cholec80/images` 등으로 자동 결정된다. `--input`/`--output`(또는 GUI 편집기의 `--images`/`--segmentation`/`--annotation`)을 직접 지정하면 그 값이 `--dataset` 기본값보다 항상 우선한다.

`data/dataset-src`는 읽기 전용 외부 원본이므로 스크립트가 그 안에 쓰지 않는다. `generate_progressive.py`의 출력(디인터레이스/재인코딩된 동영상)은 `data/dataset-src/<name>/progressive`가 아니라 `data/dataset/<name>/progressive`에 저장된다. `generate_dataset.py`는 `--dataset`만 주어졌을 때 `data/dataset/<name>/progressive`가 존재하고 비어있지 않으면 그것을, 없으면 `data/dataset-src/<name>`(원본)을 입력으로 사용한다. 즉 원본 자체가 이미 progressive 형식이라 디인터레이스가 필요 없는 데이터셋은 1단계를 건너뛰어도 된다.

모델 파일 기본 위치는 다음과 같다(데이터셋과 무관한 공용 자원).

```text
./temp/models/model.pt
```

## 패키지 구조

- `scripts/`: 사용자가 직접 실행하는 진입점 스크립트.
- `tooltip/`: 스크립트들이 공유하는 임포트 전용 모듈(`tooltip/dataset_paths.py`의 데이터셋 경로 계산 등). 직접 실행하지 않는다.

## 작업 순서

이하 예시는 `cholec80` 데이터셋 기준이며, `--dataset erop`처럼 이름만 바꾸면 다른 데이터셋에도 동일하게 적용된다.

### 1. 비디오를 progressive로 변환

```bash
python scripts/generate_progressive.py --dataset cholec80
```

기본 입력은 `./data/dataset-src/cholec80`, 기본 출력은 `./data/dataset/cholec80/progressive`이다.

문서: [docs/generate_progressive.md](./docs/generate_progressive.md)

### 2. 데이터셋 이미지 생성

```bash
python scripts/generate_dataset.py --dataset cholec80
```

기본 출력은 `./data/dataset/cholec80/images/{train,val,test}`이다.

문서: [docs/generate_dataset.md](./docs/generate_dataset.md)

### 3. segmentation 모델 다운로드

```bash
python scripts/download_model.py
```

문서: [docs/download_model.md](./docs/download_model.md)

### 4. GPU 환경 점검

GPU를 사용할 예정이면 먼저 환경 점검을 권장한다.

```bash
python -m scripts.check_gpu --device cuda:0
```

이 스크립트는 다음을 확인한다.

- PyTorch CUDA 인식 상태
- cuDNN 버전 정합성
- `nvidia-smi`, `nvcc`
- `libtorch_cuda.so`가 실제로 물고 있는 CUDA/cuDNN 라이브러리
- 최소 CUDA Conv2D smoke test
- cuDNN enabled/disabled 비교

### 5. segmentation mask 생성

```bash
python -m scripts.generate_segmentation --dataset cholec80
```

기본 입력은 `./data/dataset/cholec80/images/{train,val,test}`이고, 기본 출력은 `./data/dataset/cholec80/segmentation/{train,val,test}`이다.

주요 옵션 예시:

```bash
python -m scripts.generate_segmentation --dataset cholec80 --device cuda:0
python -m scripts.generate_segmentation --dataset cholec80 --device cpu
python -m scripts.generate_segmentation --dataset cholec80 --overwrite
```

문서: [docs/generate_segmentation.md](./docs/generate_segmentation.md)

### 6. annotation JSON 자동 생성

```bash
python -m scripts.generate_annotation --dataset cholec80
```

이 스크립트는 segmentation mask에서 contour를 찾고, 각 contour의 bounding box와 tool tip 좌표를 계산해 JSON으로 저장한다.

기본 출력은 `./data/dataset/cholec80/annotation/{train,val,test}`이다.

문서: [docs/generate_annotation.md](./docs/generate_annotation.md)

### 7. GUI 편집기로 annotation 수정

```bash
python -m scripts.annotation_editor --dataset cholec80
```

편집기에서 가능한 작업:

- split 전환
- 이미지 선택
- segmentation overlay 표시
- bbox 이동
- bbox 리사이즈
- tip 드래그 수정
- annotation 추가
- annotation 삭제
- `Save` 또는 `Ctrl+S`로 저장

문서: [docs/annotation_editor.md](./docs/annotation_editor.md)

![Screen](./screen.png)

## scripts 요약

- `scripts/download_model.py`
  - Hugging Face Hub에서 MONAI segmentation 모델 다운로드
- `scripts/generate_progressive.py`
  - 원본 비디오를 ffmpeg로 디인터레이스 및 재인코딩
- `scripts/generate_dataset.py`
  - 비디오 프레임 추출 후 `train/val/test` 분할 이미지 생성
- `scripts/check_gpu.py`
  - PyTorch CUDA/cuDNN 런타임 점검 및 문제 분석
- `scripts/generate_segmentation.py`
  - 모델을 이용해 이미지별 binary segmentation mask 생성
- `scripts/generate_annotation.py`
  - segmentation mask로부터 bbox/tip annotation JSON 생성
- `scripts/annotation_editor.py`
  - annotation 수동 보정을 위한 GUI 편집기
- `scripts/pipeline.py`
  - 전체 파이프라인 단계를 실행하고 진행 상태를 보여주는 통합 GUI. 상단 `Dataset` 드롭다운으로 `data/dataset-src`/`data/dataset` 아래 발견된 데이터셋을 전환하며 실행/상태를 확인할 수 있다.
- `tooltip/dataset_paths.py`
  - `--dataset` 옵션의 기본 경로 계산과 데이터셋 목록 조회를 담당하는 공용 모듈(직접 실행하지 않음)

## 문서 목록

- [docs/commands.md](./docs/commands.md)
- [docs/download_model.md](./docs/download_model.md)
- [docs/generate_progressive.md](./docs/generate_progressive.md)
- [docs/generate_dataset.md](./docs/generate_dataset.md)
- [docs/generate_segmentation.md](./docs/generate_segmentation.md)
- [docs/generate_annotation.md](./docs/generate_annotation.md)
- [docs/annotation_editor.md](./docs/annotation_editor.md)
- [docs/pipeline.md](./docs/pipeline.md)
