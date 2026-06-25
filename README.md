# tooltip-detector

복강경(laparoscope) 수술 영상에서 수술도구의 끝부분(팁)을 탐지하는 딥러닝 파이프라인이다.
EfficientNet-B2 인코더와 U-Net 디코더를 결합한 모델이 각 프레임에서 도구 팁의 위치를 거리 기반 히트맵으로 출력한다.

## 모델 구조

```text
입력 이미지 (3, 480, 736)
    │
    ▼
EfficientNet-B2 Encoder   ← ImageNet 사전학습 구조 (스크래치 학습)
    │  스킵 연결 (5개 스테이지)
    ▼
U-Net Decoder             ← 5단계 업샘플링
    │
    ▼
Segmentation Head         ← Conv2d → 2채널 출력
    │
    ├─ 채널 0: 배경
    └─ 채널 1: 도구 영역 (sigmoid → 거리 기반 히트맵)
```

- 출력 채널 1에 sigmoid를 적용한 값이 팁까지의 거리 기반 히트맵이다.
- 히트맵에서 임계값 이상의 피크를 추출하여 팁 위치를 예측한다.

## 요구 사항

- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
- CUDA 12.8 호환 GPU (CPU 추론도 가능하나 학습 시 GPU 권장)

## 설치

```bash
uv sync
```

## 디렉터리 구조

```text
tooltip-detector/
├── ttd/
│   ├── model.py             # EfficientNet-B2 + U-Net 모델 정의
│   ├── dataset.py           # SurgicalToolDataset (히트맵 타겟 생성)
│   ├── train.py             # 학습 스크립트
│   ├── eval.py              # 평가 스크립트
│   └── dataset-browser.py   # 데이터셋 시각화 GUI
├── bin/
│   ├── train-model(.bat)        # 학습 실행
│   ├── eval-model(.bat)         # 평가 실행
│   └── dataset-browser(.bat)    # 데이터셋 브라우저 실행
├── docs/
│   ├── dataset-guide.md         # 데이터셋 구조 및 포맷
│   ├── train-guide.md           # 학습 절차 및 설정
│   ├── eval-guide.md            # 평가 방법 및 지표
│   └── dataset-browser-guide.md # GUI 브라우저 사용법
├── data/
│   ├── dataset/                 # 학습 데이터 (annotation / images / segmentation)
│   ├── model/                   # 학습된 모델 가중치 (best.pt, last.pt)
│   └── results/                 # 평가 결과
└── temp/
    └── model.pt                 # 참조 모델 (구조 확인용)
```

## 데이터셋

5개의 복강경 수술 세션에서 추출한 총 **180,706 프레임** (736 × 480 px).

| 스플릿 | 프레임 수 | 비율 |
| ------ | --------- | ---- |
| train  | 108,424   | 60 % |
| val    | 36,140    | 20 % |
| test   | 36,142    | 20 % |

분할 기준은 세션 내 시간 순서(temporal split)이다. 각 프레임마다 RGB 이미지, 이진 분할 마스크, 도구 팁 좌표 어노테이션이 제공된다.

데이터셋 포맷 상세: [docs/dataset-guide.md](docs/dataset-guide.md)

## 작업 순서

### 1. 데이터셋 확인 (선택)

```bash
bin/dataset-browser
```

원본 이미지, 바운딩 박스, 팁 마커, 거리 기반 히트맵을 시각적으로 탐색한다.

### 2. 학습

```bash
bin/train-model
```

- 에포크마다 validation 실행
- `data/model/best.pt` — 검증 손실 최저 모델 자동 저장
- `data/model/last.pt` — 최종 에포크 모델 자동 저장
- `--resume` 플래그로 `last.pt`에서 이어서 학습 가능

### 3. 평가

```bash
bin/eval-model
```

테스트 세트에서 팁 탐지 정확도를 측정하고 결과를 `data/results/`에 저장한다. 실행마다 덮어쓴다.

| 저장 파일      | 내용                                                 |
| -------------- | ---------------------------------------------------- |
| `summary.json` | 전체 지표 + 세션별 지표 + 실행 파라미터              |
| `per_tip.csv`  | GT 팁 1개당 1행 (좌표, 예측값, 거리, 탐지 성공 여부) |

## 스크립트 요약

| 스크립트              | 설명                | 주요 인수                                      |
| --------------------- | ------------------- | ---------------------------------------------- |
| `bin/train-model`     | 모델 학습           | `--epochs`, `--batch-size`, `--lr`, `--resume` |
| `bin/eval-model`      | 팁 탐지 정확도 평가 | `--threshold`, `--nms-radius`, `--model`       |
| `bin/dataset-browser` | 데이터셋 시각화 GUI | `--split`, `--data-root`                       |

직접 실행 예시:

```bash
# 에포크 60회, 배치 8로 학습
uv run python -m ttd.train --epochs 60 --batch-size 8

# 임계값 조정 후 평가
uv run python -m ttd.eval --threshold 0.3 --nms-radius 15
```

## 평가 지표

| 지표               | 설명                                   |
| ------------------ | -------------------------------------- |
| Hit-rate @ 10 px   | 예측 팁이 GT 팁 10 px 이내에 있는 비율 |
| Hit-rate @ 20 px   | 예측 팁이 GT 팁 20 px 이내에 있는 비율 |
| Hit-rate @ 50 px   | 예측 팁이 GT 팁 50 px 이내에 있는 비율 |
| Mean / Median dist | 매칭된 팁의 평균·중앙값 픽셀 거리      |
| P90 dist           | 픽셀 거리 90 백분위수                  |
| Miss rate          | 탐지 후보 없음으로 실패한 팁 비율      |

## 문서 목록

| 문서                                                           | 내용                                                                 |
| -------------------------------------------------------------- | -------------------------------------------------------------------- |
| [docs/dataset-guide.md](docs/dataset-guide.md)                 | 데이터셋 디렉터리 구조, 파일 포맷, 어노테이션 스펙, 히트맵 타겟 수식 |
| [docs/train-guide.md](docs/train-guide.md)                     | 학습 실행 방법, 인수 설명, 증강 파이프라인, 체크포인트, 출력 예시    |
| [docs/eval-guide.md](docs/eval-guide.md)                       | 평가 실행 방법, 피크 탐지 알고리즘, 지표 정의, 결과 파일 구조        |
| [docs/dataset-browser-guide.md](docs/dataset-browser-guide.md) | GUI 화면 구성, 조작 방법, 키보드 단축키                              |
