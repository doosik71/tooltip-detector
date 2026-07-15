# tooltip-detector

복강경(laparoscope) 수술 영상에서 수술도구의 끝부분(팁)을 탐지하는 딥러닝 모델 기반 실험 도구이다.
EfficientNet-B2 인코더와 U-Net 디코더를 결합한 모델이 각 프레임에서 도구 팁의 위치를 거리 기반 히트맵으로 출력한다.
현재는 `monai`와 `monai_mini` 두 가지 모델 변형을 지원하며, 학습·평가·GUI 추론 모두 멀티 모델 선택이 가능하다.
정지 이미지/데이터셋 기반 탐지 시각화(`tooltip-detector`) 외에, 동영상을 실시간으로 처리하며 내시경 카메라가 이동해야 할 방향을 화살표로 안내하는 `tooltip-tracker` GUI도 제공한다.

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

## 지원 모델

| 모델         | 설명                                                   | 체크포인트 경로           | 평가 결과 경로             |
| ------------ | ------------------------------------------------------ | ------------------------- | -------------------------- |
| `monai`      | EfficientNet-B2 + U-Net 풀 모델, MONAI 호환 구조       | `data/models/monai/`      | `data/results/monai/`      |
| `monai_mini` | EfficientNet-B2 + U-Net 경량 모델, 인코더 블록 수 절반 | `data/models/monai_mini/` | `data/results/monai_mini/` |

기본 실행 시 학습/평가/GUI는 각 모델 타입에 대응하는 `data/models/<model-type>/best.pt`를 사용한다.

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
│   ├── model/
│   │   ├── monai.py               # EfficientNet-B2 + U-Net (풀 모델, MONAI 호환)
│   │   └── monai_mini.py          # EfficientNet-B2 + U-Net (경량 모델, 인코더 블록 수 절반)
│   ├── dataset.py                 # SurgicalToolDataset (히트맵 타겟 생성)
│   ├── train.py                   # 학습 스크립트
│   ├── eval.py                    # 평가 스크립트
│   ├── compare-speed.py           # 모델 속도 비교 스크립트
│   ├── tooltip-detector.py        # 탐지 결과 시각화 GUI (이미지/데이터셋)
│   ├── tooltip-tracker.py         # 동영상 실시간 탐지·추적 GUI
│   ├── camera_motion_vector.py    # tooltip-tracker의 화살표(카메라 이동 방향) 스무딩 구현
│   └── dataset-browser.py         # 데이터셋 시각화 GUI
├── bin/
│   ├── train-model(.bat)         # 학습 실행
│   ├── eval-model(.bat)          # 평가 실행
│   ├── compare-speed(.bat)       # 모델 속도 비교 실행
│   ├── tooltip-detector(.bat)    # 탐지 결과 시각화 실행
│   ├── tooltip-tracker(.bat)     # 동영상 실시간 추적 실행
│   └── dataset-browser(.bat)     # 데이터셋 브라우저 실행
├── docs/
│   ├── dataset-guide.md          # 데이터셋 구조 및 포맷
│   ├── train-guide.md            # 학습 절차 및 설정
│   ├── eval-guide.md             # 평가 방법 및 지표
│   ├── tooltip-detector.md       # 탐지 GUI 상세설계서 및 사용자설명서
│   ├── tooltip-tracker.md        # 동영상 추적 GUI 상세설계서 및 사용자설명서
│   └── dataset-browser-guide.md  # GUI 브라우저 사용법
├── images/
│   └── tooltip-detector.png      # Tooltip Detector GUI 스크린샷
├── data/
│   ├── dataset/                  # 학습 데이터 (annotation / images / segmentation)
│   ├── models/                   # 학습된 모델 가중치 (모델 타입별 best.pt, last.pt)
│   └── results/                  # 평가 결과
└── temp/
    └── monai.pt                  # 참조 모델 (구조 확인용)
```

## 데이터셋

5개의 복강경 수술 세션에서 추출한 총 **180,706 프레임** (736 × 480 px).

| 스플릿 | 프레임 수 | 비율 |
| ------ | --------: | ---: |
| train  |   108,424 | 60 % |
| val    |    36,140 | 20 % |
| test   |    36,142 | 20 % |

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

- 기본 모델 타입: `monai`
- 다른 모델을 학습하려면 `--model-type monai_mini`를 지정
- 에포크마다 validation 실행
- `data/models/<model-type>/best.pt` — 검증 손실 최저 모델 자동 저장
- `data/models/<model-type>/last.pt` — 최종 에포크 모델 자동 저장
- `--resume` 플래그로 해당 모델 타입의 `last.pt`에서 이어서 학습 가능

#### 학습 타겟 변환

원본 어노테이션은 도구 팁 좌표(x, y)와 이진 분할 마스크(segmentation mask)로 구성된다. 이를 그대로 학습 타겟으로 사용하는 대신, 분할 마스크 내 각 픽셀에 팁까지의 거리에 반비례하는 gradient 값(0~1)을 부여한 히트맵으로 변환한다.

```text
이진 분할 마스크  +  팁 좌표 (x, y)
        │
        ▼
  연결요소별 팁 거리 계산
        │
        ▼
  팁 픽셀         → 1.0
  마스크 내 픽셀  → 1 - (팁까지 거리 / 컴포넌트 내 최대 거리)
  배경 픽셀       → 0.0
        │
        ▼
  거리 기반 히트맵 (float32, H × W)
```

이 히트맵을 학습 타겟으로 삼아 `MSE(sigmoid(pred[:, 1]), target)` 손실로 학습한다. 이진 마스크 대신 연속적인 gradient를 사용하면 팁 근방에 더 높은 학습 신호가 집중되어 팁 위치 예측 정밀도가 향상된다.

#### 전이 학습 (Transfer Learning)

MONAI 프레임워크 기반으로 동일한 아키텍처(EfficientNet-B2 인코더 + U-Net 디코더)를 사전학습한 가중치(`temp/monai.pt`)를 초기값으로 사용한다. `model.py`의 모듈 명칭(ADN, conv, adn.N 등)을 MONAI 호환 구조로 설계하여 state_dict를 직접 로드할 수 있게 했다.

```text
ttd.model.build(model_type)
       │
       ▼
TooltipDetector  (선택한 모델 타입의 인코더 + 디코더 + 분할 헤드)
       │
       ▼  파인튜닝
data/models/<model-type>/best.pt
```

현재 학습 스크립트는 선택한 `model_type`에 해당하는 아키텍처를 생성해 복강경 수술 데이터셋으로 전체 네트워크를 파인튜닝한다.

### 3. 평가

```bash
bin/eval-model
```

- 기본 모델 타입: `monai`
- 다른 모델을 평가하려면 `--model-type monai_mini`를 지정
- 테스트 세트에서 팁 탐지 정확도를 측정하고 결과를 `data/results/<model-type>/`에 저장한다.

| 저장 파일      | 내용                                                 |
| -------------- | ---------------------------------------------------- |
| `summary.json` | 전체 지표 + 세션별 지표 + 실행 파라미터              |
| `per_tip.csv`  | GT 팁 1개당 1행 (좌표, 예측값, 거리, 탐지 성공 여부) |

### 3-1. 속도 비교

```bash
bin/compare-speed
```

- 기본 설정은 test 셋에서 임의 샘플 1,000건을 골라 `monai`와 `monai_mini`의 추론 속도를 비교한다.
- 결과는 `data/results/speed-comparison.json`에 저장된다.

### 4. 탐지 결과 시각화

```bash
bin/tooltip-detector
```

학습된 모델로 데이터셋 프레임 또는 임의 이미지를 추론하고 결과를 인터랙티브하게 시각화한다.
GUI 상단의 `Model` 드롭다운으로 `monai`와 `monai_mini`를 전환할 수 있으며, 선택한 모델 타입의 `best.pt`를 다시 로드한다.

![Tooltip Detector GUI](images/tooltip-detector.png)

- **왼쪽 패널**: 원본 이미지 + segmentation mask overlay + GT 어노테이션(색상 원·바운딩 박스) + 예측 팁(빨간 X) + 오차 선(노랑)
- **오른쪽 패널**: 모델 출력 히트맵 (hot 컬러맵) + 예측 피크 마커
- **Threshold / NMS radius 슬라이더**: 실시간으로 파라미터를 조정하며 결과 확인
- **Inference time 표시**: 현재 프레임의 모델 forward 시간을 ms 단위로 표시
- **누적 통계 패널**: 탐색한 프레임의 hit-rate, 평균 거리 등 평가 지표를 실시간 집계

### 5. 동영상 실시간 추적

```bash
bin/tooltip-tracker
```

사용자가 선택한 동영상 파일을 프레임 단위로 순차 처리하며, 화면 중앙에서 탐지된 수술도구 팁 방향으로 화살표를 그려 내시경 카메라가 이동해야 할 방향을 안내한다. 화살표의 방향·길이는 프레임 간 급격히 바뀌지 않도록 스무딩되며, 오탐지(팁 미탐지·개수 이상·방향 모순)는 별도의 색상·경고 도형으로 표시된다.

- **`Model` 드롭다운**: `monai` / `monai_mini` 전환
- **`Method` 드롭다운**: 화살표 스무딩 구현 전환 (`CameraMotionVectorMagnitudeBlend` 기본값 / `CameraMotionVectorBlend` / `CameraMotionVectorKalman`)
- **Play / Pause, 탐색 바**: 벽시계 기준 재생(필요 시 프레임 드롭) 및 임의 프레임 이동
- **Threshold / NMS radius 슬라이더**: `tooltip-detector`와 동일한 피크 탐지 파라미터

상세 설계와 스무딩 로직, 오탐지 판정 규칙: [docs/tooltip-tracker.md](docs/tooltip-tracker.md)

## 스크립트 요약

| 스크립트               | 설명                       | 주요 인수                                                      |
| ---------------------- | -------------------------- | -------------------------------------------------------------- |
| `bin/train-model`      | 모델 학습                  | `--model-type`, `--epochs`, `--batch-size`, `--lr`, `--resume` |
| `bin/eval-model`       | 팁 탐지 정확도 평가        | `--model-type`, `--threshold`, `--nms-radius`, `--model`       |
| `bin/compare-speed`    | 모델 추론 속도 비교        | `--model-types`, `--num-samples`, `--batch-size`, `--workers`  |
| `bin/tooltip-detector` | 탐지 결과 시각화 GUI       | `--model-type`, `--model`, `--data-root`                       |
| `bin/tooltip-tracker`  | 동영상 실시간 추적 GUI     | `--model-type`, `--model`                                      |
| `bin/dataset-browser`  | 데이터셋 시각화 GUI        | `--split`, `--data-root`                                       |

직접 실행 예시:

```bash
# 경량 모델을 에포크 60회, 배치 8로 학습
uv run python -m ttd.train --model-type monai_mini --epochs 60 --batch-size 8

# 경량 모델 평가
uv run python -m ttd.eval --model-type monai_mini --threshold 0.3 --nms-radius 15

# 두 모델의 속도 비교
uv run python ttd/compare-speed.py --num-samples 1000
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

## 평가 결과

테스트 세트 기준 `threshold=0.5`, `NMS radius=20 px` 설정의 모델별 성능 요약.

| 모델         | Miss rate | Hit@10 px | Hit@20 px | Hit@50 px | Median dist | Mean dist |  P90 dist |
| ------------ | --------: | --------: | --------: | --------: | ----------: | --------: | --------: |
| `monai`      |    2.97 % |   38.10 % |   64.96 % |   80.95 % |    12.53 px |  39.76 px | 111.00 px |
| `monai_mini` |    3.20 % |   29.61 % |   62.00 % |   80.16 % |    14.14 px |  41.35 px | 115.52 px |

- `monai`는 현재 기준 모델이며, 정밀도(Hit@10 px)가 더 높다.
- `monai_mini`는 경량 모델로, 정확도와 속도의 trade-off 비교를 염두에 두고 추가되었다.

속도 비교 결과 (`data/results/speed-comparison.json`, CPU, test 샘플 1,000건, batch=16, workers=0):

| 모델         | 파라미터 수 | Forward / frame | Wall / frame | Forward FPS | Wall FPS |
| ------------ | ----------: | --------------: | -----------: | ----------: | -------: |
| `monai`      |  11,450,828 |       277.00 ms |    287.49 ms |        3.61 |     3.48 |
| `monai_mini` |   5,502,870 |       204.57 ms |    215.06 ms |        4.89 |     4.65 |

- `monai_mini`가 `monai`보다 forward 기준 **1.35배**, wall-clock 기준 **1.34배** 빠르다.
- 파라미터 수는 `monai_mini`가 `monai`의 약 48 % 수준으로 줄어든다.
- 정확도는 `monai`가 우세하지만, CPU 추론 속도까지 함께 보면 `monai_mini`는 경량 대안으로 의미가 있다.

상세 분석:
[docs/eval-results-monai.md](docs/eval-results-monai.md)
[docs/eval-results-monai_mini.md](docs/eval-results-monai_mini.md)

## 문서 목록

| 문서                                                               | 내용                                                                 |
| ------------------------------------------------------------------ | -------------------------------------------------------------------- |
| [docs/model-monai.md](docs/model-monai.md)                         | EfficientNet-B2 + U-Net 풀 모델 아키텍처 상세 설명                   |
| [docs/model-monai-mini.md](docs/model-monai-mini.md)               | EfficientNet-B2 + U-Net 경량 모델 아키텍처 상세 설명                 |
| [docs/dataset-guide.md](docs/dataset-guide.md)                     | 데이터셋 디렉터리 구조, 파일 포맷, 어노테이션 스펙, 히트맵 타겟 수식 |
| [docs/train-guide.md](docs/train-guide.md)                         | 학습 실행 방법, 인수 설명, 증강 파이프라인, 체크포인트, 출력 예시    |
| [docs/eval-guide.md](docs/eval-guide.md)                           | 평가 실행 방법, 피크 탐지 알고리즘, 지표 정의, 결과 파일 구조        |
| [docs/tooltip-detector.md](docs/tooltip-detector.md)               | 탐지 GUI 화면 구성, 추론 흐름, 조작 방법, 설계 상세                  |
| [docs/tooltip-tracker.md](docs/tooltip-tracker.md)                 | 추적 GUI 화면 구성, 화살표 스무딩 로직, 오탐지 판정 규칙, 설계 상세  |
| [docs/dataset-browser-guide.md](docs/dataset-browser-guide.md)     | GUI 화면 구성, 조작 방법, 키보드 단축키                              |
| [docs/eval-results-monai.md](docs/eval-results-monai.md)           | `monai` 테스트 세트 평가 결과 분석                                   |
| [docs/eval-results-monai_mini.md](docs/eval-results-monai_mini.md) | `monai_mini` 테스트 세트 평가 결과 분석                              |
