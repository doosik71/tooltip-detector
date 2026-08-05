# 학습 가이드

`TooltipDetector` 모델을 학습하기 위한 절차와 설정 옵션을 설명한다.

## 사전 준비

의존성 설치:

```bash
uv sync
```

학습에 필요한 디렉터리 구조 (`<dataset-name>`은 `--dataset`으로 지정, 예: `erop`, `cholec80`):

```text
data/dataset/<dataset-name>/
├── annotation/  train / val / test
├── images/      train / val / test
└── segmentation/ train / val / test
```

데이터셋 구조와 파일 포맷, 데이터셋별 통계는 [dataset-guide.md](dataset-guide.md)를 참고한다.

## 실행

```bash
run train-model --dataset cholec80          # Linux / macOS
run.bat train-model --dataset cholec80      # Windows
```

## 인수

| 인수                 | 기본값                                                | 설명                                                     |
| ------------------ | -------------------------------------------------- | ------------------------------------------------------ |
| `--dataset`        | (필수)                                               | `data/dataset/` 아래 데이터셋 이름 (`erop` / `cholec80`)       |
| `--model-type`     | `monai`                                            | 모델 아키텍처 (`monai` / `monai_mini`)                       |
| `--target-mode`    | `gradient-seg`                                     | 학습 타겟 생성 방식 (`gradient-seg` / `gaussian-tip`)          |
| `--gaussian-sigma` | `15.0`                                             | gaussian 표준편차(px). `--target-mode gaussian-tip`일 때만 사용 |
| `--data-root`      | `data/dataset`                                     | `<dataset-name>/` 서브디렉터리를 담는 루트 디렉터리                   |
| `--model-dir`      | `data/models/<dataset>/<target-mode>/<model-type>` | 체크포인트 저장 디렉터리                                          |
| `--epochs`         | `30`                                               | 총 에포크 수                                                |
| `--batch-size`     | `16`                                               | 배치 크기                                                  |
| `--lr`             | `1e-4`                                             | 초기 학습률 (Adam optimizer)                                |
| `--workers`        | `4`                                                | DataLoader 워커 수                                        |
| `--no-resume`      | off (기본값은 재개)                                 | `<model-dir>`의 기존 체크포인트를 무시하고 처음부터 학습          |

타겟 생성 방식(`gradient-seg`/`gaussian-tip`)의 차이는 [dataset-guide.md](dataset-guide.md)의 "학습 타겟 생성 방법"을 참고한다.

예시 — 에포크·배치 크기 변경:

```bash
run train-model --dataset cholec80 --epochs 60 --batch-size 8
```

예시 — segmentation mask 없이 팁 좌표만으로 학습:

```bash
run train-model --dataset cholec80 --target-mode gaussian-tip --gaussian-sigma 15
```

## 데이터 증강

학습 시 아래 순서로 무작위 증강을 적용한다. 공간 변환은 이미지와 타겟 마스크에 동시 적용된다.

| 변환                | 확률 | 파라미터                                             |
| ------------------- | ---- | ---------------------------------------------------- |
| `HorizontalFlip`    | 0.5  | —                                                    |
| `VerticalFlip`      | 0.2  | —                                                    |
| `Affine`            | 0.6  | 이동 ±5 %, 스케일 0.85–1.15, 회전 ±20°               |
| `RandomResizedCrop` | 0.5  | scale 0.7–1.0, 원본 가로세로비 ±10 %                 |
| `Resize`            | 항상 | 480 × 736 px                                         |
| `ColorJitter`       | 0.6  | brightness/contrast ±0.3, saturation ±0.2, hue ±0.05 |
| `GaussianBlur`      | 0.3  | kernel 3–7                                           |
| `GaussNoise`        | 0.3  | —                                                    |
| `Normalize`         | 항상 | ImageNet mean/std                                    |

검증(val)·평가(test) 시에는 `Resize` + `Normalize`만 적용한다.

## 손실 함수

모델 출력의 채널 1(tool channel)에 sigmoid를 적용한 값과 히트맵 타겟 사이의 MSE 손실을 사용한다.

```text
loss = MSE(sigmoid(pred[:, 1]), target)
```

타겟 히트맵 생성 방법은 [dataset-guide.md](dataset-guide.md)의 "학습 타겟 생성 방법" 항목을 참고한다.

## 옵티마이저 및 스케줄러

- **Optimizer**: Adam (`lr=1e-4`)
- **Scheduler**: CosineAnnealingLR (`T_max=epochs`) — 에포크 종료 후 step

## 체크포인트

각 에포크 종료 시 세 개의 파일이 `data/models/<dataset>/<target-mode>/<model-type>/`에 저장된다.

| 파일                  | 저장 조건        | 설명                                                        |
| --------------------- | ---------------- | ----------------------------------------------------------- |
| `last.pt`             | 매 에포크        | 가장 최근 에포크의 모델 가중치                               |
| `best.pt`             | val loss 개선 시 | 검증 손실이 가장 낮은 모델 가중치                            |
| `train-status.json`   | 매 에포크        | 재개용 진행 기록 (완료 에포크 수, best val loss, 실행 하이퍼파라미터, 누적 학습 시간) |
| `metric.csv`          | 매 에포크        | 에포크별 학습 곡선 기록 (아래 "학습 곡선" 항목 참고), 재개 시 이어서 추가(append)됨 |

`data/models/<dataset>/<target-mode>/<model-type>/` 디렉터리는 없으면 자동으로 생성된다. 데이터셋·모델 타입·타겟 생성 방식이 다르면 서로 다른 디렉터리에 저장되므로 체크포인트가 덮어써지지 않는다.

### 학습 재개 (기본 동작, `--no-resume`으로 끄기)

외부 요인(서버 재부팅, OOM, 타임아웃 등)으로 학습이 중단돼도 **같은 명령을 그대로 다시 실행하면** 이어서 학습한다 — 재개가 기본 동작이며 별도 플래그가 필요 없다.

```bash
run train-model --dataset cholec80              # 최초 실행
run train-model --dataset cholec80              # 중단 후 다시 실행하면 자동으로 이어서 학습
```

동작 방식:

- `<model-dir>`에 `train-status.json`이 있으면 그 안의 `completed_epochs + 1` 에포크부터, 기록된 `best_val_loss`와 학습률 스케줄(`CosineAnnealingLR`) 진행 위치를 그대로 이어서 학습한다. `best.pt`를 재개 이전보다 더 나쁜 모델로 덮어쓰지 않도록 `best_val_loss`도 함께 복원된다.
- optimizer(Adam)의 momentum·분산 추정치는 저장하지 않으므로 재개 시 새로 초기화된다 — 학습률 스케줄만 정확히 이어지고, 옵티마이저 내부 상태는 근사적으로만 복원된다.
- 재개 명령의 `--epochs`가 새로운 총 목표 에포크 수가 된다. 이전 실행의 `--epochs`와 달라도 되며(예: 30 → 50), 학습률 스케줄은 새 `--epochs` 기준으로 재계산된다.
- 이미 `--epochs` 목표를 달성한 상태에서 재개하면 아무 것도 하지 않고 안내 메시지만 출력한다. 더 학습시키려면 더 큰 `--epochs`를 지정한다.
- `train-status.json`이 없고 `last.pt`만 있는 경우(이 기능이 추가되기 전에 저장된 체크포인트 등)는 가중치만 불러오고 에포크 카운트는 1부터 다시 시작한다.
- `metric.csv`도 이어서 기록된다(기존 행 뒤에 새 행을 추가) — 재개 여부와 무관하게 전체 학습 과정의 학습 곡선을 한 파일로 볼 수 있다.

`--no-resume`을 지정하면 `<model-dir>`에 있는 `last.pt`·`best.pt`·`train-status.json`·`metric.csv`를 전부 무시하고 처음부터 새로 학습한다(에포크가 진행되면서 이 파일들은 덮어써진다).

```bash
run train-model --dataset cholec80 --no-resume
```

## 학습 곡선 (`metric.csv`)

각 에포크 종료 시 `metric.csv`에 한 행씩 추가된다. train/val 각각에 대해, 모델 출력 히트맵 `sigmoid(pred[:, 1])`과 타겟 히트맵의 픽셀별 오차(`pred - target`)로부터 계산한 통계다 — GT 팁 좌표를 이용한 탐지 기반 오차가 아니라 히트맵 회귀 자체의 오차이므로 매 에포크 계산해도 비용이 거의 들지 않는다.

| 컬럼                              | 설명                                                  |
| --------------------------------- | ----------------------------------------------------- |
| `epoch`                           | 에포크 번호                                            |
| `train_loss` / `val_loss`         | `MSE(sigmoid(pred[:, 1]), target)`                     |
| `train_mae` / `val_mae`           | 평균 절대 오차 `mean(\|pred - target\|)`               |
| `train_me` / `val_me`             | 평균 신호 오차(편향) `mean(pred - target)`             |
| `train_std` / `val_std`           | 오차의 표준편차                                        |
| `lr`                              | 해당 에포크 시작 시점의 학습률                         |
| `epoch_sec`                       | 해당 에포크 소요 시간(초)                              |
| `elapsed_sec`                     | 누적 학습 시간(초), 재개 이전 세션 포함                |

팁 탐지 정확도(hit-rate, 픽셀 거리 등)를 보려면 학습이 끝난 뒤 `run eval-model`을 사용한다 ([eval-guide.md](eval-guide.md) 참고).

## 진행 상태 출력

에포크별로 학습/검증 진행 상황이 터미널에 실시간으로 표시된다. 아래는 `erop` 데이터셋으로 학습했을 때의 실제 출력 예시다.

```text
Device     : cuda
Dataset    : erop
Model type : monai
Target mode: gradient-seg
Train      : 108,424 samples  (6,777 batches)
Val        : 36,140 samples  (2,259 batches)
Epochs     : 30   batch=16   lr=1.00e-04
Checkpoints: data/models/erop/gradient-seg/monai/best.pt  /  data/models/erop/gradient-seg/monai/last.pt
Metrics    : data/models/erop/gradient-seg/monai/metric.csv

Epoch   1/30  lr=1.00e-04
  [train] |████████████░░░░░░░░░░░░░░░░░░|  800/6777  loss=0.021345  00:48<05:43
  [val  ] |██████████████████████████████| 2259/2259  loss=0.018921  01:12<00:00
   ★ train=0.021345  val=0.018921  epoch=02:01  elapsed=02:01  eta=58:22
    best.pt updated  (val_loss=0.018921)

Epoch   2/30  lr=9.98e-05
  ...
```

- `|████░░░|` 진행 바: 현재 배치 / 전체 배치
- `loss=`: 현재까지의 평균 손실
- `00:48<05:43`: 경과 시간 < 남은 예상 시간
- `★`: 해당 에포크에서 val loss 최저치 경신
