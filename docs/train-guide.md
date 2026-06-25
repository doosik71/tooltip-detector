# 학습 가이드

`TooltipDetector` 모델을 학습하기 위한 절차와 설정 옵션을 설명한다.

## 사전 준비

의존성 설치:

```bash
uv sync
```

학습에 필요한 디렉터리 구조:

```text
data/dataset/
├── annotation/  train / val / test
├── images/      train / val / test
└── segmentation/ train / val / test
```

데이터셋 구조와 파일 포맷은 [dataset-guide.md](dataset-guide.md)를 참고한다.

## 실행

```bash
bin/train-model          # Linux / macOS
bin\train-model.bat      # Windows
```

직접 실행:

```bash
uv run python -m ttd.train
```

## 인수

| 인수           | 기본값         | 설명                                  |
| -------------- | -------------- | ------------------------------------- |
| `--data-root`  | `data/dataset` | 데이터셋 루트 디렉터리                |
| `--model-dir`  | `data/model`   | 체크포인트 저장 디렉터리              |
| `--epochs`     | `30`           | 총 에포크 수                          |
| `--batch-size` | `16`           | 배치 크기                             |
| `--lr`         | `1e-4`         | 초기 학습률 (Adam optimizer)          |
| `--workers`    | `4`            | DataLoader 워커 수                    |
| `--resume`     | off            | `<model-dir>/last.pt`에서 이어서 학습 |

예시 — 에포크·배치 크기 변경:

```bash
uv run python -m ttd.train --epochs 60 --batch-size 8
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

각 에포크 종료 시 두 개의 파일이 저장된다.

| 파일                 | 저장 조건        | 설명                              |
| -------------------- | ---------------- | --------------------------------- |
| `data/model/last.pt` | 매 에포크        | 가장 최근 에포크의 모델 가중치    |
| `data/model/best.pt` | val loss 개선 시 | 검증 손실이 가장 낮은 모델 가중치 |

`data/model/` 디렉터리는 없으면 자동으로 생성된다.

학습 재개:

```bash
bin/train-model --resume
```

`--resume` 플래그를 사용하면 `data/model/last.pt`를 불러와 이어서 학습한다. 파일이 없으면 처음부터 시작한다.

## 진행 상태 출력

에포크별로 학습/검증 진행 상황이 터미널에 실시간으로 표시된다.

```text
Device     : cuda
Train      : 108,424 samples  (6,777 batches)
Val        : 36,140 samples  (2,259 batches)
Epochs     : 30   batch=16   lr=1.00e-04
Checkpoints: data/model/best.pt  /  data/model/last.pt

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
