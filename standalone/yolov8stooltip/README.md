# yolov8stooltip: 도구 박스와 팁을 한 쌍으로 내는 YOLOv8s-pose 모델

[baseline/yolov8sclone](../../baseline/yolov8sclone/)은 `tool` 박스와 `tip` 박스를 **서로 무관한 두 클래스**로 탐지한다. 어느 팁이 어느 도구의 것인지는 출력에 없어서, 쓰는 쪽이 후처리 규칙으로 짝을 지어야 한다(예: [scripts/interactive-trainer.py](../../scripts/interactive-trainer.py)의 `pair_detections`). 그 결과 도구 박스가 가까이 겹친 프레임에서는 다른 도구의 팁을 고르는 일이 수작업 레이블링 중에 자주 발견됐다.

이 서브 프로젝트는 백본·넥은 yolov8sclone 그대로 두고, 탐지 헤드만 **YOLOv8-pose** 구조로 바꾼 새 모델이다. 탐지 하나가 도구 박스, `tool` 점수, 그 도구의 팁 (x, y)를 **같은 앵커 포인트에서 함께** 낸다. 팁이 처음부터 자기 박스에 묶여 나오므로 예측 뒤 짝짓기가 필요 없고, 학습도 데이터셋 레이블의 `{bbox, tip}` 1대1 대응을 그대로 쓴다.

기존 어느 모델과도 같지 않아 베이스라인이 아니고, 학습·평가 구조(라벨 할당, 손실, NMS, AP 평가)가 루트 프로젝트의 히트맵 모델(`ttd`)과 전혀 달라 별도 서브 프로젝트로 두었다. 코드는 yolov8sclone의 `common/`을 복사해 고친 이 프로젝트 자신의 사본이며, 다른 서브 프로젝트의 코드를 import하지 않는다.

## yolov8sclone과 무엇이 다른가

|              | yolov8sclone (`tooltip`)                 | yolov8stooltip                                       |
| ------------ | ---------------------------------------- | ---------------------------------------------------- |
| 클래스       | `tool`, `tip` 2개                        | `tool` 1개                                           |
| 팁 표현      | 팁 중심의 32 × 32 px 상자 (별도 탐지)    | 각 `tool` 탐지에 딸린 키포인트 (x, y)                |
| 박스-팁 대응 | 없음. 후처리 규칙으로 추정               | 같은 앵커에서 나오므로 구조적으로 1대1               |
| 헤드         | Detect (박스 분기 + 클래스 분기)         | Pose (Detect + 키포인트 분기 `cv4`)                  |
| 손실         | box 7.5 / cls 0.5 / dfl 1.5              | 같은 세 항 + **pose 12.0** (OKS형 키포인트 손실)     |
| 라벨 한 행   | `[class, cx, cy, w, h]` (도구·팁 따로)   | `[class, cx, cy, w, h, kx, ky, kv]` (도구 하나)      |
| 파라미터     | 11.14 M                                  | 11.42 M                                              |

백본(0〜9)과 넥(10〜21)은 층 하나까지 같다. 바뀐 것은 22번 헤드뿐이다.

## 구조

```text
입력 640 × 640 (736 × 480 프레임을 letterbox)
    │
    ▼  백본 + 넥: yolov8sclone과 동일 → N3 (80×80), N4 (40×40), N5 (20×20)
    │
    ▼  22 Pose 헤드 (레벨마다 같은 앵커 포인트에서 세 분기)
  박스 분기:     Conv → Conv → Conv1×1 → 4 × 16 DFL 로짓 → 도구 박스
  클래스 분기:   Conv → Conv → Conv1×1 → 1 로짓            → tool 점수
  키포인트 분기: Conv → Conv → Conv1×1 → 2 값              → 그 도구의 팁 (x, y)
    │
    ▼  decode: 앵커 8,400개 × [cx, cy, w, h, score, kx, ky]
    ▼  NMS: 박스 기준으로 줄이며 팁 좌표는 자기 박스를 따라간다
  (n, 8) = [x1, y1, x2, y2, score, class, kx, ky]
```

키포인트는 YOLOv8-pose와 같은 방식으로 복원한다. 격자 칸 중심 a, stride s인 앵커의 원시값 r은 `(2r + a − 0.5) × s` 픽셀이 된다. 팁이 앵커 칸 안에 있을 필요가 없어서, 도구 박스 안쪽 앵커가 박스 끝의 팁을 가리킬 수 있다.

팁에는 visibility 채널이 없다. 데이터셋의 모든 도구에 팁이 있어서 가르칠 근거가 없기 때문이다.

### 손실

```text
Loss = 7.5 · L_box + 0.5 · L_cls + 1.5 · L_dfl + 12.0 · L_pose
L_pose = 양성 앵커 평균 (1 − exp(−e)),   e = d² / ((2σ)² · area · 2)
```

- 앞의 세 항은 yolov8sclone과 같다. TaskAlignedAssigner가 도구 하나에 양성 앵커를 고르면, 그 앵커들은 **바로 그 도구의** 팁으로 키포인트를 회귀한다. 박스를 고르는 할당이 팁도 함께 고르므로 박스와 팁이 한 쌍으로 학습된다.
- `d`는 팁 오차, `area`는 도구 박스 넓이다(둘 다 격자 단위). OKS처럼 오차를 도구 크기에 비례해 잰다.
- **σ = 0.05** (`--kpt-sigma`). YOLOv8-pose는 사용자 정의 키포인트에 σ = 1/키포인트 수를 쓰는데, 키포인트가 하나면 1이 된다. 이 값으로는 300 × 150 px 도구에서 팁이 20 px 틀려도 e ≈ 0.001로 손실이 거의 없다. COCO의 키포인트별 σ는 0.025(눈)〜0.107(엉덩이)이고, 0.05면 같은 오차가 e ≈ 0.4, 60 × 60 px 도구의 10 px 오차가 e ≈ 1.4다.
- mosaic 크롭이 도구 박스 일부는 남기고 팁만 잘라내면, 그 팁은 키포인트 손실에서 뺀다(라벨의 `kv` = 0). 캔버스 가장자리로 팁을 끌어당기지 않기 위해서다. `kv`는 어노테이션이 아니라 학습 중에만 쓰는 마스크다.

## 평가 지표

`eval-model`은 모든 거리를 원본 프레임(736 × 480) 기준으로 잰다.

| 분류      | 지표                                                     | 설명                                                                                                                                                            |
| --------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| detection | `tool` AP@0.5, AP@0.5:0.95, P, R                         | 도구 박스 탐지 성능                                                                                                                                             |
| detection | `tip` AP@0.5, AP@0.5:0.95, P, R                          | 예측 팁 둘레에 32 px 상자(`--tip-box-size`)를 만들어 도구 점수로 채점. 베이스라인의 `tip` AP와 같은 상자 크기에서 같은 뜻이다                                   |
| tip       | miss rate, Hit-rate @ 10/20/50 px, 거리 중앙값·평균·P90 | 루트 프로젝트 `scripts/eval-model.py`와 같은 매칭 규칙(최근접 + Hungarian 1:1)                                                                                  |
| pairs     | box matched, own tip @ 10/20/50 px                       | **이 모델을 만든 이유를 재는 지표.** GT 도구와 예측 도구를 박스 IoU ≥ 0.5로 1:1 매칭한 뒤, 예측 도구의 팁이 **그 GT 도구 자신의** 팁에서 N px 안에 있는 비율 |

tip 지표는 팁을 점의 집합으로만 보므로, 팁이 다른 도구에 붙어 나와도 위치만 가까우면 맞은 것으로 센다. pairs 지표는 이런 뒤바뀜을 실패로 센다.

학습 중 체크포인트는 베이스라인과 같은 기준으로 고른다. 검증 mAP@0.5:0.95(`tool`과 32 px `tip` 상자의 평균)가 가장 높은 에포크다.

## 디렉터리 구조

```text
standalone/yolov8stooltip/
├── run(.bat)                 # `run <script> [args...]` → scripts/<script>.py 실행
├── common/
│   ├── model.py              # YOLOv8s 백본·넥 + Pose 헤드, 키포인트 디코딩
│   ├── boxes.py              # xywh/xyxy, CIoU, letterbox, 키포인트를 함께 넘기는 NMS
│   ├── assigner.py           # TaskAlignedAssigner (yolov8sclone과 동일)
│   ├── loss.py               # box + cls + dfl + OKS형 키포인트 손실
│   ├── dataset.py            # 어노테이션 → 도구당 (박스, 팁) 한 행, mosaic 증강
│   ├── metrics.py            # AP@0.5, AP@0.5:0.95, 비교용 팁 상자 생성
│   ├── tipmetrics.py         # Hit-rate @ N px (루트와 같은 규칙) + 도구-팁 쌍 지표
│   ├── inference.py          # 체크포인트 포맷 + 프레임 1장 추론
│   └── progress.py           # tqdm 공통 설정
├── scripts/
│   ├── train-model.py        # 학습
│   └── eval-model.py         # 평가 (탐지 AP + 팁 hit-rate + 쌍 지표)
└── data/                     # 체크포인트·평가 결과 (git 추적 제외)
    ├── model/<dataset>/      #   model.pt, model-last.pt, train-status.json, metric.csv
    └── results/<dataset>/<split>/   #   summary.json, per_tip.csv
```

## 설치

없다. yolov8sclone처럼 자체 `.venv` 없이 루트 환경을 그대로 쓴다. `run`이 `uv run --project <저장소 루트>`를 호출한다.

```bash
uv sync        # 루트에서 한 번 (이미 했다면 불필요)
```

## 사용법

어느 디렉터리에서 실행해도 되고, Windows에서는 `run.bat`을 쓴다. 이 장비에서는 cuda:0의 PCIe 대역이 떨어져 있으므로 `--device cuda:1`〜`cuda:3`을 지정한다.

### 1. 학습

```bash
./standalone/yolov8stooltip/run train-model --dataset cholec80 --device cuda:1 \
    --frame-stride 5 --val-frames 1500 --workers 12
./standalone/yolov8stooltip/run train-model --dataset erop --device cuda:2 \
    --frame-stride 5 --val-frames 1500 --workers 12
```

위 명령은 yolov8sclone 본 학습([commands.md](../../baseline/yolov8sclone/docs/commands.md))과 같은 설정이다. 사전학습 가중치 없이 무작위 초기화에서 시작한다.

| 인수                       | 기본값 | 설명                                                                        |
| -------------------------- | ------ | --------------------------------------------------------------------------- |
| `--dataset`                | (필수) | `data/dataset/` 아래 디렉터리 이름 (`cholec80` / `erop`)                    |
| `--epochs`                 | 150    |                                                                             |
| `--batch-size`             | 16     |                                                                             |
| `--lr`                     | 0.01   | SGD, momentum 0.937, nesterov. 3에포크 warmup 뒤 cosine 감쇠                |
| `--kpt-sigma`              | 0.05   | 키포인트 손실의 OKS σ. 작을수록 팁 오차에 엄격하다                          |
| `--tip-box-size`           | 32     | 검증 `tip` AP용 상자 한 변(원본 프레임 px). 학습 자체에는 쓰이지 않는다     |
| `--frame-stride`           | 1      | N프레임마다 1장만 학습에 쓴다                                               |
| `--val-frames`             | 2000   | 에포크마다 평가할 val 프레임 수 상한 (0이면 전체)                           |
| `--output-dir`             |        | 체크포인트 디렉터리 (기본 `data/model/<dataset>/`)                          |
| `--no-ema` / `--no-resume` |        | EMA 끄기 / 처음부터 다시 학습. `model-last.pt`가 있으면 기본 동작이 재개다 |

### 2. 평가

```bash
./standalone/yolov8stooltip/run eval-model --dataset cholec80 --device cuda:1
./standalone/yolov8stooltip/run eval-model --dataset erop --device cuda:2
```

결과는 `data/results/<dataset>/<split>/`에 `summary.json`과 `per_tip.csv`로 저장된다. `per_tip.csv`는 루트 프로젝트·베이스라인과 열 이름이 같아 같은 방식으로 다시 집계할 수 있다. `--split`(기본 `test`), `--limit`, `--frame-stride`, `--conf`(기본 0.25), `--map-conf`(AP 곡선용, 기본 0.001)를 받는다.

## 검증 현황

<!-- 짧은 검증 학습 결과는 아래에 기록한다. -->

## 한계

- **본 학습은 아직 하지 않았다.** 위 검증은 구현이 학습되는지 확인한 짧은 실행이다. yolov8sclone과 성능을 비교하려면 같은 설정으로 150에포크를 학습해야 한다(yolov8sclone 기준 cholec80 4.1시간).
- **σ는 조정하지 않은 값이다.** 0.05는 COCO 범위에서 고른 출발점이며, 팁 정밀도와 학습 안정성 사이의 실제 최적값은 실험으로 정해야 한다.
- **도구 크기에 비례하는 허용 오차.** OKS형 손실이라 큰 도구일수록 같은 픽셀 오차를 덜 벌한다. 팁 위치는 도구 크기와 무관하게 픽셀로 평가되므로, 큰 도구의 팁 정밀도가 상대적으로 덜 학습될 수 있다.
- **interactive-trainer는 아직 yolov8sclone을 쓴다.** 이 모델로 바꾸면 `pair_detections`의 후처리 짝짓기가 필요 없어진다.
- 데모 GUI는 없다.
