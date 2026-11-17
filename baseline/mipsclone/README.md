# RTMDet-Pose 수술도구·팁 탐지 베이스라인 (mipsclone)

[ROBUST_MIPS_toolpose](https://github.com/cai4cai/ROBUST_MIPS_toolpose)의 포즈 추정 모델(RTMPose)과, 그 앞단에서 도구 박스를 공급하는 탐지기 RTMDet에서 **구조 아이디어를 가져와** 하나의 end-to-end 모델로 합쳤다. 한 번의 순전파로 도구 박스와 각 도구의 팁을 함께 낸다.

원본을 충실히 재현하는 것이 목적이 아니다. 원본 가중치나 수치와 대조하지 않으며, COCO나 ImageNet 사전학습 없이 cholec80에서 처음부터 학습한다. 설계 결정과 그 근거는 [docs/IMPLEMENTATION-PLAN.md](docs/IMPLEMENTATION-PLAN.md)에 있다.

**추가 의존성이 없다.** 루트 프로젝트 환경(torch, torchvision, numpy, opencv, scipy, tqdm)만 쓰고 OpenMMLab 패키지는 쓰지 않는다.

## 무엇을 탐지하는가

어노테이션의 도구 하나가 인스턴스 하나다. 인스턴스는 도구 박스와 그 도구의 팁(키포인트 1개)으로 이루어진다.

| 출력        | 정의                                   | 출처                 |
| ----------- | -------------------------------------- | -------------------- |
| `tool` 박스 | 어노테이션의 바운딩 박스               | `annotations[].bbox` |
| 팁 (K = 1)  | 그 박스에 대해 포즈 가지가 낸 키포인트 | `annotations[].tip`  |

형제 클론(yolov8sclone, yolo26clone, cladnet)은 팁을 32 px 상자인 별도 클래스로 학습하므로, 어느 팁이 어느 도구의 것인지 모델이 알지 못한다. 이 모델은 팁을 **그 도구의 키포인트로** 학습한다.

## 구조

```text
입력 640 × 640
  │
  ▼ CSPNeXt 백본 → CSPNeXt-PAFPN       (RTMDet / RTMPose 공통)
  P3 (/8) · P4 (/16) · P5 (/32)
  │                       │
  ▼ RTMDet SepBN 헤드      ▼ 포즈 가지 (RTMPose의 RTMCC 헤드)
  도구 점수 + 박스          RoIAlign(P3, 박스 × 1.25) → Conv × 4 → 7×7 Conv
  → NMS                     → ScaleNorm + Linear → GAU → SimCC x / y (256 bin)
```

원래 RTMPose는 탐지 박스로 이미지를 잘라 자기 백본에 다시 넣는다. 여기서는 포즈 가지가 탐지기와 **같은 특징 맵(P3)** 에서 박스 영역을 RoIAlign으로 읽는다. 그래서 백본과 넥이 프레임당 한 번만 돌고, 탐지 손실과 포즈 손실이 같은 특징을 함께 학습한다.

| 크기 (`--scale`) | 파라미터 | 구성                                                   |
| ---------------- | -------: | ------------------------------------------------------ |
| `tiny`           |    5.83M |                                                        |
| `s` (기본)       |   10.08M | 백본 4.38M, 넥 3.88M, 탐지 헤드 0.60M, 포즈 가지 1.22M |
| `m`              |   26.63M |                                                        |

참고로 yolov8sclone은 11.14M(2클래스), yolo26clone은 9.95M(2클래스)이다.

### 손실

```text
L = L_QFL + 2 · L_GIoU + pose_weight · L_SimCC
```

- **탐지 (RTMDet 그대로):** DynamicSoftLabelAssigner, Quality Focal Loss(목표값은 예측 박스의 IoU), IoU로 가중한 GIoU.
- **포즈:**
  - 정답 도구 박스를 중심 ±10 %, 크기 ±10 % 범위에서 흔든 RoI로 학습한다.
  - RTMPose의 KL 손실(beta 10, 목표 분포에 softmax)을 쓰되, bin에 대해 평균 대신 합을 쓴다. 공유 백본에서 포즈 그래디언트가 묻히지 않게 하려는 것이다.
  - 팁이 RoI 밖이면 그 인스턴스의 포즈 손실은 0이다.

### 학습 레시피

RTMDet 설정을 단일 GPU에 맞췄다.

- 옵티마이저와 스케줄: AdamW(lr 1e-3, wd 0.05), 1,000 iter 웜업, 전반부 고정, 후반부 cosine.
- 가중치 평균: ExpMomentumEMA. 검증과 저장에는 평균 가중치를 쓴다.
- 증강: 모자이크를 쓰다가 마지막 10 에포크는 무작위 리사이즈-크롭으로 바꾼다.

## 디렉터리 구조

```text
baseline/mipsclone/
├── run(.bat)              # `run <script> [args...]` → scripts/<script>.py
├── common/
│   ├── model.py           # 백본, 넥, 탐지 헤드, 포즈 가지, 추론(NMS + 팁 디코딩)
│   ├── loss.py            # assigner, QFL, GIoU, SimCC KL
│   ├── boxes.py           # IoU/GIoU, 거리 디코딩, letterbox
│   ├── dataset.py         # 도구 박스 + 팁 쌍, 모자이크 / 리사이즈-크롭
│   ├── inference.py       # 체크포인트 포맷, 프레임 단위 추론
│   ├── metrics.py         # 탐지 AP        (yolo26clone에서 복사)
│   ├── tipmetrics.py      # 팁 지표        (yolo26clone에서 복사, 같은 판정 규칙)
│   └── progress.py        #                (yolo26clone에서 복사)
├── scripts/
│   ├── train-model.py
│   └── eval-model.py
├── docs/IMPLEMENTATION-PLAN.md
└── data/                  # git 추적 제외
    ├── model/<dataset>/   #   model.pt, model-last.pt, metric.csv, train-status.json
    └── results/<dataset>/<split>[-gtbox]/   # summary.json, per_tip.csv
```

## 사용법

자체 `.venv`가 없고 루트 환경을 쓴다(`uv sync`를 루트에서 한 번 실행). 어느 디렉터리에서 실행해도 되고, Windows에서는 `run.bat`을 쓴다. `cuda:0`은 PCIe 성능이 떨어지므로 `cuda:1`〜`cuda:3`을 지정한다.

### 학습

```bash
./baseline/mipsclone/run train-model --dataset cholec80 --frame-stride 5 --val-frames 1500 --workers 12 --device cuda:1
```

| 인수              | 기본값 | 설명                                              |
| ----------------- | ------ | ------------------------------------------------- |
| `--dataset`       | (필수) | `data/dataset/` 아래 이름                         |
| `--scale`         | `s`    | `tiny` / `s` / `m` / `l`                          |
| `--epochs`        | 150    |                                                   |
| `--stage2-epochs` | 10     | 모자이크 없이 학습할 마지막 에포크 수             |
| `--batch-size`    | 32     |                                                   |
| `--lr`            | 1e-3   | AdamW                                             |
| `--pose-weight`   | 1.0    | 포즈 손실 가중치                                  |
| `--roi-size`      | 16     | RoIAlign 출력 한 변                               |
| `--roi-padding`   | 1.25   | RoI를 박스보다 넓히는 배율                        |
| `--bins`          | 256    | SimCC 축당 bin 수                                 |
| `--frame-stride`  | 1      | N 프레임마다 1장만 학습에 쓴다                    |
| `--val-frames`    | 2000   | 에포크마다 평가할 val 프레임 수 상한 (0이면 전체) |
| `--no-resume`     |        | `model-last.pt`가 있어도 처음부터 다시 학습       |

`model-last.pt`가 있으면 **기본 동작이 재개**다. optimizer, AMP scaler, EMA, iteration 수까지 복원한다. 최적 모델(`model.pt`)은 val의 팁 F1 @ 20 px(헝가리안 매칭, conf 0.25) 기준으로 고른다.

### 평가

```bash
./baseline/mipsclone/run eval-model --dataset cholec80 --device cuda:1              # 자체 탐지 박스
./baseline/mipsclone/run eval-model --dataset cholec80 --device cuda:1 --boxes gt   # 정답 박스 (상한)
```

- **탐지 지표:** `tool`의 AP@0.5, AP@0.5:0.95, precision, recall.
- **팁 지표:** conf 0.25 이상인 박스마다 팁 하나. miss rate, Hit-rate @ 10/20/50 px, 헝가리안 TP/FP/FN, 거리. 형제 베이스라인과 같은 판정 규칙이다.
- 거리는 원본 프레임 좌표(736 × 480)에서 잰다.
- **`--boxes gt`는 비교에 쓰지 않는다.** cholec80에서는 팁의 97 %가 박스 가장자리 근처에 있어, 정답 박스가 팁 위치를 상당 부분 알려 준다.
  cholec80 test(10프레임마다 하나, 도구 14,402개)에서 모델 없이 박스만으로 낸 기준선은 다음과 같다.
  - 정답 박스의 네 모서리 중 팁에 가장 가까운 것을 고르면(정답을 아는 오라클) hit@20이 75.4 %다.
  - 모든 박스에서 같은 상대 위치(왼쪽 아래 부근)를 찍으면 37.8 %다.

## 검증

- 모델 생성, 순전파·역전파(AMP), 포즈 손실의 그래디언트가 백본까지 전달되는지 확인했다.
- 증강(모자이크, 리사이즈-크롭, 반전) 후 박스와 팁이 맞는지 표본을 그려 확인했다.
- 학습, 재개, 모자이크 전환, 체크포인트 저장이 소량 설정에서 끝까지 동작함을 확인했다.
- **짧은 학습에서 실제로 학습됨을 확인했다.** `--frame-stride 10`(7,104프레임)으로 8 에포크(약 1,770 iter) 학습한 뒤 test를 50프레임마다 하나씩(1,965프레임, 팁 2,968개) 평가한 결과는 아래와 같다.
  이 결과는 동작 확인용이다. 학습량과 평가 범위가 형제 베이스라인(150 에포크, test 전수)과 달라서 비교할 수 없다.

  | 박스        | tool AP@0.5 | hit@20 | 중앙 거리 | 헝가리안 20 px P / R | ms/frame |
  | ----------- | ----------: | -----: | --------: | -------------------- | -------: |
  | 자체 탐지   |       0.912 | 81.8 % |    5.1 px | 0.482 / 0.817        |      8.3 |
  | 정답 (상한) |             | 89.6 % |    4.3 px | 0.894 / 0.894        |          |

  precision이 낮은 것은 conf 0.25를 넘는 중복·저품질 박스가 많기 때문이다. 박스 하나가 팁 하나를 내므로, 남는 박스는 그대로 FP 팁이 된다.
- **150 에포크 본 학습과 test 전수 평가는 아직 하지 않았다.**

## 한계

- **하이퍼파라미터는 튜닝하지 않았다.** lr, 포즈 손실 가중치, RoI 크기, bin 수, RoI 흔들기 폭은 근거 있는 출발값일 뿐이다.
- **도구 골격을 학습하지 않는다.** cholec80에는 팁 하나뿐이라, 원 논문의 장점인 entry, hinge, tip1, tip2 구조 학습이 재현되지 않는다. GAU도 키포인트 사이 attention 없이 게이트가 달린 MLP처럼 동작한다.
- **MixUp과 flip test는 넣지 않았다.** RTMDet은 MixUp을, RTMPose는 flip test를 쓴다.
- **데모 GUI, `generate-summary`, dashboard·tooltip-tracker 연동은 아직 없다.**

## 참고

- RTMDet: Lyu et al., *RTMDet: An Empirical Study of Designing Real-Time Object Detectors*, 2022
- RTMPose: Jiang et al., *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*, 2023
- ROBUST-MIPS: Han et al., arXiv:2508.21096, 2025
- SimCC: Li et al., *SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation*, ECCV 2022
