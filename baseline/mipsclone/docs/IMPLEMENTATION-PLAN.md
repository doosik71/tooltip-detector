# mipsclone 구현 계획

ROBUST_MIPS_toolpose(MMPose 기반 수술도구 포즈 추정)와 그 앞단의 탐지기 RTMDet(MMDetection)에서 **모델 구조의 아이디어를 가져와**, 도구 박스와 팁을 한 번에 내는 **하나의 end-to-end 모델(RTMDet-Pose)** 을 순수 PyTorch로 새로 만든다. 학습은 `cholec80`에서 무작위 초기화로 시작한다.

- 작성일: 2026-11-17 (같은 날 1차 계획을 이 방향으로 대체)
- 참고 구현: `/home/doosik/work/ROBUST_MIPS_toolpose` (MMPose, RTMPose-m 설정), `/mnt/work/doosik/mmdetection` (MMDetection 3.3.0, RTMDet 설정)

## 1. 방향 결정

| 항목          | 결정                                                                      | 이유                                                                                  |
| ------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 원본과의 관계 | 충실한 재현이 아니라 구조 아이디어 차용. 원본 수치 대조는 하지 않는다     | 원본과의 비교는 목표가 아니다. 목표는 cholec80에서 형제 베이스라인과 비교하는 것이다  |
| 탐지기        | RTMDet을 순수 PyTorch로 재구현해 포함                                     | 원본 저장소는 탐지 결과 JSON만 제공한다. 탐지기 없이는 실사용 조건으로 평가할 수 없다 |
| 통합 방식     | 백본·넥을 공유하는 단일 모델. 포즈 가지가 탐지기의 특징 맵을 RoI로 읽는다 | 2단계 파이프라인(이미지 재크롭)보다 연산이 적고, 두 손실이 같은 특징을 함께 학습한다  |
| 사전학습      | 사용하지 않음 (COCO, ImageNet 모두)                                       | 형제 클론(yolov8sclone, yolo26clone, cladnet)과 같은 조건                             |
| 의존성        | 루트 환경만 사용 (torch, torchvision, numpy, opencv, scipy, tqdm)         | 형제 클론과 같은 규약. OpenMMLab 패키지는 쓰지 않는다                                 |

## 2. 전제: 팁이 하나뿐인 데이터에서도 학습 가능한가

1차 검토 결론이다.

- **모델:** 키포인트 수 K = 1이어도 SimCC 분류기, KL 손실, 헤드가 그대로 동작한다. GAU는 키포인트 사이의 attention이 의미를 잃어 게이트가 달린 MLP처럼 동작하지만, residual 연결이 있어 학습은 막히지 않는다.
- **데이터:** 학습셋 표본 22,604개(5프레임마다 하나)를 측정했다.
  - 팁은 100 % 박스 안에 있고, 그중 96.9 %는 가장자리 근처(박스 중심을 0, 가장자리를 1로 잰 값이 0.9 이상)에 있다.
  - 크롭 안에 다른 도구의 팁이 함께 들어오는 비율은 26 %다.
- **해석:**
  - 정답 박스를 주고 잰 팁 성능은 박스가 팁 위치를 상당 부분 알려 주므로 상한으로만 쓴다. test에서 정답 박스의 가장 가까운 모서리를 고르기만 해도(오라클) hit@20이 75.4 %다.
  - 비교에는 모델이 스스로 찾은 박스로 잰 수치를 쓴다.
  - 원 논문의 장점인 도구 골격(entry, hinge, tip1, tip2) 학습은 cholec80에서는 재현되지 않는다.

## 3. 모델: RTMDet-Pose

```text
입력 640 × 640 (letterbox)
  │
  ▼ CSPNeXt 백본 (RTMDet-s: deepen 0.33, widen 0.5)
  C3 (/8, 128) · C4 (/16, 256) · C5 (/32, 512)
  │
  ▼ CSPNeXt-PAFPN (출력 128채널)
  P3 · P4 · P5
  │                         │
  ▼ RTMDet SepBN 헤드        ▼ 포즈 가지 (RTMPose RTMCC 헤드의 아이디어)
  도구 점수 + ltrb 거리       RoIAlign(P3, 박스 × 1.25) → 16 × 16
  (stride 8/16/32 점 prior)   → 3×3 Conv × 4 → 7×7 Conv (K = 1) → 펼침 256
                              → ScaleNorm + Linear(256) → GAU → SimCC x / y (각 256 bin)
```

| 부분      | 출처                     | 이 구현의 내용                                                                                                                         |
| --------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| 백본      | RTMDet / RTMPose 공통    | CSPNeXt P5. 스템 3 Conv, 4 스테이지(CSPNeXtBlock: 3×3 + 5×5 depthwise), SPP, 채널 attention                                            |
| 넥        | RTMDet                   | CSPNeXt-PAFPN (top-down, bottom-up, 출력 3×3 Conv)                                                                                     |
| 탐지 헤드 | RTMDet                   | SepBN 헤드: Conv는 레벨 간 공유, BN과 예측층은 레벨별. 클래스 1개(`tool`), 거리 × stride                                               |
| 포즈 가지 | RTMPose (구조) + 이 구현 | RTMPose는 이미지 크롭을 별도 백본에 넣지만, 여기서는 공유 P3에서 RoIAlign으로 읽는다. 얇은 공유 특징을 보완하려고 Conv 4개를 앞에 둔다 |
| 크기 (s)  |                          | 전체 10.08M (백본 4.38M, 넥 3.88M, 탐지 헤드 0.60M, 포즈 가지 1.22M). 탐지 부분 8.86M은 공식 RTMDet-s(8.89M)와 비슷하다                |

`--scale`로 `tiny`(5.8M), `s`(10.1M, 기본), `m`(26.6M), `l`을 고를 수 있다.

## 4. 손실

```text
L = L_QFL + 2 · L_GIoU + pose_weight · L_SimCC     (pose_weight 기본 1)
```

- **탐지 (RTMDet 그대로):** DynamicSoftLabelAssigner(topk 13, IoU 가중치 3, soft center 반경 3)로 양성을 고른다. Quality Focal Loss는 점수를 예측 박스의 IoU로 학습시킨다. GIoU 손실은 IoU로 가중한다. 두 손실 모두 양성 IoU의 합으로 정규화한다.
- **포즈 (RTMPose의 KLDiscretLoss에서 출발):**
  - 학습용 RoI는 정답 도구 박스를 중심 ±10 %, 크기 ±10 % 범위에서 흔든 뒤 1.25배로 넓혀 만든다. 탐지기가 낸 박스처럼 부정확한 박스에 미리 익숙해지게 하려는 것이다.
  - 목표 분포는 bin 좌표의 가우시안(σ 3 bin)에 softmax(×10)를 씌워 만든다. 예측은 log_softmax(×10)로 바꾼다.
  - 팁이 RoI 밖이면 그 인스턴스의 포즈 손실은 0이다.
  - **원본과 다른 점 1:** KL을 bin 평균이 아니라 bin 합으로 계산한다. 백본을 공유하므로, 평균을 쓰면 포즈 그래디언트가 탐지 그래디언트보다 수백 배 작아진다.
  - **원본과 다른 점 2:** 배치 안에서 인스턴스 합이 아니라 평균을 쓴다.
  - **원본과 다른 점 3:** 가우시안 중심을 정수 bin으로 반올림하지 않는다.

## 5. 학습 레시피

RTMDet 설정을 단일 GPU에 맞췄다.

| 항목          | 값                                                                                                        | 비고                                                                 |
| ------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 옵티마이저    | AdamW, lr 1e-3, weight decay 0.05 (2D 이상 가중치에만)                                                    | RTMDet은 배치 256에서 lr 4e-3. 배치 32에 맞춘 값이며 튜닝하지 않았다 |
| 스케줄        | 1,000 iter 선형 웜업 → 전반부 고정 → 후반부 cosine (최저 0.05배)                                          | RTMDet과 같다                                                        |
| EMA           | ExpMomentumEMA (momentum 2e-4, BN 통계 포함). 검증과 저장은 EMA 가중치로                                  | RTMDet과 같다                                                        |
| 증강          | 모자이크 + 좌우 반전 + HSV. 마지막 `--stage2-epochs`(기본 10) 에포크는 모자이크 대신 무작위 리사이즈-크롭 | RTMDet의 2단계 전환. MixUp은 넣지 않았다                             |
| 기타          | AMP (손실과 포즈 가지는 fp32), 그래디언트 클리핑 10                                                       | 클리핑은 형제 클론과 맞춘 값                                         |
| 에포크·데이터 | 150 에포크, `--frame-stride 5`, `--val-frames 1500` 권장                                                  | 형제 클론의 cholec80 명령과 같다                                     |
| 최적 모델     | val의 팁 F1 @ 20 px (헝가리안, conf 0.25)                                                                 | 목표가 팁이므로 탐지 AP 대신 이것을 쓴다                             |

## 6. 평가

[scripts/eval-model.py](../scripts/eval-model.py)

- **탐지:** `tool`의 AP@0.5, AP@0.5:0.95, precision, recall.
- **팁:** conf 0.25 이상인 박스마다 팁 하나를 냈을 때의 miss rate, Hit-rate @ 10/20/50 px, 헝가리안 TP/FP/FN, 거리 통계. 판정 규칙은 형제 클론과 같다(`tipmetrics.py`를 yolo26clone에서 그대로 복사).
- **`--boxes gt`:** 정답 박스로 포즈 가지만 평가한다. 결과는 `<split>-gtbox/`에 따로 저장하고 상한으로만 쓴다.
- 모든 거리는 원본 프레임 좌표(736 × 480)에서 잰다.

## 7. 파일 구조

```text
baseline/mipsclone/
├── run(.bat)
├── README.md
├── common/
│   ├── model.py       # CSPNeXt, PAFPN, RTMDet SepBN 헤드, RoI 포즈 헤드(GAU, SimCC), 추론
│   ├── loss.py        # DynamicSoftLabelAssigner, QFL, GIoU, SimCC KL
│   ├── boxes.py       # IoU/GIoU, 거리 디코딩, letterbox
│   ├── dataset.py     # 도구 박스 + 팁 쌍, 모자이크 / 리사이즈-크롭
│   ├── inference.py   # 체크포인트, 프레임 단위 추론
│   ├── metrics.py     # (yolo26clone에서 복사) 탐지 AP
│   ├── tipmetrics.py  # (yolo26clone에서 복사) 팁 지표
│   └── progress.py    # (yolo26clone에서 복사)
├── scripts/
│   ├── train-model.py
│   └── eval-model.py
├── docs/IMPLEMENTATION-PLAN.md
└── data/              # git 추적 제외 (model/<dataset>/, results/<dataset>/<split>/)
```

## 8. 검증 현황

| 항목                                  | 상태                                                                                           |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 모델 생성, 파라미터 수                | 완료. 탐지 부분이 공식 RTMDet-s와 비슷한 크기                                                  |
| 순전파, 손실, 역전파 (AMP)            | 완료. 포즈 손실의 그래디언트가 백본까지 전달됨                                                 |
| 증강 후 박스·팁 정렬                  | 완료. 모자이크, 리사이즈-크롭, 좌우 반전 표본을 그려 확인                                      |
| 학습, 재개, 모자이크 전환, 체크포인트 | 완료. 소량 설정(711프레임, 3 에포크 + 재개 1 에포크)                                           |
| 짧은 학습에서 실제로 학습되는지       | 완료. 8 에포크(약 1,770 iter) 뒤 val tool AP50 0.88, 팁 hit@20 74 %                            |
| 평가 스크립트 (`det`, `gt` 두 모드)   | 완료. test 1,965프레임에서 hit@20 81.8 % (자체 박스), 89.6 % (정답 박스). 수치는 README에 있다 |
| 150 에포크 본 학습과 test 평가        | **미실시.** 학습 시간이 길어 사용자 확인 후 진행                                               |

## 9. 남은 일과 열린 질문

- **하이퍼파라미터:** lr(1e-3), `pose_weight`(1), RoI 크기(16), bin 수(256), RoI 흔들기 폭(±10 %)은 근거 있는 출발값일 뿐 튜닝하지 않았다.
- **팁 precision:** 짧은 학습에서는 conf 0.25를 넘는 중복·저품질 박스가 많아, 헝가리안 20 px precision이 0.48이었다. 본 학습 뒤에도 낮으면 conf 임계값이나 NMS IoU를 조정해야 한다.
- **RoI 레벨:** 지금은 P3만 읽는다. 큰 도구에는 P4·P5를 함께 읽는 편이 나을 수 있다.
- **범위 밖 (1차):** 데모 GUI, `generate-summary`, dashboard·tooltip-tracker 연동, `erop` 실험.
