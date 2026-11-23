# detrtooltip: 도구마다 박스와 팁 1〜2개를 내는 DETR 모델

수술도구마다 bounding box 하나와 팁 1〜2개(`tip`, 선택적인 `tip2`)를 내는 탐지 모델이다. 벌어진 가위나 bipolar forceps처럼 끝이 둘인 도구를 다루려고 만들었다. 목적은 [scripts/interactive-trainer.py](../../scripts/interactive-trainer.py)에서 쓸 정확한 모델이며, 속도는 우선순위가 낮다.

설계와 결정 근거는 [docs/IMPLEMENTATION-PLAN.md](docs/IMPLEMENTATION-PLAN.md)에 있다.

## 1. 모델

```text
입력 736 × 480 (원본 해상도)
  → ResNet-50 인코더 + FPN 디코더 (P2〜P5)                       백본, 1단계에서 사전학습
  → Deformable DETR 인코더 6층 (P3〜P5) → 2-stage 제안 상위 30개   헤드
  → Deformable DETR 디코더 6층, 박스와 팁 반복 정제
  → 쿼리마다 점수, 박스, 팁 슬롯 A·B, P(B 존재)
  → 팁 정제: P2에서 팁 주변 15 × 15 창 → 히트맵 → soft-argmax
```

| 부분 | 파라미터 |
| ---- | -------: |
| 백본 |  26.85 M |
| 헤드 |  12.67 M |
| 전체 |  39.53 M |

- 사전학습 가중치(ImageNet 등)는 쓰지 않는다.
- 팁 슬롯에는 순서가 없다. 정답 팁이 둘이면 두 가지 대응 중 손실이 작은 쪽을 쓴다.
- `tip`은 슬롯 A, `tip2`는 P(B 존재) > 0.5일 때의 슬롯 B다.
- multi-scale deformable attention은 `grid_sample`로 구현했다(CUDA 확장 없음).

## 2. 학습 단계

| 단계 | 스크립트         | 학습 대상                                      | 데이터                                         |
| ---- | ---------------- | ---------------------------------------------- | ---------------------------------------------- |
| 1    | `pretrain-model` | 백본 (마스킹 복원)                             | train, val, test 전체 184,578장. 레이블 불필요 |
| 2    | `train-model`    | 전체. 백본은 1단계에서 시작, lr은 헤드의 0.1배 | video42를 뺀 전체                              |

처음에는 2단계를 헤드만 학습(백본 고정)하는 단계와 전체 미세조정 단계로 나눴으나, 짧은 학습에서 헤드만 학습하는 쪽이 모든 에포크에서 뒤처져(6.1절) 없앴다. interactive-trainer에서는 여전히 백본을 고정하고 헤드만 학습한다.

실행 명령은 2.4절에 있다. GPU는 cuda:1〜3을 쓴다. cuda:0은 PCIe 대역폭이 낮아 학습이 느리다.

### 2.1 레이블

프레임마다 아래 순서로 처음 찾은 파일 하나를 쓴다.

| 순서 | 출처                 | "`tip2` 없음"을 정답으로 쓰는가                        |
| ---: | -------------------- | ------------------------------------------------------ |
|    1 | `annotation-manual/` | 예                                                     |
|    2 | `annotation-opus/`   | 예                                                     |
|    3 | `annotation/`        | 아니오 (대부분 팁을 하나만 내는 yolov8stooltip의 예측) |

`tip2`가 있는 프레임은 학습 프레임의 약 1.2 %(2,169장)뿐이라, 이런 프레임을 가중치 10, 그 밖의 manual·opus 프레임을 3, `annotation/` 프레임을 1로 복원 추출한다(`--two-tip-weight`, `--trusted-weight`).

**test split도 학습에 쓴다.** 지금은 객관적인 성능 비교가 아니라 interactive-trainer에서 쓸 좋은 모델이 목적이기 때문이다. 어노테이션이 끝나면 train split만으로 다시 학습한다.

### 2.2 checkpoint 선택

`--holdout-videos`(기본 `video42`)는 학습에서 빼고 checkpoint 선택에만 쓴다. 기준은 `tip`과 `tip2`를 모두 넣은 팁 F1 @ 20 px(헝가리안, conf 0.5)이다. video42는 3,713장 모두 opus 또는 manual 레이블이 있고, `tip2`가 있는 프레임이 225장이다.

### 2.3 주요 옵션

| 스크립트         | 옵션                                                                                     | 기본값                                       |
| ---------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------- |
| `pretrain-model` | `--epochs`, `--epoch-size`, `--batch-size`, `--lr`, `--mask-ratio`, `--patch`, `--depth` | 30, 60000, 32, 5e-4, 0.6, 32, 50             |
| `train-model`    | `--init`, `--epochs`, `--epoch-size`, `--batch-size`, `--lr`, `--backbone-lr-factor`     | `pretrain.pt`, 30, 30000, 8, 1e-4, 0.1       |
| `train-model`    | `--holdout-videos`, `--select-stride`, `--conf`                                          | `video42`, 1, 0.5                            |
| `eval-model`     | `--family`, `--model`, `--videos`, `--split`, `--conf`                                   | `detrtooltip`, `model.pt`, `video42`, -, 0.5 |

`train-model --init none`은 무작위 초기화 백본으로 학습한다. 사전학습의 효과를 비교할 때 쓴다. `--init`에 학습이 끝난 `model.pt`를 주면 그 모델에서 이어 학습한다.

### 2.4 학습 명령 예시

저장소 루트에서 실행한다. 학습이 길어 터미널을 닫아도 계속 돌도록 `nohup`으로 띄우고, 출력은 `data/log/`에 남긴다. 예상 시간은 A6000 1장, 기본 설정 기준이다.

| 단계 | 기본 설정                   | 예상 시간 |
| ---- | --------------------------- | --------: |
| 1    | 30 에포크 × 6만 장, 배치 32 |  약 6시간 |
| 2    | 30 에포크 × 3만 장, 배치 8  | 약 14시간 |

**1단계와 2단계를 이어서 실행** (1단계가 실패하면 2단계는 시작하지 않는다):

```bash
cd ~/tooltip-detector
mkdir -p standalone/detrtooltip/data/log
nohup bash -c '
  ./standalone/detrtooltip/run pretrain-model --dataset cholec80 --device cuda:1 &&
  ./standalone/detrtooltip/run train-model --dataset cholec80 --device cuda:1
' > standalone/detrtooltip/data/log/cholec80.log 2>&1 &
```

**단계별로 따로 실행:**

```bash
# 1단계: data/model/cholec80/pretrain.pt
nohup ./standalone/detrtooltip/run pretrain-model --dataset cholec80 --device cuda:1 \
  > standalone/detrtooltip/data/log/pretrain.log 2>&1 &

# 2단계: 1단계가 끝난 뒤. data/model/cholec80/model.pt
nohup ./standalone/detrtooltip/run train-model --dataset cholec80 --device cuda:1 \
  > standalone/detrtooltip/data/log/train.log 2>&1 &
```

**진행 확인:** 진행 막대가 `\r`로 한 줄을 덮어쓰므로 줄바꿈으로 바꿔 본다.

```bash
tr '\r' '\n' < standalone/detrtooltip/data/log/cholec80.log | grep -E 'epoch [0-9]+:|saved|done|Error'
cat standalone/detrtooltip/data/model/cholec80/pretrain-metric.csv   # 1단계 에포크별 손실
cat standalone/detrtooltip/data/model/cholec80/metric.csv            # 2단계 에포크별 지표
```

1단계는 에포크마다 `data/model/cholec80/pretrain-sample.jpg`(원본, 가린 입력, 복원 결과)를 덮어쓴다. 복원이 나아지는지 눈으로 확인할 수 있다.

**중단 후 재개:** 같은 명령을 다시 실행하면 마지막으로 끝난 에포크 다음부터 이어 간다(1단계는 `pretrain.pt`, 2단계는 `model-last.pt`). 처음부터 다시 하려면 `--no-resume`을 붙인다.

**평가와 비교:** 2단계가 끝난 뒤 video42에서 평가하고, 같은 프레임에서 yolov8stooltip과 비교한다.

```bash
./standalone/detrtooltip/run eval-model --dataset cholec80 --device cuda:1
./standalone/detrtooltip/run eval-model --dataset cholec80 --device cuda:1 --family yolov8stooltip
```

결과는 `data/results/cholec80/detrtooltip-video42/summary.json`과 `data/results/cholec80/yolov8stooltip-video42/summary.json`에 저장된다.

## 3. 평가

`eval-model`은 학습과 같은 레이블 규칙으로 아래 지표를 낸다. `--family yolov8stooltip`을 주면 [standalone/yolov8stooltip](../yolov8stooltip/)의 모델을 같은 프레임, 같은 규칙으로 평가한다.

| 지표                        | 뜻                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| tool AP50, AP50:95          | 도구 박스 AP                                                                                |
| tip hit@N, Hungarian P/R/F1 | 모든 팁(`tip`, `tip2`)을 점의 집합으로 본 지표. 형제 프로젝트의 `tipmetrics.py`와 같은 규칙 |
| own tip @N                  | 박스 IoU ≥ 0.5로 짝지은 도구에서, 정답 팁마다 그 도구의 예측 팁이 N px 안에 있는 비율       |
| two-tip tools @N            | 위 지표를 팁이 둘인 도구로 한정한 값                                                        |
| tip2 P/R/F1                 | "이 도구에 두 번째 팁이 있다"의 정밀도와 재현율. manual·opus 프레임만                       |

## 4. 파일 구조

```text
standalone/detrtooltip/
├── run(.bat)
├── common/
│   ├── model.py        # 백본, 복원 헤드, Deformable DETR, 팁 정제, 후처리
│   ├── msdeform.py     # multi-scale deformable attention (grid_sample)
│   ├── loss.py         # 복원 손실; 헝가리안 매칭, 순서 무관 팁 손실, tip2 존재 손실
│   ├── boxes.py        # 박스 변환, GIoU
│   ├── dataset.py      # 레이블 우선순위, 샘플링 가중치, 증강
│   ├── evaluation.py   # 학습 중 선택과 eval-model이 함께 쓰는 평가
│   ├── inference.py    # checkpoint 형식, 프레임 단위 추론 (Detector)
│   ├── metrics.py      # (yolov8stooltip에서 복사) 탐지 AP
│   ├── tipmetrics.py   # (yolov8stooltip에서 복사) 팁 지표 + 1〜2개 팁용 PairEvaluator
│   └── progress.py     # (yolov8stooltip에서 복사)
├── scripts/
│   ├── pretrain-model.py
│   ├── train-model.py
│   └── eval-model.py
├── docs/IMPLEMENTATION-PLAN.md
└── data/               # git 추적 제외
    ├── model/<dataset>/        # pretrain.pt, model.pt, model-last.pt, metric.csv
    ├── log/                    # 2.4절 명령의 출력
    └── results/<dataset>/<name>/summary.json
```

## 5. interactive-trainer 연동

checkpoint에는 `"family": "detrtooltip"`이 들어 있다. 연동은 아직 하지 않았다. interactive-trainer가 `tip2`와 `label`을 보존하도록 먼저 고쳐야 하기 때문이다([계획서 7절](docs/IMPLEMENTATION-PLAN.md)).

## 6. 검증 현황

| 항목                          | 결과                                                                                                                        |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 순전파·역전파 (bf16 autocast) | 완료. 배치 4에서 메모리 7.6 GB                                                                                              |
| 레이블 우선순위, 증강 후 정렬 | 완료. 출처별 표본과 증강 표본을 그려 확인                                                                                   |
| 과적합 (프레임 8장, 600 step) | 완료. 팁 F1@20 1.0, AP50 0.995, `tip2` F1 1.0. 정답과 순서가 바뀐 팁 쌍도 맞게 매칭됨                                       |
| 학습 속도 (A6000 1장)         | 1단계 약 86장/초 (배치 32), 2단계 약 18장/초 (배치 8, 백본 고정 시 약 21장/초). 대부분 deformable attention의 `grid_sample` |
| 짧은 학습과 사전학습 효과     | 완료. 아래 표                                                                                                               |
| 본 학습 (1단계, 2단계)        | **미실시**                                                                                                                  |

### 6.1 짧은 학습 결과

사전학습 3 에포크(2만 장씩, 12분), 2단계는 실행마다 3 에포크(8천 장씩). video42를 4장마다 하나씩(929장) 평가했다. 팁 지표는 conf 0.5 이상만 세므로, 점수가 아직 낮은 초기에는 0에 가깝다.

| 실행                            | tool AP50 (1 → 2 → 3 에포크) | AP50:95 | 팁 F1@20 | own tip @20 |
| ------------------------------- | ---------------------------- | ------: | -------: | ----------: |
| 헤드만 학습 (백본 고정, 없앰)   | 0.170 → 0.462 → 0.592        |   0.351 |    0.033 |      1.72 % |
| 전체 학습 (사전학습 백본, 채택) | 0.250 → 0.742 → **0.803**    |   0.487 |    0.272 |     18.64 % |
| 전체 학습 (무작위 초기화)       | 0.019 → 0.135 → 0.183        |   0.048 |    0.000 |      0.00 % |
| 참고: yolov8stooltip (본 학습)  | 0.926                        |   0.672 |    0.655 |     59.32 % |

- 12분짜리 사전학습만으로도, 같은 조건에서 무작위 초기화보다 AP50이 0.18에서 0.80으로 올랐다.
- `tip2` F1은 세 실행 모두 0이다. 짧은 학습에서는 존재 확률이 아직 0.5를 넘지 못한다.
- yolov8stooltip 수치는 150 에포크 본 학습 결과라 직접 비교할 대상은 아니다.
