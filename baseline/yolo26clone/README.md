# YOLO26 클론 수술도구 탐지 베이스라인

[baseline/yolo26](../yolo26/)이 쓰는 `yolo26s.pt`는 Ultralytics 피클이라
**`ultralytics` 패키지 없이는 언피클조차 되지 않는다.** 이 서브 프로젝트는 같은 아키텍처를
**순수 PyTorch로 다시 구현**해, 학습·평가·데모 어디서도 `ultralytics`에 의존하지 않는다.
[yolov8sclone](../yolov8sclone/)이 [yolov8s](../yolov8s/)에 대해 한 일을 YOLO26에 대해 한 것이다.

**추가 의존성이 하나도 없다.** 루트 프로젝트가 이미 쓰는 패키지(torch, torchvision, numpy,
opencv, pillow, scipy, tqdm)만으로 구현했다.

## 구현이 맞다는 증거

`ultralytics` 없이 아키텍처를 재현했다는 것을 어떻게 확인하는가. yolov8sclone은 파라미터 수를
모듈 단위로 대조하는 데서 멈췄지만, 여기서는 참조 체크포인트를 이미 받아 두었으므로
**네 단계까지 확인한다.**

| 질문                       | 확인 방법                                  | 결과                                |
| -------------------------- | ------------------------------------------ | ----------------------------------- |
| 모양이 맞는가              | 모듈별 파라미터 수                         | 10,009,784 (nc=80) 완전 일치        |
| 파라미터가 맞는가          | 참조 `state_dict`를 그대로 로드            | 708 / 708 키, 누락 0, 초과 0        |
| **같은 것을 계산하는가**   | 참조의 입력으로 참조의 출력을 재현         | **최대 오차 0.000e+00 (완전 일치)** |
| **같은 것을 학습하는가**   | 참조의 GT 박스로 참조의 손실을 재현        | **최대 오차 0.000e+00 (완전 일치)** |

마지막 항목이 중요하다. assigner와 손실 함수는 대조할 가중치가 없으므로, 출력만 맞춰서는
검증되지 않는 부분이다.

모듈별 파라미터 수 (`nc=80`, BatchNorm 버퍼 제외):

| 모듈           | 구성                                 |          클론 |          참조 |
| -------------- | ------------------------------------ | ------------: | ------------: |
| 0〜10          | 백본 (Conv, C3k2, SPPF, C2PSA)       |     5,441,984 |     5,441,984 |
| 13, 16         | top-down (C3k2)                      |       613,376 |       613,376 |
| 17, 19, 20, 22 | bottom-up (Conv, C3k2)               |     2,960,640 |     2,960,640 |
| 23             | Detect (one2many + one2one 두 분기)  |       993,784 |       993,784 |
| **합계**       |                                      | **10,009,784** | **10,009,784** |

출력 비교 (참조가 고정 시드로 만든 640 × 640 입력 1장):

| 비교 대상                     | 최대 절대 오차 |
| ----------------------------- | -------------: |
| one2many 박스 / 점수          |    0.000e+00   |
| one2one 박스 / 점수           |    0.000e+00   |
| 최종 탐지 결과 (top-k 후처리) |    0.000e+00   |

손실 비교 (참조가 고정 시드로 만든 GT 박스 6개, 가중치 box 7.5 / cls 0.5 / l1 1.5):

| 분기                        |     클론 |     참조 |
| --------------------------- | -------: | -------: |
| one2many (topk 10) 합계     | 10.710819 | 10.710819 |
| one2one (topk 7, topk2 1) 합계 | 10.634089 | 10.634089 |
| 혼합 합계 (w = 0.8)         | 10.695474 | 10.695474 |

box·cls·l1 세 항목도 각각 일치한다. 부동소수점 오차 수준이 아니라 **비트 단위로 같다.**
재현 명령은 두 줄이다.

```bash
# 1) 참조 가중치를 평문 텐서로 추출 (ultralytics가 있는 형제 프로젝트 환경에서 1회)
uv run --project baseline/yolo26 python baseline/yolo26clone/scripts/export-reference.py
# 2) 대조
./baseline/yolo26clone/run verify-clone
```

### 파라미터 수만으로는 잡히지 않았던 것

이 3단계 검증이 실제로 잡아낸 버그가 있다. 참조 구현은 모델을 만든 뒤 **모든 BatchNorm의
`eps`를 1e-3, `momentum`을 0.03으로 바꾼다.** PyTorch 기본값은 1e-5와 0.1이다. 파라미터 수도
`state_dict` 키도 이 값에 영향받지 않으므로 앞의 두 단계는 통과하지만, 출력은 **첫 번째 Conv
레이어에서부터 224만큼** 어긋났다. 수치 대조가 없었다면 "구현이 맞다"고 보고한 뒤 다른
모델을 학습시켰을 것이다.

## 무엇을 탐지하는가

이 저장소의 어노테이션은 도구마다 `bbox`와 `tip`을 갖고 클래스 레이블은 없다. 그래서
[yolov8sclone](../yolov8sclone/), [cladnet](../cladnet/), [yolo26](../yolo26/)과 **똑같이**
두 클래스로 학습한다.

| 클래스 | 정의                                                         | 출처                 |
| ------ | ------------------------------------------------------------ | -------------------- |
| `tool` | 어노테이션의 바운딩 박스 그대로                              | `annotations[].bbox` |
| `tip`  | 팁 좌표를 중심으로 한 **32 × 32 px 박스** (`--tip-box-size`) | `annotations[].tip`  |

예측된 `tip` 박스의 중심이 팁 좌표이므로, 탐지 지표(AP)와 이 프로젝트의 팁 지표
(Hit-rate @ N px)를 같은 모델에서 함께 잴 수 있다. 32 px이라는 값의 근거는
[yolov8sclone README](../yolov8sclone/README.md)에 있고, 네 베이스라인이 같은 값을 쓴다.

다만 YOLO26의 assigner는 **한 변이 stride(16 px)보다 짧은 GT 박스를 16 px로 늘려서**
"박스 안에 들어오는 앵커"를 센다(`common/assigner.py`). 작은 객체가 앵커를 하나도 갖지 못하는
것을 막는 장치이며, 팁처럼 작은 박스에는 직접 영향이 있다. 32 px 팁 박스는 letterbox 후
27.8 px이라 이 하한에 걸리지 않지만, 10 px로 줄이면 걸린다.

## YOLOv8s 클론과 무엇이 다른가

두 클론을 같은 데이터로 학습하면 이 차이들의 효과를 직접 볼 수 있다. 아래는 코드에 있는
것만 적은 것이다.

|             | YOLOv8s 클론                                | YOLO26 클론                                                        |
| ----------- | ------------------------------------------- | ------------------------------------------------------------------ |
| 추론 후처리 | NMS 필요                                    | **없음.** 객체당 박스 하나, top-k 선택만 한다                      |
| 헤드        | 분기 하나                                   | **분기 둘.** one2many(topk 10) + one2one(topk 7, topk2 1)          |
| 박스 회귀   | DFL, 변마다 16개 빈의 분포                  | **DFL 없음** (`reg_max=1`), 세 번째 손실항이 L1                    |
| 라벨 할당   | TaskAlignedAssigner                         | 같음 + **박스 크기 하한(stride)** + **2차 top-k**                  |
| CSP 블록    | C2f                                         | C3k2 (블록 자리에 C3k나 어텐션 블록이 들어갈 수 있다)              |
| 백본 말단   | SPPF                                        | SPPF(풀링 횟수 인자화, residual) + **C2PSA** 어텐션 스테이지       |
| 분류 분기   | 3×3 Conv 두 개                              | **DWConv 3×3 + 1×1** 두 단                                         |
| 옵티마이저  | SGD                                         | **MuSGD** (Muon + SGD 하이브리드), `--optimizer sgd`로 전환 가능   |
| 파라미터    | 11,136,374 (2클래스)                        | **9,949,412** (2클래스)                                            |

## 구조

```text
입력 640 × 640
    │
    ▼  백본
  0 Conv 3→32 s2                        P1/2
  1 Conv 32→64 s2                       P2/4
  2 C3k2 64→128 (e=0.25)
  3 Conv 128→128 s2 ─ 4 C3k2 ─────────► P3/8   (80×80)
  5 Conv 256→256 s2 ─ 6 C3k2(c3k) ────► P4/16  (40×40)
  7 Conv 256→512 s2 ─ 8 C3k2(c3k)
  9 SPPF(k=5, n=3, residual) ─ 10 C2PSA► P5/32  (20×20)
    │
    ▼  넥 (PAN-FPN)
  top-down : up(P5) ⊕ P4 → 13 C3k2 ;  up(·) ⊕ P3 → 16 C3k2  ──► N3
  bottom-up: 17 Conv s2 ⊕ 13 → 19 C3k2                       ──► N4
             20 Conv s2 ⊕ P5 → 22 C3k2(attn)                 ──► N5
    │
    ▼  23 Detect (end-to-end, 앵커프리)
  one2many 분기: 박스 Conv→Conv→1×1 → l,t,r,b ;  클래스 DWConv→Conv→1×1 → nc
  one2one  분기: 같은 구조를 하나 더. 추론은 이쪽만 읽는다
```

one2one 분기는 **detach된 특징**을 받는다. 그 분기의 그래디언트는 헤드만 학습시키고, 공유
백본은 one2many 분기가 혼자 만든다.

### 손실

```text
L = w · L_one2many + (1 - w) · L_one2one,   각 분기는 7.5·L_box + 0.5·L_cls + 1.5·L_l1
```

- `L_box` = 1 − CIoU
- `L_cls` = BCE. 타깃이 0/1이 아니라 TaskAlignedAssigner가 계산한 정렬 점수다.
- `L_l1` = 네 거리의 L1. **DFL이 없으므로** 분포가 아니라 값을 직접 맞춘다. 거리를 픽셀로
  되돌린 뒤 이미지 크기로 나누므로, P5 앵커의 오차와 P3 앵커의 오차가 같은 무게를 갖는다.
- `w`는 학습이 진행되며 **0.8에서 0.1로 선형 감쇠**한다. 초반에는 배우기 쉬운 조밀한
  one2many 할당이 모델을 끌고 가고, 후반에는 실제로 추론에 쓰이는 one2one 분기가 끌고 간다.
  보고되는 손실 값은 one2one 분기의 것이다.

### MuSGD

`optimizer=auto`가 이 규모의 학습에서 고르는 옵티마이저를 그대로 옮겼다. 모든 파라미터는
보통의 SGD 업데이트를 받고, **행렬 모양 파라미터(2D·4D, 즉 모든 conv와 linear 가중치)는
직교화된 업데이트를 하나 더 받는다.**

1. 그래디언트의 지수이동평균을 유지한다 (momentum)
2. 행렬로 펴서 **Newton-Schulz 5회 반복**으로 특이값을 1 근처로 민다 (SVD의 UV^T 대용)
3. `sqrt(max(1, 행/열))`로 스케일해 자기 학습률(그룹 학습률의 0.2배)로 적용한다

직교화는 가중치 행렬의 각 방향이 움직이는 정도를 고르게 만들어, 한 방향이 스텝을 독점하지
않게 한다. 벡터(bias, BatchNorm)는 직교화할 행렬 구조가 없으므로 SGD 경로만 탄다.
참조 구현과 같이 **분류 헤드(`cv3`, `one2one_cv3`)는 학습률 3배**로 학습한다.

## 디렉터리 구조

```text
baseline/yolo26clone/
├── run(.bat)                 # `run <script> [args...]` → scripts/<script>.py 실행
├── common/
│   ├── model.py              # YOLO26 아키텍처, 앵커 생성, end-to-end 디코딩·top-k
│   ├── assigner.py           # TaskAlignedAssigner + 박스 크기 하한 + 2차 top-k
│   ├── loss.py               # BCE(cls) + CIoU(box) + L1, 두 분기의 가중합
│   ├── optim.py              # MuSGD (Muon + SGD 하이브리드)
│   ├── boxes.py              # xywh/xyxy, CIoU, letterbox (NMS는 없다)
│   ├── dataset.py            # 저장소 어노테이션 → 2클래스 라벨, mosaic 증강
│   ├── metrics.py            # AP@0.5, AP@0.5:0.95, precision, recall
│   ├── tipmetrics.py         # Hit-rate @ N px (루트 프로젝트와 동일한 매칭 규칙)
│   ├── inference.py          # 체크포인트 포맷 + 프레임 1장 추론
│   ├── sources.py            # 영상 / 추출 프레임 디렉터리 읽기 (데모용)
│   └── draw.py               # 예측·GT 오버레이
├── scripts/
│   ├── export-reference.py   # 참조 체크포인트 → 평문 텐서 (이것만 ultralytics 필요)
│   ├── verify-clone.py       # 파라미터·가중치·출력 3단계 대조
│   ├── train-model.py        # 학습
│   ├── eval-model.py         # 평가 (탐지 AP + 팁 hit-rate)
│   ├── demo.py               # 탐지 결과 시각화 GUI
│   └── generate-summary.py   # data/ → docs/summary-results.md
└── data/                     # 체크포인트·평가 결과·추출한 참조 가중치 (git 추적 제외)
    ├── reference/            #   export-reference.py의 산출물
    ├── model/<dataset>/      #   학습 산출물 (model.pt, model-last.pt, metric.csv, ...)
    └── results/<dataset>/<split>/   # summary.json, per_tip.csv
```

## 설치

**없다.** yolov8sclone과 마찬가지로 자체 `.venv`가 없고 루트 프로젝트의 환경을 그대로 쓴다.

```bash
uv sync        # 루트에서 한 번 (이미 했다면 불필요)
```

`scripts/export-reference.py` 하나만 예외로 `ultralytics`가 필요하며, 형제 프로젝트의
환경으로 실행한다 (위 "구현이 맞다는 증거" 참고).

## 사용법

스크립트는 모두 `run`으로 실행한다. 어느 디렉터리에서 실행해도 되고, Windows에서는
`run.bat`을 쓴다. 데이터셋별 전체 재현 명령은 [docs/commands.md](docs/commands.md)에 있다.

### 1. 구현 검증 (선택)

```bash
./baseline/yolo26clone/run verify-clone
```

참조 파일이 없으면 파라미터 표만 출력한다. `--nc`/`--scale`로 다른 크기의 파라미터 수도 볼 수 있다.

### 2. 학습

```bash
./baseline/yolo26clone/run train-model --dataset cholec80
```

| 인수                       | 기본값  | 설명                                                                                      |
| -------------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `--dataset`                | (필수)  | `data/dataset/` 아래 디렉터리 이름 (`cholec80` / `erop`)                                  |
| `--tip-box-size`           | 32      | 팁 박스 한 변의 길이 (원본 프레임 px). 체크포인트에 기록된다                              |
| `--scale`                  | `s`     | YOLO26 깊이·폭 스케일 (`n`/`s`/`m`/`l`/`x`)                                               |
| `--optimizer`              | `musgd` | `sgd`로 바꾸면 형제 클론들과 같은 옵티마이저가 된다                                       |
| `--epochs`                 | 150     |                                                                                           |
| `--batch-size`             | 16      |                                                                                           |
| `--lr`                     | 0.01    | momentum 0.9 (MuSGD 기준값), nesterov                                                     |
| `--image-size`             | 640     |                                                                                           |
| `--frame-stride`           | 1       | N프레임마다 1장만 학습에 쓴다. 영상 프레임은 서로 거의 같으므로 에포크 시간을 크게 줄인다 |
| `--val-frames`             | 2000    | 에포크마다 평가할 val 프레임 수 상한 (0이면 전체)                                         |
| `--max-det`                | 300     | 검증에서 프레임당 박스 수 상한. NMS가 없는 대신 이 값이 있다                              |
| `--no-ema` / `--no-resume` |         | EMA 끄기 / 처음부터 다시 학습                                                             |

체크포인트는 `data/model/<dataset>/`에, 평가 결과는 `data/results/<dataset>/<split>/`에
들어간다. `model-last.pt`가 있으면 **기본 동작이 재개**이며 optimizer·스케줄러·EMA 상태와
두 분기의 혼합 가중치까지 복원된다.

### 3. 평가

```bash
./baseline/yolo26clone/run eval-model --dataset cholec80
```

두 종류의 수치를 함께 낸다.

- **탐지 지표**: `tool`·`tip` 각각의 AP@0.5, AP@0.5:0.95, precision, recall
- **팁 지표**: miss rate, Hit-rate @ 10/20/50 px, 오차 거리 중앙값·평균·P90.
  루트 프로젝트 `scripts/eval-model.py`와 **같은 매칭 규칙**을 쓴다.

거리는 letterbox된 640 × 640이 아니라 **원본 프레임 좌표계(736 × 480)** 에서 잰다.
`--iou`에 해당하는 인수는 없다. 대신 `--max-det`이 있다.

### 4. 데모 GUI

```bash
./baseline/yolo26clone/run demo --dataset cholec80
```

조작은 [yolov8s 데모](../yolov8s/README.md)와 같지만 **IoU 슬라이더가 없다.** NMS를 돌리지
않으므로 있어도 아무 일도 하지 않는 조작이 되기 때문이다.

### 5. 수치 요약 문서 생성

```bash
./baseline/yolo26clone/run generate-summary
```

`data/model/`·`data/results/`의 파일만 읽어 [docs/summary-results.md](docs/summary-results.md)를
만든다. 그 수치를 해석한 실험결과보고서는
[docs/experimental-results.md](docs/experimental-results.md)에 있다.

## 구현 메모

- **BatchNorm `eps`=1e-3, `momentum`=0.03.** PyTorch 기본값이 아니라 참조 구현이 모델을 만든
  뒤 덮어쓰는 값이다. 위 "파라미터 수만으로는 잡히지 않았던 것" 참고.
- **AMP를 써도 손실은 fp32로 계산한다.** fp16에서는 CIoU의 박스 넓이 곱이 half의 표현
  범위를 넘어 `inf - inf = NaN`이 된다. 합성곱이 대부분인 forward는 autocast 아래 그대로
  두고 헤드 출력만 `.float()`으로 올린다. 특징 맵은 앵커 격자(100 미만의 반정수)를 만드는
  데만 쓰이므로 fp16 그대로 두어도 정확히 표현된다.
- **추론에만 걸리는 최적화가 둘 있다.** `Detector`가 체크포인트를 로드한 뒤 켜며,
  프레임당 9.8 → 6.0 ms(cholec80 test 전수)를 만든다. **체크포인트 포맷과 학습 경로는
  건드리지 않는다.**
  - `Detect.one2one_only`: 추론이 읽지 않는 one2many 분기를 건너뛴다(약 1.0 ms).
    탐지 결과는 비트 단위로 같다. 기본값이 꺼져 있는 이유는 `verify-clone`이 `.eval()`
    상태에서 `one2many`를 참조와 대조하고, 학습의 검증 루프도 val loss를 내는 데 그
    분기를 쓰기 때문이다. 학습 모드에서는 플래그를 켜도 무시된다.
  - `fuse()`: 모든 `Conv`의 BatchNorm을 합성곱에 접는다(약 1.8 ms). 참조 구현도
    `AutoBackend`가 로드할 때 같은 일을 한다. `state_dict` 키가 바뀌므로 반드시
    `load_state_dict` **이후에** 불러야 하고, 학습 모드 모델에는 거부한다. 실수
    연산으로는 정확하지만 TF32 합성곱에서는 반올림이 달라져, test 전수 기준 헝가리안
    TP가 0.02 % 움직인다.
- 그래디언트는 참조 구현과 같이 norm 10으로 클리핑한다.
- 레이어 11과 14의 upsample은 같은 인스턴스다. 상태가 없는 모듈이라 결과가 같고, 그 덕분에
  모듈 번호가 참조 설정과 어긋나지 않는다.

## 한계

- **학습 결과는 두 데이터셋뿐이고, 후처리 파라미터 탐색은 하지 않았다.** cholec80·erop을
  각각 150 에포크 학습·평가한 수치가 [docs/summary-results.md](docs/summary-results.md)에,
  그 분석이 [docs/experimental-results.md](docs/experimental-results.md)에 있다.
  `conf=0.25`·`max_det=300`은 형제 베이스라인과 맞춘 기본값일 뿐 이 과제에 맞춰 고른 값이 아니다.
- **학습 루프까지 YOLO26을 따라가지는 않는다.** 모델·손실·assigner·옵티마이저는 참조를
  그대로 옮겼지만, 바깥 루프는 형제 클론들과 동일하게 맞췄다: 3에포크 선형 warmup 뒤
  **cosine** 감쇠(참조는 linear), **그래디언트 누적 없음**(참조는 nominal batch 64에 맞춰
  누적하고 weight decay도 그에 맞춰 스케일한다). 세 재구현 베이스라인의 차이가 모델과
  손실의 차이이도록 하려는 선택이며, 따라서 [baseline/yolo26](../yolo26/)의 수치와 이
  프로젝트의 수치가 다르면 그 차이에는 학습 루프 몫이 섞여 있다.
- **사전학습 가중치를 쓰지 않는다.** 참조 체크포인트는 80클래스 COCO 가중치이고, 여기서는
  구현을 검증하는 데만 쓴다. 무작위 초기화에서 시작하므로 [baseline/yolo26](../yolo26/)의
  기본 설정(COCO 파인튜닝)보다 불리하다.
- **라이선스.** 이 구현과 그 학습 결과는 원본 가중치를 쓰지 않으므로 AGPL을 승계하지 않는다.
  다만 `data/reference/`에 추출해 둔 파일은 AGPL 가중치 그 자체이므로 git 추적에서 제외되어
  있고 배포해서는 안 된다.
- **GUI는 실행 검증하지 못했다.** 개발 환경에 디스플레이가 없어 tkinter 창을 띄우지 못했다.
  창 구성을 뺀 나머지(체크포인트 저장·로드, 소스 탐색, 추론, 오버레이 그리기)는 확인했다.

## 참고

- 아키텍처: Ultralytics YOLO26, <https://docs.ultralytics.com/models/yolo26>
- 참조 구현: <https://github.com/ultralytics/ultralytics> (`ultralytics` 8.4.131)
- Muon: Jordan et al., *Muon: An optimizer for hidden layers in neural networks*, 2024
- TaskAlignedAssigner: Feng et al., *TOOD: Task-aligned One-stage Object Detection*, ICCV 2021
- end-to-end 헤드(one2many + one2one): Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024
