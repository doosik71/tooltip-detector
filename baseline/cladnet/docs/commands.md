# 실행 명령 모음 (CLAD-Net 베이스라인)

CLAD-Net 베이스라인의 학습·평가를 재현하는 명령을 실행 순서대로 모아둔다.
각 인수의 의미는 [README](../README.md)를 참고한다.

모든 명령은 **저장소 루트에서** 실행하는 것을 기준으로 적었다. `run`이 스스로
`baseline/cladnet`으로 이동하므로 다른 디렉터리에서 실행해도 결과는 같다.
Windows에서는 `./baseline/cladnet/run` 대신 `baseline\cladnet\run.bat`을 사용한다.

실험은 **데이터셋(`cholec80`, `erop`) 2개**로 구성되며, 학습·평가 명령도 데이터셋마다
하나씩이다. 산출물은 데이터셋별로 나뉜다.

| 데이터셋   | 학습 산출물                             | 평가 산출물                                       |
| ---------- | --------------------------------------- | ------------------------------------------------- |
| `cholec80` | `baseline/cladnet/data/model/cholec80/` | `baseline/cladnet/data/results/cholec80/<split>/` |
| `erop`     | `baseline/cladnet/data/model/erop/`     | `baseline/cladnet/data/results/erop/<split>/`     |

## 1. 설치

루트와 별개의 uv 프로젝트다 (이유는 [README의 설치 절](../README.md#설치) 참고).

```bash
uv sync --project baseline/cladnet
```

## 2. 데이터셋 배치 확인

루트 프로젝트와 **같은** `data/dataset/`을 읽는다. 별도의 변환 과정은 없다.

```bash
ls data/dataset/cholec80   # annotation  images  segmentation
ls data/dataset/erop       # annotation  images  segmentation
```

`annotation/`과 `images/` 아래에 `train`/`val`/`test` 디렉터리가 있어야 한다.

## 3. 학습

체크포인트는 `baseline/cladnet/data/model/<dataset>/`에 저장된다 (`model.pt`, `model-last.pt`,
`train-status.json`, `metric.csv`). 데이터셋별로 디렉터리가 분리되므로 아래 두 명령은
서로 덮어쓰지 않는다.

```bash
# ── cholec80 ────────────────────────────────────────────────────────────
./baseline/cladnet/run train-model --dataset cholec80 \
    --epochs 30 --frame-stride 5 --val-frames 1500 \
    --batch-size 16 --workers 12 --device cuda:1

# ── erop ────────────────────────────────────────────────────────────────
./baseline/cladnet/run train-model --dataset erop \
    --epochs 30 --frame-stride 5 --val-frames 1500 \
    --batch-size 16 --workers 12 --device cuda:2
```

- `--frame-stride 5` — 영상 프레임은 서로 거의 같으므로 5장마다 1장만 쓴다. 에포크 시간이
  1/5로 줄어든다. 논문 그대로 전 프레임을 쓰려면 이 인수를 뺀다.
- `--val-frames 1500` — 에포크마다 재는 val 프레임 수 상한. 0이면 전체.
- `--epochs 30` — 논문은 150이다. 30은 루트 프로젝트의 학습 조건과 맞춘 값이다.
- `--device` — 한 실행은 GPU 하나만 쓰므로 두 데이터셋을 다른 GPU에 나눠 동시에 돌릴 수 있다.
  뒤에 `&`를 붙이면 백그라운드로 함께 진행된다.

중단된 학습은 **같은 명령을 다시 실행하면 이어서** 진행된다 (`model-last.pt`에 optimizer·
스케줄러·EMA 상태까지 저장된다). 처음부터 다시 학습하려면 `--no-resume`을 덧붙인다.

같은 데이터셋으로 설정만 바꿔 여러 번 학습할 때는 서로 덮어쓰지 않도록 `--output-dir`를
따로 준다.

```bash
./baseline/cladnet/run train-model --dataset cholec80 --rm-combine mean \
    --output-dir baseline/cladnet/data/model/cholec80-rm-mean
```

## 4. 평가

평가할 체크포인트는 `--dataset`에서 정해진다 (`data/model/<dataset>/model.pt`). 결과는
`data/results/<dataset>/<split>/`에 `summary.json`·`per_tip.csv`로 저장된다.

```bash
# ── cholec80 ────────────────────────────────────────────────────────────
./baseline/cladnet/run eval-model --dataset cholec80 --split test --device cuda:1

# ── erop ────────────────────────────────────────────────────────────────
./baseline/cladnet/run eval-model --dataset erop --split test --device cuda:2
```

`--split`의 기본값은 `test`이고 `train`/`val`도 지정할 수 있다.

```bash
./baseline/cladnet/run eval-model --dataset cholec80 --split val --device cuda:1
```

한 번에 두 종류의 수치가 나온다.

- **탐지 지표** — `tool`·`tip` 각각의 AP@0.5, AP@0.5:0.95, precision, recall (논문의 지표)
- **팁 지표** — miss rate, Hit-rate @ 10/20/50 px, 오차 거리 통계. 루트 프로젝트
  `scripts/eval-model.py`와 같은 매칭 규칙을 쓰므로 tooltip-detector의 수치와 직접 비교된다.

test 스플릿 전체가 오래 걸리면 `--frame-stride`로 부분 평가할 수 있다. 다만 이렇게 얻은
수치는 전체 평가 결과와 직접 비교하면 안 된다.

```bash
./baseline/cladnet/run eval-model --dataset cholec80 --frame-stride 10 --device cuda:1
```

`--model`로 다른 체크포인트를, `--output-dir`로 다른 저장 위치를 지정할 수 있다.

```bash
./baseline/cladnet/run eval-model --dataset cholec80 \
    --model baseline/cladnet/data/model/cholec80/model-last.pt \
    --output-dir baseline/cladnet/data/results/cholec80/test-last
```

## 5. 탐지 결과 시각화 (선택)

GUI는 `--dataset`을 받지 않으므로, 학습된 모델이 여러 개면 `--weights`로 직접 고른다.
`Source` 드롭다운에서 원본 영상과 추출 프레임 디렉터리를 모두 열 수 있다.

```bash
./baseline/cladnet/run demo --weights baseline/cladnet/data/model/cholec80/model.pt
./baseline/cladnet/run demo --weights baseline/cladnet/data/model/erop/model.pt
```
