# 실행 명령 모음 (CLAD-Net 베이스라인)

[실험결과 보고서](experimental-results.md)의 수치를 그대로 재현하는 명령이다. 인수의 의미는
[README](../README.md)에 있다. 저장소 루트에서 실행하는 것을 기준으로 적었지만 `run`이 스스로
`baseline/cladnet`으로 이동하므로 어느 디렉터리에서 실행해도 결과는 같다. Windows에서는
`./baseline/cladnet/run` 대신 `baseline\cladnet\run.bat`을 쓴다.

## 1. 설치

루트와 별개의 uv 프로젝트다.

```bash
uv sync --project baseline/cladnet
```

## 2. 데이터셋 확인

루트 프로젝트와 같은 `data/dataset/`을 읽으며 별도의 변환 과정은 없다.

```bash
ls data/dataset/cholec80   # annotation  images  segmentation
ls data/dataset/erop       # annotation  images  segmentation
```

## 3. 학습

데이터셋마다 한 번씩, 산출물은 `baseline/cladnet/data/model/<dataset>/<label-set>/`에
나뉘어 저장된다. `<label-set>`은 `--label-set`이 고른 학습 모드다 (아래 3.1).

```bash
./baseline/cladnet/run train-model --dataset cholec80 --frame-stride 5 --val-frames 1500 --workers 12
./baseline/cladnet/run train-model --dataset erop     --frame-stride 5 --val-frames 1500 --workers 12
```

### 3.1 학습 모드 (`--label-set`)

무엇을 레이블로 쓰는지에 따라 두 모드가 있다. 값은 그대로 산출물 디렉터리 이름이 된다.

| 모드                | 학습 클래스      | 산출물 디렉터리                       |
| ------------------- | ---------------- | ------------------------------------- |
| `tooltip` (기본값)  | `tool`, `tip`    | `data/model/<dataset>/tooltip/`       |
| `tiponly`           | `tip`            | `data/model/<dataset>/tiponly/`       |

`tiponly`는 수술도구 상자 어노테이션 없이 팁 상자만으로 학습했을 때 팁 탐지 성능이
어떻게 되는지를 재는 조건이다. 설계와 판정 기준은
[tip-only 학습 실험 계획](../../../docs/tip-only-experiment-plan.md)에 있다.
`--label-set`을 뺀 나머지 인수는 두 모드에서 같아야 비교할 수 있다.

```bash
./baseline/cladnet/run train-model --dataset cholec80 --label-set tiponly --frame-stride 5 --val-frames 1500 --workers 12
./baseline/cladnet/run train-model --dataset erop     --label-set tiponly --frame-stride 5 --val-frames 1500 --workers 12
```

나머지는 전부 기본값이다 (`--epochs 150`, `--batch-size 16`, `--lr 0.01`, `--image-size 640`,
`--tip-box-size 32`, `--rm-combine sum`, EMA 켬).
한 실행은 GPU 하나만 쓰므로 뒤에 `&`를 붙이면 두 데이터셋을 동시에 돌릴 수 있다
(cholec80 9.2시간, erop 14.2시간).
중단되면 같은 명령을 다시 실행할 때 이어서 진행되고, 처음부터 다시 하려면 `--no-resume`을 붙인다.

## 4. 평가

`data/model/<dataset>/<label-set>/model.pt`를 test 스플릿 전수로 평가해
`data/results/<dataset>/<label-set>/test/`에 `summary.json`·`per_tip.csv`를 쓴다.

```bash
./baseline/cladnet/run eval-model --dataset cholec80 --split test
./baseline/cladnet/run eval-model --dataset erop     --split test

./baseline/cladnet/run eval-model --dataset cholec80 --label-set tiponly --split test
./baseline/cladnet/run eval-model --dataset erop     --label-set tiponly --split test
```

기본값인 `conf=0.25`, NMS IoU 0.45, AP 곡선용 `--map-conf 0.001`을 쓰고, 팁 박스 크기는
체크포인트에 기록된 값을 그대로 읽는다.

## 5. 수치 요약 문서 생성

`data/model/`·`data/results/`를 다시 읽어 [summary-results.md](summary-results.md)를 새로 쓴다.
재학습·재평가 뒤에는 이 명령을 다시 실행한다. 모드마다 문서가 하나씩이다.

```bash
./baseline/cladnet/run generate-summary
./baseline/cladnet/run generate-summary --label-set tiponly --output docs/summary-results-tiponly.md
```

## 6. 탐지 결과 시각화 (선택)

`--dataset`을 주면 `data/model/<dataset>/<label-set>/model.pt`를 찾아 연다
(생략하면 사전순 첫 번째, `--label-set`을 생략하면 `tooltip`).

```bash
./baseline/cladnet/run demo --dataset cholec80
./baseline/cladnet/run demo --dataset erop
```
