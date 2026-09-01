# 실행 명령 모음 (YOLO26 클론 베이스라인)

인수의 의미는 [README](../README.md)에 있다. 저장소 루트에서 실행하는 것을 기준으로 적었지만
`run`이 스스로 경로를 잡으므로 어느 디렉터리에서 실행해도 결과는 같다. Windows에서는
`./baseline/yolo26clone/run` 대신 `baseline\yolo26clone\run.bat`을 쓴다.

## 1. 설치

자체 `.venv`가 없고 루트 프로젝트의 환경을 그대로 쓴다. 루트에서 한 번만 실행한다.

```bash
uv sync
```

## 2. 데이터셋 확인

루트 프로젝트와 같은 `data/dataset/`을 읽으며 별도의 변환 과정은 없다.

```bash
ls data/dataset/cholec80   # annotation  images  segmentation
ls data/dataset/erop       # annotation  images  segmentation
```

## 3. 구현 검증 (선택, 1회)

참조 가중치를 평문 텐서로 뽑은 뒤 대조한다. 첫 줄만 `ultralytics`가 필요하므로 형제
프로젝트의 환경으로 실행한다. 참조 가중치는 `baseline/yolo26/data/pretrained/yolo26s.pt`에
있어야 하며, 없으면 `./baseline/yolo26/run train-model`이 한 번 받아 온다.

```bash
uv run --project baseline/yolo26 python baseline/yolo26clone/scripts/export-reference.py
./baseline/yolo26clone/run verify-clone
```

## 4. 학습

데이터셋마다 한 번씩, 산출물은 `baseline/yolo26clone/data/model/<dataset>/<label-set>/`에
나뉘어 저장된다. `<label-set>`은 `--label-set`이 고른 학습 모드다 (아래 3.1).

```bash
./baseline/yolo26clone/run train-model --dataset cholec80 --frame-stride 5 --val-frames 1500 --workers 12
./baseline/yolo26clone/run train-model --dataset erop     --frame-stride 5 --val-frames 1500 --workers 12
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
./baseline/yolo26clone/run train-model --dataset cholec80 --label-set tiponly --frame-stride 5 --val-frames 1500 --workers 12
./baseline/yolo26clone/run train-model --dataset erop     --label-set tiponly --frame-stride 5 --val-frames 1500 --workers 12
```

나머지는 전부 기본값이다 (`--epochs 150`, `--batch-size 16`, `--lr 0.01`, `--image-size 640`,
`--scale s`, `--tip-box-size 32`, `--optimizer musgd`, EMA 켬).
형제 클론들과 같은 인수이므로 세 베이스라인의 학습 조건이 일치한다.
한 실행은 GPU 하나만 쓰므로 뒤에 `&`를 붙이면 두 데이터셋을 동시에 돌릴 수 있다.
중단되면 같은 명령을 다시 실행할 때 이어서 진행되고, 처음부터 다시 하려면 `--no-resume`을 붙인다.

옵티마이저의 몫을 떼어 보려면 SGD 실행을 따로 쌓는다. `--output-dir`에 준 상대 경로는
`baseline/yolo26clone/`을 기준으로 해석된다.

```bash
./baseline/yolo26clone/run train-model --dataset cholec80 --frame-stride 5 --val-frames 1500 \
    --optimizer sgd --momentum 0.937 --output-dir data/model-sgd/cholec80
```

## 5. 평가

`data/model/<dataset>/<label-set>/model.pt`를 test 스플릿 전수로 평가해
`data/results/<dataset>/<label-set>/test/`에 `summary.json`·`per_tip.csv`를 쓴다.

```bash
./baseline/yolo26clone/run eval-model --dataset cholec80 --split test
./baseline/yolo26clone/run eval-model --dataset erop     --split test

./baseline/yolo26clone/run eval-model --dataset cholec80 --label-set tiponly --split test
./baseline/yolo26clone/run eval-model --dataset erop     --label-set tiponly --split test
```

기본값인 `conf=0.25`, AP 곡선용 `--map-conf 0.001`, `--max-det 300`을 쓰고, 팁 박스 크기는
체크포인트에 기록된 값을 그대로 읽는다. 이 헤드는 NMS가 없으므로 `--iou`에 해당하는 인수가 없다.

속도를 형제 베이스라인과 비교하려면 같은 GPU를 유휴 상태로 두고 같은 장치를 지정해야 한다.

## 6. 수치 요약 문서 생성

`data/model/`·`data/results/`를 다시 읽어 [summary-results.md](summary-results.md)를 새로 쓴다.
재학습·재평가 뒤에는 이 명령을 다시 실행한다. 모드마다 문서가 하나씩이다.

```bash
./baseline/yolo26clone/run generate-summary
./baseline/yolo26clone/run generate-summary --label-set tiponly --output docs/summary-results-tiponly.md
```

위의 SGD 실행처럼 별도 디렉터리에 쌓은 실험은 접미사로 지정한다.

```bash
./baseline/yolo26clone/run generate-summary --suffix "-sgd" --output docs/summary-sgd.md
```

## 7. 탐지 결과 시각화 (선택)

`--dataset`을 주면 `data/model/<dataset>/<label-set>/model.pt`를 찾아 연다
(생략하면 사전순 첫 번째, `--label-set`을 생략하면 `tooltip`).

```bash
./baseline/yolo26clone/run demo --dataset cholec80
./baseline/yolo26clone/run demo --dataset erop
```
