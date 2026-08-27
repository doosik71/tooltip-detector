# 실행 명령 모음 (YOLOv8s 클론 베이스라인)

[실험결과 보고서](experimental-results.md)의 수치를 그대로 재현하는 명령이다. 인수의 의미는
[README](../README.md)에 있다. 저장소 루트에서 실행하는 것을 기준으로 적었지만 `run`이 스스로
경로를 잡으므로 어느 디렉터리에서 실행해도 결과는 같다. Windows에서는
`./baseline/yolov8sclone/run` 대신 `baseline\yolov8sclone\run.bat`을 쓴다.

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

## 3. 학습

데이터셋마다 한 번씩, 산출물은 `baseline/yolov8sclone/data/model/<dataset>/`에 나뉘어 저장된다.

```bash
./baseline/yolov8sclone/run train-model --dataset cholec80 --frame-stride 5 --val-frames 1500 --workers 12
./baseline/yolov8sclone/run train-model --dataset erop     --frame-stride 5 --val-frames 1500 --workers 12
```

나머지는 전부 기본값이다 (`--epochs 150`, `--batch-size 16`, `--lr 0.01`, `--image-size 640`,
`--scale s`, `--tip-box-size 32`, EMA 켬).
한 실행은 GPU 하나만 쓰므로 뒤에 `&`를 붙이면 두 데이터셋을 동시에 돌릴 수 있다
(cholec80 4.1시간, erop 5.8시간).
중단되면 같은 명령을 다시 실행할 때 이어서 진행되고, 처음부터 다시 하려면 `--no-resume`을 붙인다.

## 4. 평가

`data/model/<dataset>/model.pt`를 test 스플릿 전수로 평가해
`data/results/<dataset>/test/`에 `summary.json`·`per_tip.csv`를 쓴다.

```bash
./baseline/yolov8sclone/run eval-model --dataset cholec80 --split test
./baseline/yolov8sclone/run eval-model --dataset erop     --split test
```

기본값인 `conf=0.25`, NMS IoU 0.45, AP 곡선용 `--map-conf 0.001`을 쓰고, 팁 박스 크기는
체크포인트에 기록된 값을 그대로 읽는다.
보고서 §8의 속도 수치는 세 모델을 같은 GPU(`cuda:3`)에서 잰 것이므로, 비교하려면 유휴 GPU에서
같은 장치로 평가해야 한다.

## 5. 수치 요약 문서 생성

`data/model/`·`data/results/`를 다시 읽어 [summary-results.md](summary-results.md)를 새로 쓴다.
재학습·재평가 뒤에는 이 명령을 다시 실행한다.

```bash
./baseline/yolov8sclone/run generate-summary
```

## 6. 탐지 결과 시각화 (선택)

`--dataset`을 주면 `data/model/<dataset>/model.pt`를 찾아 연다 (생략하면 사전순 첫 번째).

```bash
./baseline/yolov8sclone/run demo --dataset cholec80
./baseline/yolov8sclone/run demo --dataset erop
```
