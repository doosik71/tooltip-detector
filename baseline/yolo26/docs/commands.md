# 실행 명령 모음 (YOLO26 베이스라인)

인수의 의미는 [README](../README.md)에 있다. 저장소 루트에서 실행하는 것을 기준으로 적었지만
`run`이 스스로 경로를 잡으므로 어느 디렉터리에서 실행해도 결과는 같다. Windows에서는
`./baseline/yolo26/run` 대신 `baseline\yolo26\run.bat`을 쓴다.

## 1. 설치

`ultralytics`가 요구하는 `opencv-python`이 루트 환경을 깨뜨리므로 자체 `.venv`를 쓴다.

```bash
uv sync --project baseline/yolo26
```

## 2. 데이터셋 준비

학습 전에 데이터셋마다 한 번 실행한다. 어노테이션을 Ultralytics가 읽는 형식으로 바꾸며,
산출물은 `baseline/yolo26/data/yolo/<dataset>/`에 들어간다. 각각 2〜3초면 끝난다.

```bash
./baseline/yolo26/run prepare-dataset --dataset cholec80
./baseline/yolo26/run prepare-dataset --dataset erop
```

기본값은 `--tip-box-size 32`, `--frame-stride 5`, `--val-frames 2000`, `--splits train val`이다.
이 세 값이 곧 학습 세트의 정의이므로, 바꾸려면 이 단계를 다시 실행한다 (`--force`).

준비 결과 (기본 인수 기준):

| 데이터셋   | train 프레임 | tool 박스 | tip 박스 | val 프레임 |
| ---------- | -----------: | --------: | -------: | ---------: |
| `cholec80` |       14,207 |    25,622 |   25,622 |      1,914 |
| `erop`     |       21,685 |    30,328 |   30,328 |      1,903 |

## 3. 학습

데이터셋마다 한 번씩, 산출물은 `baseline/yolo26/data/model/<dataset>/`에 나뉘어 저장된다.

```bash
./baseline/yolo26/run train-model --dataset cholec80 --device cuda:1 --workers 12
./baseline/yolo26/run train-model --dataset erop     --device cuda:2 --workers 12
```

나머지는 전부 기본값이다 (`--scale s`, `--epochs 150`, `--batch-size 16`, `--image-size 640`,
`--optimizer auto`, COCO 사전학습에서 시작).
한 실행은 GPU 하나만 쓰므로 뒤에 `&`를 붙이면 두 데이터셋을 동시에 돌릴 수 있다.
중단되면 같은 명령을 다시 실행할 때 이어서 진행되고, 처음부터 다시 하려면 `--no-resume`을 붙인다.

스크래치 학습(형제 베이스라인과 같은 조건)은 출력을 분리해서 돌린다. `--output-dir`이나
`--model`에 준 상대 경로는 `baseline/yolo26/`을 기준으로 해석된다.

```bash
./baseline/yolo26/run train-model --dataset cholec80 --no-pretrained \
    --output-dir data/model-scratch/cholec80
./baseline/yolo26/run eval-model  --dataset cholec80 --split test \
    --model data/model-scratch/cholec80/model.pt \
    --output-dir data/results-scratch/cholec80/test
```

## 4. 평가

`data/model/<dataset>/model.pt`를 test 스플릿 전수로 평가해
`data/results/<dataset>/test/`에 `summary.json`·`per_tip.csv`를 쓴다.

```bash
./baseline/yolo26/run eval-model --dataset cholec80 --split test --device cuda:1
./baseline/yolo26/run eval-model --dataset erop     --split test --device cuda:1
```

기본값인 `conf=0.25`, AP 곡선용 `--map-conf 0.001`, `--max-det 300`을 쓰고, 팁 박스 크기는
체크포인트의 `model-info.json`에 적힌 값을 그대로 읽는다. YOLO26은 NMS가 없으므로 다른
베이스라인의 `--iou`에 해당하는 인수가 없다.

속도 수치를 형제 베이스라인과 비교하려면 같은 GPU를 유휴 상태로 두고 같은 장치를 지정해야 한다.

## 5. 수치 요약 문서 생성

`data/model/`·`data/results/`를 다시 읽어 [summary-results.md](summary-results.md)를 새로 쓴다.
재학습·재평가 뒤에는 이 명령을 다시 실행한다.

```bash
./baseline/yolo26/run generate-summary
```

위의 스크래치 학습처럼 `data/model-scratch/`·`data/results-scratch/`에 쌓은 실험은
접미사로 지정한다.

```bash
./baseline/yolo26/run generate-summary --suffix "-scratch" --output docs/summary-scratch.md
```

## 6. 탐지 결과 시각화 (선택)

`--dataset`을 주면 `data/model/<dataset>/model.pt`를 찾아 연다 (생략하면 사전순 첫 번째).

```bash
./baseline/yolo26/run demo --dataset cholec80
./baseline/yolo26/run demo --dataset erop
```
