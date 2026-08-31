# 실행 명령 모음 (YOLOv8s 베이스라인)

인수의 의미는 [README](../README.md)에 있다. 저장소 루트에서 실행하는 것을 기준으로 적었지만
`run`이 스스로 경로를 잡으므로 어느 디렉터리에서 실행해도 결과는 같다. Windows에서는
`./baseline/yolov8s/run` 대신 `baseline\yolov8s\run.bat`을 쓴다.

## 1. 설치

`ultralytics`가 요구하는 `opencv-python`이 루트 환경의 `opencv-python-headless`와 같은 `cv2`
패키지에 파일을 쓰므로 자체 `.venv`를 쓴다. 루트 환경과 섞으면 기존 학습·평가 환경이 깨진다.

```bash
uv sync --project baseline/yolov8s
```

## 2. 데이터셋 준비

학습 전에 데이터셋마다 한 번 실행한다. 어노테이션을 Ultralytics가 읽는 형식으로 바꾸며,
산출물은 `baseline/yolov8s/data/yolo/<dataset>/`에 들어간다. 각각 2〜3초면 끝난다.

```bash
./baseline/yolov8s/run prepare-dataset --dataset cholec80
./baseline/yolov8s/run prepare-dataset --dataset erop
```

기본값은 `--tip-box-size 32`, `--frame-stride 5`, `--val-frames 2000`, `--splits train val`이다.
이 세 값이 곧 학습 세트의 정의이므로, 바꾸려면 이 단계를 다시 실행한다 (`--force`).

준비 결과 (기본 인수 기준, 실측):

| 데이터셋   | train 프레임 | tool 박스 | tip 박스 | val 프레임 | 소요  |
| ---------- | -----------: | --------: | -------: | ---------: | ----: |
| `cholec80` |       14,207 |    25,622 |   25,622 |      1,914 | 1.8 s |
| `erop`     |       21,685 |    30,328 |   30,328 |      1,903 | 2.4 s |

이 수치는 [baseline/yolo26](../../yolo26/docs/commands.md)의 준비 결과와 완전히 같다.
두 베이스라인이 같은 프레임, 같은 라벨로 학습한다는 뜻이며, 그래서 결과를 나란히 놓을 수 있다.

## 3. 학습

데이터셋마다 한 번씩, 산출물은 `baseline/yolov8s/data/model/<dataset>/`에 나뉘어 저장된다.

```bash
./baseline/yolov8s/run train-model --dataset cholec80 --device cuda:1 --workers 12
./baseline/yolov8s/run train-model --dataset erop     --device cuda:2 --workers 12
```

나머지는 전부 기본값이다 (`--scale s`, `--epochs 150`, `--batch-size 16`, `--image-size 640`,
`--optimizer SGD`, `--lr 0.01`, COCO 사전학습에서 시작).
한 실행은 GPU 하나만 쓰므로 뒤에 `&`를 붙이면 두 데이터셋을 동시에 돌릴 수 있다.
중단되면 같은 명령을 다시 실행할 때 이어서 진행되고, 처음부터 다시 하려면 `--no-resume`을 붙인다.
이미 끝난 학습을 같은 명령으로 다시 실행하면 아무 일도 하지 않고 그 사실을 알린다.

소요 시간은 cholec80 1에포크를 `cuda:1`에서 실측한 82초가 전부다. 150에포크면 3.4시간이지만,
첫 에포크에는 라벨 스캔과 캐시 생성이 포함되므로 이후 에포크는 이보다 빠르다. erop은 학습
프레임이 1.5배이므로 그만큼 길어진다.

### 스크래치 학습 (형제 베이스라인과 같은 조건)

기본값인 COCO 사전학습은 스크래치로 학습한 [yolov8sclone](../../yolov8sclone/)·[cladnet](../../cladnet/)보다
유리하다. 세 모델을 같은 조건에서 비교하려면 `--no-pretrained`가 필요하다. 출력을 분리해서
돌리면 기본 회차의 결과를 덮지 않는다. `--output-dir`이나 `--model`에 준 상대 경로는
`baseline/yolov8s/`를 기준으로 해석된다.

```bash
./baseline/yolov8s/run train-model --dataset cholec80 --no-pretrained \
    --output-dir data/model-scratch/cholec80 --device cuda:1
./baseline/yolov8s/run eval-model  --dataset cholec80 --split test \
    --model data/model-scratch/cholec80/model.pt \
    --output-dir data/results-scratch/cholec80/test --device cuda:1
```

### 옵티마이저를 바꿔 볼 때

기본값 `SGD`는 YOLOv8이 발표된 레시피이자 yolov8sclone이 쓴 것이다. Ultralytics 8.4의
`--optimizer auto`는 **모델과 무관하게** 이 길이의 학습에서 MuSGD를 고르므로, YOLO26과 같은
옵티마이저로 맞춰 보고 싶을 때만 쓴다. `auto`일 때 `--lr`은 무시된다.

```bash
./baseline/yolov8s/run train-model --dataset cholec80 --optimizer auto \
    --output-dir data/model-musgd/cholec80 --device cuda:3
```

## 4. 평가

`data/model/<dataset>/model.pt`를 test 스플릿 전수로 평가해
`data/results/<dataset>/test/`에 `summary.json`·`per_tip.csv`를 쓴다.

```bash
./baseline/yolov8s/run eval-model --dataset cholec80 --split test --device cuda:1
./baseline/yolov8s/run eval-model --dataset erop     --split test --device cuda:1
```

기본값인 `conf=0.25`, NMS `iou=0.45`, AP 곡선용 `--map-conf 0.001`, `--max-det 300`을 쓰고,
팁 박스 크기는 체크포인트의 `model-info.json`에 적힌 값을 그대로 읽는다. `--iou`의 기본값이
Ultralytics의 0.7이 아니라 0.45인 것은 yolov8sclone과 같은 운용 지점에서 재기 위해서다.

평가는 준비된 YOLO 데이터셋이 아니라 `data/dataset/`의 원본 프레임과 어노테이션을 직접 읽는다.
따라서 `--frame-stride 5`로 준비했더라도 test는 전수(cholec80 98,234 / erop 36,142 프레임)로
평가된다.

빠르게 확인만 하려면 프레임 수를 줄인다.

```bash
./baseline/yolov8s/run eval-model --dataset cholec80 --split test --limit 2000 --device cuda:1
```

## 5. 수치 요약 문서 생성

`data/model/`·`data/results/`를 다시 읽어 [summary-results.md](summary-results.md)를 새로 쓴다.
재학습·재평가 뒤에는 이 명령을 다시 실행한다. 모델이나 데이터셋을 로드하지 않는다.

```bash
./baseline/yolov8s/run generate-summary
```

위의 스크래치 학습처럼 `data/model-scratch/`·`data/results-scratch/`에 쌓은 실험은 접미사로
지정한다. **값이 하이픈으로 시작하므로 `=`로 붙여 써야 한다**. 띄어 쓰면 argparse가 다음 인수를
플래그로 읽어 실패한다.

```bash
./baseline/yolov8s/run generate-summary --suffix=-scratch --output docs/summary-scratch.md
```

## 6. 탐지 결과 시각화 (선택)

학습한 `tool`/`tip` 모델을 영상에 돌려 본다. `--weights`로 체크포인트를 지정한다.

```bash
./baseline/yolov8s/run demo --weights data/model/cholec80/model.pt
./baseline/yolov8s/run demo --weights data/model/erop/model.pt
```

`--weights`를 생략하면 이 서브 프로젝트가 원래 갖고 있던 **공개 7클래스 체크포인트**
(`data/yolov8s_cholec80.pt`)가 열린다. 그것은 도구 종류를 분류하는 다른 모델이며 팁을 내놓지
않는다. 먼저 내려받아야 한다.

```bash
./baseline/yolov8s/run download-model
./baseline/yolov8s/run demo
```

`eval-model`에 7클래스 체크포인트를 넘기면 `tip` 클래스가 없다는 이유로 중단된다. 팁 지표가
정의되지 않는 모델이라 의도된 동작이다.

## 전체 순서 한 번에

```bash
uv sync --project baseline/yolov8s

./baseline/yolov8s/run prepare-dataset --dataset cholec80
./baseline/yolov8s/run prepare-dataset --dataset erop

./baseline/yolov8s/run train-model --dataset cholec80 --device cuda:1 --workers 12 &
./baseline/yolov8s/run train-model --dataset erop     --device cuda:2 --workers 12 &
wait

./baseline/yolov8s/run eval-model --dataset cholec80 --split test --device cuda:1
./baseline/yolov8s/run eval-model --dataset erop     --split test --device cuda:1

./baseline/yolov8s/run generate-summary
```
