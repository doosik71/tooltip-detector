# 실행 명령 모음

실험 재현에 필요한 명령을 실행 순서대로 모아둔다. 각 명령의 상세 설명은 해당 가이드 문서를 참고한다.
Windows에서는 `./run` 대신 `run.bat`을 사용한다.

실험은 **데이터셋(`cholec80`, `erop`) × 타겟 생성 방식(`gradient-seg`, `gaussian-tip`) × 모델 타입
(`monai`, `monai_mini`)** 의 8개 조합으로 구성되며, 학습·평가 명령도 조합마다 하나씩 있다.
아래 학습/평가 절에는 8개 조합을 모두 명시적으로 적어 둔다 (`--target-mode`·`--model-type`의
기본값은 각각 `gradient-seg`·`monai`이지만, 재현 시 혼동을 막기 위해 항상 함께 적는다).

## 1. 설치

```bash
uv sync
```

## 2. 데이터셋 배치 확인

`data/dataset/`은 프로젝트 외부에서 구축되는 read-only 마운트다 (상세: [데이터셋 가이드](dataset-guide.md)).
학습 전에 각 데이터셋이 아래 구조로 놓여 있는지 확인한다.

```bash
ls data/dataset/cholec80  # annotation  images  segmentation
ls data/dataset/erop      # annotation  images  segmentation
```

`annotation/`이 한 단계 더 안쪽(`data/dataset/<name>/<name>/annotation/`)에 있으면 학습·평가
스크립트가 샘플을 하나도 찾지 못하고 종료한다. 이 경우 안쪽 디렉터리의 내용을 위로 올린다.

## 3. 실험 현황 확인 (선택)

8개 조합 각각의 학습·평가 진행 상황을 표로 확인한다 (상세: [dashboard 가이드](dashboard-guide.md)).

```bash
./run dashboard                        # 모든 데이터셋
./run dashboard --dataset cholec80     # cholec80 행이 처음부터 선택된 상태로 시작
```

## 4. 데이터셋 확인 (선택)

```bash
./run dataset-browser --dataset cholec80
./run dataset-browser --dataset erop

# 스플릿 지정 (기본: train)
./run dataset-browser --dataset cholec80 --split test
```

## 5. 학습

상세: [학습 가이드](train-guide.md). 체크포인트는 `data/models/<dataset>/<target-mode>/<model-type>/`에 저장된다.

```bash
# ── cholec80 ────────────────────────────────────────────────────────────
./run train-model --dataset cholec80 --target-mode gradient-seg --model-type monai
./run train-model --dataset cholec80 --target-mode gradient-seg --model-type monai_mini
./run train-model --dataset cholec80 --target-mode gaussian-tip --model-type monai
./run train-model --dataset cholec80 --target-mode gaussian-tip --model-type monai_mini

# ── erop ────────────────────────────────────────────────────────────────
./run train-model --dataset erop --target-mode gradient-seg --model-type monai
./run train-model --dataset erop --target-mode gradient-seg --model-type monai_mini
./run train-model --dataset erop --target-mode gaussian-tip --model-type monai
./run train-model --dataset erop --target-mode gaussian-tip --model-type monai_mini
```

GPU가 여러 개인 장비에서 사용할 장치를 지정하려면 `--device cuda:<N>`을 덧붙인다.
한 실행은 GPU 한 대만 쓰므로, 조합이 다른 실험을 GPU별로 나눠 동시에 돌릴 수 있다.

```bash
./run train-model --dataset cholec80 --target-mode gradient-seg --model-type monai      --device cuda:0 &
./run train-model --dataset cholec80 --target-mode gradient-seg --model-type monai_mini --device cuda:1 &
```

중단된 학습은 같은 명령을 다시 실행하면 이어서 진행된다. 처음부터 다시 학습하려면 `--no-resume`을 덧붙인다.

## 6. 평가

상세: [평가 가이드](eval-guide.md). 결과는 `data/results/<dataset>/<target-mode>/<model-type>/`에
`summary.json`·`per_tip.csv`로 저장된다. 학습에 쓴 것과 같은 조합으로 실행한다.

```bash
# ── cholec80 ────────────────────────────────────────────────────────────
./run eval-model --dataset cholec80 --target-mode gradient-seg --model-type monai
./run eval-model --dataset cholec80 --target-mode gradient-seg --model-type monai_mini
./run eval-model --dataset cholec80 --target-mode gaussian-tip --model-type monai
./run eval-model --dataset cholec80 --target-mode gaussian-tip --model-type monai_mini

# ── erop ────────────────────────────────────────────────────────────────
./run eval-model --dataset erop --target-mode gradient-seg --model-type monai
./run eval-model --dataset erop --target-mode gradient-seg --model-type monai_mini
./run eval-model --dataset erop --target-mode gaussian-tip --model-type monai
./run eval-model --dataset erop --target-mode gaussian-tip --model-type monai_mini

# 평가에 사용할 GPU 지정
./run eval-model --dataset cholec80 --target-mode gradient-seg --model-type monai --device cuda:1
```

## 7. 속도 비교

한 번의 실행이 `--model-types`의 모든 모델(기본: `monai monai_mini`)을 함께 벤치마크하므로,
**데이터셋 × 타겟 생성 방식** 4개 조합으로 실행한다. 결과는
`data/results/<dataset>/<target-mode>/speed-comparison.json`에 저장된다.

```bash
./run compare-speed --dataset cholec80 --target-mode gradient-seg
./run compare-speed --dataset cholec80 --target-mode gaussian-tip
./run compare-speed --dataset erop     --target-mode gradient-seg
./run compare-speed --dataset erop     --target-mode gaussian-tip
```

## 8. 탐지 결과 시각화 (GUI)

상세: [tooltip-detector 설명서](tooltip-detector.md). 실행 후 GUI의 `Dataset`/`Target`/`Model`
드롭다운으로 조합을 바꿀 수 있으므로, 아래 인자는 초기 선택값일 뿐이다.

```bash
./run tooltip-detector --dataset cholec80 --target-mode gradient-seg --model-type monai
./run tooltip-detector --dataset cholec80 --target-mode gaussian-tip --model-type monai
./run tooltip-detector --dataset erop     --target-mode gradient-seg --model-type monai
./run tooltip-detector --dataset erop     --target-mode gaussian-tip --model-type monai
```

## 9. 동영상 실시간 추적 (GUI)

상세: [tooltip-tracker 설명서](tooltip-tracker.md). 동영상 파일은 GUI에서 연다.

인자는 체크포인트 경로 하나뿐이다. 인자 없이 실행하면 디스크에 있는 모델 목록을 출력하고 종료한다.

```bash
./run tooltip-tracker
```

이 프로젝트의 히트맵 모델:

```bash
./run tooltip-tracker data/models/cholec80/gradient-seg/monai/best.pt
./run tooltip-tracker data/models/cholec80/gaussian-tip/monai/best.pt
./run tooltip-tracker data/models/erop/gradient-seg/monai/best.pt
./run tooltip-tracker data/models/erop/gaussian-tip/monai/best.pt
```

재구현 베이스라인 탐지기(`tip` 박스의 중심을 팁으로 쓴다):

```bash
./run tooltip-tracker baseline/yolov8sclone/data/model/cholec80/model.pt
./run tooltip-tracker baseline/cladnet/data/model/cholec80/model.pt
./run tooltip-tracker baseline/yolo26clone/data/model/erop/model.pt
```

`baseline/yolov8s`와 `baseline/yolo26`은 열 수 없다. `ultralytics`가 필요하고, 그것이 요구하는
`opencv-python`이 루트 환경의 `opencv-python-headless`를 깨뜨리기 때문이다. 두 모델의 아키텍처는
위 재구현 3종이 그대로 커버한다.

모델은 실행 시점에 고정된다. 바꾸려면 tracker를 다시 실행한다.
