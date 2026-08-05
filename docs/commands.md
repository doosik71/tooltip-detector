# 실행 명령 모음

실험 재현에 필요한 명령을 모아둔다. 각 명령의 상세 설명은 해당 가이드 문서를 참고한다.
Windows에서는 `run` 대신 `run.bat`을 사용한다.

## 설치

```bash
uv sync
```

## 데이터셋 확인 (선택)

```bash
run dataset-browser --dataset cholec80
```

## 학습

```bash
# monai / gradient-seg
run train-model --dataset cholec80

# monai / gaussian-tip
run train-model --dataset cholec80 --target-mode gaussian-tip

# monai_mini / gradient-seg
run train-model --dataset cholec80 --model-type monai_mini

# monai_mini / gaussian-tip
run train-model --dataset cholec80 --model-type monai_mini --target-mode gaussian-tip
```

## 평가

```bash
# monai / gradient-seg
run eval-model --dataset cholec80

# monai / gaussian-tip
run eval-model --dataset cholec80 --target-mode gaussian-tip

# monai_mini / gradient-seg
run eval-model --dataset cholec80 --model-type monai_mini

# monai_mini / gaussian-tip
run eval-model --dataset cholec80 --model-type monai_mini --target-mode gaussian-tip
```

## 속도 비교

```bash
run compare-speed --dataset cholec80
```

## 탐지 결과 시각화 (GUI)

```bash
run tooltip-detector --dataset cholec80
run tooltip-detector --dataset cholec80 --target-mode gaussian-tip
```

## 동영상 실시간 추적 (GUI)

```bash
run tooltip-tracker --dataset cholec80
run tooltip-tracker --dataset cholec80 --target-mode gaussian-tip
```
