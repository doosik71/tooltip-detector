# 실행 명령 모음

실험 재현에 필요한 명령을 모아둔다. 각 명령의 상세 설명은 해당 가이드 문서를 참고한다.
Windows에서는 `run` 대신 `run.bat`을 사용한다.

## 설치

```bash
uv sync
```

## 데이터셋 확인 (선택)

```bash
run dataset-browser
```

## 학습

```bash
# monai / gradient-seg
run train-model

# monai / gaussian-tip
run train-model --target-mode gaussian-tip

# monai_mini / gradient-seg
run train-model --model-type monai_mini

# monai_mini / gaussian-tip
run train-model --model-type monai_mini --target-mode gaussian-tip
```

## 평가

```bash
# monai / gradient-seg
run eval-model

# monai / gaussian-tip
run eval-model --target-mode gaussian-tip

# monai_mini / gradient-seg
run eval-model --model-type monai_mini

# monai_mini / gaussian-tip
run eval-model --model-type monai_mini --target-mode gaussian-tip
```

## 속도 비교

```bash
run compare-speed
```

## 탐지 결과 시각화 (GUI)

```bash
run tooltip-detector
run tooltip-detector --target-mode gaussian-tip
```

## 동영상 실시간 추적 (GUI)

```bash
run tooltip-tracker
run tooltip-tracker --target-mode gaussian-tip
```
