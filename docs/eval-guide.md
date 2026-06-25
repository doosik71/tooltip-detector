# 평가 가이드

`TooltipDetector` 모델의 팁 탐지 정확도를 테스트 세트로 정량 평가하는 방법을 설명한다.

## 개요

모델이 출력한 2채널 히트맵의 채널 1(tool channel)에 sigmoid를 적용하여 팁 위치 후보(피크)를 추출한다. 추출된 후보와 정답(GT) 팁 좌표를 매칭하여 픽셀 거리 기반 정확도 지표를 계산한다.

평가 결과는 `data/results/YYYYMMDD_HHMMSS/` 디렉터리에 자동 저장된다.

## 사전 준비

학습이 완료된 가중치 파일이 필요하다:

```text
data/model/best.pt   ← 기본 경로
```

## 실행

```bash
bin/eval-model          # Linux / macOS
bin\eval-model.bat      # Windows
```

직접 실행:

```bash
uv run python -m ttd.eval
```

임계값 조정:

```bash
uv run python -m ttd.eval --threshold 0.3 --nms-radius 15
```

## 인수

| 인수 | 기본값 | 설명 |
| --- | --- | --- |
| `--model` | `data/model/best.pt` | 모델 가중치 파일 경로 |
| `--data-root` | `data/dataset` | 데이터셋 루트 디렉터리 |
| `--results-dir` | `data/results` | 결과 저장 루트 디렉터리 |
| `--threshold` | `0.5` | 피크 탐지 임계값 (히트맵 값 기준) |
| `--nms-radius` | `20` | 두 피크 사이의 최소 픽셀 거리 (NMS) |
| `--batch-size` | `16` | 추론 배치 크기 |
| `--workers` | `4` | DataLoader 워커 수 |
| `--device` | 자동 | torch device (예: `cuda:0`, `cpu`) |

## 결과 저장

실행마다 타임스탬프 디렉터리가 생성된다:

```text
data/results/
└── 20260625_143022/
    ├── summary.json
    └── per_tip.csv
```

여러 실행 결과가 디렉터리별로 분리되므로 다른 모델이나 임계값 조합을 비교할 수 있다.

### summary.json

전체 지표, 세션별 지표, 실행 파라미터를 포함한다.

```json
{
  "timestamp": "20260625_143022",
  "model_path": "data/model/best.pt",
  "threshold": 0.5,
  "nms_radius": 20,
  "data_root": "data/dataset",
  "n_frames_with_tools": 30930,
  "n_gt_tips": 31005,
  "n_missed": 318,
  "miss_rate_pct": 1.03,
  "mean_dist_px": 8.43,
  "median_dist_px": 5.21,
  "p90_dist_px": 18.67,
  "hit_rate_10px_pct": 72.4,
  "hit_rate_20px_pct": 88.1,
  "hit_rate_50px_pct": 96.3,
  "per_session": {
    "doctor_250620_084707": {
      "n_gt_tips": 4121,
      "n_missed": 38,
      "mean_dist_px": 7.9
    }
  }
}
```

### per_tip.csv

GT 팁 1개당 1행. 탐지 성공/실패 여부와 예측 좌표, 거리를 기록한다.

| 컬럼 | 설명 |
| --- | --- |
| `frame` | 프레임 베이스명 (확장자 제외) |
| `session` | 세션 ID |
| `gt_x`, `gt_y` | GT 팁 픽셀 좌표 |
| `pred_x`, `pred_y` | 매칭된 예측 피크 좌표 (탐지 실패 시 빈 값) |
| `dist_px` | 유클리드 픽셀 거리 (탐지 실패 시 빈 값) |
| `missed` | `1` = 탐지 실패, `0` = 탐지 성공 |

예시:

```csv
frame,session,gt_x,gt_y,pred_x,pred_y,dist_px,missed
doctor_250620_084707_00000042,doctor_250620_084707,154,176,151,179,3.16,0
doctor_250620_084707_00000042,doctor_250620_084707,186,240,,,,1
```

`per_tip.csv`를 이용하면:

- `missed=1` 행만 필터링하여 탐지 실패 프레임 목록 추출
- `dist_px` 히스토그램으로 오차 분포 분석
- 특정 세션 또는 프레임 구간의 성능 집계

## 피크 탐지 알고리즘

1. **임계값 적용** — 히트맵 값 ≥ `threshold`인 픽셀만 남긴다.
2. **연결요소 분석** — 이진 마스크에 `scipy.ndimage.label`을 적용한다.
3. **피크 선택** — 각 연결요소에서 최대값 픽셀을 피크 후보로 선택한다.
4. **NMS** — 이미 선택된 피크로부터 `nms-radius` 픽셀 이내에 있는 낮은 값의 피크를 제거한다. 피크는 값이 높은 순서로 처리된다.

피크 결과: `(x, y, value)` 목록, 값 내림차순 정렬.

## GT 팁 매칭

각 GT 팁에 대해 피크 후보 집합 전체 중 가장 가까운 후보를 선택하고 유클리드 픽셀 거리를 기록한다. 피크 후보가 없으면 해당 GT 팁은 `missed`로 처리된다.

## 평가 지표

| 지표 | 설명 |
| --- | --- |
| `n_gt_tips` | 테스트 세트 전체 GT 팁 수 |
| `n_missed` | 후보 없음으로 탐지 실패한 팁 수 |
| `miss_rate` | 탐지 실패율 (%) |
| `mean_dist` | 매칭된 팁의 평균 픽셀 거리 |
| `median_dist` | 중앙값 픽셀 거리 |
| `p90_dist` | 90 백분위수 픽셀 거리 |
| `hit_rate @ 10 px` | 거리 ≤ 10 px인 팁 비율 (%) |
| `hit_rate @ 20 px` | 거리 ≤ 20 px인 팁 비율 (%) |
| `hit_rate @ 50 px` | 거리 ≤ 50 px인 팁 비율 (%) |

세션별(`per_session`) `n_gt_tips`, `n_missed`, `mean_dist`도 함께 출력된다.

## 터미널 출력 예시

```text
Device      : cuda
Model       : data/model/best.pt
Threshold   : 0.5   NMS radius: 20 px
Results dir : data/results/20260625_143022
Test frames : 36,142

  36000/36142  gt_tips_seen=30921  missed=312
  36142/36142  gt_tips_seen=31005  missed=318

==========================================================
  Evaluation Results
==========================================================
  Model                              data/model/best.pt
  Threshold / NMS radius             0.5  /  20 px
----------------------------------------------------------
  Frames with tools                      30,930
  GT tips total                          31,005
  Missed (no prediction)                    318  (1.0%)
----------------------------------------------------------
  Mean distance                        8.43 px
  Median distance                      5.21 px
  P90 distance                        18.67 px
----------------------------------------------------------
  Hit-rate @  10 px                      72.4 %
  Hit-rate @  20 px                      88.1 %
  Hit-rate @  50 px                      96.3 %
----------------------------------------------------------
  Per-session:
    doctor_250620_084707  tips= 4121  missed=  38  mean=7.9 px
    ...
==========================================================

  summary.json  → data/results/20260625_143022/summary.json
  per_tip.csv   → data/results/20260625_143022/per_tip.csv  (31,005 rows)
```

## 임계값 튜닝

`--threshold`와 `--nms-radius`는 정확도에 영향을 준다.

- **임계값 낮춤** → 더 많은 피크 탐지 → miss 감소, 오탐 증가 가능성
- **임계값 높임** → 피크 수 감소 → 고신뢰도 탐지만 유지, miss 증가 가능성
- **NMS 반경 낮춤** → 인접한 두 도구 팁을 별개로 탐지 가능
- **NMS 반경 높임** → 같은 도구의 중복 피크 억제

실행마다 결과가 별도 디렉터리에 저장되므로, 여러 조합으로 평가를 반복하여 `summary.json`끼리 비교할 수 있다.
