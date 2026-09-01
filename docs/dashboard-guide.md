# Dashboard 사용 설명서

`scripts/dashboard.py`는 **학습·평가가 어디까지 진행됐는지**를 표로 보여주고, 선택한 칸의
작업을 수행하려면 어떤 명령을 실행해야 하는지 알려주는 GUI다.

표는 탭 두 개로 나뉜다. 두 프로젝트의 실험 축이 다르기 때문이다.

| 탭             | 한 행의 의미                              | 산출물 경로                                                   |
| -------------- | ----------------------------------------- | ------------------------------------------------------------- |
| `This project` | 데이터셋 × 타겟 생성 방식 × 모델 타입 (8) | `data/models/`·`data/results/`                                |
| `Baselines`    | 베이스라인 × 데이터셋 (10)                | `baseline/<name>/data/model/`·`baseline/<name>/data/results/` |

베이스라인은 `tool`·`tip` 박스를 직접 회귀하는 탐지기라 타겟 생성 방식이라는 축이 없고,
서브 프로젝트마다 자기 러너(`./baseline/<name>/run`)를 쓴다. 그래서 하나의 표로 합치지 않고
탭을 나눴다. 아래쪽 상세 패널과 `Command` 상자는 두 탭이 공유하며, 항상 **앞에 나와 있는
탭**의 선택 칸을 설명한다.

모델이나 데이터셋을 전혀 로드하지 않고 위 경로의 파일만 확인하므로, 학습이 돌고 있는
중에도 가볍게 띄워 둘 수 있다.

## 실행

```bash
run dashboard          # Linux / macOS
run.bat dashboard      # Windows
```

`--dataset`은 다른 스크립트와 달리 **선택 인자**다. 모든 데이터셋이 항상 표에 나오며,
`--dataset`을 주면 해당 데이터셋의 첫 행이 처음부터 선택되어 있을 뿐이다.

`--dataset`을 주면 **두 탭 모두** 해당 데이터셋의 첫 행이 선택된 채로 열린다.

| 인수              | 기본값         | 설명                                            |
| ----------------- | -------------- | ----------------------------------------------- |
| `--dataset`       | (없음)         | 처음 선택할 행의 데이터셋 (`erop` / `cholec80`) |
| `--models-root`   | `data/models`  | 체크포인트 트리 루트                            |
| `--results-root`  | `data/results` | 평가 결과 트리 루트                             |
| `--baseline-root` | `baseline`     | 베이스라인 서브 프로젝트 루트                   |

## 화면 구성

```text
┌─ Refresh    scanned 14:06:10          click a Train/Eval cell  Left/Right : task ───┐
│ ┌ This project ┬ Baselines ┐                                                        │
│ │ Dataset  │ Target mode  │ Model type │ Train                  │ Eval           │  │
│ │ cholec80 │ gradient-seg │ monai      │ not started            │ not run        │  │
│ │ erop     │ gradient-seg │ monai      │ done  30/30 epochs     │ done  hit@20 ..│  │
│ │ erop     │ gradient-seg │ monai_mini │ 12/30 epochs (resum..) │ not run        │  │
├─ Train : erop / gradient-seg / monai  [done  30/30 epochs] ─────────────────────────┤
│ Model dir          : data/models/erop/gradient-seg/monai                            │
│ Epochs             : 30 / 30 completed                                              │
│ Best val loss      : 0.001400                                                       │
│ ...                                                                                 │
├─ Command ───────────────────────────────────────────────────────────────────────────┤
│ ./run train-model --dataset erop --target-mode gradient-seg --model-type monai[Copy]│
│ Already complete. The same command exits with "Nothing to do": ...                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

- **상태 표** — 조합 하나가 한 행이고, `Train`·`Eval` 두 열이 그 조합의 진행 상태를 보여준다.
  행 색상은 초록(학습·평가 모두 완료), 주황(진행 중이거나 한쪽만 완료), 회색(아무것도 안 함)이다.
  `Baselines` 탭은 `Baseline`·`Dataset` 두 열로 행을 가리키며, `Train`·`Eval`은 두 탭 모두
  마지막 두 열이다.
- **상세 패널** — 선택한 칸의 근거가 된 파일들(경로, 크기, 수정 시각)과 기록된 지표를 보여준다.
- **Command** — 그 칸의 작업을 수행하는 명령. `Copy` 버튼으로 클립보드에 복사한다.
  아래 회색 줄은 현재 상태에서 그 명령을 실행하면 무슨 일이 벌어지는지에 대한 안내다.
  다른 문서에서는 `run train-model ...`로 줄여 쓰지만, 이 상자의 명령은 터미널에
  그대로 붙여넣어 실행하는 용도이므로 저장소 루트 기준 경로(`./run`, Windows는
  `.\run.bat`)로 표시된다. 저장소 루트는 `PATH`에 없어서 `run`만으로는 실행되지 않는다.
  베이스라인 행은 그 서브 프로젝트의 러너(`./baseline/<name>/run`)로 표시되고, 학습 명령에는
  그 베이스라인의 `docs/commands.md`가 보고 수치를 재현할 때 쓰는 인수가 함께 붙는다
  (`--frame-stride 5 --val-frames 1500` 등). 스크립트 기본값이 아니라서 빼면 디스크에 있는
  체크포인트와 다른 프레임으로 학습하게 되기 때문이다.

## 조작 방법

| 조작                   | 동작                                      |
| ---------------------- | ----------------------------------------- |
| `Train`/`Eval` 칸 클릭 | 그 행의 해당 작업을 선택                  |
| `↑` `↓`                | 조합(행) 이동                             |
| `←` `→`                | 같은 행에서 Train ↔ Eval 전환             |
| 탭 클릭                | `This project` ↔ `Baselines` 표 전환      |
| `F5` / `Refresh`       | 즉시 다시 스캔 (5초마다 자동 갱신도 된다) |
| `Copy`                 | 표시된 명령을 클립보드로 복사             |

## 상태 판정 기준

상태는 전부 파일 존재 여부와 내용으로만 판정한다.

### Train — `data/models/<dataset>/<target-mode>/<model-type>/`

| 표시                      | 조건                                                                     |
| ------------------------- | ------------------------------------------------------------------------ |
| `done  N/N epochs`        | `train-status.json`의 `completed_epochs` ≥ `epochs`                      |
| `M/N epochs  (resumable)` | `train-status.json`은 있으나 `completed_epochs` < `epochs` (중단된 학습) |
| `done  (weights only)`    | `best.pt`/`last.pt`는 있으나 `train-status.json`이 없음                  |
| `not started`             | 체크포인트가 하나도 없음                                                 |

`done (weights only)`는 학습 재개 기능이 생기기 전에 만들어진 체크포인트에서 나타난다.
에포크 수를 알 수 없으므로, 같은 명령으로 다시 학습하면 에포크 카운트가 1부터 시작한다
(가중치는 `last.pt`에서 이어받는다).

### Eval — `data/results/<dataset>/<target-mode>/<model-type>/`

| 표시                        | 조건                                            |
| --------------------------- | ----------------------------------------------- |
| `done  hit@20 64.96%`       | `summary.json`이 있고 hit-rate가 기록되어 있음  |
| `done`                      | `summary.json`은 있으나 hit-rate 항목이 없음    |
| `! unreadable summary.json` | 파일은 있으나 JSON 파싱 실패 (중간에 끊긴 파일) |
| `not run`                   | `summary.json` 없음                             |

### 베이스라인 — `baseline/<name>/data/model|results/<dataset>/`

판정 규칙은 같고 파일 이름과 키 이름만 다르다.

| 축         | 이 프로젝트                     | 베이스라인                                          |
| ---------- | ------------------------------- | --------------------------------------------------- |
| 체크포인트 | `best.pt` / `last.pt`           | `model.pt` / `model-last.pt`                        |
| 에포크     | `completed_epochs` / `epochs`   | `epochs_completed` / `epochs_total`                 |
| 선정 기준  | best val loss                   | best mAP@0.5:0.95 (클론) 또는 fitness (Ultralytics) |
| 평가 결과  | `data/results/.../summary.json` | `data/results/<dataset>/test/summary.json`          |
| 팁 지표    | `summary.json` 최상위           | `summary.json`의 `tip` 하위                         |

베이스라인 상세 패널에는 팁 지표 외에 탐지 지표(mAP@0.5, mAP@0.5:0.95)와 측정된 속도
(ms/frame, fps, 측정 GPU)도 함께 나온다. `yolo26`·`yolo26clone`은 end-to-end 헤드라 NMS IoU가
없으므로 그 자리에 `max det`이 표시된다.

## 알려진 제약

- 학습이 **지금 돌고 있는지**는 알 수 없다. `M/N epochs (resumable)`은 "중단됐다"와
  "다른 터미널에서 진행 중이다"를 구분하지 않는다. 상세 패널의 `Last epoch at`
  시각과 5초 자동 갱신으로 진행 여부를 짐작할 수 있다.
- 평가가 **최신 체크포인트 기준인지**는 검사하지 않는다. 학습을 더 돌린 뒤에는
  `Eval`이 `done`이어도 다시 평가해야 최신 결과가 된다.
- 속도 비교(`compare-speed`) 결과는 이 표에 포함되지 않는다.
- 베이스라인의 이전 회차(`data/model-16x16`·`data/results-16x16`)는 표에 나오지 않는다.
  현재 회차로 대체된 실험이라, 어느 쪽이 현행인지 흐려지지 않도록 제외했다.
- `Baselines` 탭의 `Eval`은 test 스플릿 결과만 본다. 다른 스플릿으로 평가해
  `data/results/<dataset>/val/`에 쌓은 결과가 있어도 표에는 반영되지 않는다.
