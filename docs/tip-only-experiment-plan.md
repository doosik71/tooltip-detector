# tip-only 학습 실험 계획

## 1. 연구 질문

`baseline/` 아래 탐지 베이스라인은 모두 **`tool`(수술도구 상자)과 `tip`(팁 좌표를 32 × 32 px
상자로 바꾼 것) 2클래스**를 하나의 탐지기로 학습한다. 이 실험은 그 공통 전제를 제거한다.

> **수술도구 상자 어노테이션 없이 팁 상자만으로 학습하면 팁 탐지 성능이 어떻게 되는가?**

결론이 갖는 실무적 의미는 두 갈래다.

| 관측                             | 판단                                                                    |
| -------------------------------- | ----------------------------------------------------------------------- |
| 팁 지표에 유의미한 차이가 없다   | 팁 좌표만 레이블링해도 되므로 어노테이션 비용을 줄일 수 있다            |
| tool 학습 쪽이 유의미하게 앞선다 | 도구 상자는 팁 학습에 대한 보조 감독 신호로 값을 하므로 비용을 지불한다 |

루트 프로젝트의 히트맵 모델은 이미 같은 축의 실험(`gradient-seg` 대 `gaussian-tip`, 즉
분할 마스크의 유무)을 마쳤고 "마스크 레이블링 비용이 정확도로 회수되지 않는다"는 결론을
얻었다. 이 실험은 **탐지 계열에서 상자 레이블에 대해 같은 질문**을 던지는 것이고, 두 결과를
나란히 놓으면 "팁 좌표 외의 부가 레이블이 팁 탐지에 얼마나 기여하는가"라는 하나의 주제가 된다.

### 1.1 용어

두 학습 모드를 부르는 이름을 먼저 고정한다. 이 이름이 그대로 실행 옵션의 값이자 산출물
디렉터리의 이름이 된다.

| 모드      | 학습하는 클래스      | 의미                                    |
| --------- | -------------------- | --------------------------------------- |
| `tooltip` | `tool` + `tip` (2개) | 지금까지의 모든 베이스라인 학습. 기본값 |
| `tiponly` | `tip` (1개)          | 이 실험이 새로 추가하는 조건            |

`tooltip`은 저장소 이름(`tooltip-detector`)과 철자가 같지만 여기서는 **"tool과 tip을 함께
학습하는 모드"** 를 가리킨다. 문서에서 모드를 뜻할 때는 항상 코드 서체로 적는다.

## 2. 실험 설계

### 2.1 대상

스크래치 학습 클론 3종 × 데이터셋 2개 = **6개 조합**을 새로 학습한다.

| 베이스라인     | 아키텍처 특성                       | 라벨 할당          |
| -------------- | ----------------------------------- | ------------------ |
| `cladnet`      | CSPDarknet53 + CLAA 넥, 앵커 기반   | 정적 (앵커 비율)   |
| `yolov8sclone` | 앵커프리 + DFL                      | 동적 (TaskAligned) |
| `yolo26clone`  | 앵커프리, DFL 없음, end-to-end 헤드 | 동적 (TaskAligned) |

`baseline/yolov8s`와 `baseline/yolo26`은 **학습 대상에서** 제외한다. 둘은 `ultralytics` +
COCO 사전학습 가중치를 쓰는 참조 구현이라 학습 조건이 다르고,
[docs/baseline-report.md](baseline-report.md)의 비교군에서도 같은 이유로 빠져 있다. 세 클론은
사전학습 없이 동일 레시피(150 에포크, `--frame-stride 5`, 640 × 640, 팁 32 px)로 학습되므로
**바뀌는 변인이 학습 모드 하나뿐**이다.

다만 **디렉터리 규칙 변경은 5종 전부에 적용한다**(§3.1). 두 참조 구현만 옛 경로에 남겨 두면
대시보드와 요약 스크립트가 베이스라인마다 다른 경로 규칙을 알아야 하기 때문이다.

### 2.2 `tiponly`의 정의

**`tip` 1클래스 모델**로 학습한다. `tool` 상자는 라벨 생성 단계에서 아예 만들지 않고,
탐지 헤드의 클래스 채널도 1개로 줄인다. "팁만 레이블링했다면 무엇을 학습했을 것인가"라는
비용 절감 시나리오에 그대로 대응하는 설정이다.

헤드의 클래스 채널이 하나 줄면서 파라미터 수가 미세하게 달라진다. 실측값은 다음과 같다.

| 베이스라인     | `tooltip` (nc=2) | `tiponly` (nc=1) |           차이 |
| -------------- | ---------------: | ---------------: | -------------: |
| `cladnet`      |        7,488,483 |        7,487,610 | -873 (0.012 %) |
| `yolov8sclone` |       11,136,374 |       11,135,987 | -387 (0.003 %) |
| `yolo26clone`  |        9,949,412 |        9,948,638 | -774 (0.008 %) |

세 경우 모두 0.02 % 미만이므로 관측될 성능 차이를 모델 용량으로 설명할 수는 없다.

### 2.3 통제하는 것과 통제하지 못하는 것

**동일하게 유지:** 에포크 수(150), `--frame-stride 5`, `--val-frames 1500`, 배치 16,
학습률·옵티마이저·스케줄, 입력 해상도 640, 팁 상자 크기 32 px, 증강(mosaic·HSV), EMA,
데이터 분할, 평가 조건(test 전수, `conf 0.25`, NMS IoU 0.45, AP 곡선용 `map-conf 0.001`).

**통제하지 못하는 것 세 가지를 미리 적어 둔다.**

1. **`cladnet`의 앵커.** `common/model.py`의 `ANCHORS`는 cholec80 train 스플릿에서
   tool 상자와 상수 크기 tip 상자를 함께 k-means하여 얻은 값이다. `tiponly`에서도 이 앵커를
   **그대로 둔다.** 다시 뽑으면 변인이 둘이 되기 때문이다. 대신 tool 크기에 맞춰진 큰 앵커는
   `tiponly`에서 양성을 하나도 받지 못하므로, 유효 용량이 줄어드는 방향의 불리함이 `cladnet`에만
   존재한다. `cladnet`의 `tiponly` 결과가 다른 둘과 다른 방향으로 나오면 이 항목을 먼저 의심하고,
   팁 전용 앵커로 다시 뽑는 추가 실험을 별도 변인으로 돌린다.
2. **학습 예산의 실질적 변화.** GT 상자 수가 절반 가까이 줄어 에포크당 손실 신호의 총량이
   달라진다. 이는 제거할 수 없는 처치의 일부다(팁만 레이블링하면 실제로 그렇게 된다).
3. **단일 실행.** 조합당 1회만 학습한다. 셀 하나의 작은 차이는 실행 간 변동과 구분되지 않는다.
   따라서 판정은 개별 셀이 아니라 **6개 셀에 걸친 방향의 일관성**으로 내린다(§5).

### 2.4 비교 지표

팁 지표는 두 모드에서 정의가 동일하므로 그대로 비교한다.

| 지표                                             | 비교 가능 여부                                  |
| ------------------------------------------------ | ----------------------------------------------- |
| Miss rate, Hit@10/20/50 px, median/mean/p90 dist | 가능 (주 판정 지표는 **Hit@10 px**와 miss rate) |
| 헝가리안 일대일 정밀도·재현율                    | 가능                                            |
| `tip` AP@0.5, AP@0.5:0.95                        | 가능                                            |
| `tool` AP, 전체 mAP                              | **불가**. `tiponly`에는 `tool` 클래스가 없다    |

`tooltip` 쪽의 현재 test 수치가 비교 기준선이다.

| 베이스라인 / 데이터셋     | Miss rate | Hit@10 px | Median dist | `tip` AP@0.5 |
| ------------------------- | --------: | --------: | ----------: | -----------: |
| `cladnet` / cholec80      |    9.56 % |   56.65 % |     4.93 px |       0.5773 |
| `cladnet` / erop          |    6.84 % |   68.47 % |     3.23 px |       0.6983 |
| `yolov8sclone` / cholec80 |    8.82 % |   59.87 % |     3.60 px |       0.6661 |
| `yolov8sclone` / erop     |    4.79 % |   75.91 % |     2.04 px |       0.8213 |
| `yolo26clone` / cholec80  |    9.71 % |   59.27 % |     3.45 px |       0.6321 |
| `yolo26clone` / erop      |    5.96 % |   74.24 % |     1.96 px |       0.7995 |

## 3. 구현 계획

### 3.1 설계 방침: 데이터셋 아래 모드 서브폴더

산출물 경로에 **모드 단계를 하나 추가한다.** 기존 산출물은 삭제하지 않고 `tooltip/`
서브폴더로 옮기며, 새 학습은 그 옆 `tiponly/`에 쓴다.

```text
baseline/<name>/data/
├── model/<dataset>/
│   ├── tooltip/          # 기존 산출물이 이동해 오는 자리
│   │   ├── model.pt
│   │   ├── model-last.pt
│   │   ├── train-status.json
│   │   └── metric.csv
│   └── tiponly/          # 신규
└── results/<dataset>/
    ├── tooltip/<split>/  # summary.json, per_tip.csv
    └── tiponly/<split>/
```

`yolov8s`·`yolo26`은 Ultralytics 자체 실행 디렉터리를 함께 두므로
`data/model/<dataset>/tooltip/ultralytics/`가 된다. 두 서브 프로젝트가 변환해 두는
`data/yolo/<dataset>/`(준비된 YOLO 데이터셋)은 `tooltip` 라벨 전용이므로 이번 범위에서
건드리지 않는다.

**경로를 인수로 받지 않고 모드에서 유도한다.** 스크립트에는 `--label-set {tooltip,tiponly}`
옵션 하나만 추가하고(기본값 `tooltip`), 경로는 전부 그 값에서 계산한다. 값 문자열이 곧
디렉터리 이름이므로 `--output-dir`를 직접 적을 일이 없다. 기존의 `--output-dir`는 예외적
용도(스모크 테스트 등)로 남겨 둔다.

세 클론만 두 값을 모두 받는다. `yolov8s`·`yolo26`은 §2.1대로 `tiponly` 학습을 하지 않으므로
옵션을 추가하지 않고 경로에 `tooltip` 단계를 상수로 넣는다. 나중에 두 참조 구현까지
확장하려면 옵션 추가와 `nc: 1` 데이터셋 준비가 함께 필요하다.

**기존 접미사 관례와의 관계.** `cladnet`·`yolov8sclone`에는 이전 회차의 산출물이
`data/model-16x16`·`data/results-16x16`에 남아 있고 `generate-summary.py`의 `--suffix`가
이를 읽는다. 이 관례는 **회차**를 가르고, 새 서브폴더는 **모드**를 가르므로 축이 다르다.
두 축을 함께 쓸 수 있도록 접미사 디렉터리 안에도 같은 모드 단계를 넣는다
(`data/model-16x16/<dataset>/tooltip/`). 그래야 경로 규칙이 하나로 유지된다.

### 3.2 마이그레이션 (코드 변경 직후 1회)

`baseline/*/data`는 각 서브 프로젝트의 `.gitignore`에 걸려 git이 추적하지 않으므로
파일 이동으로 처리한다. 같은 파일시스템 안이라 즉시 끝난다.

대상은 5개 베이스라인 × `model`·`results`, 그리고 `cladnet`·`yolov8sclone`의
`model-16x16`·`results-16x16`까지 포함해 **총 24개 데이터셋 디렉터리**다.

```bash
for base in baseline/*/data/model baseline/*/data/model-16x16 \
            baseline/*/data/results baseline/*/data/results-16x16; do
  [ -d "$base" ] || continue
  for ds in "$base"/*/; do
    [ -d "$ds" ] || continue
    [ -d "$ds/tooltip" ] && continue          # 이미 옮겼으면 건너뛴다
    mkdir -p "$ds/tooltip"
    find "$ds" -mindepth 1 -maxdepth 1 ! -name tooltip -exec mv {} "$ds/tooltip/" \;
  done
done
```

이동 뒤 확인할 것:

- `model/<dataset>/tooltip/`에 `model.pt`·`model-last.pt`·`train-status.json`·`metric.csv`가
  모두 있고 `model/<dataset>/` 바로 아래에는 `tooltip/`만 남아 있다
- `results/<dataset>/tooltip/test/`에 `summary.json`·`per_tip.csv`가 있다
- `yolov8s`·`yolo26`은 `model/<dataset>/tooltip/ultralytics/`가 온전하다
- 파일 개수와 총 용량이 이동 전후로 같다

이 스크립트는 멱등이다. `tooltip/`이 이미 있으면 그 데이터셋을 건너뛰므로 두 번 돌려도
`tooltip/tooltip/`이 생기지 않는다.

### 3.3 클론 3종 공통 변경

세 클론은 파일 구조와 호출 지점이 대칭이므로 같은 변경을 세 번 적용한다.

**`common/dataset.py`**

- `LABEL_SETS = {"tooltip": ("tool", "tip"), "tiponly": ("tip",)}`,
  `DEFAULT_LABEL_SET = "tooltip"` 추가
- `class_names(label_set)`, `tip_class(label_set)` 헬퍼 추가.
  `tiponly`에서 `tip` 인덱스는 1이 아니라 **0**이 된다
- `load_annotation(..., label_set=...)`: `tooltip`이면 지금 그대로, `tiponly`면 `bbox` 행을
  만들지 않고 `tip` 행의 클래스 인덱스를 0으로 쓴다
- `_MIN_AREA_KEPT`의 키를 정수 인덱스에서 **클래스 이름**으로 바꾼다. 현재
  `{TOOL_CLASS: 0.2, TIP_CLASS: 0.8}`는 인덱스 0을 tool로 가정하므로, `tiponly`에서 tip이
  0번이 되면 tip에 tool의 느슨한 mosaic 유지 기준 0.2가 적용된다. 잘린 tip 상자가 학습에
  섞이는 조용한 오염이므로 반드시 함께 고친다
- `SurgicalDetectionDataset.__init__`에 `label_set` 인수 추가 후 `load_annotation`으로 전달

**`common/inference.py`**

- `model_dir(dataset, label_set=DEFAULT_LABEL_SET)` →
  `data/model/<dataset>/<label_set>/`
- `results_dir(dataset, label_set=DEFAULT_LABEL_SET)` →
  `data/results/<dataset>/<label_set>/`
- `trained_datasets(label_set=...)`, `default_model_path(dataset, label_set=...)`도 같은 인수를
  받는다. 글롭 패턴의 단계 수가 하나 늘어난다
- `Detector`는 이미 체크포인트의 `class_names`를 읽어 `build(num_classes=len(...))`로 모델을
  만든다. **수정 불필요.** `tiponly` 체크포인트가 그대로 로드된다

**`scripts/train-model.py`**

- `--label-set {tooltip,tiponly}` 추가 (기본 `tooltip`)
- `CLASS_NAMES` 상수 참조를 `class_names(args.label_set)`로 교체 (모델 빌드,
  `DetectionEvaluator`, 체크포인트의 `class_names` 필드, best 재구성 경로의 4개 지점)
- 기본 `--output-dir`를 `model_dir(args.dataset, args.label_set)`로
- `train-status.json`은 `vars(args)`를 그대로 쓰므로 `label_set`이 자동으로 기록된다

**`scripts/eval-model.py`**

- `--label-set` 추가. 기본 `--model`과 출력 경로를 이 값에서 유도한다
- 실제 클래스 구성은 **체크포인트의 `class_names`를 신뢰**한다 (`tip_box_size`를 다루는 방식과
  동일한 원칙). `--label-set`와 체크포인트가 불일치하면 즉시 오류로 멈춘다
- 모듈 상수 `TIP_CLASS` 참조를 `detector.class_names.index("tip")`로 교체.
  `baseline/yolov8s/scripts/eval-model.py`의 `class_index()`가 이미 쓰는 방식이다
- 데이터셋 GT도 같은 `label_set`으로 만들어 `tip` 행을 뽑는다
- `summary.json`에 `label_set`과 `class_names`를 기록한다

**`scripts/generate-summary.py`**

- `--label-set` 추가. 스캔 루트가 `data/model{suffix}/<dataset>/<label-set>/`가 되도록
  글롭을 한 단계 늘린다. `--suffix`(회차 축)는 그대로 남는다
- 파라미터 수를 계산하는 `build(num_classes=len(CLASS_NAMES))` 한 줄이 모드에 맞는 클래스 수를
  쓰도록 고친다
- 문서 머리말에 어느 모드의 수치인지 표시한다

**`scripts/demo.py`**

- `--label-set` 추가해 열 체크포인트를 고른다. `Detector`가 클래스 이름을 스스로 읽으므로
  표시 로직은 변경 불필요

### 3.4 참조 구현 2종 (`yolov8s`·`yolo26`) 변경

옵션은 추가하지 않고 경로만 맞춘다.

- `common/dataset.py`의 `model_dir()`·`results_dir()`가 `tooltip` 단계를 상수로 포함하도록
  수정
- `train-model.py`·`eval-model.py`·`demo.py`·`generate-summary.py`는 그 헬퍼를 통해
  경로를 얻으므로 호출부 수정은 최소로 그친다. 직접 경로를 조립하는 곳이 있으면 헬퍼로 모은다

### 3.5 루트 프로젝트 변경

**`ttd/tip_source.py`**

- `describe_checkpoint()`의 경로 판정이
  `baseline/<name>/data/model/<dataset>/<file>.pt` 형태(꼬리 5단계)만 인식한다.
  모드 단계가 늘었으므로 `baseline/<name>/data/model/<dataset>/<label-set>/<file>.pt`
  (꼬리 6단계)를 인식하도록 고치고, `label` 문자열에 모드를 넣는다
  (`cladnet/cholec80/tiponly`)
- 팁 클래스는 이미 이름(`_TIP_CLASS_NAME = "tip"`)으로 찾으므로 그 외 변경 불필요
- 인수 없이 실행했을 때 출력하는 가용 모델 목록의 글롭도 한 단계 늘린다

**`scripts/dashboard.py`**

- `BASELINES` 각 항목에 `label_sets` 필드 추가. 기본 `("tooltip",)`, 클론 3종만
  `("tooltip", "tiponly")`
- Baselines 탭 행 수: 현재 10행 → **16행** (`yolov8s`·`yolo26` 각 2행 + 클론 3종 × 데이터셋 2개
  × 모드 2개 = 12행)
- `Labels` 열 추가 (`tooltip` / `tiponly`)
- `baseline_model_dir`·`baseline_results_dir`에 모드 단계 반영, 클론 행의 학습·평가 명령
  문자열에 `--label-set <mode>` 추가

### 3.6 문서

| 문서                                               | 변경                                                            |
| -------------------------------------------------- | --------------------------------------------------------------- |
| `baseline/<name>/README.md`                        | 디렉터리 구조 절의 경로를 모드 단계 포함으로 갱신               |
| `baseline/<clone>/README.md`                       | "무엇을 탐지하는가" 절에 두 모드 설명 추가                      |
| `baseline/<clone>/docs/commands.md`                | `--label-set tiponly` 학습·평가·요약 재현 명령 추가             |
| `baseline/<clone>/docs/experimental-results.md`    | `tiponly` 결과 절 추가. `-16x16` 회차를 가리키는 경로 문구 갱신 |
| `baseline/<clone>/docs/summary-results-tiponly.md` | `generate-summary --label-set tiponly` 산출물 (신규)            |
| `docs/baseline-report.md`                          | "tool 감독 신호의 기여" 절 추가 (6개 셀 비교표 + 결론)          |
| `notebook/baseline-report-graph.ipynb`             | 두 모드 대비 그래프 추가. 산출물 경로 갱신                      |
| `docs/dashboard-guide.md`                          | `Labels` 열과 16행 구성 설명                                    |
| `README.md`                                        | 대시보드 행 수 설명, 베이스라인 절, 트래커 경로 규칙 표 갱신    |

`README.md`의 트래커 경로 규칙 표에 있는
`baseline/<name>/data/model/<dataset>/model.pt` 항목은 모드 단계를 포함하도록 반드시 함께
고친다. 이 표가 `ttd/tip_source.py`의 판정 규칙을 사용자에게 설명하는 유일한 자리다.

## 4. 실행 계획

### 4.1 1단계. 구현·마이그레이션·스모크 검증 (학습 전, 반나절)

순서는 **코드 변경 → 마이그레이션(§3.2) → 검증**이다. 마이그레이션 전에는 새 코드가 기존
체크포인트를 찾지 못하므로, 둘을 한 작업으로 묶어 끝낸다.

GPU 47시간을 쓰기 전에 다음을 모두 통과시킨다.

1. **마이그레이션 무손실.** §3.2의 확인 항목 네 가지를 통과한다. 파일 개수·총 용량이 이동
   전후로 같아야 한다.
2. **회귀 없음.** `--label-set`을 생략한 실행이 변경 전과 **동일하게 동작**한다(경로는 모드
   단계만큼 깊어지지만 학습 거동은 같다). 세 클론에 대해
   `--frame-stride 500 --epochs 1 --no-resume --output-dir <임시경로>`로 1 에포크를 돌려
   변경 전후의 학습 손실이 일치하는지 본다.
3. **재개.** 마이그레이션 뒤 `run train-model --dataset cholec80`(옵션 없음)이
   `model/<dataset>/tooltip/model-last.pt`를 찾아 "150/150 완료"로 즉시 끝나는지 확인한다.
   경로를 잘못 잡으면 학습이 처음부터 다시 시작되므로 이 확인이 가장 중요하다.
4. **라벨 생성.** `tiponly` 라벨의 행 수가 GT 팁 수와 정확히 같고 `tool` 행이 0개인지,
   클래스 인덱스가 전부 0인지 프레임 표본으로 확인한다.
5. **mosaic 유지 기준.** `_MIN_AREA_KEPT`가 이름 기준으로 바뀌어 `tiponly`에서도 tip에 0.8이
   적용되는지 단언한다.
6. **체크포인트 자기기술.** `tiponly` 체크포인트의 `class_names`가 `["tip"]`이고,
   `Detector`가 이를 읽어 nc=1로 모델을 세우는지 확인한다.
7. **경로 분리.** `tiponly` 학습이 `model/<dataset>/tiponly/`에만 쓰고 `tooltip/`을 건드리지
   않는지 확인한다.
8. **평가 경로.** `tiponly` 체크포인트를 `--split val --limit 200`으로 평가해
   `results/<dataset>/tiponly/val/`에 산출물이 생기고 `tool` AP 항목이 없는지 확인한다.
9. **불일치 차단.** `tooltip` 체크포인트에 `--label-set tiponly`를 주면 오류로 멈추는지 확인한다.
10. **트래커.** `tiponly` 체크포인트로 `run tooltip-tracker`가 열리고, 인수 없이 실행했을 때
    목록에 두 모드가 모두 나오는지 확인한다.
11. **대시보드.** Baselines 탭이 16행으로 뜨고 기존 10개 조합이 전부 "완료"로 판정되는지
    확인한다. 마이그레이션 누락을 잡는 두 번째 그물이다.

### 4.2 2단계. 학습 (약 18시간, 3 GPU 병렬)

`cuda:0`은 PCIe 성능 저하가 있으므로 `cuda:1`〜`cuda:3`만 쓴다. 기존 `tooltip` 학습의
실측 시간을 그대로 예산으로 삼는다(`tiponly`는 같거나 조금 빠를 것이다).

| GPU      | 작업                                                                     |   예상 |
| -------- | ------------------------------------------------------------------------ | -----: |
| `cuda:1` | `cladnet` / erop                                                         | 14.2 h |
| `cuda:2` | `cladnet` / cholec80 → `yolo26clone` / cholec80                          | 15.0 h |
| `cuda:3` | `yolo26clone` / erop → `yolov8sclone` / erop → `yolov8sclone` / cholec80 | 18.1 h |

합계 47.3 GPU-h, 벽시계 약 18시간이다. 명령 형태는 다음과 같다.

```bash
./baseline/cladnet/run train-model --dataset erop --label-set tiponly \
    --frame-stride 5 --val-frames 1500 --workers 12 --device cuda:1
```

`--label-set`을 제외한 모든 인수는 기존 `tooltip` 학습과 동일하다.

### 4.3 3단계. 평가와 요약 (1시간 미만)

```bash
./baseline/<clone>/run eval-model --dataset <ds> --label-set tiponly --split test
./baseline/<clone>/run generate-summary --label-set tiponly \
    --output docs/summary-results-tiponly.md
```

test 전수(36k 프레임)를 프레임당 5〜9 ms로 처리하므로 조합당 5분 내외다. 기존 `tooltip`
요약도 새 경로에서 다시 생성해 두 문서를 같은 세대로 맞춘다.

### 4.4 4단계. 보고서 작성

`docs/baseline-report.md`에 절을 추가하고 그래프 노트북을 갱신한다.

## 5. 판정 기준 (사전 확정)

결과를 보고 기준을 고르지 않도록 미리 정한다. 주 지표는 **Hit@10 px**, 보조 지표는
**miss rate**와 **`tip` AP@0.5**다. 판정 단위는 `tooltip` 대비 `tiponly`의 차이이며,
6개 셀 전체를 함께 본다.

| 조건                                                          | 결론                                                            |
| ------------------------------------------------------------- | --------------------------------------------------------------- |
| 6개 셀 모두 Hit@10 px 차이가 ±1 %p 이내이거나 방향이 엇갈린다 | **차이 없음.** 팁만 레이블링하여 어노테이션 비용을 줄일 수 있다 |
| 5개 이상 셀에서 같은 방향으로 2 %p 이상 차이가 난다           | **유의미.** 그 방향의 레이블 정책을 채택한다                    |
| 그 사이                                                       | 데이터셋·아키텍처별로 나누어 서술하고 결론을 유보한다           |

어느 경우든 보고서에는 **조합당 단일 실행이라는 한계**를 명시하고, 셀 하나의 작은 차이를
근거로 삼지 않는다. `cladnet`만 방향이 다르면 §2.3의 앵커 항목을 원인 후보로 검토한다.

## 6. 성공 기준

1. 기존 10개 조합의 학습·평가 산출물이 `tooltip/` 아래로 **손실 없이** 이동했고, 옵션 없는
   기존 명령이 그 산출물을 그대로 찾아낸다.
2. `--label-set`을 생략한 모든 명령이 변경 전과 동일하게 동작한다.
3. `tiponly` 학습·평가가 `tooltip/` 산출물을 한 번도 건드리지 않는다.
4. 6개 `tiponly` 조합의 `model.pt`·`summary.json`·`per_tip.csv`가 생성된다.
5. 대시보드 Baselines 탭이 16행으로 두 모드의 진행 상태를 함께 보여준다.
6. `docs/baseline-report.md`가 §5의 기준으로 판정한 결론을 담는다.

## 7. 범위 밖

- `baseline/yolov8s`·`baseline/yolo26`의 `tiponly` **학습**. 경로 규칙만 맞추고 옵션은 넣지
  않는다. 적용하려면 `prepare-dataset.py`가 `nc: 1` 데이터셋을 따로 준비해야 하고,
  사전학습 가중치를 쓰므로 세 클론과 조건이 다르다.
- 두 참조 구현의 `data/yolo/<dataset>/`(준비된 YOLO 데이터셋) 레이아웃 변경.
- 루트 프로젝트의 히트맵 모델. 이미 `gradient-seg` 대 `gaussian-tip`으로 같은 축의 답을 갖고 있다.
- 다중 시드 반복. §2.3에서 한계로 기록하고, §5의 기준을 셀 단위가 아닌 방향 일관성으로 둔 것이
  그에 대한 대응이다.
- `cladnet`의 팁 전용 앵커 재산출. §5에서 필요하다고 판정될 때만 별도 변인으로 돌린다.
