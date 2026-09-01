# YOLO26 수술도구 탐지 베이스라인

Ultralytics가 공개한 최신 모델 **YOLO26**을 이 저장소의 복강경 데이터셋으로 직접 학습하고
평가하는 서브 프로젝트다 (여기서는 `ultralytics` 8.4.131의 구현을 쓴다).
옆의 [baseline/yolov8s](../yolov8s/)가 2024년에 공개된 YOLOv8s 가중치를 *그대로 돌려 보는*
데모라면, 이쪽은 같은 계열의 최신 모델을 *우리 데이터로 학습해* tooltip-detector와 같은
지표로 채점한다.

모델과 학습 루프는 전부 `ultralytics` 패키지의 것이다. 이 서브 프로젝트가 하는 일은 세 가지다.

1. 이 저장소의 JSON 어노테이션을 Ultralytics가 읽는 YOLO 데이터셋으로 바꾼다.
2. 학습 산출물을 다른 베이스라인과 같은 파일 레이아웃으로 정리한다.
3. 평가를 루트 프로젝트와 **같은 매칭 규칙**으로 다시 계산해, 팁 지표를 직접 비교할 수 있게 한다.

## 다른 베이스라인과의 관계

| 서브 프로젝트                              | 모델                | 가중치            | 학습    | 평가 지표                    |
| ------------------------------------------ | ------------------- | ----------------- | ------- | ---------------------------- |
| [yolov8s](../yolov8s/)                     | YOLOv8s (7클래스)   | 공개 HF 체크포인트 | 안 함  | 없음 (관측용 데모)           |
| [yolov8sclone](../yolov8sclone/)           | YOLOv8s 재구현      | 스크래치          | 직접   | 탐지 AP + 팁 Hit-rate        |
| [cladnet](../cladnet/)                     | CLAD-Net 재구현     | 스크래치          | 직접   | 탐지 AP + 팁 Hit-rate        |
| **yolo26** (이 폴더)                       | YOLO26 (Ultralytics) | COCO 사전학습 (기본) | 직접 | 탐지 AP + 팁 Hit-rate        |
| [yolo26clone](../yolo26clone/)             | YOLO26 재구현       | 스크래치          | 직접   | 탐지 AP + 팁 Hit-rate        |

재구현 베이스라인들은 `ultralytics` 의존성을 없애는 것이 목적이고, 이 베이스라인은 반대로
**공개된 YOLO26을 있는 그대로** 쓰는 것이 목적이다. 둘은 짝이다. 여기서 내려받은
`data/pretrained/yolo26s.pt`가 [yolo26clone](../yolo26clone/)의 검증 기준이 되며, 그 프로젝트는
파라미터·가중치·출력·손실을 이 참조와 대조해 재구현이 맞다는 것을 보인다.

## YOLO26이 YOLOv8과 무엇이 다른가

설치된 `ultralytics` 8.4의 구현에서 확인한 차이다. 마케팅 용어가 아니라 코드에 있는 것만 적는다.

| 항목        | YOLOv8s                                   | YOLO26s                                                          |
| ----------- | ----------------------------------------- | ---------------------------------------------------------------- |
| 추론 후처리 | NMS 필요                                  | **end-to-end**: 객체당 박스 하나, NMS 없음 (`end2end: True`)     |
| 박스 회귀   | DFL, 변마다 16개 빈의 분포                | **DFL 없음** (`reg_max: 1`), 세 번째 손실항이 `l1_loss`          |
| CSP 블록    | C2f                                       | C3k2                                                             |
| 백본 말단   | SPPF                                      | SPPF + C2PSA (attention 블록)                                    |
| 손실        | box 7.5 / cls 0.5 / dfl 1.5, 단일 분기    | one2many(topk 10)와 one2one(topk 7) 두 분기의 가중합. one2many 가중치가 학습이 진행되며 0.8에서 0.1로 감쇠한다 |
| 옵티마이저  | SGD                                       | `optimizer=auto`가 이 규모에서 **MuSGD**를 고른다                |
| 파라미터    | 11,136,xxx (2클래스)                      | **9,949,412** (2클래스, `--scale s`)                             |

실무적으로 가장 큰 차이는 첫 줄이다. **NMS가 없으므로 조정할 IoU 임계값도 없다.** 다른
베이스라인의 `--iou`에 해당하는 인수가 이 서브 프로젝트에는 없고, 대신 프레임당 박스 수
상한인 `--max-det`만 있다.

## 무엇을 탐지하는가

이 저장소의 어노테이션은 도구마다 `bbox`와 `tip`을 갖고 클래스 레이블은 없다. 그래서
[yolov8sclone](../yolov8sclone/), [cladnet](../cladnet/)과 **똑같이** 두 클래스로 학습한다.

| 클래스 | 정의                                                             | 출처                 |
| ------ | ---------------------------------------------------------------- | -------------------- |
| `tool` | 어노테이션의 바운딩 박스 그대로                                  | `annotations[].bbox` |
| `tip`  | 팁 좌표를 중심으로 한 **32 × 32 px 박스** (`--tip-box-size`)     | `annotations[].tip`  |

예측된 `tip` 박스의 중심이 팁 좌표이므로, 탐지 지표(AP)와 이 프로젝트의 팁 지표
(Hit-rate @ N px)를 같은 모델에서 함께 잴 수 있다.

팁 박스 크기는 형식이 아니라 실제 하이퍼파라미터다. 앵커프리 모델은 라벨 할당이 "박스 안에
들어오는 앵커 포인트" 중에서 이뤄지므로, 박스가 작으면 팁은 할당 정원을 채우지 못한다.
32 px이라는 값의 근거(640 × 640 letterbox 후 앵커 수 측정)는
[yolov8sclone README](../yolov8sclone/README.md)에 있고, 세 베이스라인이 같은 값을 쓰기 때문에
그대로 따랐다. 이 값은 `prepare-dataset` 단계에서 라벨에 구워지며 `prepare-status.json`과
학습된 모델의 `model-info.json`에 기록된다. 평가는 그 값을 읽어 라벨을 다시 만들므로, 학습
때 못 본 크기로 채점되는 일은 없다.

## 디렉터리 구조

```text
baseline/yolo26/
├── pyproject.toml            # 독립 uv 프로젝트 (아래 "설치" 참고)
├── run(.bat)                 # `run <script> [args...]` → scripts/<script>.py 실행
├── common/
│   ├── dataset.py            # 어노테이션 → 2클래스 YOLO 라벨, 이 베이스라인의 모든 경로 규칙
│   ├── inference.py          # 체크포인트 + 사이드카 메타데이터, 프레임 1장 추론
│   ├── metrics.py            # AP@0.5, AP@0.5:0.95, precision, recall
│   ├── tipmetrics.py         # Hit-rate @ N px (루트 프로젝트와 동일한 매칭 규칙)
│   ├── boxes.py              # 탐지 지표가 쓰는 IoU 행렬
│   ├── progress.py           # 공통 tqdm 설정
│   ├── sources.py            # 영상 / 추출 프레임 디렉터리 읽기 (데모용)
│   └── draw.py               # 예측·GT 오버레이
├── scripts/
│   ├── prepare-dataset.py    # 어노테이션 → Ultralytics YOLO 데이터셋 (학습 전 1회)
│   ├── train-model.py        # 학습
│   ├── eval-model.py         # 평가 (탐지 AP + 팁 hit-rate)
│   ├── demo.py               # 탐지 결과 시각화 GUI
│   └── generate-summary.py   # data/ → docs/summary-results.md
├── docs/
│   └── commands.md           # 실행 명령 모음
└── data/                     # 준비된 데이터셋·체크포인트·평가 결과 (git 추적 제외)
    ├── pretrained/           #   내려받은 COCO 가중치 (yolo26s.pt 등)
    ├── yolo/<dataset>/       #   Ultralytics가 읽는 데이터셋 (아래 참고)
    ├── model/<dataset>/      #   학습 산출물
    │   ├── model.pt          #     최고 성능 체크포인트
    │   ├── model-last.pt     #     마지막 에포크 (재개용)
    │   ├── model-info.json   #     팁 박스 크기·입력 크기·스케일·데이터셋·에포크
    │   ├── train-status.json #     진행 상황과 실행 인수
    │   ├── metric.csv        #     에포크별 학습 곡선
    │   └── ultralytics/      #     Ultralytics 자체 run 디렉터리 (weights/, results.csv, 그림)
    └── results/<dataset>/<split>/
        ├── summary.json      #     전체 지표 + 실행 파라미터
        └── per_tip.csv       #     GT 팁 1개당 1행
```

## 설치

이 서브 프로젝트는 **저장소 루트와 별개의 uv 프로젝트**다. `ultralytics`가 요구하는
`opencv-python`과 루트 환경의 `opencv-python-headless`가 같은 `cv2` 패키지에 파일을 쓰기
때문에, 한 환경에 섞으면 기존 학습·평가 환경이 깨진다.
[yolov8s](../yolov8s/)와 같은 이유로 자체 `.venv`를 쓴다.

```bash
uv sync --project baseline/yolo26
```

yolov8s의 의존성에 `scipy`(팁 지표의 Hungarian 매칭)와 `tqdm`(진행 표시)이 더 필요하다.
Ultralytics 8.4는 둘 다 의존하지 않는다.

## 사용법

모든 스크립트는 `run`으로 실행한다. 어느 디렉터리에서 실행해도 되고, Windows에서는
`run.bat`을 쓴다. 인자 없이 `run`만 실행하면 스크립트 목록이 나온다.
데이터셋별 전체 재현 명령은 [docs/commands.md](docs/commands.md)에 있다.

### 1. 데이터셋 준비 (학습 전 1회)

```bash
./baseline/yolo26/run prepare-dataset --dataset cholec80
```

Ultralytics는 자기 디렉터리 규칙으로 이미지와 라벨을 읽는데 `data/dataset/`은 읽기 전용
마운트이고 어노테이션도 JSON이다. 그 사이를 메우는 변환 단계이며, 결과는
`data/yolo/<dataset>/`에 만들어진다.

| 인수                | 기본값       | 설명                                                                    |
| ------------------- | ------------ | ----------------------------------------------------------------------- |
| `--dataset`         | (필수)       | `data/dataset/` 아래 디렉터리 이름 (`cholec80` / `erop`)                |
| `--splits`          | `train val`  | 준비할 스플릿. 평가는 원본 어노테이션을 직접 읽으므로 test는 필요 없다  |
| `--tip-box-size`    | 32           | 팁 박스 한 변의 길이 (원본 프레임 px)                                   |
| `--frame-stride`    | 5            | train 프레임을 N장마다 1장만 쓴다. 영상 프레임은 서로 거의 같다         |
| `--val-frames`      | 2000         | 에포크마다 평가할 val 프레임 수 상한, 스플릿 전체에 고르게 분포 (0이면 전체) |
| `--force`           |              | 이미 같은 조건으로 준비돼 있어도 다시 만든다                            |

만들어지는 것:

```text
data/yolo/<dataset>/
├── dataset.yaml         # train-model이 Ultralytics에 넘기는 파일
├── prepare-status.json  # 팁 박스 크기, 프레임 수, 어떻게 잘랐는지
├── images/<split>       # <repo>/data/dataset/<dataset>/images/<split>로 가는 심볼릭 링크
├── labels/<split>/*.txt # 목록에 오른 프레임마다 YOLO 라벨 파일 하나
└── <split>.txt          # 이 데이터셋이 쓰는 프레임 목록
```

이미지는 복사가 아니라 심볼릭 링크다. 데이터셋마다 18만 장이 넘는 PNG를 복제할 이유가 없고, Ultralytics가
이미지 경로의 `/images/`를 `/labels/`로 바꿔 라벨을 찾기 때문에 링크만 준비된 트리 안에
있으면 된다. 라벨은 목록에 오른 프레임에만 만들어지므로 `--frame-stride`는 준비 비용도 함께
줄인다. 같은 인수로 다시 실행하면 아무 일도 하지 않고, 조건이 하나라도 다르면 그 스플릿을
처음부터 다시 만든다. 준비된 데이터셋이 `prepare-status.json`의 내용과 항상 일치해야 하기
때문이다.

`--frame-stride`와 `--val-frames`가 학습이 아니라 준비 단계의 인수인 것은 이 구조 때문이다.
준비된 데이터셋 하나가 곧 학습 세트 하나의 정의이며, 비교 실험을 하려면 아래처럼 준비를 다시
한다 (`--force`).

### 2. 학습

```bash
./baseline/yolo26/run train-model --dataset cholec80
```

| 인수                | 기본값 | 설명                                                                       |
| ------------------- | ------ | -------------------------------------------------------------------------- |
| `--dataset`         | (필수) | 먼저 `prepare-dataset`을 돌려 둔 데이터셋                                  |
| `--scale`           | `s`    | YOLO26 깊이·폭 스케일 (`n`/`s`/`m`/`l`/`x`). `s`가 YOLOv8s 베이스라인과 같은 크기 |
| `--epochs`          | 150    |                                                                            |
| `--batch-size`      | 16     |                                                                            |
| `--image-size`      | 640    |                                                                            |
| `--optimizer`       | `auto` | 이 규모에서는 MuSGD로 해석된다. `SGD`, `AdamW` 등을 직접 지정할 수 있다    |
| `--lr`              | 0.01   | `--optimizer auto`일 때는 무시된다 (Ultralytics가 학습률까지 정한다)       |
| `--patience`        | 0      | 조기 종료까지 기다릴 에포크 수 (0이면 조기 종료하지 않는다)                |
| `--workers`         | 8      |                                                                            |
| `--device`          |        | 예: `cuda:1`                                                               |
| `--no-pretrained`   |        | COCO 가중치 대신 무작위 초기화에서 시작                                    |
| `--no-amp`          |        | 혼합 정밀도 끄기 (Ultralytics의 AMP 점검 다운로드도 함께 건너뛴다)         |
| `--no-resume`       |        | 기존 `model-last.pt`를 무시하고 처음부터                                   |

기본은 `yolo26<scale>.pt`(COCO 사전학습)에서 시작하는 파인튜닝이다. `--no-pretrained`를 주면
무작위 초기화에서 시작하는데, 이는 [yolov8sclone](../yolov8sclone/)·[cladnet](../cladnet/)과
같은 조건이다. **세 베이스라인을 같은 조건으로 비교하려면 이 옵션이 필요하다.**

`model-last.pt`가 있으면 **기본 동작이 재개**이며 optimizer·스케줄러·EMA 상태까지 복원된다.
이미 끝난 학습을 같은 명령으로 다시 실행하면 아무 일도 하지 않고 그 사실을 알린다. 재개할
때 Ultralytics는 에포크 수도 체크포인트에서 읽으므로, 더 오래 학습하려면 `--no-resume`으로
새로 시작해야 한다.

체크포인트는 매 에포크 `data/model/<dataset>/`으로 복사된다. 중간에 끊겨도 그 시점까지의
`model.pt`와 완전한 `metric.csv`가 남는다.

### 3. 평가

```bash
./baseline/yolo26/run eval-model --dataset cholec80
```

두 종류의 수치를 함께 낸다.

- **탐지 지표**: `tool`·`tip` 각각의 AP@0.5, AP@0.5:0.95, precision, recall
- **팁 지표**: miss rate, Hit-rate @ 10/20/50 px, 오차 거리 중앙값·평균·P90.
  예측된 `tip` 박스의 중심을 팁 좌표로 삼고, 루트 프로젝트 `scripts/eval-model.py`와
  **같은 매칭 규칙**(최근접 매칭 + Hungarian 1:1 매칭)을 쓴다.

두 지표 모두 Ultralytics의 validator가 아니라 이 서브 프로젝트가 직접 계산한다. 형제
베이스라인들이 보고하는 정의와 한 글자도 다르지 않아야 비교가 성립하기 때문이다. 프레임과
어노테이션도 준비된 YOLO 데이터셋이 아니라 `data/dataset/`에서 직접 읽으므로, 평가 결과는
학습을 어떻게 준비했는지와 무관하다.

| 인수             | 기본값 | 설명                                                              |
| ---------------- | ------ | ----------------------------------------------------------------- |
| `--dataset`      | (필수) |                                                                   |
| `--split`        | `test` |                                                                   |
| `--conf`         | 0.25   | 팁 지표를 계산할 때의 신뢰도 임계값                               |
| `--map-conf`     | 0.001  | AP 곡선용 하한. 모델은 프레임당 한 번만 돌린다                    |
| `--max-det`      | 300    | 프레임당 박스 수 상한. NMS가 없는 대신 이 값이 있다               |
| `--frame-stride` | 1      | N프레임마다 1장만 평가                                            |
| `--limit`        |        | 평가할 프레임 수 상한                                             |
| `--model`        |        | 체크포인트 경로 (기본 `data/model/<dataset>/model.pt`)            |

거리는 letterbox된 640 × 640이 아니라 **원본 프레임 좌표계(736 × 480)** 에서 잰다.
결과는 `data/results/<dataset>/<split>/`에 `summary.json`과 `per_tip.csv`로 저장된다.

### 4. 데모 GUI

```bash
./baseline/yolo26/run demo --dataset cholec80
```

영상을 프레임 단위로 처리하며 `tool` 박스와 `tip` 박스(중심에 십자 마커)를 그린다.
조작은 [yolov8s 데모](../yolov8s/README.md)와 같지만 **IoU 슬라이더가 없다.** YOLO26은
end-to-end 헤드라 NMS를 돌리지 않으므로, 있어도 아무 일도 하지 않는 조작이 되기 때문이다.

`--dataset`을 주면 `data/model/<dataset>/model.pt`를 찾아 열고(생략하면 사전순 첫 번째),
`--weights`로 임의의 체크포인트를 직접 지정할 수도 있다.

### 5. 수치 요약 문서 생성

```bash
./baseline/yolo26/run generate-summary
```

`data/model/`·`data/results/`의 파일만 읽어 `docs/summary-results.md`를 만든다. 학습 곡선,
탐지·팁 지표 외에 `summary.json`에 없는 오차 거리 분포·프레임 단위 탐지 실패율·팁 수별
성능·세션별 편차를 `per_tip.csv`에서 다시 계산해 포함한다. 재학습·재평가 뒤에 다시 실행한다.

## 구현 메모

- **입력 채널 순서.** Ultralytics는 넘겨받은 numpy 배열을 BGR로 간주해 내부에서 뒤집는다.
  이 저장소는 어디서나 RGB를 쓰므로 `common/inference.py`가 넘기기 직전에 한 번 뒤집는다.
  이 처리가 없으면 학습 때와 추론 때의 채널 순서가 어긋난다. 같은 프레임에 대해
  `predict(파일 경로)`와 `detect(RGB 배열)`의 박스 좌표가 정확히 일치하는 것으로 확인했다.
- **에포크 미러링.** `on_fit_epoch_end` 콜백에서 체크포인트를 복사하고 `metric.csv`를 쓴다.
  이 콜백은 학습이 끝난 뒤 best.pt를 다시 검증할 때 한 번 더 불리는데, 그 호출은 학습 손실이
  없고 에포크 번호도 학습 길이를 넘으므로 건너뛴다.
- **다운로드 위치.** Ultralytics의 AMP 점검은 전역 설정에 적힌 위치로 `yolo26n.pt`를
  내려받는다. 이 머신에서는 그 위치가 저장소 루트였다. 학습 스크립트가 프로세스 안에서만
  그 경로를 `data/pretrained/`로 돌려놓으므로 사용자의 설정 파일은 건드리지 않는다.
- **AGPL-3.0.** `ultralytics` 패키지와 `yolo26*.pt` 가중치는 AGPL-3.0으로 배포된다.
  재구현 베이스라인 두 개와 달리 이 서브 프로젝트의 산출물은 그 조건을 따른다.

## 한계

- **아직 학습 결과가 없다.** 이 서브 프로젝트는 스크립트와 파이프라인까지다.
  `docs/summary-results.md`와 실험 보고서는 실제 학습을 돌린 뒤에 만들어진다.
- **재구현이 아니다.** 모델·손실·학습 루프가 모두 `ultralytics`에 있으므로, 형제
  베이스라인들이 확보한 "의존성 없이 돌아간다"는 성질은 여기에 없다.
- **기본 설정은 공정 비교가 아니다.** 기본값인 COCO 사전학습은 스크래치로 학습한
  yolov8sclone·cladnet보다 유리하다. 세 모델을 같은 조건에서 비교하려면 `--no-pretrained`를
  쓰고, 그 사실을 수치와 함께 밝혀야 한다.
- **GUI는 이 환경에서 실행 검증하지 못했다.** 디스플레이가 없어 tkinter 창을 띄우지 못했다.
  창 구성을 뺀 나머지(체크포인트 로드, 소스 탐색, 추론, 오버레이 그리기)는 확인했다.

## 참고

- YOLO26: <https://docs.ultralytics.com/models/yolo26>
- Ultralytics: <https://github.com/ultralytics/ultralytics>
- 형제 베이스라인 비교 보고서: [docs/baseline-report.md](../../docs/baseline-report.md)
