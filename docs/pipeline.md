# `scripts/pipeline.py`

`scripts/pipeline.py`는 tooltip-annotator의 전체 작업 흐름(progressive 변환 → 데이터셋 생성 → 모델 다운로드 → segmentation → annotation → 수동 편집)을 하나의 GUI 창에서 순서대로 실행하고, 각 단계의 진행 상태와 실행 로그를 함께 보여주는 통합 파이프라인 GUI다.

각 단계는 `bin/` 디렉터리의 런처 스크립트를 호출해 실행되며, 런처는 내부적으로 `uv run python -m scripts.<단계>`를 수행한다.

## 사용자 문서

### 목적

여러 스크립트를 터미널에서 하나씩 실행하는 대신, 버튼 클릭만으로 단계를 실행하고 진행 상황을 한눈에 보기 위한 GUI다. 주요 기능은 다음과 같다.

- 상단 `Dataset` 드롭다운으로 `data/dataset-src`/`data/dataset` 아래 발견된 데이터셋 중 하나를 선택
- 6개 파이프라인 단계를 버튼으로 실행(선택된 데이터셋 기준)
- 각 단계의 완료/진행/대기 상태를 아이콘과 색으로 표시
- 실행 중인 단계의 표준 출력/오류를 실시간 로그 창에 표시
- 실행 중인 단계 중단(Stop), 상태 새로고침(Refresh), 로그 비우기(Clear Log)

### 실행 전 요구 사항

- Python `3.12` 이상
- 프로젝트 의존성 설치
- `tkinter` (GUI)
- 각 단계 스크립트가 요구하는 의존성(예: `ffmpeg`, `opencv-python`, `torch`, `monai` 등)
- `uv` (런처 스크립트가 `uv run`을 사용한다)

각 단계 자체의 요구 사항은 해당 단계 문서를 참고한다.

### 기본 실행 방법

프로젝트 루트에서 실행한다.

```bash
python -m scripts.pipeline
```

`uv`를 사용 중이면 다음처럼 실행할 수 있다.

```bash
uv run python -m scripts.pipeline
```

런처 스크립트로도 실행할 수 있다.

```bash
bin/pipeline
```

이 GUI는 별도의 CLI 옵션을 받지 않는다. 모든 경로는 프로젝트 루트 기준으로 고정되어 있다(`data/`, `temp/`, `bin/`).

### 화면 구성

창은 위에서 아래로 네 영역으로 구성된다.

1. **Dataset**: 데이터셋 선택 드롭다운. `data/dataset-src`와 `data/dataset` 하위 폴더 이름의 합집합이 정렬되어 표시된다. 하나도 없으면 안내 문구가 옆에 뜨고 `download_model`을 제외한 모든 단계의 `Run`/`Launch` 버튼이 비활성화된다.
2. **Pipeline Steps**: 6개 단계 목록. 각 단계는 상태 아이콘, 단계 이름, 진행 상세, 설명, 그리고 `Run`(또는 6단계는 `Launch`) 버튼으로 구성된다.
3. **컨트롤 바**: `Refresh`, `Stop`, `Clear Log` 버튼.
4. **Log**: 실행 중인 단계의 출력이 실시간으로 표시되는 검은 배경의 로그 창.

### 파이프라인 단계

| 단계 | 이름                  | 실행 스크립트           | 동작 방식        |
| ---- | --------------------- | ----------------------- | ---------------- |
| 1    | Generate Progressive  | `generate_progressive`  | Run (로그 캡처)  |
| 2    | Generate Dataset      | `generate_dataset`      | Run (로그 캡처)  |
| 3    | Download Model        | `download_model`        | Run (로그 캡처)  |
| 4    | Generate Segmentation | `generate_segmentation` | Run (로그 캡처)  |
| 5    | Generate Annotation   | `generate_annotation`   | Run (로그 캡처)  |
| 6    | Annotation Editor     | `annotation_editor`     | Launch (별도 창) |

1~5단계는 GUI 내부에서 자식 프로세스로 실행되고 출력이 로그 창으로 흘러온다. 6단계는 GUI를 막지 않는 **별도 창**으로 편집기를 띄우며, 편집기 출력은 로그로 캡처하지 않는다.

`download_model`(3단계)을 제외한 모든 단계는 선택된 데이터셋 이름을 `--dataset <name>` 인자로 붙여 실행된다. 모델은 데이터셋과 무관한 공용 자원이라 `download_model`은 인자 없이 실행된다.

### 상태 아이콘의 의미

각 단계의 상태는 선택된 데이터셋 기준으로 `data/`, `temp/` 아래 파일 개수를 세어 자동 판정한다(3단계 `Download Model`은 데이터셋과 무관).

| 상태    | 유니코드 아이콘 | ASCII 대체 | 색     | 의미                                  |
| ------- | --------------- | ---------- | ------ | ------------------------------------- |
| done    | ✓               | `+`        | 초록   | 단계가 완료된 것으로 판정됨           |
| partial | ~               | `~`        | 주황   | 일부만 처리됨 (입력 대비 출력이 부족) |
| pending | ○               | `.`        | 회색   | 아직 출력이 없음                      |
| waiting | –               | `-`        | 연회색 | 선행 입력이 없어 실행할 수 없음       |
| ready   | ✓               | `+`        | 파랑   | (6단계) 편집할 annotation이 준비됨    |
| running | ▶               | `>`        | 파랑   | 현재 실행 중                          |

> 시스템에 적절한 유니코드 심볼 폰트가 없으면 자동으로 ASCII 아이콘으로 대체된다.

상태 옆의 상세 텍스트는 단계별로 다르다(예: `train:120  val:40  test:40`, `12/12 videos`, `model.pt` 등).

### 단계별 상태 판정 기준

데이터셋이 선택되지 않은 상태(목록이 비어 있음)에서는 `download_model`을 제외한 모든 단계가 `waiting`/"no dataset selected"로 표시되고 버튼이 비활성화된다.

- **1. Generate Progressive**
  - 입력: `data/dataset-src/<dataset>`의 비디오 수, 출력: `data/dataset/<dataset>/progressive`의 비디오 수.
  - 입력이 0이면 `waiting`, 출력 ≥ 입력이면 `done`, 일부면 `partial`, 0이면 `pending`.
- **2. Generate Dataset**
  - `data/dataset/<dataset>/images/{train,val,test}`에 이미지가 하나라도 있으면 `done`, 없으면 `pending`.
- **3. Download Model**
  - `temp/models/model.pt`가 있으면 `done`, 없으면 `pending`(데이터셋과 무관).
- **4. Generate Segmentation**
  - 입력: `data/dataset/<dataset>/images/...`의 이미지 수, 출력: `data/dataset/<dataset>/segmentation/...`의 mask 수.
  - mask가 0이면 `pending`, mask ≥ 이미지면 `done`, 그 외 `partial`.
- **5. Generate Annotation**
  - 입력: `data/dataset/<dataset>/segmentation/...`의 mask 수, 출력: `data/dataset/<dataset>/annotation/...`의 JSON 수.
  - JSON이 0이면 `pending`, JSON ≥ mask면 `done`, 그 외 `partial`.
- **6. Annotation Editor**
  - `data/dataset/<dataset>/annotation/...`에 JSON이 하나라도 있으면 `ready`, 없으면 `waiting`(5단계 먼저 실행).

### 실행/중단 동작

- 한 번에 하나의 단계만 실행된다. 어떤 단계를 실행하면 다른 단계 버튼은 모두 비활성화되고, 끝나면 다시 활성화된다.
- 실행 중에는 `Stop` 버튼이 활성화된다. `Stop`은 현재 실행 중인 자식 프로세스에 종료 신호(`terminate`)를 보낸다.
- 단계가 끝나면 로그 끝에 `✓ done` 또는 `✗ error (exit N)`가 표시되고, 상태가 자동으로 새로고침된다.
- 6단계(편집기)는 별도 창으로 뜨며, GUI는 곧바로 다시 조작 가능한 상태로 돌아온다. 편집기는 `Stop` 대상이 아니다(편집기 창에서 직접 닫는다).

### 컨트롤 버튼

- **Refresh**: 파일 시스템을 다시 스캔해 각 단계 상태를 갱신한다. 실행 중에는 동작하지 않는다.
- **Stop**: 실행 중인 단계를 중단한다.
- **Clear Log**: 로그 창을 비운다.

### `--dataset` 외에는 기본 옵션으로만 실행된다는 점 (중요)

이 GUI는 각 단계를 선택된 데이터셋의 `--dataset <name>`만 붙여 실행한다(`download_model`은 그마저도 없이 실행). 그 외에는 각 스크립트의 **기본 옵션**으로만 동작하며, `--overwrite`나 `--device`, 커스텀 `--input`/`--output` 같은 옵션을 GUI에서 지정할 수 없다.

특히 `generate_segmentation`과 `generate_annotation`은 `--overwrite` 없이 실행되므로, **이미 결과 파일이 있으면 건너뛴다**. 즉 이 GUI로는 기존 mask/annotation을 다시 생성(덮어쓰기)할 수 없다. 결과를 강제로 재생성하려면 터미널에서 직접 옵션을 지정해 실행해야 한다.

```bash
uv run python -m scripts.generate_segmentation --dataset erop --overwrite
uv run python -m scripts.generate_annotation --dataset erop --overwrite
```

마찬가지로 디바이스 지정(`--device`)이나 경로 변경 등도 GUI에서는 불가능하며, CLI 실행이 필요하다.

### 문제 해결

#### GUI가 뜨지 않거나 `tkinter` 관련 에러가 나는 경우

`tkinter`가 설치되어 있어야 한다. 배포판에 따라 별도 패키지(`python3-tk` 등) 설치가 필요할 수 있다.

#### `Run`을 눌렀는데 `launch failed`가 로그에 뜨는 경우

`bin/<단계>` 런처가 없거나 실행 권한이 없을 수 있다. `bin/` 디렉터리의 스크립트에 실행 권한이 있는지, `uv`가 설치되어 있는지 확인한다.

#### 상태가 갱신되지 않는 경우

터미널에서 직접 단계를 돌렸거나 파일을 바꾼 경우, `Refresh`를 눌러 다시 스캔해야 반영된다. 단계 실행 중에는 `Refresh`가 동작하지 않는다.

#### 진행 상세에서 이미지 개수가 실제보다 적게 표시되는 경우

데이터셋/이미지 개수 집계는 소문자 확장자(`.png`, `.jpg`, `.jpeg`)만 센다. 대문자 확장자 이미지는 집계에서 빠질 수 있다(아래 "현재 설계 제약" 참고). 단계 자체의 실행에는 영향이 없다.

## 개발 문서

### 파일 구조

스크립트는 폰트 선택 헬퍼, 진행 판정 로직, GUI 클래스로 구성된다.

- `_pick_ui_font(size, weight)`
  - 사용 가능한 UI 폰트를 우선순위대로 선택한다.
- `_pick_symbol_font(size)`
  - 유니코드 심볼 커버리지가 넓은 폰트를 선택한다. 없으면 `None`.
- `_pick_log_font(size)`
  - 로그용 고정폭 폰트를 선택한다.
- `_count(directory, extensions)`
  - 디렉터리에서 지정 확장자 파일 수를 센다.
- `STEPS`
  - 6개 단계의 메타데이터(id, label, desc, script, editor) 리스트.
- `_step_progress(step_id, dataset)`
  - 선택된 데이터셋 기준으로 단계별 `(status, detail)`을 계산한다(`download_model`은 `dataset`을 무시).
- `_STATUS_UNICODE` / `_STATUS_ASCII`
  - 상태별 (아이콘, 색) 매핑. 심볼 폰트 유무에 따라 선택된다.
- `PipelineApp`
  - GUI 본체. 아래 메서드로 구성된다.
    - `__init__(root)`: 폰트/상태 스타일/큐/`dataset_var` 초기화, UI 빌드, 첫 새로고침, 폴링 시작.
    - `_build_ui()`: 데이터셋 드롭다운, 단계 행, 컨트롤 바, 로그 창을 생성한다.
    - `_reload_datasets()`: `tooltip.dataset_paths.list_datasets()`로 드롭다운 값을 다시 채우고, 현재 선택이 목록에 없으면 첫 번째 항목으로 되돌린다.
    - `_on_dataset_changed(event)`: 드롭다운 선택이 바뀌면 `_refresh()`를 호출한다.
    - `_refresh()`: 데이터셋 목록을 다시 불러온 뒤 각 단계 상태를 다시 계산해 표시한다. 데이터셋이 없으면 `download_model` 외 모든 단계를 `waiting`으로 표시하고 버튼을 비활성화한다.
    - `_set_running(idx)`: 실행 중 단계를 표시하고 버튼 상태를 토글한다.
    - `_run(idx)`: 해당 단계를 선택된 데이터셋으로 실행한다(`download_model` 제외 `--dataset <name>` 전달, 편집기는 별도 창, 나머지는 작업 스레드).
    - `_stop()`: 실행 중 프로세스를 종료한다.
    - `_poll()`: 로그 큐를 비우고 100ms마다 자신을 다시 예약한다.
    - `_append_log(text)` / `_clear_log()`: 로그 출력/초기화.
- `main()`
  - Tk 루트를 만들고 `PipelineApp`을 띄운 뒤 메인 루프를 돈다.

### 핵심 구현 세부 사항

#### 1. 경로 기준

모듈 상단에서 파일 위치 기준으로 프로젝트 루트를 고정한다.

```python
ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
DATA = ROOT / "data"
TEMP = ROOT / "temp"
```

따라서 작업 디렉터리와 무관하게 동일한 경로를 본다. 자식 프로세스도 `cwd=str(ROOT)`로 실행된다.

#### 2. 단계 실행 방식 (Run vs Launch)

`_run(idx)`는 먼저 `STEPS[idx]["id"] != "download_model"`이면 데이터셋이 선택되어 있는지 확인하고(없으면 로그에 에러만 남기고 종료), 실행할 인자 목록에 `--dataset <name>`을 붙인다. 이후 `STEPS[idx]["editor"]` 값으로 분기한다.

- **편집기(editor=True)**: `subprocess.Popen(args, cwd=ROOT)`로 별도 창을 띄우고, 출력은 캡처하지 않는다. 곧바로 큐에 `None`(완료 신호)을 넣어 GUI 잠금을 해제한다.
- **일반 단계(editor=False)**: 데몬 스레드에서 `subprocess.Popen(args, ...)`을 `stdout=PIPE, stderr=STDOUT, text=True, bufsize=1`로 실행하고, 출력 라인을 하나씩 `_log_queue`에 넣는다. 프로세스가 끝나면 종료 코드에 따라 `✓ done`/`✗ error` 메시지와 완료 신호(`None`)를 큐에 넣는다.

자식 프로세스는 `PYTHONUNBUFFERED=1` 환경에서 실행되어 출력 버퍼링을 줄인다.

#### 3. 스레드–GUI 통신 (큐 + 폴링)

Tkinter는 메인 스레드에서만 위젯을 안전하게 갱신할 수 있으므로, 작업 스레드는 위젯을 직접 건드리지 않고 `queue.Queue`에만 메시지를 넣는다.

`_poll()`은 메인 스레드에서 100ms마다 큐를 비우며,

- 문자열이면 로그에 추가하고,
- `None`이면 "단계 완료"로 보고 실행 상태를 해제(`_set_running(None)`)한 뒤 상태를 새로고침한다.

이 구조로 멀티스레드 환경에서도 GUI 갱신이 메인 스레드에서만 일어난다.

#### 4. 진행 판정 (`_step_progress`)

각 단계 상태는 별도 메타 파일 없이, 선택된 데이터셋 아래 **출력 파일 개수**로 추정한다. 입력 대비 출력 비율로 `done`/`partial`/`pending`을 구분하고, 선행 입력이 없으면 `waiting`을 반환한다. 이는 가볍고 단순하지만, "건너뛴 것"과 "실제로 처리한 것"을 구분하지는 못한다(파일 존재 여부만 본다).

#### 4.5. 데이터셋 드롭다운 (`_reload_datasets`)

드롭다운 값은 `tooltip.dataset_paths.list_datasets()`(`data/dataset-src`와 `data/dataset` 하위 폴더 이름의 합집합, 정렬됨)로 매 `_refresh()`마다 다시 채워진다. 즉 GUI를 실행한 채로 새 데이터셋 심볼릭 링크를 추가해도 `Refresh` 버튼이나 다음 자동 새로고침 시점에 드롭다운에 나타난다. 현재 선택값이 새 목록에 없으면(처음 실행 시 등) 정렬된 목록의 첫 번째 항목으로 자동 선택된다. 목록이 비어 있으면 선택값을 빈 문자열로 비우고 안내 문구를 표시한다.

#### 5. 로그 정리 (ANSI 제거)

`_append_log()`는 정규식으로 ANSI 이스케이프 시퀀스와 캐리지 리턴을 제거한다.

```python
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')
```

`tqdm` 같은 진행바가 만들어내는 제어 문자를 정리해 로그 창이 깨지지 않게 한다. 다만 `\r` 기반 in-place 갱신은 제거되므로, 진행바는 여러 줄로 누적되어 보일 수 있다.

#### 6. 폰트 선택과 상태 스타일 대체

`_pick_*` 헬퍼는 사용 가능한 폰트 패밀리 목록에서 우선순위대로 폰트를 고른다. 심볼 폰트가 없으면 `_STATUS_ASCII`로 대체해, 유니코드 아이콘이 깨지는 환경에서도 상태를 읽을 수 있게 한다.

### 예외와 종료 방식

- 단계 실행 중 발생하는 예외는 작업 스레드/런치 분기에서 잡아 `✗ launch failed: ...` 형태로 로그에 표시하고, 완료 신호를 보내 GUI를 정상 상태로 되돌린다. 즉 한 단계가 실패해도 GUI 자체는 죽지 않는다.
- `Stop`은 `terminate()`만 호출하며 강제 종료(`kill`)나 타임아웃 처리는 하지 않는다.

### 현재 설계 제약

- `--dataset` 외에는 각 단계를 기본 옵션으로만 실행하므로, `--overwrite`/`--device`/경로 변경 등 CLI 옵션을 GUI에서 줄 수 없다.
- 진행 판정이 파일 개수 기반이라, 건너뛴 결과와 새로 생성한 결과를 구분하지 못한다.
- 데이터셋/이미지 개수 집계(`IMAGE_EXTS`)는 소문자 `.png/.jpg/.jpeg`만 센다. 비디오 집계(`VIDEO_EXTS`)는 대소문자 변형을 포함하지만, 이미지/마스크/JSON은 소문자만 본다.
- 단계 간 의존성을 강제하지 않는다(선행 단계 없이도 버튼은 눌린다. 실패는 로그로만 표시).
- `Stop`은 자식 프로세스에 부드러운 종료만 시도하며, 응답하지 않는 프로세스를 강제로 죽이지는 않는다.
- 편집기(6단계)는 별도 창이라 로그 캡처와 `Stop` 대상에서 제외된다.
- 단계 실행 중에도 데이터셋 드롭다운 자체는 계속 조작할 수 있다(실행 중 버튼은 비활성화되지만 드롭다운은 아니다). 실행 중에 다른 데이터셋으로 바꾸면 실제로 실행 중인 프로세스는 원래 선택했던 데이터셋 그대로 계속 동작하지만, 완료 후 상태 새로고침은 바뀐 선택값 기준으로 표시되어 잠시 혼동을 줄 수 있다.

### 확장 권장 사항

1. 단계별 옵션(`--overwrite`, `--device`, 경로 등)을 GUI에서 설정할 수 있는 입력 추가
2. 진행 판정을 파일 개수 외에 매니페스트/타임스탬프 기반으로 보강
3. 이미지/마스크/JSON 집계도 대문자 확장자를 포함하도록 통일
4. 선행 단계 미완료 시 실행 버튼 비활성화 등 의존성 가드 추가
5. `Stop` 시 타임아웃 후 `kill` 폴백 추가
6. "전체 실행(Run All)"처럼 단계를 연속 실행하는 기능 추가
7. 실행 중 데이터셋 드롭다운을 잠가 위 혼동을 원천 차단

### 다른 코드에서 재사용할 때

이 모듈은 GUI 앱이라 함수 단위 재사용보다는 전체 실행이 기본이다. 다만 진행 판정 로직(`_step_progress`, `_count`)은 GUI와 분리되어 있어, 파이프라인 상태를 비-GUI 환경에서 조회하는 용도로 활용할 수 있다.

```python
from scripts.pipeline import _step_progress
from tooltip.dataset_paths import list_datasets

dataset = list_datasets()[0]
for step_id in ("generate_dataset", "generate_segmentation", "generate_annotation"):
    status, detail = _step_progress(step_id, dataset)
    print(step_id, status, detail)
```

단, `_count`/`_step_progress`는 모듈 상단의 `DATA`/`TEMP` 상수(프로젝트 루트 기준)를 직접 참조하므로, 다른 데이터 경로에 적용하려면 경로 처리를 인자로 분리하는 리팩터링이 필요하다.
