# `scripts/generate_dataset.py`

`scripts/generate_dataset.py`는 동영상 파일에서 일정 간격으로 프레임을 추출하고, 각 프레임을 `train`, `val`, `test` 세 분할로 나누어 PNG 이미지 데이터셋을 만드는 스크립트다.

`--dataset <name>` 옵션으로 데이터셋 이름만 주면 입력/출력 경로가 자동 계산된다. 입력은 `data/dataset/<name>/progressive`가 존재하고 비어있지 않으면 그것을, 아니면 원본 `data/dataset-src/<name>`을 사용한다. 출력은 `data/dataset/<name>/images`다. 출력 이미지는 지정한 해상도에 맞춰 letterbox 방식으로 리사이즈된다.

## 사용자 문서

### 목적

전처리된 동영상에서 학습용 이미지 프레임을 추출해 `train/val/test` 구조의 데이터셋을 생성한다.

이 프로젝트의 기본 작업 순서는 보통 다음과 같다.

1. `scripts/generate_progressive.py --dataset <name>`로 `data/dataset/<name>/progressive` 준비(원본이 이미 progressive 형식이면 생략 가능)
2. `scripts/generate_dataset.py --dataset <name>`로 프레임 추출 및 데이터셋 생성

### 실행 전 요구 사항

- Python `3.12` 이상
- 프로젝트 의존성 설치
- `opencv-python` 설치
- `tqdm` 설치
- 입력 동영상이 들어 있는 디렉터리

`pyproject.toml` 기준 관련 의존성은 다음과 같다.

- `opencv-python`
- `tqdm`

### 기본 실행 방법

프로젝트 루트에서 실행한다.

```bash
python scripts/generate_dataset.py --dataset erop
```

`uv`를 사용 중이면 다음처럼 실행할 수 있다.

```bash
uv run python scripts/generate_dataset.py --dataset erop
```

`--dataset`을 주면 아래 경로를 사용한다.

- 입력 디렉터리: `data/dataset/<dataset>/progressive`가 존재하고 비어있지 않으면 그것, 아니면 `data/dataset-src/<dataset>`
- 출력 디렉터리: `data/dataset/<dataset>/images`
- 프레임 간격: `10`
- 출력 크기: `736x480`
- 분할 비율: `train=60`, `val=20`, `test=20`
- 랜덤 시드: `0`

### CLI 옵션

- `--dataset`
  - 데이터셋 이름(예: `erop`, `cholec80`). `--input`/`--output`을 명시하지 않았을 때 기본 경로를 계산하는 데 쓰인다.
- `--input`
  - 입력 동영상 디렉터리. 기본값은 위 "`--dataset`을 주면 아래 경로를 사용한다" 참고. 명시하면 `--dataset` 기본값보다 항상 우선한다.
- `--output`
  - 출력 데이터셋 디렉터리. 기본값은 `data/dataset/<dataset>/images`. 명시하면 `--dataset` 기본값보다 항상 우선한다.
- `--frame`
  - 몇 프레임마다 한 장을 추출할지 정하는 간격. 기본값은 `10`.
- `--width`
  - 출력 이미지 너비. 기본값은 `736`.
- `--height`
  - 출력 이미지 높이. 기본값은 `480`.
- `--train`
  - 학습 세트 비율. 기본값은 `60`.
- `--val`
  - 검증 세트 비율. 기본값은 `20`.
- `--test`
  - 테스트 세트 비율. 기본값은 `20`.
- `--seed`
  - 비디오별 deterministic split에 사용하는 시드. 기본값은 `0`.
- `--verify`
  - 재개 시 기존 이미지의 유효성 검사 방식. `fast`(기본값)는 PNG 헤더/트레일러만 빠르게 확인하고, `full`은 이미지를 완전히 디코딩해 내부 손상까지 검사한다.

### 실행 예시

`--dataset`으로 기본 경로를 사용해 데이터셋을 생성하는 예시:

```bash
python scripts/generate_dataset.py --dataset cholec80
```

프레임을 더 촘촘히 추출하고 출력 해상도를 바꾸는 예시:

```bash
python scripts/generate_dataset.py --dataset cholec80 --frame 5 --width 1280 --height 720
```

입력과 출력 경로를 직접 지정하는 예시(`--dataset` 기본값을 무시):

```bash
python scripts/generate_dataset.py --input ./data/dataset/erop/progressive --output ./data/dataset_v2/images
```

분할 비율을 바꾸는 예시:

```bash
python scripts/generate_dataset.py --dataset cholec80 --train 70 --val 15 --test 15
```

### 입력 대상 파일

입력 디렉터리 바로 아래에서 다음 확장자의 파일만 읽는다.

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.mpeg`
- `.mpg`
- `.m4v`
- `.wmv`

확장자 비교는 소문자 기준으로 수행하므로 대문자 확장자도 처리된다. 하위 디렉터리는 재귀 탐색하지 않는다.

### 출력 구조

실행 시 출력 디렉터리 아래에 다음 하위 폴더가 생성된다.

```text
./data/dataset/<dataset>/
└── images/
    ├── train/
    ├── val/
    └── test/
```

이미지 파일명은 다음 형식이다.

```text
{원본동영상이름}_{프레임번호8자리}.png
```

예시:

```text
sample01_00000030.png
```

### 동작 방식

1. CLI 인자를 파싱한다.
2. `--dataset`으로 `--input`/`--output` 기본값을 해석한다.
3. OpenCV import 여부와 인자 값의 유효성을 검사한다.
4. 출력 디렉터리 아래 `train`, `val`, `test` 폴더를 생성한다. `--dataset`을 쓰면 실제 생성 위치는 `data/dataset/<dataset>/images/{train,val,test}`다.
5. 입력 디렉터리의 비디오 파일 목록을 정렬해 수집한다.
6. 각 비디오에 대해 `0, frame_step, 2*frame_step, ...` 위치의 프레임 번호 목록을 만든다.
7. 비디오 경로와 시드를 기반으로 난수 생성기를 만들고, 프레임 번호를 `train/val/test`로 나눈다.
8. 비디오를 처음부터 끝까지 읽으면서 선택된 프레임만 저장한다.
9. 각 프레임은 원본 종횡비를 유지한 채 letterbox 방식으로 리사이즈된다.
10. 모든 비디오 처리가 끝나면 split별 저장 개수를 출력한다.

### 분할 비율의 의미

`--train 60 --val 20 --test 20`은 퍼센트 전용 옵션이 아니라 상대 비율이다. 따라서 아래 두 설정은 같은 의미다.

```bash
--train 60 --val 20 --test 20
--train 3 --val 1 --test 1
```

총 추출 프레임 수가 비율로 정확히 나누어떨어지지 않으면, 소수점 몫이 큰 split부터 1장씩 추가 배정한다.

### deterministic split

같은 비디오 경로와 같은 `--seed`를 사용하면 split 배정 결과는 재현 가능하다. 다만 비디오 파일의 절대 경로가 바뀌면 seed material도 바뀌므로 동일 파일이라도 split 결과가 달라질 수 있다.

### 이미지 리사이즈 방식

출력 이미지는 원본 비율을 유지하면서 지정한 `width x height` 안에 맞게 축소 또는 확대한 뒤, 남는 영역을 검은색 패딩으로 채운다.

즉, 원본 비율을 강제로 찌그러뜨리지 않는다.

### 종료 및 출력

- 입력 디렉터리에 비디오가 없으면 `No video files found ...`를 출력하고 종료 코드 `0`으로 끝난다.
- 정상 완료 시 `Extraction complete.`와 split별 저장 개수를 출력하고 종료 코드 `0`을 반환한다.
- OpenCV import 실패, 잘못된 인자, 입력 경로 오류, 비디오 열기 실패, 이미지 저장 실패 등은 예외로 중단된다.

### 문제 해결

#### `opencv-python is required` 에러가 나는 경우

OpenCV가 설치되지 않은 상태다. 프로젝트 의존성을 설치한 뒤 다시 실행하면 된다.

#### `Input directory does not exist` 에러가 나는 경우

`--input` 경로가 실제로 존재하는지 확인해야 한다. `--dataset`만 썼다면 `data/dataset/<dataset>/progressive` 또는 `data/dataset-src/<dataset>`가 준비되어 있어야 한다(둘 다 없으면 이 에러가 난다).

#### `Either --dataset or --input must be provided.` 에러가 나는 경우

`--dataset`도 `--input`도 주지 않은 경우다. 둘 중 하나는 반드시 지정해야 한다. `--output`도 마찬가지다.

#### `Input path is not a directory` 에러가 나는 경우

`--input`으로 파일 경로를 넘겼거나 잘못된 경로를 넘긴 경우다. 디렉터리 경로를 지정해야 한다.

#### 프레임이 너무 적거나 많은 경우

`--frame` 값을 조정하면 된다.

- 더 많이 추출하려면 작은 값 사용
- 더 적게 추출하려면 큰 값 사용

#### 출력 이미지 비율이 기대와 다른 경우

이 스크립트는 stretch가 아니라 letterbox를 사용하므로 상하 또는 좌우에 검은 패딩이 생길 수 있다. 이는 정상 동작이다.

## 개발 문서

### 파일 구조

스크립트는 다음 함수들로 구성된다.

- `parse_args()`
  - CLI 인자(`--dataset` 포함)를 정의하고 파싱한다.
- `validate_args(args)`
  - OpenCV 의존성과 인자 값을 검증한다. `--dataset` 기반 기본 경로 해석은 `main()`에서 `validate_args()` 호출 전에 끝난다.
- `list_video_files(input_dir)`
  - 입력 디렉터리의 비디오 파일을 정렬해 반환한다.
- `allocate_split_counts(total_items, ratios)`
  - 분할 비율에 따라 split별 목표 개수를 계산한다.
- `create_split_rng(video_path, seed)`
  - 비디오별 deterministic RNG를 만든다.
- `assign_splits(frame_numbers, ratios, rng)`
  - 추출 대상 프레임 번호를 split 이름에 매핑한다.
- `resize_with_letterbox(frame, width, height)`
  - 종횡비를 유지한 채 black padding으로 리사이즈한다.
- `is_valid_image(path, full=False)`
  - 기존 이미지가 유효한지 검사한다. `full=False`는 PNG 헤더/트레일러만, `full=True`는 완전 디코딩으로 검사한다.
- `_has_png_envelope(path)`
  - PNG 시그니처와 IEND 트레일러 유무를 빠르게 확인하는 경량 검사다.
- `write_image_atomic(path, image)`
  - 임시 파일에 쓰고 `os.replace`로 옮겨 부분 파일이 남지 않게 저장한다.
- `save_frames_for_video(...)`
  - 비디오를 읽고 선택된 프레임만 저장한다. 유효하지 않은 기존 파일은 재추출한다.
- `ensure_output_dirs(output_dir)`
  - `train`, `val`, `test` 디렉터리를 생성한다.
- `main()`
  - `tooltip.dataset_paths.resolve_video_input()`/`resolve_path()`로 `--input`/`--output`을 확정한 뒤 전체 파이프라인을 조합하고 종료 코드 `0`을 반환한다.

### 핵심 구현 세부 사항

#### 0. `--dataset` 경로 해석

`main()`은 `parse_args()` 직후 다음을 수행한다.

```python
args.input = resolve_video_input(args.input, args.dataset)
args.output = resolve_path(args.output, args.dataset, images_dir, "--output")
```

`resolve_video_input()`(`tooltip/dataset_paths.py`)은 이 스크립트 전용 헬퍼로, 명시적 `--input`이 있으면 그대로 쓰고, 없으면 `data/dataset/<dataset>/progressive`가 존재하고 비어있지 않은지 확인해 있으면 그것을, 없으면 원본 `data/dataset-src/<dataset>`을 반환한다. 즉 progressive 변환을 거쳤는지 여부와 무관하게 `--dataset` 하나로 항상 올바른 입력을 찾는다.

#### 1. 비디오별 deterministic RNG

split RNG는 전역 `random.seed()`를 쓰지 않고, 아래 재료로 독립 생성된다.

- `video_path.resolve()`
- 사용자 지정 `seed`

이 둘을 문자열로 합친 뒤 `blake2b` 해시를 만들고, 그 결과를 정수로 변환해 `random.Random`에 넣는다. 따라서 비디오마다 서로 다른 셔플 순서를 갖는다.

#### 2. split 배정은 비디오 단위로 독립적

프레임은 전체 데이터셋 기준으로 한 번에 섞이지 않고, 각 비디오 내부에서만 섞여서 split이 정해진다. 즉, 모든 비디오가 대체로 같은 비율로 `train/val/test`를 갖게 된다.

이 특성은 균형 측면에서는 예측 가능하지만, 프로젝트 목적에 따라서는 비디오 간 분리보다 약한 독립성을 가질 수 있다.

#### 3. split 개수 계산

`allocate_split_counts()`는 비율을 실수 개수로 환산한 뒤 버림하여 기본 개수를 만들고, 남은 remainder를 소수 부분이 큰 split부터 1개씩 배정한다.

예를 들어 총 7개 프레임에 `60/20/20`을 적용하면 대략 `4.2 / 1.4 / 1.4`가 되며, 최종 배정은 `4 / 2 / 1` 또는 소수 정렬 결과에 따른 동등한 배분이 된다.

#### 4. 전체 프레임 순회 방식

추출 대상 프레임만 직접 seek 하지는 않고, 비디오를 0번 프레임부터 끝까지 순차적으로 읽는다.

```python
while True:
    success, frame = capture.read()
    if not success:
        break
```

이 방식은 구현이 단순하고 안정적이지만, `frame_step`이 큰 긴 영상에서는 읽지 않는 프레임도 모두 디코딩하므로 성능상 비효율이 생길 수 있다.

#### 5. letterbox 리사이즈

리사이즈는 `scale = min(width / source_width, height / source_height)`를 사용해 출력 박스 안에 맞춘 뒤, `cv2.copyMakeBorder()`로 부족한 영역을 검은색으로 채운다.

따라서 모델 입력 크기를 강제로 맞추면서도 원본 비율 왜곡은 방지한다.

#### 6. 저장 정책과 재개(resume) 안전성

이미지 포맷은 항상 `.png`다. 출력은 **원자적(atomic) 쓰기**로 저장된다. 즉, 메모리에서 인코딩한 뒤 같은 디렉터리의 임시 파일(`.<이름>.tmp`)에 먼저 쓰고 `os.replace`로 최종 경로에 옮긴다. 따라서 쓰는 도중 프로세스가 강제 종료되어도 최종 경로에는 완전한 파일 또는 이전 파일만 남으며, 잘린 파일이 생기지 않는다(`write_image_atomic`).

재개 시 skip 판정은 단순 존재 여부가 아니라 **유효성 검사**를 거친다(`is_valid_image`).

- `--verify fast`(기본값): PNG 시그니처(파일 앞 8바이트)와 IEND 트레일러(파일 끝 8바이트)만 확인한다. 강제 종료로 잘린 파일을 거의 즉시 잡아내며, 18만 장 규모 재개에서도 빠르다. 다만 트레일러가 멀쩡한 내부 손상은 잡지 못한다.
- `--verify full`: 이미지를 완전히 디코딩(`cv2.imread`)해 모든 손상을 검사한다. 느리지만 철저하다.

유효하지 않은(없거나 손상된) 파일은 재추출 대상이 되어 덮어써지고, 손상 파일을 발견하면 재추출 개수를 로그로 알린다.

### 예외와 종료 방식

`main()`은 예외를 별도로 잡지 않는다. 따라서 다음 문제는 traceback과 함께 즉시 종료된다.

- OpenCV import 실패
- 잘못된 인자 값
- 입력 경로 오류
- 비디오 열기 실패
- 이미지 저장 실패

반면 비디오 파일이 하나도 없는 경우는 예외가 아니라 정상 상황으로 취급해 종료 코드 `0`으로 끝난다.

### 현재 설계 제약

- 입력 디렉터리의 하위 폴더를 재귀 탐색하지 않는다.
- split이 전체 데이터셋 기준이 아니라 비디오별로 독립 배정된다.
- 추출 대상 프레임만 seek하지 않고 전체 프레임을 순차적으로 읽는다.
- 저장 포맷이 PNG로 고정되어 있다.
- 기본 `fast` 검사는 PNG 트레일러만 보므로, 트레일러가 온전한 내부 손상은 `--verify full`로만 잡힌다.
- 강제 종료 시 임시 파일(`.<이름>.tmp`)이 정리되지 않고 남을 수 있다(다음 실행에 영향은 없으나 수동 정리가 필요할 수 있다).
- 로그 출력이 `print()`와 `tqdm`에 묶여 있어 구조화된 로그에는 적합하지 않다.

### 확장 권장 사항

1. 재귀 입력 탐색 옵션 추가
2. JPEG 저장 품질 또는 출력 포맷 선택 옵션 추가
3. 전체 데이터셋 기준 split 또는 비디오 단위 split을 선택할 수 있게 확장
4. 긴 영상 성능 개선을 위해 프레임 seek 또는 샘플링 전략 최적화
5. 실행 시작 시 잔여 임시 파일(`.<이름>.tmp`)을 자동 정리
6. 예외를 정리해 사용자 친화적인 에러 메시지와 종료 코드를 제공

### 다른 코드에서 재사용할 때

현재 구현은 함수 분리가 비교적 잘 되어 있어 필요한 부분만 import해서 쓸 수 있다.

```python
from pathlib import Path
from scripts.generate_dataset import save_frames_for_video

saved_counts = save_frames_for_video(
    video_path=Path("./data/dataset/erop/progressive/sample01.mp4"),
    output_dir=Path("./data/dataset/erop/images"),
    frame_step=10,
    ratios=(60, 20, 20),
    output_width=736,
    output_height=480,
    seed=0,
)
```

다만 OpenCV 객체와 파일 시스템에 직접 의존하는 구조라 순수 함수형 재사용성은 제한적이다. 라이브러리 수준으로 발전시키려면 I/O와 split 계산 로직을 더 분리하는 편이 낫다.
