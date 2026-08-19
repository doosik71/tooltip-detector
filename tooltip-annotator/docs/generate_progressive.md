# `scripts/generate_progressive.py`

`scripts/generate_progressive.py`는 데이터셋 원본 동영상 디렉터리를 순회하면서 `ffmpeg`로 디인터레이스 및 재인코딩을 수행하고, 결과를 데이터셋별 progressive 디렉터리에 저장하는 스크립트다.

기본 입력은 `./data/dataset-src/<dataset>`, 기본 출력은 `./data/dataset/<dataset>/progressive`이며, `--dataset` 옵션으로 데이터셋 이름만 주면 이 두 경로가 자동으로 계산된다.

- 변환 필터: `yadif`
- 비디오 코덱: `libx264`
- 품질 설정: `-crf 20`
- 오디오 처리: `-c:a copy`

## 사용자 문서

### 목적

인터레이스가 포함될 수 있는 원본 동영상을 progressive 형식에 맞게 정리해 후속 프레임 추출 작업에 사용하기 위한 전처리 스크립트다.

이 프로젝트 기준으로는 `scripts/generate_dataset.py`를 실행하기 전에 데이터셋별 progressive 입력 영상을 준비하는 단계로 볼 수 있다. 다만 원본 동영상이 이미 progressive 형식이라 디인터레이스가 필요 없는 데이터셋이라면, 이 단계를 건너뛰고 `generate_dataset.py --dataset <name>`을 바로 실행해도 된다(자세한 내용은 [docs/generate_dataset.md](./generate_dataset.md)의 입력 fallback 설명 참고).

### 실행 전 요구 사항

- Python `3.12` 이상
- 시스템에 `ffmpeg` 설치
- 입력 동영상이 `./data/dataset-src/<dataset>` 아래에 존재
- `./data/dataset/<dataset>/progressive`에 쓸 수 있는 권한

스크립트는 Python 표준 라이브러리만 사용하지만, 실제 변환은 외부 프로그램 `ffmpeg`에 의존한다.

### 기본 실행 방법

프로젝트 루트에서 실행한다.

```bash
python scripts/generate_progressive.py --dataset erop
```

`uv`를 사용 중이라면 다음처럼 실행할 수 있다.

```bash
uv run python scripts/generate_progressive.py --dataset erop
```

### CLI 옵션

- `--dataset`
  - 데이터셋 이름(예: `erop`, `cholec80`). `--input`/`--output`을 명시하지 않았을 때 기본 경로를 계산하는 데 쓰인다.
- `--input`
  - 입력 동영상 디렉터리. 기본값은 `data/dataset-src/<dataset>`. 명시하면 `--dataset` 기본값보다 항상 우선한다.
- `--output`
  - 출력 디렉터리. 기본값은 `data/dataset/<dataset>/progressive`. 명시하면 `--dataset` 기본값보다 항상 우선한다.

`--dataset`과 `--input`/`--output` 둘 다 주지 않으면 `Either --dataset or --input must be provided.`(또는 `--output`) 에러로 즉시 실패한다.

### 실행 예시

`--dataset`만으로 기본 경로를 사용하는 예시:

```bash
python scripts/generate_progressive.py --dataset cholec80
```

경로를 직접 지정하는 예시(다른 데이터셋 컨벤션을 따르지 않는 임시 작업 등):

```bash
python scripts/generate_progressive.py --input ./data/tmp_videos --output ./data/tmp_progressive
```

`--dataset`과 `--output`을 함께 줘서 출력 위치만 바꾸는 예시:

```bash
python scripts/generate_progressive.py --dataset erop --output ./data/dataset/erop_v2/progressive
```

### 입력 대상 파일

다음 확장자의 파일만 변환 대상으로 인식한다.

- `.mp4`
- `.avi`
- `.mkv`
- `.mov`
- `.MP4`
- `.AVI`
- `.MKV`
- `.MOV`

입력 디렉터리 바로 아래 파일만 검사하며, 하위 디렉터리를 재귀적으로 탐색하지는 않는다.

### 출력 결과

변환된 파일은 원본과 같은 파일명으로 출력 디렉터리 아래에 저장된다.

예시:

```text
입력:  ./data/dataset-src/erop/sample01.mp4
출력:  ./data/dataset/erop/progressive/sample01.mp4
```

출력 파일이 이미 존재하면 해당 파일은 다시 만들지 않고 건너뛴다.

### 동작 방식

1. CLI 인자를 파싱한다.
2. `--input`/`--output`이 없으면 `--dataset`으로 기본 경로를 계산한다. 둘 다 없으면 에러로 종료한다.
3. 입력 디렉터리 존재 여부를 검사한다.
4. 출력 디렉터리가 없으면 자동 생성한다.
5. 입력 디렉터리에서 지원 확장자를 가진 파일만 수집한다.
6. 각 파일에 대해 같은 이름의 출력 파일이 이미 있으면 건너뛴다.
7. 출력 파일이 없으면 `ffmpeg`를 호출해 변환한다.
8. 전체 파일 처리가 끝나면 완료 메시지를 출력한다.

실제 `ffmpeg` 호출은 아래 옵션에 해당한다.

```bash
ffmpeg -loglevel error -i INPUT -vf yadif -c:v libx264 -crf 20 -c:a copy OUTPUT
```

각 옵션의 의미는 다음과 같다.

- `-loglevel error`
  - 에러만 출력해 로그를 단순화한다.
- `-vf yadif`
  - 디인터레이스 필터를 적용한다.
- `-c:v libx264`
  - H.264 코덱으로 비디오를 재인코딩한다.
- `-crf 20`
  - 화질과 용량의 균형을 정하는 품질 값이다.
- `-c:a copy`
  - 오디오는 다시 인코딩하지 않고 복사한다.

### 후속 작업과의 연결

이 스크립트의 기본 출력 경로는 `data/dataset/<dataset>/progressive`이고, `scripts/generate_dataset.py`가 `--dataset`만으로 입력 경로를 계산할 때 이 폴더가 존재하고 비어있지 않으면 우선 사용한다. 따라서 일반적인 작업 순서는 다음과 같다.

1. `scripts/generate_progressive.py --dataset <name>`로 전처리 영상 생성
2. `scripts/generate_dataset.py --dataset <name>`로 프레임 추출 및 `train/val/test` 분할

### 문제 해결

#### `ffmpeg`이 없다고 나오는 경우

스크립트는 `ffmpeg` 실행 파일을 찾지 못하면 아래와 같은 안내를 출력하고 종료한다.

```text
ffmpeg is not installed. Run 'sudo apt install ffmpeg'.
```

리눅스 환경에서는 보통 다음처럼 설치한다.

```bash
sudo apt install ffmpeg
```

#### `Either --dataset or --input must be provided.` 에러가 나는 경우

`--dataset`도 `--input`도 주지 않은 경우다. 둘 중 하나는 반드시 지정해야 한다. `--output`도 마찬가지다.

#### `Input directory does not exist` 에러가 나는 경우

`--dataset`으로 계산된 `data/dataset-src/<dataset>` 또는 직접 지정한 `--input` 경로가 실제로 존재하는지 확인한다.

#### 변환 대상 파일이 없다고 나오는 경우

입력 폴더에 지원 확장자의 파일이 없으면 변환을 수행하지 않고 종료한다. 이 경우 다음을 확인하면 된다.

- 파일이 실제로 입력 디렉터리 바로 아래에 있는지 확인
- 확장자가 지원 목록에 포함되는지 확인
- `--dataset` 이름이 올바른지 확인(`data/dataset-src` 아래 실제 폴더 이름과 일치해야 함)

#### 일부 파일만 건너뛰는 경우

동일한 파일명이 출력 디렉터리에 이미 있으면 스크립트는 해당 파일을 정상적으로 skip한다. 다시 변환하려면 기존 출력 파일을 직접 삭제해야 한다.

## 개발 문서

### 파일 구조

스크립트는 다른 5개 스크립트(`generate_dataset.py` 등)와 동일한 `parse_args()` → `validate_args()` → `main() -> int` 구조를 따른다.

- `parse_args()`
  - CLI 인자(`--dataset`, `--input`, `--output`)를 정의하고 파싱한다.
- `validate_args(args)`
  - 경로 해석이 끝난 뒤의 입력 디렉터리 존재 여부를 검증한다.
- `convert_to_progressive(input_dir, output_dir)`
  - 입력 디렉터리에서 대상 영상을 찾고, 각 파일에 대해 `ffmpeg` 변환을 수행한다. CLI 계층과 분리된 순수 변환 로직이다.
- `main()`
  - 인자를 파싱하고, `tooltip.dataset_paths.resolve_path()`로 `--input`/`--output`을 확정한 뒤 `validate_args()`와 `convert_to_progressive()`를 호출한다.

### 핵심 구현 세부 사항

#### 1. 경로 해석과 `--dataset`

`main()`은 `parse_args()` 직후, `validate_args()`를 호출하기 전에 다음처럼 경로를 확정한다.

```python
args.input = resolve_path(args.input, args.dataset, dataset_src_dir, "--input")
args.output = resolve_path(args.output, args.dataset, progressive_dir, "--output")
```

`resolve_path()`(`tooltip/dataset_paths.py`)는 명시적으로 준 경로가 있으면 그대로 쓰고, 없으면 `--dataset` 이름으로 기본 경로를 계산한다. 둘 다 없으면 어떤 플래그가 필요한지 알려주는 `ValueError`를 즉시 던진다.

`data/dataset-src`는 읽기 전용 원본이므로 출력 기본 경로는 그 안이 아니라 `data/dataset/<dataset>/progressive`로 계산된다(`progressive_dir()`).

#### 2. 파일 탐색 범위

입력 파일 수집은 `input_dir.iterdir()` 기반이므로 재귀 탐색이 아니다.

```python
video_files = [
    f for f in input_dir.iterdir() if f.is_file() and f.suffix in VALID_EXTENSIONS
]
```

따라서 하위 폴더에 있는 영상은 자동 처리되지 않는다.

#### 3. 스킵 정책

출력 파일 경로는 `output_dir / video_file.name`으로 결정된다. 즉, 파일명이 같으면 동일 출력 파일로 간주한다.

```python
output_file = output_dir / video_file.name
if output_file.exists():
    continue
```

이 로직은 단순하고 빠르지만, 입력 파일이 바뀌었더라도 같은 이름의 출력 파일이 이미 있으면 재변환하지 않는다.

#### 4. `ffmpeg` 실행 방식

변환은 `subprocess.run(cmd, check=True)`로 수행한다. 표준 출력 캡처는 하지 않으며, 실패 시 `CalledProcessError`를 잡아 파일 단위로 에러를 출력하고 다음 파일로 진행한다.

즉, 일부 파일 실패가 전체 배치를 즉시 중단시키지는 않는다.

#### 5. 로그와 종료 방식

`convert_to_progressive()`는 `logging`이 아니라 `print()` 기반 상태 메시지를 사용한다. `main()`은 다른 스크립트와 동일하게 `int` 종료 코드를 반환하고, `if __name__ == "__main__": raise SystemExit(main())` 패턴으로 종료 코드를 프로세스에 전달한다.

#### 6. progressive 변환의 의미

스크립트 이름은 `generate_progressive`지만, 내부적으로는 컨테이너 포맷 자체를 "progressive 포맷"으로 바꾸는 전용 로직이 있는 것이 아니라 `yadif` 디인터레이스와 H.264 재인코딩을 통해 후속 처리에 적합한 영상을 만드는 방식이다.

### 현재 설계 제약

- 입력 경로를 재귀 탐색하지 않는다.
- 지원 확장자 목록이 코드 내부 상수로만 존재한다.
- 출력 파일 존재 여부만으로 skip 판단을 한다(입력 파일 변경 여부는 확인하지 않는다).
- `ffmpeg` 경로를 지정할 수 없고 시스템 PATH에 의존한다.
- 변환 파라미터(`yadif`, `libx264`, `crf=20`)를 실행 시점에 바꿀 수 없다.

### 확장 권장 사항

다음 개선이 실용적이다.

1. `--crf`, `--overwrite`, `--recursive` 같은 추가 옵션 지원
2. `--extensions` 옵션으로 지원 확장자 관리 일원화
3. `ffmpeg` 표준 에러를 캡처해 실패 원인을 파일별로 더 명확히 출력
4. 필요하면 병렬 처리나 작업 큐를 도입해 대량 비디오 처리 성능 개선

### 다른 코드에서 재사용할 때

`convert_to_progressive()`는 CLI 계층과 분리되어 있어 직접 import해서 호출할 수 있다.

```python
from scripts.generate_progressive import convert_to_progressive

convert_to_progressive(Path("./data/dataset-src/erop"), Path("./data/dataset/erop/progressive"))
```

다만 내부 출력이 모두 `print()`에 묶여 있고 변환 옵션도 함수 인자로 노출되어 있지 않아, 라이브러리 함수로 재사용하기에는 유연성이 낮다. 재사용 범위가 커지면 변환 옵션을 명시적 인자로 분리하는 편이 낫다.
