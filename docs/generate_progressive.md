# `scripts/generate_progressive.py`

`scripts/generate_progressive.py`는 `./data/video`에 있는 동영상 파일을 순회하면서 `ffmpeg`로 디인터레이스 및 재인코딩을 수행하고, 결과를 `./data/progressive`에 저장하는 스크립트다.

현재 구현은 범용 CLI 도구가 아니라 프로젝트 전용 배치 스크립트이며, 입력/출력 경로가 코드에 고정되어 있다.

- 입력 디렉터리: `./data/video`
- 출력 디렉터리: `./data/progressive`
- 변환 필터: `yadif`
- 비디오 코덱: `libx264`
- 품질 설정: `-crf 20`
- 오디오 처리: `-c:a copy`

## 사용자 문서

### 목적

인터레이스가 포함될 수 있는 원본 동영상을 progressive 형식에 맞게 정리해 후속 프레임 추출 작업에 사용하기 위한 전처리 스크립트다.

이 프로젝트 기준으로는 `scripts/generate_dataset.py`를 실행하기 전에 `./data/progressive`용 입력 영상을 준비하는 단계로 볼 수 있다.

### 실행 전 요구 사항

- Python `3.12` 이상
- 시스템에 `ffmpeg` 설치
- 입력 동영상이 `./data/video` 아래에 존재
- `./data/progressive`에 쓸 수 있는 권한

스크립트는 Python 표준 라이브러리만 사용하지만, 실제 변환은 외부 프로그램 `ffmpeg`에 의존한다.

### 실행 방법

프로젝트 루트에서 실행한다.

```bash
python scripts/generate_progressive.py
```

프로젝트에서 `uv`를 사용 중이라면 다음처럼 실행할 수 있다.

```bash
uv run python scripts/generate_progressive.py
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

변환된 파일은 원본과 같은 파일명으로 `./data/progressive` 아래에 저장된다.

예시:

```text
입력:  ./data/video/sample01.mp4
출력:  ./data/progressive/sample01.mp4
```

출력 파일이 이미 존재하면 해당 파일은 다시 만들지 않고 건너뛴다.

### 동작 방식

1. 입력 디렉터리와 출력 디렉터리를 `Path` 객체로 준비한다.
2. 출력 디렉터리가 없으면 자동 생성한다.
3. 입력 디렉터리에서 지원 확장자를 가진 파일만 수집한다.
4. 각 파일에 대해 같은 이름의 출력 파일이 이미 있으면 건너뛴다.
5. 출력 파일이 없으면 `ffmpeg`를 호출해 변환한다.
6. 전체 파일 처리가 끝나면 완료 메시지를 출력한다.

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

이 스크립트의 기본 출력 경로는 `./data/progressive`이고, `scripts/generate_dataset.py`의 기본 입력 경로도 `./data/progressive`다. 따라서 일반적인 작업 순서는 다음과 같다.

1. `scripts/generate_progressive.py`로 전처리 영상 생성
2. `scripts/generate_dataset.py`로 프레임 추출 및 `train/val/test` 분할

### 문제 해결

#### `ffmpeg`이 없다고 나오는 경우

스크립트는 `ffmpeg` 실행 파일을 찾지 못하면 아래와 같은 안내를 출력하고 종료한다.

```text
시스템에 'ffmpeg'이 설치되어 있지 않습니다.
```

리눅스 환경에서는 보통 다음처럼 설치한다.

```bash
sudo apt install ffmpeg
```

#### 변환 대상 파일이 없다고 나오는 경우

입력 폴더에 지원 확장자의 파일이 없으면 변환을 수행하지 않고 종료한다. 이 경우 다음을 확인하면 된다.

- 파일이 실제로 `./data/video` 바로 아래에 있는지 확인
- 확장자가 지원 목록에 포함되는지 확인
- 입력 폴더 경로를 코드에서 바꿔야 하는 상황인지 확인

#### 일부 파일만 건너뛰는 경우

동일한 파일명이 `./data/progressive`에 이미 있으면 스크립트는 해당 파일을 정상적으로 skip한다. 다시 변환하려면 기존 출력 파일을 직접 삭제해야 한다.

## 개발 문서

### 파일 구조

스크립트는 하나의 핵심 함수와 하나의 고정 진입점으로 구성된다.

- `convert_to_progressive(input_dir, output_dir)`
  - 입력 디렉터리에서 대상 영상을 찾고, 각 파일에 대해 `ffmpeg` 변환을 수행한다.
- `__main__` 블록
  - `INPUT_FOLDER = "./data/video"`와 `OUTPUT_FOLDER = "./data/progressive"`를 정의한 뒤 `convert_to_progressive()`를 호출한다.

### 핵심 구현 세부 사항

#### 1. 파일 탐색 범위

입력 파일 수집은 `input_path.iterdir()` 기반이므로 재귀 탐색이 아니다.

```python
video_files = [
    f for f in input_path.iterdir() if f.is_file() and f.suffix in valid_extensions
]
```

따라서 하위 폴더에 있는 영상은 자동 처리되지 않는다.

#### 2. 스킵 정책

출력 파일 경로는 `output_path / video_file.name`으로 결정된다. 즉, 파일명이 같으면 동일 출력 파일로 간주한다.

```python
output_file = output_path / video_file.name
if output_file.exists():
    continue
```

이 로직은 단순하고 빠르지만, 입력 파일이 바뀌었더라도 같은 이름의 출력 파일이 이미 있으면 재변환하지 않는다.

#### 3. `ffmpeg` 실행 방식

변환은 `subprocess.run(cmd, check=True)`로 수행한다. 표준 출력 캡처는 하지 않으며, 실패 시 `CalledProcessError`를 잡아 파일 단위로 에러를 출력하고 다음 파일로 진행한다.

즉, 일부 파일 실패가 전체 배치를 즉시 중단시키지는 않는다.

#### 4. 로그와 종료 방식

스크립트는 `logging`이 아니라 `print()` 기반 상태 메시지를 사용한다. 메시지에는 이모지가 포함되어 있으며, CLI 자동화 관점에서는 기계 친화적인 출력 형식은 아니다.

또한 `main()` 함수가 따로 없고 `convert_to_progressive()`가 `None`을 반환하므로, 예외가 밖으로 전파되지 않는 한 프로세스 종료 코드는 일반적으로 `0`이다.

#### 5. progressive 변환의 의미

스크립트 이름은 `generate_progressive`지만, 내부적으로는 컨테이너 포맷 자체를 "progressive 포맷"으로 바꾸는 전용 로직이 있는 것이 아니라 `yadif` 디인터레이스와 H.264 재인코딩을 통해 후속 처리에 적합한 영상을 만드는 방식이다.

### 현재 설계 제약

- CLI 인자를 받지 않는다.
- 입력 경로와 출력 경로가 코드에 하드코딩되어 있다.
- 지원 확장자 목록이 코드 내부 상수로만 존재한다.
- 재귀 탐색을 지원하지 않는다.
- 출력 파일 존재 여부만으로 skip 판단을 한다.
- `ffmpeg` 경로를 지정할 수 없고 시스템 PATH에 의존한다.
- 변환 파라미터(`yadif`, `libx264`, `crf=20`)를 실행 시점에 바꿀 수 없다.

### 확장 권장 사항

다음 개선이 실용적이다.

1. `argparse`를 추가해 `--input`, `--output`, `--crf`, `--overwrite`, `--recursive`를 지원
2. `--extensions` 또는 내부 상수 정리로 지원 확장자 관리 일원화
3. `path.suffix.lower()`를 사용해 대소문자 중복 목록 제거
4. `ffmpeg` 표준 에러를 캡처해 실패 원인을 파일별로 더 명확히 출력
5. `main()`과 종료 코드 체계를 도입해 자동화 환경에서 성공/실패를 명확히 구분
6. 필요하면 병렬 처리나 작업 큐를 도입해 대량 비디오 처리 성능 개선

### 다른 코드에서 재사용할 때

현재 함수는 직접 import 해서 호출할 수 있다.

```python
from scripts.generate_progressive import convert_to_progressive

convert_to_progressive("./data/video", "./data/progressive")
```

다만 내부 출력이 모두 `print()`에 묶여 있고 변환 옵션도 함수 인자로 노출되어 있지 않아, 라이브러리 함수로 재사용하기에는 유연성이 낮다. 재사용 범위가 커지면 변환 옵션을 명시적 인자로 분리하는 편이 낫다.
