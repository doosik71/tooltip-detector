# `scripts/download_model.py`

`scripts/download_model.py`는 Hugging Face Hub에서 프로젝트가 사용하는 MONAI 모델 파일을 내려받아 로컬 `temp/` 디렉터리에 준비하는 스크립트다.

현재 구현은 범용 CLI가 아니라 프로젝트 전용 부트스트랩 스크립트이며, 아래 값을 코드에 고정해서 사용한다.

- Hugging Face 저장소: `MONAI/endoscopic_tool_segmentation`
- 다운로드 대상 파일: `models/model.pt`
- 출력 디렉터리: `./temp`

## 사용자 문서

### 목적

이 스크립트는 추론이나 데이터 생성에 필요한 사전 학습 모델을 로컬에 준비한다. 이미 동일한 경로에 모델 파일이 존재하면 다시 다운로드하지 않는다.

### 실행 전 요구 사항

- Python `3.12` 이상
- 프로젝트 의존성 설치
- 인터넷 연결
- 대상 Hugging Face 저장소에 접근 가능한 권한

`pyproject.toml` 기준 주요 의존성은 다음과 같다.

- `huggingface-hub`

### 실행 방법

프로젝트 루트에서 실행한다.

```bash
python scripts/download_model.py
```

프로젝트에서 `uv`를 사용 중이라면 다음처럼 실행해도 된다.

```bash
uv run python scripts/download_model.py
```

### 실행 결과

정상 실행 시 다음 경로에 모델 파일이 준비된다.

```text
./temp/models/model.pt
```

스크립트는 콘솔 로그를 출력하며, 성공 시 종료 코드 `0`, 실패 시 종료 코드 `1`을 반환한다.

### 동작 방식

1. 콘솔 로깅을 `INFO` 레벨로 초기화한다.
2. 출력 경로 `./temp`를 생성한다.
3. `./temp/models/model.pt`가 이미 존재하면 다운로드를 건너뛴다.
4. 파일이 없으면 `huggingface_hub.hf_hub_download()`로 모델을 다운로드한다.
5. 다운로드된 실제 경로를 로그로 남기고 종료한다.

### 인증이 필요한 저장소를 사용할 때

현재 스크립트는 `HF_TOKEN = None`으로 고정되어 있어 공개 저장소만 바로 사용할 수 있다. 비공개 저장소를 사용하려면 `scripts/download_model.py` 안의 `HF_TOKEN` 값을 토큰 문자열로 바꿔야 한다.

예시:

```python
HF_TOKEN = "hf_xxx"
```

토큰을 코드에 직접 넣는 방식은 보안상 바람직하지 않다. 개발 환경에서는 환경 변수나 CLI 인자로 전달하도록 스크립트를 확장하는 편이 낫다.

### 문제 해결

`Failed to download model` 로그가 출력되면 보통 아래 항목을 확인하면 된다.

- 인터넷 연결이 가능한지 확인
- Hugging Face 저장소 이름과 파일 경로가 유효한지 확인
- 비공개 저장소라면 유효한 토큰이 설정되어 있는지 확인
- `temp/` 디렉터리에 쓰기 권한이 있는지 확인

## 개발 문서

### 파일 구조

스크립트는 세 개의 주요 진입점으로 구성된다.

- `configure_logging()`
  - `logging.basicConfig()`로 단순한 콘솔 로깅 포맷을 설정한다.
- `download_monai_model(repo_id, filename, output_dir, token=None)`
  - 출력 디렉터리를 만들고, 로컬 파일 존재 여부를 확인한 뒤, 필요하면 Hub에서 파일을 내려받는다.
- `main()`
  - 현재 프로젝트에서 사용할 저장소와 파일 경로를 상수처럼 정의하고 다운로드 함수를 호출한다.

### 핵심 구현 세부 사항

#### 1. 캐시 확인 기준

재다운로드 방지 로직은 Hugging Face 캐시가 아니라 `output_dir / filename`의 실제 파일 존재 여부를 기준으로 동작한다.

```python
local_model_path = output_dir / filename
if local_model_path.exists():
    return local_model_path
```

따라서 파일만 존재하면 내용 검증 없이 그대로 재사용한다.

#### 2. 다운로드 위치

`hf_hub_download()` 호출 시 `local_dir=str(output_dir)`를 사용한다. `filename`이 `models/model.pt`이므로 실제 파일은 `temp/models/model.pt`에 배치된다.

#### 3. 심볼릭 링크 비활성화

`local_dir_use_symlinks=False`가 설정되어 있어 Hugging Face 캐시를 가리키는 심볼릭 링크 대신 실제 파일이 로컬 디렉터리에 배치된다. 배포 산출물이나 단순 파일 복사 흐름에는 이 구성이 더 다루기 쉽다.

#### 4. 예외 처리

`main()`은 모든 예외를 잡아 에러 로그를 남기고 종료 코드 `1`을 반환한다. 호출 스택이나 세부 예외 종류별 분기는 현재 없다.

### 현재 설계 제약

- CLI 인자를 받지 않는다.
- 모델 저장소, 파일명, 출력 경로가 모두 코드에 하드코딩되어 있다.
- Hugging Face 토큰도 코드 수정으로만 바꿀 수 있다.
- 다운로드된 파일의 무결성 검증이나 버전 고정 검증이 없다.
- `except Exception`으로 넓게 처리해 오류 원인별 대응이 어렵다.

### 확장 권장 사항

다음 순서로 개선하는 것이 실용적이다.

1. `argparse`를 추가해 `--repo-id`, `--filename`, `--output-dir`, `--token-env`를 받도록 변경
2. `HF_TOKEN` 상수 대신 환경 변수에서 토큰을 읽도록 변경
3. 다운로드 전에 대상 파일 경로와 설정값을 로그에 명시
4. Hugging Face 관련 예외를 구분 처리해 인증 실패, 파일 없음, 네트워크 실패를 분리
5. 다른 스크립트에서 재사용할 수 있도록 라이브러리 함수와 CLI 진입점을 분리

### 다른 코드에서 재사용할 때

현재 구조는 함수 분리가 되어 있어 import 후 직접 호출할 수 있다.

```python
from pathlib import Path
from scripts.download_model import download_monai_model

path = download_monai_model(
    repo_id="MONAI/endoscopic_tool_segmentation",
    filename="models/model.pt",
    output_dir=Path("./temp"),
    token=None,
)
```

다만 `scripts` 디렉터리를 라이브러리 패키지처럼 쓰는 구조는 장기적으로는 불안정할 수 있다. 재사용 범위가 넓어지면 별도 모듈로 옮기는 편이 낫다.
