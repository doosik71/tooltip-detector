# `scripts/generate_segmentation.py`

`scripts/generate_segmentation.py`는 다운로드한 MONAI segmentation 모델로 데이터셋 이미지마다 binary segmentation mask를 추론해 PNG로 저장하는 스크립트다.

`--dataset <name>` 옵션을 주면 기본 입력은 `data/dataset/<name>/images`, 기본 출력은 `data/dataset/<name>/segmentation`이며, 입력과 출력 모두 `train`, `val`, `test` 세 분할 구조를 그대로 따른다.

## 사용자 문서

### 목적

`generate_dataset.py`로 만든 이미지 데이터셋의 각 프레임에 대해, 학습된 모델로 도구(tool) 영역을 분할한 binary mask를 생성한다. 이 mask는 다음 단계인 `generate_annotation.py`의 입력이 된다.

이 프로젝트의 기본 작업 순서에서 이 스크립트의 위치는 다음과 같다.

1. `scripts/generate_dataset.py --dataset <name>`로 `data/dataset/<name>/images` 준비
2. `scripts/download_model.py`로 `./temp/models/model.pt` 준비(데이터셋과 무관한 공용 모델)
3. (GPU 사용 시) `scripts/check_gpu.py`로 CUDA/cuDNN 환경 점검
4. `scripts/generate_segmentation.py --dataset <name>`로 mask 생성
5. `scripts/generate_annotation.py --dataset <name>`로 bbox/tip annotation 생성

### 실행 전 요구 사항

- Python `3.12` 이상
- 프로젝트 의존성 설치
- `opencv-python` 설치
- `numpy` 설치
- `monai` 설치
- `torch` 설치
- 입력 이미지가 들어 있는 `images` 디렉터리 (`train/val/test` 하위 폴더 포함)
- 다운로드된 모델 체크포인트 (`./temp/models/model.pt`)
- GPU 추론을 사용할 경우 CUDA가 동작하는 PyTorch 환경

`pyproject.toml` 기준 관련 의존성은 다음과 같다.

- `opencv-python`
- `numpy`
- `monai`
- `torch`

### 기본 실행 방법

프로젝트 루트에서 실행한다.

```bash
python -m scripts.generate_segmentation --dataset erop
```

`uv`를 사용 중이면 다음처럼 실행할 수 있다.

```bash
uv run python -m scripts.generate_segmentation --dataset erop
```

`--dataset`을 주면 아래를 사용한다.

- 입력 디렉터리: `data/dataset/<dataset>/images`
- 출력 디렉터리: `data/dataset/<dataset>/segmentation`
- 모델 체크포인트: `./temp/models/model.pt`
- 디바이스: CUDA 사용 가능 시 `cuda:0`, 아니면 `cpu`
- 덮어쓰기: 비활성화 (기존 mask는 건너뜀)

### CLI 옵션

- `--dataset`
  - 데이터셋 이름(예: `erop`, `cholec80`). `--input`/`--output`을 명시하지 않았을 때 기본 경로를 계산하는 데 쓰인다.
- `--input`
  - `train/val/test` 이미지 폴더가 들어 있는 루트 디렉터리. 기본값은 `data/dataset/<dataset>/images`. 명시하면 `--dataset` 기본값보다 항상 우선한다.
- `--output`
  - `train/val/test` mask 폴더를 저장할 루트 디렉터리. 기본값은 `data/dataset/<dataset>/segmentation`. 명시하면 `--dataset` 기본값보다 항상 우선한다.
- `--model`
  - MONAI 체크포인트 경로. 기본값은 `./temp/models/model.pt`.
- `--device`
  - 사용할 torch 디바이스(`cpu`, `cuda`, `cuda:0` 등). 지정하지 않으면 자동 선택한다.
- `--overwrite`
  - 지정하면 기존 mask를 건너뛰지 않고 다시 추론해 덮어쓴다. 지정하지 않으면 이미 존재하는 mask는 건너뛴다.

### 실행 예시

`--dataset`으로 기본 경로를 사용해 mask를 생성하는 예시:

```bash
python -m scripts.generate_segmentation --dataset cholec80
```

특정 GPU를 지정하는 예시:

```bash
python -m scripts.generate_segmentation --dataset cholec80 --device cuda:0
```

CPU로 추론하는 예시:

```bash
python -m scripts.generate_segmentation --dataset cholec80 --device cpu
```

기존 mask를 모두 다시 생성하는 예시:

```bash
python -m scripts.generate_segmentation --dataset cholec80 --overwrite
```

입력/출력/모델 경로를 직접 지정하는 예시(`--dataset` 기본값을 무시):

```bash
python -m scripts.generate_segmentation \
    --input ./data/dataset/erop/images \
    --output ./data/dataset_v2/segmentation \
    --model ./temp/models/model.pt
```

### 입력 대상 파일

각 split 디렉터리(`train/val/test`) 바로 아래에서 다음 확장자의 파일만 이미지로 읽는다.

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

확장자 비교는 소문자 기준으로 수행하므로 대문자 확장자도 처리된다. 하위 디렉터리는 재귀 탐색하지 않는다.

### 출력 구조

실행 시 출력 디렉터리 아래에 다음 하위 폴더가 생성된다.

```text
./data/dataset/<dataset>/
└── segmentation/
    ├── train/
    ├── val/
    └── test/
```

mask 파일명은 입력 이미지의 확장자만 `.png`로 바꾼 형태다.

```text
{이미지파일이름}.png
```

예를 들어 입력이 `sample01_00000030.png`면 출력도 `sample01_00000030.png`(mask)가 된다.

### 출력 mask 형식

- 단일 채널 8비트 PNG.
- 픽셀값은 `0`(배경) 또는 `255`(도구) 두 값만 가진다.
- 크기는 **입력 이미지의 원본 해상도와 동일**하다. 모델 입력 크기로 리사이즈해 추론한 뒤, 다시 원본 크기로 복원해 저장하기 때문이다.

### 동작 방식

1. CLI 인자를 파싱한다.
2. OpenCV/MONAI/numpy/torch import 여부, 입력 경로, 모델 파일, split 디렉터리의 유효성을 검사한다.
3. 사용할 디바이스를 결정한다.
4. 출력 디렉터리 아래 `train`, `val`, `test` 폴더를 생성한다.
5. `--overwrite`가 아니고 모든 출력 mask가 이미 존재하면, 모델을 로드하지 않고 `All segmentation masks already exist. Nothing to do.`를 출력한 뒤 종료한다.
6. 모델을 빌드하고 체크포인트를 로드한 뒤 평가 모드로 둔다.
7. split별로 이미지 목록을 정렬해 수집한다.
8. 각 이미지를 전처리해 텐서로 만들고, 추론하고, 결과를 원본 크기 mask로 복원해 저장한다.
9. 모든 split 처리가 끝나면 split별 저장/건너뜀 개수를 출력한다.

### 디바이스 선택 방식

- `--device`를 주면 그 값을 그대로 사용한다(`cpu`, `cuda`, `cuda:0` 등).
- 주지 않으면 `torch.cuda.is_available()`로 자동 판단해 가능하면 `cuda:0`, 아니면 `cpu`를 쓴다.

GPU 환경이 의심스러우면 먼저 `python -m scripts.check_gpu --device cuda:0`으로 CUDA/cuDNN 상태를 점검하는 것을 권장한다.

### 종료 및 출력

- 모든 출력이 이미 존재하고 `--overwrite`가 아니면 `All segmentation masks already exist. Nothing to do.`를 출력하고 종료 코드 `0`으로 끝난다.
- 정상 완료 시 `Segmentation generation complete.`와 split별 `saved`/`skipped` 개수를 출력하고 종료 코드 `0`을 반환한다.
- 의존성 import 실패, 잘못된 인자, 입력 경로 오류, 모델 파일 누락, split 폴더 누락, 이미지 읽기/쓰기 실패 등은 예외로 중단된다.

### 문제 해결

#### `opencv-python is required` / `numpy is required` / `monai is required` / `torch is required` 에러

해당 라이브러리가 설치되지 않은 상태다. 프로젝트 의존성을 설치한 뒤 다시 실행한다.

#### `Input directory does not exist` 에러

`--input` 경로가 실제로 존재하는지 확인한다. `--dataset`만 썼다면 `data/dataset/<dataset>/images`가 준비되어 있어야 한다(`generate_dataset.py`를 먼저 실행).

#### `Either --dataset or --input must be provided.` 에러

`--dataset`도 `--input`도 주지 않은 경우다. 둘 중 하나는 반드시 지정해야 한다. `--output`도 마찬가지다.

#### `Model checkpoint does not exist` 에러

`--model` 경로에 체크포인트가 없다. `python scripts/download_model.py`로 모델을 먼저 받는다.

#### `Split directory does not exist` 에러

입력 디렉터리 아래에 `train`, `val`, `test` 세 폴더가 모두 있어야 한다. `generate_dataset.py`를 먼저 정상 실행했는지 확인한다.

#### `Failed to read image` / `Failed to write segmentation mask` 에러

읽으려는 이미지가 손상되었거나, 출력 경로에 쓰기 권한이 없을 수 있다. 파일과 디렉터리 권한을 확인한다.

#### CUDA out of memory / GPU 관련 오류

`--device cpu`로 전환해 추론하거나, GPU 환경을 점검한다(`scripts/check_gpu.py`).

#### 이미 생성한 mask가 갱신되지 않는 경우

기본 동작은 기존 mask를 건너뛴다. 다시 생성하려면 `--overwrite`를 지정한다.

## 개발 문서

### 파일 구조

스크립트는 다음 함수들로 구성된다.

- `parse_args()`
  - CLI 인자(`--dataset`, `--input`, `--output`, `--model`, `--device`, `--overwrite`)를 정의하고 파싱한다.
- `validate_args(args)`
  - 의존성 import, 입력 루트, 모델 파일, split 디렉터리 존재 여부를 검증한다. `--dataset` 기반 기본 경로 해석은 `main()`에서 이 함수 호출 전에 끝난다.
- `resolve_device(device_arg)`
  - 사용할 `torch.device`를 결정한다.
- `list_images(image_dir)`
  - 디렉터리에서 지원 확장자의 파일을 정렬해 반환한다.
- `ensure_output_dirs(output_root)`
  - 출력 루트 아래 `train`, `val`, `test` 디렉터리를 생성한다.
- `_all_outputs_exist(input_root, output_root)`
  - 모든 입력 이미지에 대응하는 출력 mask가 이미 존재하는지 검사한다.
- `build_model(model_path, device)`
  - FlexibleUNet을 구성하고 체크포인트를 로드해 평가 모드로 반환한다.
- `preprocess_image(image_path)`
  - 이미지를 RGB로 읽고 모델 입력 크기로 리사이즈해 정규화 텐서로 만든다.
- `postprocess_mask(logits, output_size)`
  - 로짓을 argmax 후 `255` 스케일 mask로 바꾸고 원본 크기로 복원한다.
- `infer_image(...)`
  - 한 이미지에 대해 전처리→추론→후처리→저장을 수행한다.
- `process_split(...)`
  - 한 split을 처리하고 `(saved, skipped)` 개수를 반환한다.
- `main()`
  - `tooltip.dataset_paths.resolve_path()`로 `--input`/`--output`을 확정한 뒤 전체 파이프라인을 조합하고 종료 코드 `0`을 반환한다.

### 핵심 구현 세부 사항

#### 0. `--dataset` 경로 해석

`main()`은 `parse_args()` 직후 다음을 수행한다.

```python
args.input = resolve_path(args.input, args.dataset, images_dir, "--input")
args.output = resolve_path(args.output, args.dataset, segmentation_dir, "--output")
```

`resolve_path()`(`tooltip/dataset_paths.py`)는 명시적으로 준 경로가 있으면 그대로 쓰고, 없으면 `--dataset` 이름으로 기본 경로를 계산한다. 둘 다 없으면 어떤 플래그가 필요한지 알려주는 `ValueError`를 즉시 던진다.

#### 1. 의존성 지연 import

`cv2`, `monai`, `numpy`, `torch`는 모듈 상단에서 각각 `try/except`로 import하고, 실패하면 import 에러 객체를 보관한다. 실제 검증은 `validate_args()`에서 수행해, 의존성이 없을 때 traceback 대신 사람이 읽기 쉬운 `RuntimeError` 메시지를 제공한다.

#### 2. 모델 구성

`build_model()`은 MONAI의 `FlexibleUNet`을 다음 설정으로 만든다.

```python
monai.networks.nets.FlexibleUNet(
    in_channels=3,
    out_channels=2,
    backbone="efficientnet-b2",
    pretrained=False,
    spatial_dims=2,
    is_pad=False,
    pre_conv=None,
)
```

- `in_channels=3`: RGB 입력.
- `out_channels=2`: 배경/도구 2클래스. 이후 argmax로 binary mask를 만든다.
- `pretrained=False`: backbone 사전학습 가중치를 받지 않고, 전체 가중치를 체크포인트에서 로드한다.
- `is_pad=False`: 내부 자동 패딩을 끈다. 따라서 입력 spatial 크기가 네트워크가 처리 가능한 크기여야 한다.

체크포인트는 `torch.load(model_path, map_location=device)`로 읽고, `load_state_dict(state_dict, strict=True)`로 정확히 일치하는 키만 허용해 로드한다. 즉 체크포인트는 이 아키텍처와 정확히 호환되어야 한다.

#### 3. 전처리

`preprocess_image()`는 다음을 수행한다.

1. `cv2.imread(..., IMREAD_COLOR)`로 BGR 읽기.
2. `BGR2RGB` 변환.
3. 모델 입력 크기로 `INTER_LINEAR` 리사이즈.
4. `float32`로 변환 후 `255.0`으로 나눠 `[0, 1]` 정규화.
5. `(H, W, C)` → `(1, C, H, W)`로 차원 변환.

원본 RGB 이미지도 함께 반환해, 후처리에서 원본 해상도를 복원하는 데 사용한다.

#### 4. 모델 입력 크기

모델 입력 크기는 모듈 상단 상수로 고정되어 있다.

```python
MODEL_INPUT_WIDTH = 736
MODEL_INPUT_HEIGHT = 480
```

`cv2.resize`는 `(width, height)` 순서를 받으므로, 실제 리사이즈는 `(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT) = (736, 480)`, 즉 **너비 736 × 높이 480**(가로로 긴 형태)으로 수행된다. 이는 `generate_dataset.py`의 기본 출력 이미지 크기(너비 736 × 높이 480)와 같은 방향이다.

추론 결과 mask는 `postprocess_mask()`에서 다시 **원본 이미지 크기**로 복원되므로 최종 mask의 해상도는 입력 이미지와 같다. 새 모델이나 다른 해상도의 데이터셋을 쓸 때는 이 상수와 입력 해상도의 정합성을 확인하는 것이 좋다.

#### 5. 후처리

`postprocess_mask()`는 다음을 수행한다.

```python
prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
mask = prediction * 255
restored = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
```

- 클래스 차원에서 argmax해 클래스 인덱스(`0`/`1`) mask를 만든다.
- `255`를 곱해 `0`/`255` binary mask로 바꾼다.
- `INTER_NEAREST`로 원본 크기로 복원한다. nearest 보간을 쓰는 이유는 클래스 경계에 중간값이 생기지 않게 하기 위함이다.

`output_size`는 `(original_width, original_height)`로 전달되어 입력 이미지의 원본 해상도를 복원한다.

#### 6. 추론 실행

`infer_image()`는 `torch.inference_mode()` 컨텍스트에서 추론한다. 이는 autograd를 비활성화해 메모리와 속도를 최적화한다. 텐서는 추론 직전에 `.to(device)`로 대상 디바이스에 올린다.

#### 7. skip / overwrite 정책

skip은 두 단계로 동작한다.

- 전역 단축 경로: `main()`은 `--overwrite`가 아닐 때 `_all_outputs_exist()`로 전체 출력이 이미 존재하면 모델 로드 없이 즉시 종료한다. 모델 로딩 비용을 아끼는 최적화다.
- 파일 단위: `process_split()`은 출력 mask가 존재하고 `--overwrite`가 아니면 해당 파일만 건너뛴다.

`--overwrite`를 주면 두 검사 모두 무시되고 모든 mask를 재추론한다.

#### 8. 모델 1회 로드, 전 split 공유

`build_model()`은 `main()`에서 한 번만 호출되고, 그 모델 객체를 `train/val/test` 모든 split 처리에 재사용한다. split마다 모델을 다시 로드하지 않는다.

### 예외와 종료 방식

`main()`은 예외를 별도로 잡지 않는다. 따라서 다음 문제는 traceback과 함께 즉시 종료된다.

- 의존성 import 실패
- 잘못된 인자 값
- 입력 루트/모델 파일/split 디렉터리 누락
- 이미지 읽기 실패(`preprocess_image`의 `RuntimeError`)
- mask 쓰기 실패(`infer_image`의 `RuntimeError`)
- 디바이스/추론 관련 torch 오류

반면 "모든 출력이 이미 존재"하는 경우는 예외가 아니라 정상 상황으로 취급해 조기 종료한다.

### 현재 설계 제약

- 입력 split 디렉터리의 하위 폴더를 재귀 탐색하지 않는다.
- 모델 입력 크기가 상수(너비 `736`/높이 `480`)로 고정되어 있어, 다른 해상도의 데이터셋을 쓰려면 코드를 수정해야 한다.
- 배치 추론이 아니라 이미지 한 장씩 처리한다. 대규모 데이터셋에서는 처리량이 제한될 수 있다.
- 아키텍처가 `efficientnet-b2` backbone의 FlexibleUNet 2클래스로 고정되어 있어, 다른 구조의 체크포인트는 `strict=True` 로딩에서 실패한다.
- 출력이 단일 클래스 binary mask(`0`/`255`)로 고정되어 있어 다중 클래스 출력을 표현하지 못한다.
- 로그 출력이 `print()`와 `tqdm`에 묶여 있어 구조화된 로그에는 적합하지 않다.

### 확장 권장 사항

1. 모델 입력 크기를 CLI 옵션으로 빼고, 입력 해상도와의 정합성을 검증하는 로직 추가
2. 배치 추론을 도입해 GPU 활용도와 처리량 개선
3. 아키텍처/클래스 수를 설정으로 분리해 다양한 체크포인트 지원
4. 다중 클래스 mask 출력(클래스별 색상 또는 채널) 옵션 추가
5. split 이름을 옵션화해 `train/val/test` 외 구성도 지원
6. 예외를 정리해 사용자 친화적인 에러 메시지와 종료 코드 제공

### 다른 코드에서 재사용할 때

추론 핵심 함수들은 비교적 독립적이라 필요한 부분만 가져다 쓸 수 있다.

```python
from pathlib import Path
from scripts.generate_segmentation import build_model, infer_image, resolve_device

device = resolve_device("cuda:0")
model = build_model(Path("./temp/models/model.pt"), device)
infer_image(
    image_path=Path("./data/dataset/erop/images/train/sample01_00000030.png"),
    output_path=Path("./out/sample01_00000030.png"),
    model=model,
    device=device,
)
```

다만 전처리/후처리가 모델 입력 크기 상수와 OpenCV I/O에 직접 묶여 있어, 다른 해상도나 다른 모델에 적용하려면 `preprocess_image()`/`postprocess_mask()`의 크기 처리를 손봐야 한다.
