# yolov8s Jetson 변환/추론 프로젝트

[baseline/yolov8sclone](../../../baseline/yolov8sclone/)에서 학습한
`data/model/cholec80/tooltip/model.pt`를 **Jetson Orin Nano**에서 실행하기 위한
서브 프로젝트다. 저장소 루트나 `baseline/`의 다른 서브 프로젝트와는 별개의, 이 보드
전용 `uv` 가상환경을 쓴다.

`scripts/convert-model.py`가 그 체크포인트를 TensorRT 엔진으로 변환하고,
`scripts/demo.py`가 변환된 엔진으로 실시간 GUI 데모를 띄운다. 둘 다 이 보드에서
실제로 실행해 확인했다 (아래 [변환](#모델-변환)·[데모](#데모-gui) 참고).

## 왜 독립 프로젝트인가

저장소 루트의 `pyproject.toml`은 torch/torchvision을
`https://download.pytorch.org/whl/cu128`(`pytorch-cu128`) 인덱스에서 받고
Python ≥3.12를 요구한다. 이 인덱스는 **x86_64 wheel만** 배포하므로 Jetson
(aarch64)에서는 `uv sync` 자체가 실패한다. `baseline/yolov8sclone`이 별도
가상환경 없이 루트 환경을 그대로 쓰는 서브 프로젝트인 것과 대조적으로, 이
프로젝트는 Jetson에서 동작해야 하므로 처음부터 이 보드 전용 인덱스와 인터프리터를
쓰는 독립 `uv` 프로젝트로 만들었다.

## 대상 하드웨어/소프트웨어

이 프로젝트를 구축·검증한 실제 장비 기준이다.

| 항목          | 값                                                                      |
| ------------- | ----------------------------------------------------------------------- |
| 보드          | Jetson Orin Nano                                                        |
| JetPack       | 6.2 (L4T R36.4.7)                                                       |
| OS            | Ubuntu 22.04.5 LTS, aarch64                                             |
| CUDA          | 12.6                                                                    |
| TensorRT      | 10.3.0 (apt로 설치됨, 아래 [TensorRT](#tensorrt-시스템-전역-설치) 참고) |
| 시스템 Python | 3.10.12 (`/usr/bin/python3`)                                            |

## 설치

```bash
cd standalone/jetson/yolov8s
uv venv --system-site-packages    # 시스템 dist-packages(TensorRT)를 이 .venv에서도 보이게
uv sync                            # pyproject.toml의 나머지 의존성 설치
```

`--system-site-packages`가 필요한 이유는 아래 [TensorRT](#tensorrt-시스템-전역-설치)
절 참고 — 짧게 말하면 이 프로젝트가 쓰는 `tensorrt` 파이썬 바인딩은 pip으로 못 받고
JetPack이 apt로 시스템 Python에만 깔아 두기 때문이다. `uv sync`만 실행해 `.venv`가
없는 채로 시작하면 이 플래그 없이 만들어지므로, 반드시 `uv venv` → `uv sync` 순서를
지킨다 (이미 만들어진 `.venv`가 있으면 `uv sync`만으로 재현된다).

`uv`는 아래 [`pyproject.toml`](pyproject.toml)에 고정된 버전대로 torch/torchvision을
NVIDIA의 **Jetson AI Lab** 인덱스(`https://pypi.jetson-ai-lab.io/jp6/cu126`, JetPack 6 /
CUDA 12.6용 aarch64 wheel)에서, 나머지(`numpy`, `opencv-python-headless`, `onnx`,
`pillow`, `scipy`, `tqdm`)는 PyPI에서 받는다. 설치가 끝나면 CUDA 인식을 확인한다.

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.8.0 True
```

apt로 미리 깔려 있어야 하는 것은 TensorRT(JetPack이 기본 설치)뿐이고, 나머지는 전부
`uv`가 받는다. 다만 torch 버전 선택에는 짚어 둘 사정이 있다.

### torch 버전을 2.11.0이 아니라 2.8.0으로 고정한 이유

Jetson AI Lab 인덱스(jp6/cu126)에 올라온 torch 중 가장 최신은 2.11.0(대응하는
torchvision 0.26.0)이지만, 이 보드에서 import 자체가 실패한다.

```text
ImportError: libcudss.so.0: cannot open shared object file: No such file or directory
```

`libcudss`(NVIDIA cuDSS, sparse direct solver 라이브러리)는 이 JetPack의 CUDA apt
저장소에도, 이 pip 인덱스에도 없다. NVIDIA가 이 보드용으로 따로 배포하기 전에는
2.10 이상 wheel을 쓸 수 없다는 뜻이다. 인덱스에 있는 버전을 낮은 쪽부터 확인한 결과
**2.8.0 + torchvision 0.23.0**이 깨끗하게 import되고 `cuda.is_available() == True`를
반환하는 가장 최신 조합이었다. (2.9.1/0.24.1, 2.10.0/0.25.0도 인덱스에는 있으나
검증하지 않았다 — 필요하면 `pyproject.toml`의 버전만 바꿔 다시 시도해 본다.)

### numpy를 2.x 대신 `<2`로 고정한 이유

torch==2.8.0 wheel은 NumPy 1.x ABI로 빌드돼 있어서, NumPy 2.x와 같이 쓰면 import
시점에 다음 경고가 뜨고 최악의 경우 크래시할 수 있다.

```text
UserWarning: Failed to initialize NumPy: A module that was compiled using
NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
```

`numpy>=1.26,<2`로 고정하면 이 경고 없이 깨끗하게 import된다.

## TensorRT (시스템 전역 설치)

TensorRT는 JetPack이 apt로 설치하며(`libnvinfer10`, `python3-libnvinfer` 등), Python
바인딩은 시스템 Python 3.10의 `dist-packages`에만 있다. PyPI에도, Jetson AI Lab
인덱스에도 이 보드용 `tensorrt` wheel은 올라와 있지 않아 `uv add`로 받을 수 없다.

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"   # 시스템 python3: 10.3.0
```

그래서 [설치](#설치)에서 `.venv`를 `uv venv --system-site-packages`로 만든다 — 이
옵션은 `uv sync`가 아니라 venv 생성 시점의 플래그라서 `uv sync` 한 줄로는 켤 수 없다.
시스템 쪽 패키지를 이 `.venv`에 *추가로* 보이게 할 뿐, `.venv`에 이미 설치된 torch
2.8.0 등을 시스템 버전으로 덮어쓰지는 않는다 (venv 쪽이 항상 우선). 실제로 이렇게
만든 `.venv`에서 torch와 tensorrt를 함께 import해 충돌이 없음을 확인했다.

```bash
uv run python -c "import torch, tensorrt; print(torch.__version__, torch.cuda.is_available(), tensorrt.__version__)"
# 2.8.0 True 10.3.0
```

TensorRT 엔진 실행에 흔히 쓰이는 `pycuda`/`cuda-python` 같은 GPU 버퍼 관리 라이브러리는
추가하지 않았다. `common/inference.py`는 이미 있는 torch CUDA 텐서를
`tensor.data_ptr()`로 바로 TensorRT에 바인딩해서 쓰므로 별도 의존성이 필요 없다.

## 모델 변환

```bash
./run convert-model
./run convert-model --precision fp32
./run convert-model --checkpoint ../../../baseline/yolov8sclone/data/model/erop/tiponly/model.pt
```

`baseline/yolov8sclone/data/model/<dataset>/<label-set>/model.pt`(기본값:
`cholec80/tooltip`)를 읽어 `data/model/<dataset>/<label-set>/`에 세 파일을 만든다.

| 파일              | 내용                                                                        |
| ----------------- | --------------------------------------------------------------------------- |
| `model.onnx`      | 모델 forward + `common.model.decode`를 하나로 합친 그래프 (중간 산출물)     |
| `model.engine`    | `trtexec`가 이 보드용으로 빌드한 TensorRT 엔진                              |
| `model-info.json` | `class_names`·`dataset`·`tip_box_size`·`epoch`·`metrics`·정밀도 등 사이드카 |

NMS는 **엔진 안에 넣지 않았다.** 살아남는 박스 수가 매 프레임 달라지는 연산이라
고정 shape인 TensorRT 엔진과 맞지 않고, `yolov8sclone.common.boxes.non_max_suppression`을
그대로 재사용해 Python에서 처리하는 쪽이 검증 시간도 짧다 — 자세한 트레이드오프는
`scripts/convert-model.py`의 모듈 docstring 참고.

cholec80/tooltip 체크포인트로 실측한 값이다 (Jetson Orin Nano, fp16):

| 항목                       | 값                                                                                                        |
| -------------------------- | --------------------------------------------------------------------------------------------------------- |
| 빌드 시간                  | 약 715초 (trtexec, fp16 타이밍 탐색 포함)                                                                 |
| 엔진 크기                  | 24.4 MiB                                                                                                  |
| 출력 shape                 | `1 x 8400 x 6` (640² 입력, 클래스 2개)                                                                    |
| trtexec 자체 벤치마크      | 평균 9.8 ms/frame (GPU 컴퓨트 기준)                                                                       |
| `Detector.detect()` 종단간 | 평균 약 27 ms/frame (letterbox + 엔진 + NMS + 좌표 역변환 포함, 첫 호출은 CUDA 컨텍스트 초기화로 더 걸림) |

## 데모 GUI

```bash
./run demo
./run demo --dataset erop
```

`baseline/yolov8sclone/scripts/demo.py`를 그대로 옮긴 GUI다 (레이아웃·조작은
`baseline/yolov8s/scripts/demo.py`의 관례를 따른다). 다른 점은 추론 백엔드뿐이다:
`common/inference.py`의 `Detector`가 `.pt` 대신 `scripts/convert-model.py`가 만든
`model.engine` + `model-info.json`을 읽는다. 예제 영상/프레임은
`data/dataset/<dataset>/images/<split>/`(저장소 루트의 공유 데이터셋)에서 읽는다 —
`--frames-root`로 위치를 바꿀 수 있다.

이 보드(DISPLAY 있음)에서 실제로 띄워 확인했다: `cholec80/test` 프레임에서 tool 2개·
tip 2개가 GT 원 위치와 겹치게 탐지되고, 정보 패널에 프레임당 추론 시간과 예측 팁
좌표가 표시된다.

## 디렉터리 구조

```text
standalone/jetson/yolov8s/
├── pyproject.toml            # 독립 uv 프로젝트 정의 (Jetson AI Lab 인덱스, 버전 고정 이유는 주석 참고)
├── uv.lock
├── .python-version           # 3.10 (시스템 Python과 동일, jp6/cu126 wheel이 cp310 전용이라 고정)
├── run                       # `run <script> [args...]` → scripts/<script>.py 실행 (run.bat 없음, Jetson 전용)
├── common/
│   ├── model.py               # YOLOv8s 아키텍처 (baseline/yolov8sclone/common/model.py 사본, 변경 금지)
│   ├── boxes.py                # box 변환, letterbox, NMS (동일 프로젝트의 사본)
│   ├── draw.py                 # 예측·GT 오버레이 (동일 프로젝트의 사본)
│   ├── sources.py              # 데모용 영상/프레임 소스 (baseline/yolov8s의 사본, repo_root() 깊이만 조정)
│   └── inference.py            # TensorRT 엔진 로드 + model-info.json 사이드카, 프레임 1장 추론
├── scripts/
│   ├── convert-model.py        # model.pt → model.onnx → model.engine
│   └── demo.py                  # 탐지 결과 시각화 GUI
├── data/
│   └── model/<dataset>/<label-set>/   # model.onnx / model.engine / model-info.json (git 추적 제외)
└── .venv/                     # git 추적 제외, --system-site-packages로 생성 (위 "TensorRT" 참고)
```

## 한계

- `erop`/`tiponly` 조합은 변환 스크립트가 인자로는 받지만 실제로 변환·데모를 돌려
  확인한 것은 `cholec80/tooltip` 하나뿐이다.
- torch 2.9.1/2.10.0 조합은 인덱스에 존재하지만 이 보드에서 실제로 import되는지
  확인하지 않았다 (2.11.0은 실패, 2.8.0은 성공만 확인함).
- INT8 양자화는 다루지 않았다 — 보정(calibration) 데이터셋이 필요해 별도 작업이다.
