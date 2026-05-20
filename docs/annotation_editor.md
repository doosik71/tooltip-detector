# `scripts/annotation_editor.py`

`scripts/annotation_editor.py`는 `images`, `segmentation`, `annotation` 데이터를 함께 보여 주고, 수술도구의 bounding box와 tip 좌표를 수동으로 수정할 수 있는 GUI 편집기다.

기본 경로는 다음과 같다.

- 이미지 입력: `./data/dataset/images/{train,val,test}`
- segmentation 입력: `./data/dataset/segmentation/{train,val,test}`
- annotation 입출력: `./data/dataset/annotation/{train,val,test}`

## 사용자 문서

### 목적

자동 생성된 annotation JSON을 사람이 검토하고 수정하기 위한 도구다. segmentation에서 누락된 수술도구를 추가하거나, 잘못 생성된 annotation을 삭제하거나, bbox와 tip 위치를 미세 조정할 수 있다.

### 실행 전 요구 사항

- Python `3.12` 이상
- `opencv-python`
- `numpy`
- `tkinter` 사용 가능 환경
- `./data/dataset/images` 디렉터리 존재
- `./data/dataset/segmentation` 디렉터리 존재

`tkinter`는 일반적인 데스크톱 Python 환경에 기본 포함되는 경우가 많지만, 일부 최소 설치 환경에서는 별도 패키지가 필요할 수 있다.

### 실행 방법

프로젝트 루트에서 실행한다.

```bash
python -m scripts.annotation_editor
```

가상환경을 직접 사용할 경우:

```bash
.venv/bin/python -m scripts.annotation_editor
```

경로를 직접 지정할 수도 있다.

```bash
python -m scripts.annotation_editor \
  --images ./data/dataset/images \
  --segmentation ./data/dataset/segmentation \
  --annotation ./data/dataset/annotation
```

### 화면 구성

#### 상단 툴바

- `Split`
  - `train`, `val`, `test` 중 하나를 선택한다.
- `Add Tool`
  - 새 수술도구 annotation 추가 모드로 들어간다.
- `Delete Tool`
  - 현재 선택된 annotation을 삭제한다.
- `Save`
  - 현재 이미지의 annotation JSON을 저장한다.
- `Show Segmentation`
  - segmentation overlay 표시 여부를 전환한다.

#### 좌측 패널

선택된 split 안의 이미지 목록을 표시한다. 목록에서 이미지를 클릭하면 우측 메인 화면에 해당 이미지가 로드된다.

#### 우측 메인 화면

다음 요소가 함께 표시된다.

- 원본 이미지
- segmentation mask 반투명 overlay
- annotation bbox
- annotation tip

### 편집 방법

#### bbox 이동

bbox 내부를 마우스로 드래그하면 전체 박스를 이동할 수 있다. tip도 함께 같은 만큼 이동한다.

#### bbox 크기 조정

선택된 bbox의 네 모서리 핸들을 드래그하면 박스 크기를 조정할 수 있다.

#### tip 이동

빨간색 tip 핸들을 드래그하면 tip 좌표를 직접 조정할 수 있다.

#### 새 수술도구 추가

1. `Add Tool` 버튼 클릭
2. 이미지 위에서 마우스를 드래그해 bbox 생성
3. 생성된 bbox가 새 annotation으로 추가됨
4. 필요하면 tip을 추가로 드래그해 조정

`Esc` 키로 add mode를 취소할 수 있다.

#### 잘못된 annotation 삭제

1. 삭제할 annotation 선택
2. `Delete Tool` 버튼 클릭 또는 `Delete` 키 입력

### 저장 동작

annotation 정보가 바뀌면 `Save` 버튼이 활성화된다.

다음 두 방법으로 저장할 수 있다.

- `Save` 버튼 클릭
- `Ctrl+S`

이미지나 split을 바꾸거나 창을 닫을 때 저장되지 않은 변경이 있으면 저장 여부를 묻는다.

### 저장 파일 형식

각 이미지에 대해 같은 basename의 JSON 파일을 저장한다.

예시:

```text
이미지:      ./data/dataset/images/train/sample01_00000030.png
annotation: ./data/dataset/annotation/train/sample01_00000030.json
```

JSON 구조는 다음과 같다.

```json
{
  "image": "sample01_00000030.png",
  "width": 736,
  "height": 480,
  "annotations": [
    {
      "bbox": {
        "x": 120,
        "y": 90,
        "width": 80,
        "height": 40
      },
      "tip": {
        "x": 145,
        "y": 102
      }
    }
  ]
}
```

### 문제 해결

#### 창이 열리지 않는 경우

GUI 환경이 없는 터미널 세션일 수 있다. 로컬 데스크톱 세션이나 X11/Wayland가 활성화된 환경에서 실행해야 한다.

#### 이미지가 보이는데 segmentation이 안 보이는 경우

다음 항목을 확인하면 된다.

- 같은 basename의 segmentation PNG가 존재하는지 확인
- 경로가 `./data/dataset/segmentation/{split}`인지 확인
- `Show Segmentation` 체크가 켜져 있는지 확인

#### annotation 파일이 없는데 편집은 가능한가

가능하다. 기존 JSON이 없으면 빈 annotation 상태로 열리고, 저장 시 새 JSON 파일이 생성된다.

## 개발 문서

### 입력/출력 경로 모델

편집기는 세 개의 루트 디렉터리를 사용한다.

- `--images`
  - 원본 RGB 이미지 루트
- `--segmentation`
  - binary mask 루트
- `--annotation`
  - JSON 저장 루트

각 루트 아래에는 모두 `train`, `val`, `test` 디렉터리가 있다고 가정한다.

### 내부 상태

`AnnotationEditor` 클래스는 현재 이미지와 편집 상태를 메모리에 유지한다.

주요 상태는 다음과 같다.

- `current_image_rgb`
  - 현재 원본 이미지
- `current_segmentation_mask`
  - 현재 segmentation mask
- `current_annotations`
  - 현재 편집 중인 annotation 목록
- `selected_annotation_index`
  - 현재 선택된 annotation 인덱스
- `dirty`
  - 저장되지 않은 변경 여부
- `mode`
  - `select` 또는 `add`

### 표시 방식

원본 이미지는 `Canvas`에 표시되고, segmentation은 OpenCV로 반투명 합성한 뒤 렌더링된다.

annotation overlay는 Canvas 도형으로 따로 그린다.

- bbox: 사각형
- tip: 빨간 원형 핸들
- 선택된 bbox: 강조 색상과 코너 핸들 표시

### 좌표계 처리

편집기 내부에는 두 좌표계가 있다.

- 이미지 좌표계
- Canvas 표시 좌표계

창 크기에 따라 이미지는 확대/축소되어 표시되므로, 마우스 이벤트는 `canvas -> image` 좌표 변환 후 처리한다.

### 편집 동작 설계

#### 이동

bbox 내부 hit 시 `move` 모드가 활성화된다. 드래그 중 bbox와 tip이 함께 이동한다.

#### 리사이즈

선택된 bbox의 모서리 hit 시 `resize` 모드가 활성화된다. 드래그 중 bbox만 갱신되고, tip은 화면 경계 안으로 clamp된다.

#### tip 수정

tip 핸들 hit 시 `tip` 모드가 활성화된다. 드래그 중 tip 좌표만 바뀐다.

#### 추가

`Add Tool` 후 드래그하면 preview bbox를 그리고, mouse release 시 새 annotation을 생성한다.

### 저장 정책

저장 시 현재 메모리 상태를 그대로 JSON으로 직렬화한다. 자동 저장은 하지 않는다.

### 현재 설계 제약

- undo/redo가 없다.
- 다각형 contour 편집은 지원하지 않는다.
- tip 자동 재계산 기능은 없다.
- segmentation overlay 색상과 알파 값은 코드 상수로 고정되어 있다.
- 매우 큰 데이터셋에서는 좌측 이미지 목록 필터링 기능이 없다.

### 확장 권장 사항

1. undo/redo 추가
2. 키보드 단축키 확장
3. annotation 목록 패널 추가
4. tip 자동 snap 또는 contour 기반 재계산 기능 추가
5. 이미지 이동 버튼과 검색 기능 추가
6. 변경 이력 또는 작업 로그 저장 기능 추가
