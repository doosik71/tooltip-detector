# 모델 구조 설명서: ttd/model/monai.py

`ttd/model/monai.py`에 정의된 `TooltipDetector`의 아키텍처를 설명한다.
MONAI 프레임워크의 사전학습 가중치와 호환되도록 모듈 명칭을 설계했다.

## 전체 구조

```text
입력 (B, 3, 480, 736)
    │
    ▼
EfficientNetB2Encoder
    │  스킵 연결 5개: s0, s1, s2, s4, s6
    ▼
UNetDecoder
    │
    ▼
segmentation_head: Conv2d(16 → 2, 3×3)
    │
    ▼
출력 (B, 2, 480, 736)
    ├─ 채널 0: 배경 로짓
    └─ 채널 1: 도구 영역 로짓 (sigmoid → 거리 기반 히트맵)
```

## 인코더: EfficientNetB2Encoder

EfficientNet-B2 구조를 재현한다 (width_coeff=1.1, depth_coeff=1.2).

### 스템(Stem)

| 레이어       | 입력 채널       | 출력 채널 | 커널 | 스트라이드 |
| ------------ | --------------- | --------- | ---- | ---------- |
| `_conv_stem` | 3               | 32        | 3×3  | 2          |
| `_bn0`       | BatchNorm2d(32) |

출력 공간 해상도: H/2 × W/2

### MBConv 블록(MBConvBlock)

Mobile Inverted Bottleneck Conv 블록. Squeeze-and-Excitation(SE) 주의 메커니즘을 포함한다.

```text
입력 x
  │
  ├─ [expand_ratio ≠ 1일 때] Expand Conv: 1×1, in_ch → mid_ch
  │
  ├─ Depthwise Conv: k×k, mid_ch, groups=mid_ch
  │
  ├─ SE: AdaptiveAvgPool → FC(mid→se) → SiLU → FC(se→mid) → Sigmoid → 채널별 곱
  │
  ├─ Project Conv: 1×1, mid_ch → out_ch
  │
  └─ [stride=1 & in==out] 잔차 연결: x + h
```

- `expand_ratio=1`인 경우 Expand Conv 없이 Depthwise Conv로 직행
- 활성화 함수: SiLU (Swish)
- SE 채널 수: `max(1, int(in_ch × 0.25))`

### 스테이지(Stage) 구성

7개 스테이지, 총 23개 MBConv 블록.

| 스테이지 | 입력 채널 | 출력 채널 | 커널 | 스트라이드 | 블록 수 | 공간 해상도 |
| -------- | --------- | --------- | ---- | ---------- | ------- | ----------- |
| 0        | 32        | 16        | 3×3  | 1          | 2       | H/2         |
| 1        | 16        | 24        | 3×3  | 2          | 3       | H/4         |
| 2        | 24        | 48        | 5×5  | 2          | 3       | H/8         |
| 3        | 48        | 88        | 3×3  | 2          | 4       | H/16        |
| 4        | 88        | 120       | 5×5  | 1          | 4       | H/16        |
| 5        | 120       | 208       | 5×5  | 2          | 5       | H/32        |
| 6        | 208       | 352       | 3×3  | 1          | 2       | H/32        |

각 스테이지의 첫 번째 블록만 지정된 스트라이드를 적용하고, 나머지 블록은 stride=1이다.
스테이지 내 두 번째 블록부터는 `in_channels = out_channels`이므로 잔차 연결이 활성화된다.

스테이지 모듈은 `_BlockStage`로 감싸며, 블록의 키는 전역 순서 번호(0~22)를 사용한다.
이는 MONAI state_dict의 `encoder._blocks.<stage_idx>.<global_block_idx>.*` 패턴과 일치한다.

### 스킵 연결 추출 지점

`extract_features(x)`는 5개 스테이지 출력을 반환한다.

| 인덱스 | 스테이지 | 채널 수 | 공간 해상도 |
| ------ | -------- | ------- | ----------- |
| s0     | 0        | 16      | H/2         |
| s1     | 1        | 24      | H/4         |
| s2     | 2        | 48      | H/8         |
| s4     | 4        | 120     | H/16        |
| s6     | 6        | 352     | H/32        |

스테이지 3과 5는 스킵 연결에서 제외된다.

## 디코더: UNetDecoder

인코더의 `extract_features` 출력을 받아 5단계로 업샘플링한다.

### 디코더 블록(_DecoderBlock)

각 블록은 다음 순서로 동작한다.

```text
입력 x
  │
  ├─ Bilinear 업샘플링 ×2
  │
  ├─ [skip ≠ None] skip 텐서와 채널 방향 concat
  │
  └─ _DecoderConvs: ConvBnBlock × 2
```

`_DecoderConvs`는 `_ConvBnBlock` 두 개를 직렬 연결한다.
`_ConvBnBlock`은 Conv2d(3×3, padding=1) + BatchNorm2d + ReLU 구조이며,
MONAI 호환 명칭(`conv`, `adn.N`)을 사용한다.

### 5단계 디코더 블록 구성

| 블록 | 업샘플 입력 | 스킵 채널 | concat 후 입력 | 출력 채널 | 공간 해상도 |
| ---- | ----------- | --------- | -------------- | --------- | ----------- |
| 0    | s6: 352     | s4: 120   | 472            | 256       | H/16        |
| 1    | 256         | s2:  48   | 304            | 128       | H/8         |
| 2    | 128         | s1:  24   | 152            | 64        | H/4         |
| 3    | 64          | s0:  16   | 80             | 32        | H/2         |
| 4    | 32          | (없음)    | 32             | 16        | H           |

블록 4는 원본 해상도로 복원하며 스킵 연결 없이 업샘플링만 수행한다.

## 분할 헤드(Segmentation Head)

```python
nn.Conv2d(16, num_classes=2, kernel_size=3, padding=1)
```

| 입력 채널 | 출력 채널 | 커널 | 출력 해상도 |
| --------- | --------- | ---- | ----------- |
| 16        | 2         | 3×3  | H × W       |

## 학습 손실

`채널 1`에 sigmoid를 적용한 값과 거리 기반 히트맵 타겟 사이의 MSE를 최소화한다.

```python
loss = MSE(sigmoid(output[:, 1]), heatmap_target)
```

## MONAI 가중치 호환성

`temp/monai.pt`는 MONAI `FlexibleUNet(backbone="efficientnet-b2")` 구조로 학습된 570개 키의 state_dict다.
이 모델은 MONAI state_dict를 `load_state_dict(strict=True)`로 직접 로드할 수 있도록 모듈 명칭을 맞췄다.

| 모듈 경로 예시                         | 대응 MONAI 키                         |
| -------------------------------------- | ------------------------------------- |
| `encoder._conv_stem`                   | `encoder._conv_stem`                  |
| `encoder._blocks[i].j._expand_conv`    | `encoder._blocks.i.j._expand_conv`    |
| `decoder.blocks[k].convs.conv_0.conv`  | `decoder.blocks.k.convs.conv_0.conv`  |
| `decoder.blocks[k].convs.conv_0.adn.N` | `decoder.blocks.k.convs.conv_0.adn.N` |

## 입출력 형태 요약

| 단계         | 텐서 형태         |
| ------------ | ----------------- |
| 입력         | (B, 3, 480, 736)  |
| 스템 출력    | (B, 32, 240, 368) |
| s0           | (B, 16, 240, 368) |
| s1           | (B, 24, 120, 184) |
| s2           | (B, 48, 60, 92)   |
| s4           | (B, 120, 30, 46)  |
| s6           | (B, 352, 15, 23)  |
| 디코더 블록0 | (B, 256, 30, 46)  |
| 디코더 블록1 | (B, 128, 60, 92)  |
| 디코더 블록2 | (B, 64, 120, 184) |
| 디코더 블록3 | (B, 32, 240, 368) |
| 디코더 블록4 | (B, 16, 480, 736) |
| 출력         | (B, 2, 480, 736)  |
