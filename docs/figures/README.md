# docs/figures

`docs/final-report.md`에 삽입되는 그림들이다. 각 그림은 **SVG가 원본**이고, 문서는 어디서나
렌더링되도록 같은 이름의 PNG를 링크한다.

| 파일 | 참조 위치 |
| ---- | --------- |
| `annotation-pipeline` | 그림 1 — §2.3 반자동 어노테이션 파이프라인 |
| `gradient-seg-target` | 그림 2 — §2.6 `gradient-seg` 타겟 생성 과정 |
| `model-architecture` | 그림 3 — §3.1 `TooltipDetector` 구조와 처리 흐름 |
| `error-distribution-erop-mini` | 그림 4 — §6.3 오차 거리 분포 비교 |
| `target-error-mechanism` | 그림 5 — §6.3 초과-임계 영역 트레이드오프 |
| `match-type-error` | 그림 6 — §6.7.2 매칭 유형별 오차 분해 |

## 수정 방법

SVG를 편집한 뒤 PNG를 다시 내보낸다. PNG는 2배 해상도(192 dpi)로 생성한다.

```bash
inkscape --export-type=png --export-dpi=192 \
  --export-filename=docs/figures/<name>.png docs/figures/<name>.svg
```

한글 텍스트는 `Noto Sans CJK KR` / `NanumGothic`, 코드·식별자는 `D2Coding`을 쓴다.
색 규약은 `gaussian-tip` = 파랑 `#2f6f9f`, `gradient-seg` = 주황 `#c4763a`,
단독 매칭 = 초록 `#3f8468`, 공유 매칭 = 보라 `#75619e`, 실패·미탐지 = 빨강 `#ac4b45`다.
