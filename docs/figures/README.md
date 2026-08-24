# docs/figures

`docs/final-report.md`에 삽입되는 그림들이다. 각 그림은 **SVG가 원본**이고, 문서는 어디서나
렌더링되도록 같은 이름의 PNG를 링크한다.

그림은 두 종류다. **작도** 그림은 SVG를 손으로 편집하고, **데이터** 그림은
[`notebook/final-report-graph.ipynb`](../../notebook/final-report-graph.ipynb)가 실험 산출물에서
다시 계산해 그린다.

| 파일                           | 종류     | 참조 위치                                                      |
| ------------------------------ | -------- | -------------------------------------------------------------- |
| `annotation-pipeline`          | 작도     | 그림 1 — §2.3 반자동 어노테이션 파이프라인                     |
| `gradient-seg-target`          | 작도     | 그림 2 — §2.6 `gradient-seg` 타겟 생성 과정                    |
| `model-architecture`           | 작도     | 그림 3 — §3.1 `TooltipDetector` 구조와 처리 흐름               |
| `val-loss-curves`              | 데이터   | 그림 4 — §5 에포크별 검증 손실 (표 9)                          |
| `overall-performance`          | 데이터   | 그림 5 — §6.2 8개 조합 전체 성능 (표 11)                       |
| `error-distribution-all`       | 데이터   | 그림 6 — §6.3 오차 거리 분포 (표 14)                           |
| `error-distribution-erop-mini` | 작도     | 그림 7 — §6.3 오차 거리 분포 비교 (표 14의 2·4행)              |
| `target-error-mechanism`       | 작도     | 그림 8 — §6.3 초과-임계 영역 트레이드오프                      |
| `match-type-error`             | 작도     | 그림 9 — §6.7.2 매칭 유형별 오차 분해                          |
| `hungarian-fn-by-tool-count`   | 데이터   | 그림 10 — §6.7.4 도구 수별 헝가리안 FN율 (표 24-2)             |
| `tooltip-detector`             | 스크린샷 | 그림 11 — §7 탐지 GUI 화면                                     |
| `split-unit-comparison`        | 데이터   | 그림 12 — §11.3 분할 방식별 성능 변화 (표 29·30)               |
| `phase2-val-grid`              | 작도     | `parameter-optimization.md` 그림 1 — 2-1 val 격자 탐색         |
| `phase2-test-gain`             | 작도     | `parameter-optimization.md` 그림 2 — 2-1 선택 설정의 test 성능 |
| `phase2-watershed`             | 작도     | `parameter-optimization.md` 그림 3 — 2-2 피크 분할 방식 비교   |

`phase2-*` 세 그림은 `docs/parameter-optimization.md`에 삽입되며, 수치는
`data/results/phase2/`의 결과 파일에서 그대로 옮긴 것이다.

## 수정 방법

**작도 그림** — SVG를 편집한 뒤 PNG를 다시 내보낸다. PNG는 2배 해상도(192 dpi)로 생성한다.

```bash
inkscape --export-type=png --export-dpi=192 \
  --export-filename=docs/figures/<name>.png docs/figures/<name>.svg
```

**데이터 그림** — 노트북을 다시 실행하면 같은 이름의 SVG·PNG를 덮어쓴다. 보고서의 수치를 옮겨
적지 않고 `data/models/<...>/metric.csv`, `data/results/<...>/summary.json`,
`data/results/<...>/per_tip.csv`에서 매번 다시 계산하므로, 재학습·재평가 뒤에는
`run generate-summary`와 함께 이 노트북도 다시 실행한다.

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebook/final-report-graph.ipynb
```

한글 텍스트는 `Noto Sans CJK KR` / `NanumGothic`, 코드·식별자는 `D2Coding`을 쓴다.
색 규약은 `gaussian-tip` = 파랑 `#2f6f9f`, `gradient-seg` = 주황 `#c4763a`,
단독 매칭 = 초록 `#3f8468`, 공유 매칭 = 보라 `#75619e`, 실패·미탐지 = 빨강 `#ac4b45`다.
