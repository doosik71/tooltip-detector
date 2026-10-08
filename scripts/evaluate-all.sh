#!/usr/bin/env bash

# docs/final-report.md에 인용되는 평가 결과 전체를 한 번에 다시 산출한다.
#
# 학습은 다시 하지 않는다. data/models/ 아래의 기존 체크포인트(best.pt)로 val 편차
# 추정과 test 평가, 후처리 탐색, 요약 문서·그림 생성만 다시 실행한다. 평가 대상은
# eval-model.py의 기본 규칙을 따른다. 즉 도구가 4개 이상 레이블링된 프레임은 대부분
# 반자동 라벨링의 오검출이므로 평가와 편차 추정에서 제외한다(--max-tools 3,
# final-report.md §2.7).
#
# 실행 단계
#   1. 구 프레임 분할 test 프레임 목록 확인
#      cholec80 구 프레임 분할(§11.3)의 데이터셋 디렉터리는 남아 있지 않지만, 그 test
#      프레임은 모두 현재 cholec80의 세 스플릿에 같은 어노테이션으로 들어 있다. 그 목록
#      (data/results/cholec80-frame-split/test-frames.txt)으로 구 분할 test를 복원한다.
#   2. val 편차 추정과 test 평가 (12개 조합)
#      - 표준 8개 조합 (데이터셋 2 × 타겟 2 × 모델 2, final-report.md §6)
#      - cholec80 구 프레임 분할 4개 조합 (§11.3)
#      세 개의 작업 큐가 cuda:1, cuda:2, cuda:3에서 병렬로 실행된다.
#   3. 2단계 후처리 탐색 노트북 실행 (threshold × NMS val 격자 탐색, 선택 설정의 test
#      확인, watershed 비교, §11.1). 노트북 안의 GPU 배정도 cuda:1〜3만 쓴다.
#   4. docs/results-summary.md와 보고서 그림(docs/figures/) 재생성
#
# 덮어쓰는 파일
#   - data/results/<데이터셋>/<타겟>/<모델>/summary.json, per_tip.csv
#   - data/results/cholec80-frame-split/<타겟>/<모델>/summary.json, per_tip.csv
#   - data/results/phase2/ 아래의 val 격자 탐색·test·watershed 결과
#   - data/models/<...>/bias.json (이미 bias.json이 있는 조합만 다시 추정)
#   - docs/results-summary.md, docs/figures/의 데이터 그림, 두 노트북의 실행 결과
#   이전 결과를 남겨야 하면 실행 전에 직접 복사해 둔다.
#
# GPU
#   cuda:0은 PCIe 링크가 저하되어 있어 쓰지 않는다. 학습·평가는 cuda:1〜3에서만 한다.
#
# 소요 시간
#   2026-10-06 실행 기준으로 2단계가 약 1시간 10분, 3단계가 약 3시간 걸렸다.
#
# 사용법 (어느 디렉터리에서 실행해도 된다)
#   bash scripts/evaluate-all.sh
#
# 로그
#   temp/evaluation-logs/<조합>-bias.log, <조합>-test.log, 노트북·요약 생성 로그.
#   어느 평가든 실패하면 해당 큐가 멈추고, 2단계가 끝난 뒤 스크립트가 실패로 종료한다.
set -euo pipefail

# 저장소 루트로 이동한다. 이하의 경로는 모두 루트 기준 상대 경로다.
cd "$(dirname "$0")/.."
LOG_DIR=temp/evaluation-logs
mkdir -p "$LOG_DIR"

# 모든 평가에 공통인 명령. eval-model.py의 기본값은 전체 프레임 평가(--max-tools 0)
# 이므로, 보고서 4.0판의 평가 규칙을 재현하려면 --max-tools 3을 반드시 명시한다.
EVAL=(uv run python scripts/eval-model.py --max-tools 3 --batch-size 16 --workers 8)

# ── 1. 구 프레임 분할 test 프레임 목록 ──────────────────────────────────────
# 목록은 도구 4개 이상 프레임까지 포함한 원래 test 프레임 33,896개여야 한다. 그래야
# eval-model.py가 그중 1,652개를 직접 제외하고 summary.json에 제외 수를 기록한다.
# 목록 파일이 없으면 구 분할 결과의 per_tip.csv에서 다시 만든다. 다만 그 per_tip.csv가
# 이미 도구 4개 이상 프레임을 제외하고 산출된 것이면 목록에서도 그 프레임이 빠지므로,
# 평가 결과는 같지만 summary.json의 제외 프레임 수가 0으로 기록된다.
FRAME_LIST=data/results/cholec80-frame-split/test-frames.txt
if [[ ! -f $FRAME_LIST ]]; then
    echo "경고: $FRAME_LIST 이 없어 per_tip.csv에서 다시 만든다 (제외 프레임 수가 0으로 기록될 수 있다)." >&2
    tail -n +2 data/results/cholec80-frame-split/gradient-seg/monai/per_tip.csv \
        | cut -d, -f1 | sort -u > "$FRAME_LIST"
fi
echo "구 프레임 분할 test 프레임: $(wc -l < "$FRAME_LIST")개"

# ── 2. val 편차 추정과 test 평가 ─────────────────────────────────────────────
# standard <데이터셋> <타겟> <모델> <GPU> <편차 적용>
#   표준 조합 하나를 평가한다.
#   - 체크포인트 옆에 bias.json이 이미 있으면 먼저 val에서 편차를 다시 추정해 덮어쓴다
#     (--estimate-bias). 편차 추정은 val의 20 px 이내 근거리 매칭만 쓴다.
#   - <편차 적용>이 apply이면 test에서 원본 좌표와 편차 보정 좌표를 함께 평가한다
#     (--apply-bias, erop 4개 조합). summary.json의 bias_correction에 보정 지표가 들어간다.
#   - none이면 원본 좌표만 평가한다(cholec80 4개 조합, 3.0판 이후의 표준 결과와 같은
#     조건). 이 경우에도 bias.json이 있으면 다시 추정하는데, 3단계 노트북의 선택 설정
#     test가 그 값을 쓰기 때문이다.
standard() {
    local dataset=$1 target=$2 model=$3 device=$4 bias=$5
    local name="$dataset-$target-$model"
    local common=(--dataset "$dataset" --target-mode "$target" --model-type "$model" --device "$device")
    if [[ -f data/models/$dataset/$target/$model/bias.json ]]; then
        "${EVAL[@]}" "${common[@]}" --estimate-bias > "$LOG_DIR/$name-bias.log" 2>&1
    fi
    local apply=()
    [[ $bias == apply ]] && apply=(--apply-bias)
    "${EVAL[@]}" "${common[@]}" "${apply[@]}" > "$LOG_DIR/$name-test.log" 2>&1
    echo "완료: $name ($device)"
}

# frame_split <타겟> <모델> <GPU>
#   cholec80 구 프레임 분할 체크포인트 하나를 구 분할 test 프레임 목록으로 평가한다.
#   - --frame-list: test 스플릿 대신 목록의 프레임을 세 스플릿 전체에서 찾아 평가한다.
#   - --results-dir: 표준 cholec80 결과를 덮어쓰지 않도록 구 분할 결과 경로에 저장한다.
#   - 구 분할의 val 세트는 다시 만들 수 없으므로(test 프레임만 기록이 남았다) 편차는
#     다시 추정하지 않고 2026-08-20에 추정한 bias.json을 그대로 적용한다. 편차는 보정
#     지표에만 영향을 주고 §11.3이 비교하는 원본 지표와는 무관하다.
frame_split() {
    local target=$1 model=$2 device=$3
    local name="cholec80-frame-split-$target-$model"
    "${EVAL[@]}" --dataset cholec80 --target-mode "$target" --model-type "$model" --device "$device" \
        --model "data/models/cholec80-frame-split/$target/$model/best.pt" \
        --frame-list "$FRAME_LIST" \
        --results-dir "data/results/cholec80-frame-split/$target/$model" \
        --apply-bias > "$LOG_DIR/$name-test.log" 2>&1
    echo "완료: $name ($device)"
}

# GPU 하나당 큐 하나. 큐 안의 작업은 순서대로 실행되고, 세 큐는 동시에 돈다.
# test 프레임 수(erop 약 2.7만, cholec80 약 8.6만, 구 분할 약 3.2만)를 보고 세 큐의
# 총 작업량이 비슷하도록 나눴다.
queue_1() {
    standard erop     gradient-seg monai      cuda:1 apply
    standard erop     gaussian-tip monai      cuda:1 apply
    standard cholec80 gradient-seg monai      cuda:1 none
    frame_split       gradient-seg monai      cuda:1
}
queue_2() {
    standard erop     gradient-seg monai_mini cuda:2 apply
    standard erop     gaussian-tip monai_mini cuda:2 apply
    standard cholec80 gaussian-tip monai      cuda:2 none
    frame_split       gradient-seg monai_mini cuda:2
}
queue_3() {
    standard cholec80 gradient-seg monai_mini cuda:3 none
    standard cholec80 gaussian-tip monai_mini cuda:3 none
    frame_split       gaussian-tip monai      cuda:3
    frame_split       gaussian-tip monai_mini cuda:3
}

echo "[$(date '+%F %T')] 2단계: cuda:1〜3에서 편차 추정과 test 평가"
queue_1 & pid_1=$!
queue_2 & pid_2=$!
queue_3 & pid_3=$!
# 세 큐가 모두 끝날 때까지 기다린 뒤, 하나라도 실패했으면 이후 단계를 건너뛰고 종료한다.
# 3·4단계는 2단계의 결과(bias.json, summary.json)를 읽으므로 반쯤 갱신된 결과로
# 진행하면 안 된다.
status=0
for pid in $pid_1 $pid_2 $pid_3; do
    wait "$pid" || status=1
done
if (( status )); then
    echo "평가가 실패했다. $LOG_DIR/*.log 를 확인하라." >&2
    exit 1
fi

# ── 3. 2단계 후처리 탐색 노트북 (§11.1) ────────────────────────────────────
# monai_mini 4개 조합(데이터셋 × 타겟)에 대해 val에서 threshold × NMS 25개 설정을
# 헝가리안 F1@50으로 탐색하고, 선택된 설정 하나만 test에서 확인한 뒤, 같은 설정에서
# watershed 피크 분할과 비교한다. 결과는 data/results/phase2/에 덮어쓴다.
# 노트북 실행 결과(셀 출력)도 노트북 파일에 그대로 저장된다(--inplace).
echo "[$(date '+%F %T')] 3단계: 후처리 탐색 노트북"
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebook/parameter-optimization.ipynb \
    > "$LOG_DIR/parameter-optimization.log" 2>&1

# ── 4. 요약 문서와 그림 ─────────────────────────────────────────────────────
# generate-summary.py는 data/models·data/results만 읽어 docs/results-summary.md를
# 만든다. final-report-graph.ipynb는 같은 파일들로 보고서의 데이터 그림(그림 4·5·6·10·12)을
# docs/figures/에 다시 그린다. 손으로 그린 그림(그림 7·9 등)은 갱신되지 않으므로
# 수치가 바뀌면 SVG를 직접 고쳐야 한다(docs/figures/README.md).
echo "[$(date '+%F %T')] 4단계: 결과 요약 문서와 그림"
uv run python scripts/generate-summary.py > "$LOG_DIR/generate-summary.log" 2>&1
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebook/final-report-graph.ipynb \
    > "$LOG_DIR/final-report-graph.log" 2>&1

echo "[$(date '+%F %T')] 완료. 결과: data/results/  로그: $LOG_DIR/"
