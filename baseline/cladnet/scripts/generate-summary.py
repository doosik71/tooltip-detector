#!/usr/bin/env python3
"""Collect the raw numbers behind docs/experimental-results.md into docs/summary-results.md.

Everything in the generated document is read back from the two directories that
train-model / eval-model write:

  data/model<SUFFIX>/<dataset>/
      train-status.json   completed epochs, every run argument, best val fitness
      metric.csv          one row per epoch (losses, mAP, per-class AP, lr, time)
  data/results<SUFFIX>/<dataset>/<split>/
      summary.json        detection + tip metrics, and the run parameters
      per_tip.csv         one row per ground-truth tip (coordinates, nearest
                          prediction, its score, distance, whether it was missed)

The figures summary.json does not carry -- error-distance histograms, frame-level
detection failures, miss rate by tool count, the under-detection share of the
long tail, the confidence distribution and per-session spread -- are recomputed
here from per_tip.csv, so the report never has to quote a number that cannot be
regenerated from the files on disk.

Missing runs are listed rather than silently skipped, and anything that cannot
be derived from these two directories is named as a gap rather than guessed at.

Usage:
    ./baseline/cladnet/run generate-summary
    ./baseline/cladnet/run generate-summary --suffix "" --output docs/summary-results.md
"""

import argparse
import csv
import json
import os
import statistics

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import class_names
    from common.inference import data_dir

# TEMPORARY. The completed runs live in data/model-16x16 and data/results-16x16
# while a fresh training round writes to the unsuffixed directories. Set this to
# "" once that round finishes and its results are the ones worth reporting;
# --suffix overrides it for a one-off run.
DATA_SUFFIX = ""

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "summary-results.md")

SPLIT = "test"

# Which run to summarise. `--suffix` selects a *round* (a sibling data/
# directory kept from an earlier experiment); this selects a *mode* within
# it, and the two are independent axes of the same path.
DEFAULT_LABEL_SET = "tooltip"
LABEL_SETS = ("tooltip", "tiponly")

# Loss columns differ between the two baselines; CLAD-Net carries an objectness term.
LOSS_PARTS = ("box", "obj", "cls")

# Epochs shown in the convergence table. Anything past the run's length is dropped.
CURVE_EPOCHS = (1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150)

ERROR_BINS = ((0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, None))

# An error this large is almost never a mislocated tip; it is a tool that was
# missed entirely, leaving its ground-truth tip to match some other tool's
# prediction. The generated document reports how much of the tail is that case.
UNDER_DETECTION_PX = 200.0


# ── reading ────────────────────────────────────────────────────────────────

def read_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def read_metric_csv(path: str) -> list[dict] | None:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    return rows or None


def analyse_per_tip(path: str) -> dict | None:
    """Recompute from per_tip.csv everything summary.json does not carry."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    if not rows:
        return None

    matched = [r for r in rows if r["missed"] == "0"]
    distances = [float(r["dist_px"]) for r in matched]
    scores = [float(r["score"]) for r in matched if r["score"] != ""]

    frames: dict[str, list[dict]] = {}
    sessions: dict[str, list[dict]] = {}
    for row in rows:
        frames.setdefault(row["frame"], []).append(row)
        sessions.setdefault(row["session"], []).append(row)

    # error histogram
    histogram = []
    for lo, hi in ERROR_BINS:
        count = sum(1 for d in distances if d >= lo and (hi is None or d < hi))
        histogram.append({"lo": lo, "hi": hi, "count": count})

    # miss rate by how many tools the frame holds
    by_count: dict[int, list[int]] = {}
    for tips in frames.values():
        key = min(len(tips), 4)
        bucket = by_count.setdefault(key, [0, 0])
        bucket[0] += len(tips)
        bucket[1] += sum(1 for t in tips if t["missed"] == "1")

    # how much of the long tail is under-detection rather than mislocation?
    tail = shared = 0
    for tips in frames.values():
        hits = [t for t in tips if t["missed"] == "0"]
        for row in hits:
            if float(row["dist_px"]) < UNDER_DETECTION_PX:
                continue
            tail += 1
            if any(o is not row and o["pred_x"] == row["pred_x"] and o["pred_y"] == row["pred_y"]
                   for o in hits):
                shared += 1

    per_session = {}
    for name, tips in sessions.items():
        near = [t for t in tips if t["missed"] == "0" and float(t["dist_px"]) <= 10]
        hit_distances = [float(t["dist_px"]) for t in tips if t["missed"] == "0"]
        per_session[name] = {
            "n_gt": len(tips),
            "missed": sum(1 for t in tips if t["missed"] == "1"),
            "hit10_pct": 100.0 * len(near) / len(tips),
            "median_dist_px": statistics.median(hit_distances) if hit_distances else None,
        }

    well_located = [float(r["score"]) for r in matched
                    if r["score"] != "" and float(r["dist_px"]) <= 10]

    return {
        "n_gt": len(rows),
        "n_missed": sum(1 for r in rows if r["missed"] == "1"),
        "n_matched": len(matched),
        "histogram": histogram,
        "frames_with_gt": len(frames),
        "frames_without_prediction": sum(
            1 for tips in frames.values() if all(t["missed"] == "1" for t in tips)),
        "by_tool_count": by_count,
        "tail_count": tail,
        "tail_shared": shared,
        "scores": percentiles(scores),
        "scores_well_located": percentiles(well_located),
        "n_well_located": len(well_located),
        "per_session": per_session,
    }


def percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    return {"p10": at(0.10), "median": at(0.50), "p90": at(0.90),
            "min": ordered[0], "max": ordered[-1],
            "above_0.25_pct": 100.0 * sum(1 for v in ordered if v >= 0.25) / len(ordered)}


def collect(model_root: str, results_root: str,
            label_set: str = DEFAULT_LABEL_SET) -> tuple[list[dict], list[str]]:
    """One record per dataset directory found, plus a list of what is missing."""
    if not os.path.isdir(model_root):
        return [], [f"학습 결과 디렉터리가 없다: {model_root}"]

    records, gaps = [], []
    for dataset in sorted(os.listdir(model_root)):
        model_dir = os.path.join(model_root, dataset, label_set)
        if not os.path.isdir(model_dir):
            continue
        status = read_json(os.path.join(model_dir, "train-status.json"))
        curve = read_metric_csv(os.path.join(model_dir, "metric.csv"))
        result_dir = os.path.join(results_root, dataset, label_set, SPLIT)
        summary = read_json(os.path.join(result_dir, "summary.json"))
        per_tip = analyse_per_tip(os.path.join(result_dir, "per_tip.csv"))

        if status is None:
            gaps.append(f"`{dataset}`: train-status.json 없음 ({model_dir})")
        if curve is None:
            gaps.append(f"`{dataset}`: metric.csv 없음 ({model_dir})")
        if summary is None:
            gaps.append(f"`{dataset}`: {SPLIT} 평가 결과 없음 ({result_dir})")
        if summary is not None and per_tip is None:
            gaps.append(f"`{dataset}`: per_tip.csv 없음 — 오차 분포·실패 모드 분석 불가")

        records.append({"dataset": dataset, "status": status, "curve": curve,
                        "summary": summary, "per_tip": per_tip,
                        "model_dir": model_dir, "result_dir": result_dir})
    return records, gaps


def parameter_count(label_set: str = DEFAULT_LABEL_SET) -> int | None:
    """Built rather than read: no artefact on disk records it.

    The class count is part of the model, so `tiponly` is a few hundred
    parameters smaller than `tooltip`."""
    try:
        from common.model import build, parameter_count as count
        return count(build(num_classes=len(class_names(label_set))))
    except Exception:
        return None


# ── formatting ─────────────────────────────────────────────────────────────

def num(value, nd: int = 4, suffix: str = "") -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def integer(value) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def table(headers: list[str], rows: list[list[str]], align: str) -> list[str]:
    """align: one character per column -- 'l' or 'r'."""
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(cells) + " |" for cells in rows]
    out.append("")
    return out


def bin_label(lo, hi) -> str:
    return f"{lo}–{hi} px" if hi is not None else f"≥ {lo} px"


# ── sections ───────────────────────────────────────────────────────────────

def section_overview(out: list[str], records: list[dict], gaps: list[str],
                     model_root: str, results_root: str, suffix: str,
                     label_set: str = DEFAULT_LABEL_SET) -> None:
    out += ["## 1. 무엇을 읽었나", ""]
    out += [f"- 학습 모드: `{label_set}` "
            f"({'tool 상자와 tip 상자를 함께 학습' if label_set == 'tooltip' else 'tip 상자만 학습'})",
            f"- 학습 산출물: `{os.path.relpath(model_root, os.path.dirname(data_dir()))}/<dataset>/{label_set}`",
            f"- 평가 산출물: `{os.path.relpath(results_root, os.path.dirname(data_dir()))}/<dataset>/{label_set}` (split `{SPLIT}`)",
            ""]
    if suffix:
        out += [f"> 디렉터리 접미사 `{suffix}` 가 붙어 있다. 새 학습이 접미사 없는 경로에 쓰는 동안",
                "> 완료된 실험을 보존하기 위한 임시 조치이며, 스크립트 상단의 `DATA_SUFFIX`를 `\"\"`로",
                "> 바꾸면 새 결과로 전환된다.", ""]

    rows = []
    for record in records:
        status, summary = record["status"], record["summary"]
        rows.append([
            f"`{record['dataset']}`",
            "—" if status is None else f"{status['epochs_completed']} / {status['epochs_total']}",
            "—" if status is None else num(status["best_map50_95"]),
            "있음" if summary is not None else "**없음**",
            "있음" if record["per_tip"] is not None else "**없음**",
        ])
    out += table(["데이터셋", "학습 에포크", "best val mAP@0.5:0.95", "평가 결과", "per_tip.csv"],
                 rows, "lrrll")

    params = parameter_count(label_set)
    if params is not None:
        out += [f"모델 파라미터: **{params:,}**", ""]
    if gaps:
        out += ["누락:", ""] + [f"- {g}" for g in gaps] + [""]


def section_settings(out: list[str], records: list[dict]) -> None:
    out += ["## 2. 학습 설정", "",
            "`train-status.json`의 `args`를 그대로 옮긴 것이다. 보고서의 실험 설정 표는 여기서 만든다.", ""]
    keys = ("epochs", "batch_size", "lr", "momentum", "weight_decay", "image_size",
            "tip_box_size", "frame_stride", "val_frames", "neck_channels", "head_hidden",
            "rm_combine", "no_ema", "conf", "iou", "device", "workers")
    headers = ["항목"] + [f"`{r['dataset']}`" for r in records]
    rows = []
    for key in keys:
        cells = []
        for record in records:
            args = (record["status"] or {}).get("args", {})
            cells.append("—" if key not in args else str(args[key]))
        if all(c == "—" for c in cells):
            continue
        rows.append([f"`{key}`"] + cells)
    out += table(headers, rows, "l" + "l" * len(records))
    if any("tip_box_size" not in (r["status"] or {}).get("args", {}) for r in records):
        out += ["> `tip_box_size`가 비어 있는 실행은 그 인수가 생기기 전에 학습된 것이다.",
                "> 당시 상수는 10 px이었다.", ""]


def section_convergence(out: list[str], records: list[dict],
                        label_set: str = DEFAULT_LABEL_SET) -> None:
    out += ["## 3. 학습 경과", ""]
    names = class_names(label_set)
    for record in records:
        curve = record["curve"]
        out += [f"### `{record['dataset']}`", ""]
        if curve is None:
            out += ["metric.csv가 없어 곡선을 만들 수 없다.", ""]
            continue

        headers = ["에포크", "train loss"] + [f"val {p}" for p in LOSS_PARTS] + \
                  ["val loss", "mAP@0.5", "mAP@0.5:0.95"] + \
                  [f"{n} AP50" for n in names] + ["lr"]
        rows = []
        for epoch in CURVE_EPOCHS:
            if epoch > len(curve):
                continue
            row = curve[epoch - 1]
            train = sum(float(row[f"train_{p}_loss"]) for p in LOSS_PARTS)
            rows.append([row["epoch"], num(train, 4)] +
                        [num(row[f"val_{p}_loss"], 4) for p in LOSS_PARTS] +
                        [num(row["val_loss"], 4), num(row["map50"]), num(row["map50_95"])] +
                        [num(row[f"{n}_ap50"]) for n in names] + [num(row["lr"], 6)])
        if len(curve) not in CURVE_EPOCHS:
            row = curve[-1]
            train = sum(float(row[f"train_{p}_loss"]) for p in LOSS_PARTS)
            rows.append([f"**{row['epoch']}**", num(train, 4)] +
                        [num(row[f"val_{p}_loss"], 4) for p in LOSS_PARTS] +
                        [num(row["val_loss"], 4), num(row["map50"]), num(row["map50_95"])] +
                        [num(row[f"{n}_ap50"]) for n in names] + [num(row["lr"], 6)])
        out += table(headers, rows, "r" * len(headers))

        fitness = [float(r["map50_95"]) for r in curve]
        best = fitness.index(max(fitness)) + 1
        seconds = sum(float(r["seconds"]) for r in curve)
        converged = ("마지막 에포크가 최고점 — **수렴하지 않았다**" if best == len(curve)
                     else f"최고점 이후 {len(curve) - best} 에포크 동안 갱신 없음")
        out += [f"- 최고 에포크 **{best} / {len(curve)}** (mAP@0.5:0.95 {max(fitness):.4f}) — {converged}",
                f"- 마지막 5 에포크 mAP@0.5:0.95: {', '.join(f'{v:.4f}' for v in fitness[-5:])}",
                f"- 학습 시간 {seconds / 3600:.1f} 시간 ({seconds / 60 / len(curve):.1f} 분/에포크)",
                ""]


def section_detection(out: list[str], records: list[dict],
                      label_set: str = DEFAULT_LABEL_SET) -> None:
    out += ["## 4. 탐지 성능", ""]
    first = next((r["summary"] for r in records if r["summary"]), None)
    if first:
        out += [f"평가 조건: `conf={first['conf_threshold']}`, NMS IoU `{first['iou_threshold']}`, "
                f"AP 곡선은 `conf={first['map_conf_threshold']}`까지, "
                f"입력 {first['image_size']} px, split `{first['split']}`.", ""]

    names = class_names(label_set)
    headers = ["데이터셋", "프레임", "mAP@0.5", "mAP@0.5:0.95"] + \
              [h for n in names for h in (f"{n} AP@0.5", f"{n} AP@0.5:0.95")]
    rows = []
    for record in records:
        summary = record["summary"]
        if summary is None:
            rows.append([f"`{record['dataset']}`"] + ["—"] * (len(headers) - 1))
            continue
        det = summary["detection"]
        row = [f"`{record['dataset']}`", integer(summary["n_frames"]),
               num(det["map50"]), num(det["map50_95"])]
        for name in names:
            values = det["per_class"].get(name, {})
            row += [num(values.get("ap50")), num(values.get("ap50_95"))]
        rows.append(row)
    out += table(headers, rows, "l" + "r" * (len(headers) - 1))

    rows = []
    for record in records:
        summary = record["summary"]
        if summary is None:
            continue
        for name, values in summary["detection"]["per_class"].items():
            rows.append([f"`{record['dataset']}`", f"`{name}`", integer(values["n_gt"]),
                         integer(values["n_pred"]), num(values["precision"]), num(values["recall"])])
    if rows:
        out += table(["데이터셋", "클래스", "GT", "예측 수", "precision", "recall"],
                     rows, "llrrrr")
        out += ["> 이 `precision`·`recall`은 AP 곡선을 그리기 위한 낮은 임계값에서 누적된 값이라",
                "> 운용 지점의 값이 아니다. 운용 지점은 §5의 Hungarian 표를 본다.", ""]


def section_tip(out: list[str], records: list[dict]) -> None:
    out += ["## 5. 팁 위치 정확도", "",
            "루트 프로젝트 `scripts/eval-model.py`와 같은 매칭 규칙으로 계산된 값이다.",
            "거리는 letterbox 이전의 원본 프레임 좌표계(736 × 480)에서 잰다.", ""]
    rows = []
    for record in records:
        summary = record["summary"]
        if summary is None:
            rows.append([f"`{record['dataset']}`"] + ["—"] * 7)
            continue
        tip = summary["tip"]
        rows.append([f"`{record['dataset']}`", integer(tip["n_gt_tips"]),
                     num(tip["miss_rate_pct"], 2, " %"),
                     num(tip["hit_rate_10px_pct"], 2, " %"),
                     num(tip["hit_rate_20px_pct"], 2, " %"),
                     num(tip["hit_rate_50px_pct"], 2, " %"),
                     num(tip["median_dist_px"], 2, " px"),
                     num(tip["mean_dist_px"], 2, " px")])
    out += table(["데이터셋", "GT 팁", "탐지 실패율", "Hit@10 px", "Hit@20 px", "Hit@50 px",
                  "중앙값", "평균"], rows, "l" + "r" * 7)

    rows = []
    for record in records:
        summary = record["summary"]
        if summary is None:
            continue
        for cap, values in summary["tip"]["hungarian"].items():
            rows.append([f"`{record['dataset']}`", cap, integer(values["tp"]),
                         integer(values["fp"]), integer(values["fn"]),
                         num(values["precision"]), num(values["recall"]),
                         num(values["median_dist_px"], 2, " px")])
    if rows:
        out += ["### Hungarian 1:1 매칭 (운용 지점)", ""]
        out += table(["데이터셋", "cap", "TP", "FP", "FN", "precision", "recall", "중앙값"],
                     rows, "ll" + "r" * 6)


def section_distribution(out: list[str], records: list[dict]) -> None:
    out += ["## 6. 오차 거리 분포", "",
            "`per_tip.csv`에서 다시 계산한 것이다. 분모는 "
            "**놓치지 않고 매칭된 팁**이며, 괄호 안은 전체 GT 팁 대비 비율이다.", ""]
    usable = [r for r in records if r["per_tip"]]
    if not usable:
        out += ["per_tip.csv가 없어 계산할 수 없다.", ""]
        return

    headers = ["오차"] + [f"`{r['dataset']}`" for r in usable]
    rows = []
    for index, (lo, hi) in enumerate(ERROR_BINS):
        cells = []
        for record in usable:
            stats = record["per_tip"]
            count = stats["histogram"][index]["count"]
            cells.append(f"{count:,} ({100 * count / max(stats['n_matched'], 1):.1f} % / "
                         f"{100 * count / max(stats['n_gt'], 1):.1f} %)")
        rows.append([bin_label(lo, hi)] + cells)
    out += table(headers, rows, "l" + "r" * len(usable))

    rows = []
    for record in usable:
        stats = record["per_tip"]
        share = 100 * stats["tail_shared"] / max(stats["tail_count"], 1)
        rows.append([f"`{record['dataset']}`", integer(stats["tail_count"]),
                     integer(stats["tail_shared"]), f"{share:.1f} %"])
    out += [f"### 큰 오차(≥ {UNDER_DETECTION_PX:.0f} px)의 정체", "",
            "같은 예측이 다른 GT 팁에도 최근접이었다면, 위치를 틀린 것이 아니라",
            "그 프레임의 도구 하나를 통째로 놓친 것이다.", ""]
    out += table(["데이터셋", f"≥ {UNDER_DETECTION_PX:.0f} px 오차", "예측 공유", "과소 탐지 비율"],
                 rows, "lrrr")


def section_frames(out: list[str], records: list[dict]) -> None:
    out += ["## 7. 프레임 단위 통계", ""]
    usable = [r for r in records if r["per_tip"]]
    if not usable:
        out += ["per_tip.csv가 없어 계산할 수 없다.", ""]
        return

    rows = []
    for record in usable:
        stats = record["per_tip"]
        rows.append([f"`{record['dataset']}`", integer(stats["frames_with_gt"]),
                     integer(stats["frames_without_prediction"]),
                     f"{100 * stats['frames_without_prediction'] / max(stats['frames_with_gt'], 1):.2f} %"])
    out += table(["데이터셋", "GT 팁이 있는 프레임", "팁 예측이 하나도 없는 프레임", "비율"],
                 rows, "lrrr")

    out += ["### 프레임 내 도구 수에 따른 탐지 실패율", ""]
    headers = ["프레임의 GT 팁 수"] + [f"`{r['dataset']}`" for r in usable]
    rows = []
    for key in (1, 2, 3, 4):
        label = "4개 이상" if key == 4 else f"{key}개"
        cells = []
        for record in usable:
            bucket = record["per_tip"]["by_tool_count"].get(key)
            cells.append("—" if not bucket else
                         f"{100 * bucket[1] / bucket[0]:.2f} % ({bucket[0]:,} GT)")
        rows.append([label] + cells)
    out += table(headers, rows, "l" + "r" * len(usable))


def section_confidence(out: list[str], records: list[dict]) -> None:
    thresholds = sorted({r["summary"]["conf_threshold"] for r in records if r["summary"]})
    cut = ", ".join(str(t) for t in thresholds) or "평가 시 지정한 값"
    out += ["## 8. 예측 신뢰도 분포", "",
            "`per_tip.csv`의 `score` 열이다. **분포가 잘려 있다** — 평가가 `conf`",
            f"{cut} 로 실행되어 그보다 낮은 예측은 애초에 기록되지 않았다.",
            "따라서 아래 `≥ 0.25` 열은 임계값이 0.25 이상이면 항상 100 %가 되며,",
            "잘리지 않은 분포를 보려면 `--conf`를 낮춰 `eval-model`을 다시 돌려야 한다.", ""]
    usable = [r for r in records if r["per_tip"] and r["per_tip"]["scores"]]
    if not usable:
        out += ["신뢰도를 계산할 수 있는 데이터가 없다.", ""]
        return

    rows = []
    for record in usable:
        stats = record["per_tip"]
        overall, near = stats["scores"], stats["scores_well_located"]
        rows.append([f"`{record['dataset']}`", "전체 매칭", integer(stats["n_matched"]),
                     num(overall["p10"], 3), num(overall["median"], 3),
                     num(overall["p90"], 3), num(overall["max"], 3),
                     num(overall["above_0.25_pct"], 1, " %")])
        if near:
            rows.append([f"`{record['dataset']}`", "10 px 이내", integer(stats["n_well_located"]),
                         num(near["p10"], 3), num(near["median"], 3),
                         num(near["p90"], 3), num(near["max"], 3),
                         num(near["above_0.25_pct"], 1, " %")])
    out += table(["데이터셋", "표본", "n", "p10", "중앙값", "p90", "최대", "≥ 0.25"],
                 rows, "ll" + "r" * 6)
    out += ["> 최대 신뢰도가 낮으면 objectness 학습 목표(예측·GT 박스의 IoU)에 천장이 생겼다는",
            "> 신호다. 팁 박스가 작을수록 이 값이 낮아진다.", ""]


def section_sessions(out: list[str], records: list[dict]) -> None:
    out += ["## 9. 세션별 성능", "",
            "test 스플릿이 영상 단위로 나뉜 경우, 평균값 하나로 성능을 말할 수 있는지 판단하는 근거다.", ""]
    usable = [r for r in records if r["per_tip"]]
    if not usable:
        out += ["per_tip.csv가 없어 계산할 수 없다.", ""]
        return

    rows = []
    for record in usable:
        sessions = record["per_tip"]["per_session"]
        order = sorted(sessions, key=lambda k: sessions[k]["hit10_pct"])
        low, high = sessions[order[0]], sessions[order[-1]]
        rows.append([f"`{record['dataset']}`", str(len(sessions)),
                     f"{order[0]} {low['hit10_pct']:.1f} %",
                     f"{order[-1]} {high['hit10_pct']:.1f} %",
                     f"{high['hit10_pct'] - low['hit10_pct']:.1f} pp"])
    out += table(["데이터셋", "세션 수", "Hit@10 px 최저", "최고", "편차"], rows, "lrlll")

    for record in usable:
        sessions = record["per_tip"]["per_session"]
        out += [f"<details><summary><code>{record['dataset']}</code> 전체 세션 "
                f"({len(sessions)}개)</summary>", ""]
        rows = [[name, integer(v["n_gt"]), integer(v["missed"]),
                 f"{100 * v['missed'] / max(v['n_gt'], 1):.2f} %",
                 f"{v['hit10_pct']:.2f} %", num(v["median_dist_px"], 2, " px")]
                for name, v in sorted(sessions.items())]
        out += table(["세션", "GT 팁", "놓침", "실패율", "Hit@10 px", "중앙값"], rows, "lrrrrr")
        out += ["</details>", ""]


def section_speed(out: list[str], records: list[dict]) -> None:
    out += ["## 10. 속도", "",
            "`eval-model`이 프레임 한 장을 처리한 실제 시간이다 — letterbox, forward,",
            "디코딩, NMS를 모두 포함한다. GPU가 다르면 비교할 수 없다.", ""]
    rows = []
    for record in records:
        summary = record["summary"]
        if summary is None:
            continue
        rows.append([f"`{record['dataset']}`", str(summary["device"]),
                     num(summary["ms_per_frame"], 2, " ms"), num(summary["fps"], 1),
                     integer(summary["n_frames"])])
    if rows:
        out += table(["데이터셋", "장치", "프레임당", "FPS", "프레임"], rows, "llrrr")
    else:
        out += ["평가 결과가 없어 속도를 보고할 수 없다.", ""]


def section_gaps(out: list[str]) -> None:
    out += ["## 11. 이 문서에서 만들 수 없는 값", "",
            "보고서에 필요하지만 위 두 디렉터리만으로는 나오지 않는 것들이다. 추측해 넣지 않는다.", "",
            "- **다른 모델과의 비교** — 루트 프로젝트나 형제 베이스라인의 수치. 이 스크립트는",
            "  자기 서브 프로젝트의 `data/`만 읽는다.",
            "- **신뢰도 임계값 스윕** — `--conf`를 바꿔 `eval-model`을 여러 번 돌린 결과가 필요하다.",
            "- **동일 조건 속도 비교** — 모델마다 다른 GPU에서 측정된 값이라, 같은 장치에서 다시",
            "  재야 비교할 수 있다.",
            "- **데이터셋 규모** — train/val 프레임 수는 `data/dataset/`을 읽어야 나온다.",
            "- **파라미터 수의 근거** — 위 §1의 값은 파일이 아니라 모델을 만들어 센 것이다.",
            ""]


# ── entry point ────────────────────────────────────────────────────────────

def build_document(records: list[dict], gaps: list[str], model_root: str,
                   results_root: str, suffix: str,
                   label_set: str = DEFAULT_LABEL_SET) -> str:
    out = ["# CLAD-Net 실험 수치 요약", "",
           "이 문서는 `scripts/generate-summary.py`가 생성한다. 직접 고치지 말고 스크립트를 고친다.",
           "[experimental-results.md](experimental-results.md)를 쓸 때 참고할 수치를 모아 둔 것이며,",
           "모든 값은 `data/` 아래 파일에서 다시 계산된다.", ""]
    section_overview(out, records, gaps, model_root, results_root, suffix, label_set)
    section_settings(out, records)
    section_convergence(out, records, label_set)
    section_detection(out, records, label_set)
    section_tip(out, records)
    section_distribution(out, records)
    section_frames(out, records)
    section_confidence(out, records)
    section_sessions(out, records)
    section_speed(out, records)
    section_gaps(out)
    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Summarise CLAD-Net training and evaluation artefacts into a markdown document")
    parser.add_argument("--suffix", default=DATA_SUFFIX,
                        help=f'directory suffix under data/ (default: "{DATA_SUFFIX}"; '
                             'pass "" once the current training round is the one to report)')
    parser.add_argument("--label-set", default=DEFAULT_LABEL_SET, choices=LABEL_SETS,
                        help="which training mode to summarise; also the directory "
                             f"stage under data/model/<dataset> (default: {DEFAULT_LABEL_SET})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"where the document goes (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    model_root = os.path.join(data_dir(), f"model{args.suffix}")
    results_root = os.path.join(data_dir(), f"results{args.suffix}")
    records, gaps = collect(model_root, results_root, args.label_set)

    if not records:
        raise SystemExit(f"no {args.label_set} runs under {model_root}\n"
                         "train one first, or pass --label-set / --suffix for "
                         "another directory")

    document = build_document(records, gaps, model_root, results_root, args.suffix,
                              args.label_set)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(document)

    print(f"read    : {model_root}")
    print(f"          {results_root}")
    print(f"datasets: {', '.join(r['dataset'] for r in records)}")
    for gap in gaps:
        print(f"missing : {gap}")
    print(f"written : {args.output}  ({len(document.splitlines()):,} lines)")


if __name__ == "__main__":
    main()
