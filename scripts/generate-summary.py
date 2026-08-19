#!/usr/bin/env python3
"""Collect the raw numbers behind docs/final-report.md into docs/results-summary.md.

Everything in the generated document is read back from the two directories that
train-model / eval-model / compare-speed write:

  data/models/<dataset>/<target-mode>/<model-type>/
      train-status.json      completed epochs, batch/lr/sigma, best val loss, elapsed
      metric.csv             one row per epoch (train/val loss, mae, me, std, lr, time)
  data/results/<dataset>/<target-mode>/<model-type>/
      summary.json           overall + per-session metrics, run parameters
      per_tip.csv            one row per GT tip (coordinates, matched peak, distance)
  data/results/<dataset>/<target-mode>/
      speed-comparison.json  forward/wall time and parameter count per model

The figures summary.json does not carry -- error-distance histograms, frame-level
detection failures, per-tool-count performance, shared-peak (under-detection)
analysis and the systematic coordinate offset -- are recomputed here from
per_tip.csv, so the report never has to quote a number that cannot be
regenerated from the files on disk.

Combinations are enumerated from ttd.dataset.DATASETS x ttd.dataset.TARGET_MODES
x ttd.model.REGISTRY, so stray directories (e.g. a kept-aside `monai.old`) are
ignored and missing runs are listed instead of silently skipped.

NOT derivable from these two directories, because it needs data/dataset (an
external read-only mount): total test-split frame count, tool-free frame count,
and the per-frame tool-count distribution of the train split. Those are listed
as gaps in the generated document rather than guessed at.

Usage:
    uv run python scripts/generate-summary.py
    uv run python scripts/generate-summary.py --output docs/results-summary.md
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttd.checkpoints import default_results_dir, model_dir
from ttd.dataset import DATASETS, TARGET_MODES
from ttd.model import REGISTRY as MODEL_REGISTRY

# Distance-bin edges (px) for the error histogram; each bin is [lo, hi).
DIST_BINS = ((0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, None))

# Hit-rate radii, matching eval-model.py's _HIT_THRESHOLDS.
HIT_RADII = (10, 20, 50)

# Epochs quoted in the convergence table. Missing epochs render as "-".
VAL_LOSS_MILESTONES = (1, 5, 10, 20, 30)

# Tips per frame are bucketed 1 / 2 / 3 / 4+; frames with more tools than this
# are too rare to carry their own column.
MAX_TIP_BUCKET = 4

# Only near matches define the systematic offset: a tip matched to a different
# tool's peak (hundreds of px away) says nothing about coordinate bias.
OFFSET_MATCH_RADIUS_PX = 20

# Two target-mode averages closer than this are called a tie rather than a win;
# the averages themselves are printed next to the verdict either way.
TIE_MARGIN = 0.05

# Datasets with more sessions than this get a distribution + best/worst extract
# in the main body and the full listing in an appendix.
SESSION_TABLE_MAX = 10

# How many best/worst sessions to extract when the full table moves to the appendix.
SESSION_EXTRACT_N = 5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_metric_csv(path: str) -> list[dict] | None:
    """Per-epoch training curve, oldest epoch first."""
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "epoch":      int(r["epoch"]),
            "train_loss": float(r["train_loss"]),
            "val_loss":   float(r["val_loss"]),
            "val_mae":    float(r["val_mae"]),
            "val_me":     float(r["val_me"]),
            "lr":         float(r["lr"]),
            "epoch_sec":  float(r["epoch_sec"]),
        })
    out.sort(key=lambda r: r["epoch"])
    return out


def _analyse_per_tip(path: str) -> dict | None:
    """Recompute every per-tip statistic the report needs from per_tip.csv.

    One pass groups tips by frame, because the frame is the unit that decides
    both the tool count and whether two GT tips were matched to the same peak.
    """
    if not os.path.isfile(path):
        return None

    frames: dict[str, list[dict]] = defaultdict(list)
    sessions: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            missed = r["missed"] == "1"
            tip = {
                "missed": missed,
                "gt":     (int(r["gt_x"]), int(r["gt_y"])),
                "pred":   None if missed else (int(r["pred_x"]), int(r["pred_y"])),
                "dist":   None if missed else float(r["dist_px"]),
            }
            frames[r["frame"]].append(tip)
            sessions[r["session"]].append(tip)

    n_tips = sum(len(t) for t in frames.values())
    if n_tips == 0:
        return None

    # --- overall ---------------------------------------------------------
    matched = [t for ts in frames.values() for t in ts if not t["missed"]]
    dists = np.array([t["dist"] for t in matched], dtype=np.float64)
    n_missed = n_tips - len(matched)

    hist = {}
    for lo, hi in DIST_BINS:
        n = int(np.count_nonzero(dists >= lo) if hi is None
                else np.count_nonzero((dists >= lo) & (dists < hi)))
        hist[(lo, hi)] = 100.0 * n / n_tips

    # --- frame level -----------------------------------------------------
    # A frame where every GT tip is missed is a frame the model produced no
    # peak for at all: complete detection failure, not a matching error.
    n_fail_frames = sum(1 for ts in frames.values() if all(t["missed"] for t in ts))
    tip_count_frames = Counter(len(ts) for ts in frames.values())
    n_multi_tips = sum(len(ts) for ts in frames.values() if len(ts) >= 2)

    buckets: dict[int, list[dict]] = defaultdict(list)
    for ts in frames.values():
        buckets[min(len(ts), MAX_TIP_BUCKET)].extend(ts)
    by_tip_count = {k: _tip_group_stats(v) for k, v in sorted(buckets.items())}

    # --- shared peaks (under-detection) ----------------------------------
    # Several GT tips matched to one peak means the model produced fewer peaks
    # than the frame has tools; the surplus tips get dragged onto another
    # tool's peak, which the distance metrics absorb as a huge error.
    shared, solo = [], []
    for ts in frames.values():
        ms = [t for t in ts if not t["missed"]]
        counts = Counter(t["pred"] for t in ms)
        for t in ms:
            (shared if counts[t["pred"]] > 1 else solo).append(t)

    # --- systematic offset ------------------------------------------------
    near = [t for t in matched if t["dist"] <= OFFSET_MATCH_RADIUS_PX]
    dx = [t["pred"][0] - t["gt"][0] for t in near]
    dy = [t["pred"][1] - t["gt"][1] for t in near]
    if near:
        med_dx, med_dy = float(np.median(dx)), float(np.median(dy))
        mode_dx, n_mode_dx = Counter(dx).most_common(1)[0]
        mode_dy, n_mode_dy = Counter(dy).most_common(1)[0]
    else:
        med_dx = med_dy = 0.0
        mode_dx = mode_dy = n_mode_dx = n_mode_dy = 0

    return {
        "n_tips":            n_tips,
        "n_missed":          n_missed,
        "miss_pct":          100.0 * n_missed / n_tips,
        "mean":              float(dists.mean()) if len(dists) else None,
        "median":            float(np.median(dists)) if len(dists) else None,
        "p90":               float(np.percentile(dists, 90)) if len(dists) else None,
        "hit_pct":           {r: 100.0 * np.count_nonzero(dists <= r) / n_tips
                              for r in HIT_RADII},
        "hist_pct":          hist,
        "n_frames":          len(frames),
        "n_fail_frames":     n_fail_frames,
        "fail_frame_pct":    100.0 * n_fail_frames / len(frames),
        "tip_count_frames":  dict(sorted(tip_count_frames.items())),
        "multi_tip_pct":     100.0 * n_multi_tips / n_tips,
        "by_tip_count":      by_tip_count,
        "shared":            _match_group_stats(shared, n_tips, n_multi_tips),
        "solo":              _match_group_stats(solo, n_tips, n_multi_tips),
        "offset":            {"median_dx": med_dx, "median_dy": med_dy,
                              "magnitude": math.hypot(med_dx, med_dy),
                              "mode_dx": mode_dx, "n_mode_dx": n_mode_dx,
                              "mode_dy": mode_dy, "n_mode_dy": n_mode_dy,
                              "n_near": len(near)},
        "solo_raw":          _offset_stats(solo, 0.0, 0.0),
        "solo_corrected":    _offset_stats(solo, med_dx, med_dy),
        "per_session":       {s: _tip_group_stats(ts) for s, ts in sorted(sessions.items())},
    }


def _tip_group_stats(tips: list[dict]) -> dict:
    """Miss rate, hit rates and distance stats over an arbitrary set of GT tips."""
    n = len(tips)
    d = np.array([t["dist"] for t in tips if not t["missed"]], dtype=np.float64)
    n_missed = n - len(d)
    return {
        "n_tips":   n,
        "n_missed": n_missed,
        "miss_pct": 100.0 * n_missed / n if n else None,
        "mean":     float(d.mean()) if len(d) else None,
        "median":   float(np.median(d)) if len(d) else None,
        "p90":      float(np.percentile(d, 90)) if len(d) else None,
        "hit_pct":  {r: 100.0 * np.count_nonzero(d <= r) / n for r in HIT_RADII} if n else {},
    }


def _match_group_stats(tips: list[dict], n_tips: int, n_multi_tips: int) -> dict:
    """Distance stats for the shared-peak / solo-matched split of matched tips."""
    stats = _tip_group_stats(tips)
    stats["share_of_all_pct"] = 100.0 * len(tips) / n_tips if n_tips else None
    stats["share_of_multi_pct"] = 100.0 * len(tips) / n_multi_tips if n_multi_tips else None
    return stats


def _offset_stats(tips: list[dict], off_x: float, off_y: float) -> dict:
    """Distance stats after subtracting a constant coordinate offset.

    With off_x = off_y = 0 this reproduces the recorded distances, so the same
    function yields both the before and after columns of the offset table.
    """
    if not tips:
        return {"n": 0}
    d = np.array([math.hypot(t["pred"][0] - t["gt"][0] - off_x,
                             t["pred"][1] - t["gt"][1] - off_y) for t in tips])
    return {
        "n":         len(d),
        "mean":      float(d.mean()),
        "median":    float(np.median(d)),
        "p90":       float(np.percentile(d, 90)),
        "within_5":  100.0 * np.count_nonzero(d <= 5) / len(d),
        "within_10": 100.0 * np.count_nonzero(d <= 10) / len(d),
    }


def collect(models_root: str, results_root: str) -> tuple[list[dict], list[dict]]:
    """Load every known combination; returns (complete combos, incomplete combos)."""
    combos, incomplete = [], []
    for dataset in DATASETS:
        for target_mode in TARGET_MODES:
            for model_type in MODEL_REGISTRY:
                mdir = model_dir(model_type, dataset, target_mode, models_root)
                rdir = default_results_dir(model_type, dataset, target_mode, results_root)
                combo = {
                    "dataset":     dataset,
                    "target_mode": target_mode,
                    "model_type":  model_type,
                    "label":       f"{dataset}/{target_mode}/{model_type}",
                    "model_dir":   mdir,
                    "results_dir": rdir,
                    "status":      _read_json(os.path.join(mdir, "train-status.json")),
                    "metrics":     _read_metric_csv(os.path.join(mdir, "metric.csv")),
                    "summary":     _read_json(os.path.join(rdir, "summary.json")),
                    "per_tip":     _analyse_per_tip(os.path.join(rdir, "per_tip.csv")),
                }
                missing = [name for name, key in (("train-status.json", "status"),
                                                  ("metric.csv", "metrics"),
                                                  ("summary.json", "summary"),
                                                  ("per_tip.csv", "per_tip"))
                           if combo[key] is None]
                combo["missing"] = missing
                (incomplete if missing else combos).append(combo)
                print(f"  {'ok ' if not missing else 'gap'}  {combo['label']}"
                      + (f"  (missing: {', '.join(missing)})" if missing else ""))
    return combos, incomplete


def collect_speed(results_root: str) -> list[dict]:
    """compare-speed writes one file per dataset/target-mode, not per model-type."""
    found = []
    for dataset in DATASETS:
        for target_mode in TARGET_MODES:
            path = os.path.join(results_root, dataset, target_mode, "speed-comparison.json")
            data = _read_json(path)
            if data is not None:
                found.append({"path": path, "dataset": dataset,
                              "target_mode": target_mode, "data": data})
    return found


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _num(value, nd: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:,.{nd}f}{suffix}"


def _signed(value, nd: int = 2, suffix: str = "") -> str:
    """Signed number; zero carries no sign so a "no offset" cell reads as such."""
    if value is None:
        return "-"
    if round(value, nd) == 0:
        return f"{0:.{nd}f}{suffix}"
    # U+2212 MINUS SIGN keeps the report's typography for negative values.
    return f"{'+' if value > 0 else '−'}{abs(value):,.{nd}f}{suffix}"


def _int(value) -> str:
    return "-" if value is None else f"{value:,}"


def _table(headers: list[str], rows: list[list[str]], align: str) -> list[str]:
    """Markdown table; `align` is one char per column: l / c / r."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells):
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(widths[i]) if align[i] == "r" else cell.ljust(widths[i]))
        return "| " + " | ".join(out) + " |"

    rule = []
    for i, a in enumerate(align):
        rule.append(("-" * (widths[i] + 1) + ":") if a == "r"
                    else (":" + "-" * (widths[i] + 1)) if a == "c"
                    else "-" * (widths[i] + 2))
    return [line(headers), "|" + "|".join(rule) + "|"] + [line(r) for r in rows] + [""]


def _combo_cells(combo: dict) -> list[str]:
    return [f"`{combo['dataset']}`", f"`{combo['target_mode']}`", f"`{combo['model_type']}`"]


def _bin_label(lo, hi) -> str:
    return f"{lo}+" if hi is None else f"{lo}–{hi}"


def _file_time(path: str) -> str:
    if not os.path.isfile(path):
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def _eval_time(summary: dict) -> str:
    """summary.json stores its run time as YYYYmmdd_HHMMSS."""
    try:
        return time.strftime("%Y-%m-%d %H:%M",
                             time.strptime(summary["timestamp"], "%Y%m%d_%H%M%S"))
    except (KeyError, ValueError):
        return "-"


def _find(combos: list[dict], dataset: str, target_mode: str, model_type: str) -> dict | None:
    for c in combos:
        if (c["dataset"], c["target_mode"], c["model_type"]) == (dataset, target_mode, model_type):
            return c
    return None


def _metric_of(combo: dict, key: str):
    """Overall evaluation metric, read from summary.json."""
    s = combo["summary"]
    return {
        "miss":   s["miss_rate_pct"],
        "hit10":  s["hit_rate_10px_pct"],
        "hit20":  s["hit_rate_20px_pct"],
        "hit50":  s["hit_rate_50px_pct"],
        "median": s["median_dist_px"],
        "mean":   s["mean_dist_px"],
        "p90":    s["p90_dist_px"],
    }[key]


# (column name, _metric_of key, delta unit, absolute unit, lower value is better)
METRICS = (("Miss율", "miss",   " %p", " %",  True),
           ("Hit@10", "hit10",  " %p", " %",  False),
           ("Hit@20", "hit20",  " %p", " %",  False),
           ("Hit@50", "hit50",  " %p", " %",  False),
           ("Median", "median", " px", " px", True),
           ("Mean",   "mean",   " px", " px", True),
           ("P90",    "p90",    " px", " px", True))


# ---------------------------------------------------------------------------
# Document sections
# ---------------------------------------------------------------------------

def _section_overview(out: list[str], combos: list[dict], incomplete: list[dict],
                      speed: list[dict], models_root: str, results_root: str) -> None:
    out += [
        "# 실험 수치 요약 (자동 생성)",
        "",
        "이 문서는 `scripts/generate-summary.py`가 `data/models`와 `data/results`의 파일만 읽어 "
        "생성한다. [final-report.md](final-report.md)에 인용되는 모든 수치의 기초 자료이며, "
        "직접 수정하지 말고 스크립트를 다시 실행해 갱신한다.",
        "",
        "```bash",
        "uv run python scripts/generate-summary.py",
        "```",
        "",
        f"- 생성 시각: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- 모델 루트: `{models_root}`",
        f"- 결과 루트: `{results_root}`",
        f"- 완비된 조합: {len(combos)} / {len(combos) + len(incomplete)}",
        "",
        "**출처 파일과 그 역할**",
        "",
    ]
    out += _table(
        ["파일", "제공하는 수치"],
        [["`data/models/<조합>/train-status.json`", "완료 에포크, batch/lr/σ, best val loss, 총 학습 시간"],
         ["`data/models/<조합>/metric.csv`", "에포크별 train/val loss·MAE·ME·lr·소요 시간"],
         ["`data/results/<조합>/summary.json`", "전체·세션별 평가 지표, 평가 실행 파라미터"],
         ["`data/results/<조합>/per_tip.csv`", "GT 팁 1행씩 — 오차 분포·팁 수별·공유 피크·계통 편차 분석의 원천"],
         ["`data/results/<데이터셋>/<타겟>/speed-comparison.json`", "모델별 파라미터 수와 추론 속도"]],
        "ll")

    if incomplete:
        out += ["**파일이 누락된 조합**", ""]
        out += _table(["조합", "누락 파일"],
                      [[f"`{c['label']}`", ", ".join(f"`{m}`" for m in c["missing"])]
                       for c in incomplete], "ll")

    if not speed:
        out += ["> 속도 비교 결과(`speed-comparison.json`)가 없다. "
                "`run compare-speed --dataset <이름>`으로 생성한다.", ""]

    out += [
        "**이 두 폴더에서 얻을 수 없는 수치.** 아래 항목은 `data/dataset/`(외부 read-only 마운트)이 "
        "필요하므로 이 문서에 포함되지 않는다. 보고서에서 인용할 때는 데이터셋 문서를 근거로 "
        "삼아야 한다.",
        "",
        "- test 스플릿의 전체 프레임 수, 그중 도구가 없는 프레임 수 "
        "(`per_tip.csv`에는 GT 팁이 있는 프레임만 기록된다)",
        "- train/val 스플릿의 프레임 수·팁 수, 프레임당 도구 수 분포",
        "- 데이터셋 총 프레임 수·총 팁 어노테이션 수, 세션 구성",
        "",
        "---",
        "",
    ]


def _section_training(out: list[str], combos: list[dict]) -> None:
    out += [
        "## 1. 학습 기록",
        "",
        "`train-status.json`(학습 설정·총 소요 시간)과 `metric.csv`(에포크 수·에포크별 소요 시간)에서 "
        "읽는다. `초/에포크`는 `metric.csv`의 `epoch_sec` 평균이고, `총 시간`은 "
        "`train-status.json`의 `elapsed_seconds`다 — 학습을 재개한 조합에서는 두 값의 곱이 "
        "총 시간과 정확히 일치하지 않을 수 있다.",
        "",
    ]
    rows = []
    for c in combos:
        st, ms = c["status"], c["metrics"]
        epoch_secs = [m["epoch_sec"] for m in ms]
        best_epoch = min(ms, key=lambda m: m["val_loss"])["epoch"]
        rows.append(_combo_cells(c) + [
            f"{st['completed_epochs']} / {st['epochs']}",
            _int(st["batch_size"]),
            f"{st['lr']:g}",
            f"{st['gaussian_sigma']:g}" if c["target_mode"] == "gaussian-tip" else "-",
            _num(st["best_val_loss"], 6),
            str(best_epoch),
            _num(st["elapsed_seconds"] / 3600.0, 1),
            _num(sum(epoch_secs) / len(epoch_secs), 0),
            st.get("timestamp", "-"),
        ])
    out += _table(
        ["데이터셋", "타겟 방식", "모델", "에포크", "배치", "lr", "σ",
         "best val loss", "best 에포크", "총 시간(h)", "초/에포크", "학습 완료 시각"],
        rows, "lllrrrrrrrrl")

    total_h = sum(c["status"]["elapsed_seconds"] for c in combos) / 3600.0
    out += [f"- 기록된 {len(combos)}개 조합의 학습 시간 합계: **{total_h:,.1f} 시간** "
            f"(약 {total_h / 24:.1f}일)"]

    by_model = defaultdict(list)
    for c in combos:
        ms = c["metrics"]
        by_model[c["model_type"]].append(sum(m["epoch_sec"] for m in ms) / len(ms))
    if len(by_model) == 2:
        (a, sa), (b, sb) = by_model.items()
        mean_a, mean_b = sum(sa) / len(sa), sum(sb) / len(sb)
        slow, fast = ((a, mean_a), (b, mean_b)) if mean_a > mean_b else ((b, mean_b), (a, mean_a))
        out += [f"- 에포크 소요 시간 비교(GPU, 순전파+역전파+데이터 로딩 포함): "
                f"`{slow[0]}` {slow[1]:,.0f}초 대 `{fast[0]}` {fast[1]:,.0f}초 — "
                f"**{slow[1] / fast[1]:.2f}배**"]
    out += ["", "---", ""]


def _section_convergence(out: list[str], combos: list[dict]) -> None:
    out += [
        "## 2. 학습 수렴 (에포크별 검증 손실)",
        "",
        "`metric.csv`의 `val_loss` 열이다. 손실은 **같은 타겟 방식 안에서만** 비교할 수 있다 — "
        "`gaussian-tip` 타겟은 대부분의 화소가 0에 가까운 희소 히트맵이므로 MSE가 구조적으로 작다.",
        "",
    ]
    rows = []
    for c in combos:
        ms = {m["epoch"]: m for m in c["metrics"]}
        best = min(c["metrics"], key=lambda m: m["val_loss"])
        cells = [f"`{c['label']}`"]
        cells += [_num(ms[e]["val_loss"], 6) if e in ms else "-" for e in VAL_LOSS_MILESTONES]
        cells += [_num(best["val_loss"], 6), str(best["epoch"])]
        rows.append(cells)
    out += _table(["조합"] + [f"에포크 {e}" for e in VAL_LOSS_MILESTONES] + ["최저", "최저 에포크"],
                  rows, "l" + "r" * (len(VAL_LOSS_MILESTONES) + 2))

    out += [
        "**파생 지표.** `초기 감소 비중`은 (에포크 1 → 5 감소량) / (에포크 1 → 최저 감소량)으로, "
        "학습이 얼마나 앞쪽 에포크에 집중됐는지를 나타낸다. `말미 감소율`은 마지막 10 에포크 구간의 "
        "상대 감소폭으로, 0에서 멀수록 학습 예산이 부족했다는 뜻이다.",
        "",
    ]
    rows = []
    for c in combos:
        ms = {m["epoch"]: m for m in c["metrics"]}
        last = max(ms)
        best = min(c["metrics"], key=lambda m: m["val_loss"])["val_loss"]
        first = ms[min(ms)]["val_loss"]
        early = (100.0 * (first - ms[5]["val_loss"]) / (first - best)
                 if 5 in ms and first > best else None)
        tail_from = last - 10
        tail = (100.0 * (ms[tail_from]["val_loss"] - ms[last]["val_loss"]) / ms[tail_from]["val_loss"]
                if tail_from in ms else None)
        rows.append([f"`{c['label']}`", _num(first, 6), _num(ms[last]["val_loss"], 6),
                     _num(early, 1, " %"),
                     _num(tail, 2, " %") if tail is not None else "-",
                     f"{tail_from} → {last}",
                     f"{ms[last]['lr']:.2e}"])
    out += _table(["조합", "에포크 1", "최종 에포크", "초기 감소 비중", "말미 감소율",
                   "말미 구간", "최종 lr"], rows, "lrrrrlr")

    out += ["**모델 크기별 val loss 격차** (풀 − 경량, 같은 데이터셋·타겟 방식에서 최저 val loss 기준)", ""]
    rows = []
    models = list(MODEL_REGISTRY)
    for dataset in DATASETS:
        for target_mode in TARGET_MODES:
            pair = [_find(combos, dataset, target_mode, m) for m in models]
            if not all(pair):
                continue
            a, b = (p["status"]["best_val_loss"] for p in pair)
            rows.append([f"`{dataset}`", f"`{target_mode}`", _num(a, 6), _num(b, 6),
                         _signed(100.0 * (a - b) / b, 2, " %")])
    if rows:
        out += _table(["데이터셋", "타겟 방식", f"`{models[0]}`", f"`{models[1]}`",
                       "상대 격차"], rows, "llrrr")
    out += ["---", ""]


def _section_evaluation(out: list[str], combos: list[dict]) -> None:
    out += [
        "## 3. 평가 전체 지표",
        "",
        "`summary.json`을 그대로 옮긴 값이다. Hit-rate의 분모는 미탐지 팁을 포함한 전체 GT 팁 수이고, "
        "거리 통계(mean/median/P90)의 모집단은 매칭에 성공한 팁만이다.",
        "",
    ]
    rows = []
    for c in combos:
        s = c["summary"]
        rows.append(_combo_cells(c) + [
            _int(s["n_gt_tips"]), _int(s["n_missed"]), _num(s["miss_rate_pct"], 2, " %"),
            _num(s["hit_rate_10px_pct"], 2, " %"), _num(s["hit_rate_20px_pct"], 2, " %"),
            _num(s["hit_rate_50px_pct"], 2, " %"),
            _num(s["median_dist_px"], 2, " px"), _num(s["mean_dist_px"], 2, " px"),
            _num(s["p90_dist_px"], 2, " px"),
        ])
    out += _table(["데이터셋", "타겟 방식", "모델", "GT 팁", "미탐지", "Miss율",
                   "Hit@10", "Hit@20", "Hit@50", "Median", "Mean", "P90"],
                  rows, "lll" + "r" * 9)

    out += ["**평가 실행 파라미터와 시각**", ""]
    rows = []
    for c in combos:
        s = c["summary"]
        rows.append([f"`{c['label']}`", _num(s["threshold"], 2), f"{s['nms_radius']} px",
                     _int(s["n_frames_with_tools"]), f"`{s['model_path']}`", _eval_time(s)])
    out += _table(["조합", "threshold", "NMS radius", "도구 보유 프레임", "체크포인트", "평가 시각"],
                  rows, "lrrrll")

    # per_tip.csv must reproduce summary.json; a mismatch means one of the two
    # files is stale and every downstream table would silently disagree.
    drift = []
    for c in combos:
        s, p = c["summary"], c["per_tip"]
        for label, a, b in (("n_gt_tips", s["n_gt_tips"], p["n_tips"]),
                            ("n_missed", s["n_missed"], p["n_missed"]),
                            ("n_frames_with_tools", s["n_frames_with_tools"], p["n_frames"]),
                            ("mean_dist_px", s["mean_dist_px"], p["mean"]),
                            ("median_dist_px", s["median_dist_px"], p["median"]),
                            ("hit_rate_10px_pct", s["hit_rate_10px_pct"], p["hit_pct"][10])):
            if a is None or b is None:
                continue
            if abs(a - b) > 0.011:
                drift.append([f"`{c['label']}`", label, _num(a, 3), _num(b, 3)])
    if drift:
        out += ["> **경고 — `summary.json`과 `per_tip.csv`가 어긋난다.** 두 파일 중 하나가 "
                "과거 실행의 잔재일 수 있으므로, 해당 조합은 `run eval-model`로 다시 평가한 뒤 "
                "인용해야 한다.", ""]
        out += _table(["조합", "항목", "summary.json", "per_tip.csv 재계산"], drift, "llrr")
    else:
        out += ["- `per_tip.csv`에서 재계산한 전체 지표가 8개 조합 모두 `summary.json`과 "
                "일치한다(소수점 둘째 자리 기준). 아래 5–10절의 분석은 같은 파일에서 나온 값이다.", ""]
    out += ["---", ""]


def _section_axes(out: list[str], combos: list[dict]) -> None:
    out += [
        "## 4. 축별 영향",
        "",
        "세 축(타겟 방식·데이터셋·모델 크기) 각각에 대해, 나머지 두 축을 고정한 쌍의 차이다. "
        "Miss·Hit는 %p, 거리는 px 단위이며 부호는 표 제목의 뺄셈 방향을 따른다.",
        "",
    ]
    models = list(MODEL_REGISTRY)
    axes = []
    if len(TARGET_MODES) == 2:
        a, b = TARGET_MODES[1], TARGET_MODES[0]
        axes.append((f"4.1 타겟 방식 (`{a}` − `{b}`)", ["데이터셋", "모델"],
                     [([f"`{d}`", f"`{m}`"],
                       _find(combos, d, a, m), _find(combos, d, b, m))
                      for d in DATASETS for m in models]))
    if len(DATASETS) == 2:
        a, b = DATASETS[1], DATASETS[0]
        axes.append((f"4.2 데이터셋 (`{a}` − `{b}`)", ["타겟 방식", "모델"],
                     [([f"`{t}`", f"`{m}`"],
                       _find(combos, a, t, m), _find(combos, b, t, m))
                      for t in TARGET_MODES for m in models]))
    if len(models) == 2:
        a, b = models[0], models[1]
        axes.append((f"4.3 모델 크기 (`{a}` − `{b}`)", ["데이터셋", "타겟 방식"],
                     [([f"`{d}`", f"`{t}`"],
                       _find(combos, d, t, a), _find(combos, d, t, b))
                      for d in DATASETS for t in TARGET_MODES]))

    summaries = []
    for title, head, pairs in axes:
        out += [f"### {title}", ""]
        rows = []
        collected = defaultdict(list)
        for cells, left, right in pairs:
            if not (left and right):
                continue
            deltas = [_metric_of(left, key) - _metric_of(right, key)
                      for _, key, _, _, _ in METRICS]
            for (_, key, _, _, _), value in zip(METRICS, deltas):
                collected[key].append(value)
            rows.append(cells + [_signed(v, 2, m[2]) for v, m in zip(deltas, METRICS)])
        if not rows:
            continue
        out += _table(head + [m[0] for m in METRICS], rows,
                      "l" * len(head) + "r" * len(METRICS))
        summaries.append((title, collected))

    out += ["### 4.4 축별 영향 크기 요약 (Hit@10 px 기준)", ""]
    rows = []
    for title, collected in summaries:
        v = collected["hit10"]
        rows.append([title.split(" ", 1)[1], _signed(min(v), 2, " %p"),
                     _signed(max(v), 2, " %p"), _signed(sum(v) / len(v), 2, " %p")])
    out += _table(["축", "최소", "최대", "평균"], rows, "lrrr")

    out += [f"### 4.5 타겟 방식별 평균 지표 (데이터셋·모델 4개 조합 평균)", "",
            f"`우세`는 두 평균의 차이가 {TIE_MARGIN} 미만이면 `대등`으로 적는다.", ""]
    rows = []
    for name, key, _, abs_unit, lower_is_better in METRICS:
        means = []
        for target_mode in TARGET_MODES:
            vals = [_metric_of(c, key) for c in combos if c["target_mode"] == target_mode]
            means.append(sum(vals) / len(vals) if vals else None)
        cells = [name] + [_num(m, 2, abs_unit) for m in means]
        if len(means) == 2 and all(m is not None for m in means):
            if abs(means[0] - means[1]) < TIE_MARGIN:
                better = "대등"
            else:
                best = min(means) if lower_is_better else max(means)
                better = f"`{TARGET_MODES[means.index(best)]}`"
        else:
            better = "-"
        rows.append(cells + [better])
    out += _table(["지표"] + [f"`{t}`" for t in TARGET_MODES] + ["우세"],
                  rows, "l" + "r" * len(TARGET_MODES) + "l")
    out += ["---", ""]


def _section_distribution(out: list[str], combos: list[dict]) -> None:
    out += [
        "## 5. 오차 거리 분포",
        "",
        "`per_tip.csv`의 `dist_px`를 구간별로 집계한 비율(전체 GT 팁 대비 %, 구간은 [하한, 상한))이다. "
        "미탐지 팁은 거리값이 없으므로 별도 열로 둔다. 각 행의 합은 100 %다.",
        "",
    ]
    rows = []
    for c in combos:
        p = c["per_tip"]
        rows.append([f"`{c['label']}`", _num(p["miss_pct"], 2)]
                    + [_num(p["hist_pct"][b], 2) for b in DIST_BINS])
    out += _table(["조합", "미탐지"] + [_bin_label(*b) for b in DIST_BINS],
                  rows, "l" + "r" * (len(DIST_BINS) + 1))

    out += ["**누적 비율** (전체 GT 팁 대비, 미탐지 포함 분모)", ""]
    rows = []
    for c in combos:
        p = c["per_tip"]
        cum, cells = 0.0, []
        for b in DIST_BINS[:-1]:
            cum += p["hist_pct"][b]
            cells.append(_num(cum, 2, " %"))
        rows.append([f"`{c['label']}`"] + cells)
    out += _table(["조합"] + [f"≤ {hi} px" for _, hi in DIST_BINS[:-1]],
                  rows, "l" + "r" * (len(DIST_BINS) - 1))

    out += [
        "---",
        "",
        "## 6. 프레임 단위 완전 탐지 실패율",
        "",
        "GT 팁이 있는 프레임 중 **모든 팁이 미탐지**인 프레임, 즉 모델이 임계값을 넘는 피크를 "
        "하나도 만들지 못한 프레임의 비율이다. 팁 단위 Miss율과 달리 프레임 단위 실패를 센다.",
        "",
    ]
    rows = []
    for c in combos:
        p = c["per_tip"]
        rows.append([f"`{c['label']}`", _int(p["n_fail_frames"]), _int(p["n_frames"]),
                     _num(p["fail_frame_pct"], 2, " %")])
    out += _table(["조합", "실패 프레임", "도구 보유 프레임", "비율"], rows, "lrrr")
    out += ["---", ""]


def _section_tip_count(out: list[str], combos: list[dict]) -> None:
    out += [
        "## 7. 프레임 내 팁 수별 성능",
        "",
        "`per_tip.csv`를 프레임별로 묶어 그 프레임의 GT 팁 수로 분류한 결과다. "
        f"팁 {MAX_TIP_BUCKET}개 이상은 한 구간으로 합쳤다.",
        "",
        "**test 스플릿의 팁 수 구성** (GT 어노테이션에서만 결정되므로 같은 데이터셋의 "
        "네 조합에서 동일하다. 도구가 없는 프레임은 `per_tip.csv`에 기록되지 않아 빠져 있다.)",
        "",
    ]
    rows = []
    for dataset in DATASETS:
        group = [c for c in combos if c["dataset"] == dataset]
        if not group:
            continue
        p = group[0]["per_tip"]
        counts = p["tip_count_frames"]
        # Same GT for every combination of a dataset; flag it if not.
        if any(c["per_tip"]["tip_count_frames"] != counts for c in group):
            rows.append([f"`{dataset}`", "조합별로 다름 — 어노테이션 갱신 이력 확인 필요",
                         "", "", "", "", ""])
            continue
        buckets = defaultdict(int)
        for n_tips, n_frames in counts.items():
            buckets[min(n_tips, MAX_TIP_BUCKET)] += n_frames
        cells = [f"`{dataset}`", _int(p["n_frames"]), _int(p["n_tips"])]
        for k in range(1, MAX_TIP_BUCKET + 1):
            n = buckets.get(k, 0)
            cells.append(f"{n:,} ({100.0 * n / p['n_frames']:.1f} %)")
        cells.append(_num(p["multi_tip_pct"], 1, " %"))
        rows.append(cells)
    out += _table(["데이터셋", "도구 보유 프레임", "GT 팁"]
                  + [f"팁 {k}개" if k < MAX_TIP_BUCKET else f"팁 {k}개+"
                     for k in range(1, MAX_TIP_BUCKET + 1)]
                  + ["다중 도구 프레임의 팁 비중"],
                  rows, "lrr" + "r" * MAX_TIP_BUCKET + "r")

    out += ["**팁 수별 지표** (Miss율 % / Hit@10 % / 평균 거리 px / 중앙값 거리 px)", ""]
    rows = []
    for c in combos:
        cells = [f"`{c['label']}`"]
        for k in range(1, MAX_TIP_BUCKET + 1):
            g = c["per_tip"]["by_tip_count"].get(k)
            cells.append("-" if not g else
                         f"{g['miss_pct']:.2f} / {g['hit_pct'][10]:.1f} / "
                         f"{g['mean']:.1f} / {g['median']:.1f}")
        rows.append(cells)
    out += _table(["조합"] + [f"팁 {k}개" if k < MAX_TIP_BUCKET else f"팁 {k}개+"
                             for k in range(1, MAX_TIP_BUCKET + 1)],
                  rows, "l" + "r" * MAX_TIP_BUCKET)
    out += ["---", ""]


def _section_shared_peaks(out: list[str], combos: list[dict]) -> None:
    out += [
        "## 8. 공유 피크 (과소 탐지) 분석",
        "",
        "평가 프로토콜은 GT 팁마다 독립적으로 최근접 피크를 고르므로 일대일 배정이 아니다. "
        "한 프레임의 여러 GT 팁이 **동일한 예측 좌표**에 매칭되면, 모델이 그 프레임의 도구 수보다 "
        "적은 피크를 만들었다는 뜻이다(과소 탐지). 아래는 매칭된 팁을 `공유`와 `단독`으로 나눈 결과다.",
        "",
    ]
    rows = []
    for c in combos:
        sh, so = c["per_tip"]["shared"], c["per_tip"]["solo"]
        rows.append([f"`{c['label']}`",
                     _int(sh["n_tips"]), _num(sh["share_of_all_pct"], 2, " %"),
                     _num(sh["share_of_multi_pct"], 2, " %"),
                     _num(sh["mean"], 2), _num(sh["median"], 2), _num(sh["p90"], 2),
                     _num(so["mean"], 2), _num(so["median"], 2), _num(so["p90"], 2),
                     _num(so["hit_pct"].get(10), 2, " %")])
    out += _table(["조합", "공유 팁", "공유 비율<br>(전체 GT)", "공유 비율<br>(다중 도구 팁)",
                   "공유<br>평균", "공유<br>중앙값", "공유<br>P90",
                   "단독<br>평균", "단독<br>중앙값", "단독<br>P90", "단독<br>Hit@10"],
                  rows, "lrrr" + "r" * 7)
    out += [
        "- `공유 비율(전체 GT)`의 분모는 미탐지를 포함한 전체 GT 팁 수, "
        "`공유 비율(다중 도구 팁)`의 분모는 GT 팁이 2개 이상인 프레임에 속한 팁 수다.",
        "- 거리 통계는 모두 px 단위이며 매칭된 팁만 모집단으로 한다.",
        "",
        "---",
        "",
    ]


def _section_offset(out: list[str], combos: list[dict]) -> None:
    out += [
        "## 9. 예측 좌표의 계통 편차",
        "",
        f"부호 있는 좌표 차이(dx = pred_x − gt_x, dy = pred_y − gt_y)를 "
        f"**{OFFSET_MATCH_RADIUS_PX} px 이내로 매칭된 팁**에 한해 집계한 값이다. 먼 거리 매칭은 "
        "다른 도구의 피크를 가리키므로 좌표 편차를 재는 데 쓸 수 없어 제외한다. "
        "좌표가 정수이므로 중앙값·최빈값도 정수 단위로 나온다.",
        "",
    ]
    rows = []
    for c in combos:
        o = c["per_tip"]["offset"]
        rows.append([f"`{c['label']}`", _int(o["n_near"]),
                     _signed(o["median_dx"], 0, " px"), _signed(o["median_dy"], 0, " px"),
                     _num(o["magnitude"], 1, " px"),
                     f"{_signed(o['mode_dx'], 0, ' px')} ({o['n_mode_dx']:,}건)",
                     f"{_signed(o['mode_dy'], 0, ' px')} ({o['n_mode_dy']:,}건)"])
    out += _table(["조합", f"≤ {OFFSET_MATCH_RADIUS_PX} px 매칭", "중앙값 dx", "중앙값 dy",
                   "편차 크기", "최빈 dx", "최빈 dy"], rows, "lrrrrrr")

    out += [
        "**편차 보정 효과.** 위 중앙값 편차를 예측 좌표에서 뺀 뒤 거리를 다시 계산한 결과다. "
        "모집단은 8절의 **단독 매칭** 팁으로 한정했다 — 공유 매칭 팁은 애초에 다른 도구를 "
        "가리키므로 좌표 보정과 무관하다. 편차가 (0, 0)인 조합은 보정 전후가 같다.",
        "",
    ]
    rows = []
    for c in combos:
        o = c["per_tip"]["offset"]
        raw, cor = c["per_tip"]["solo_raw"], c["per_tip"]["solo_corrected"]
        if not raw.get("n"):
            continue
        rows.append([f"`{c['label']}`",
                     f"({_signed(o['median_dx'], 0)}, {_signed(o['median_dy'], 0)})",
                     _int(raw["n"]),
                     _num(raw["median"], 2), _num(cor["median"], 2),
                     _num(raw["mean"], 2), _num(cor["mean"], 2),
                     _num(raw["p90"], 2), _num(cor["p90"], 2),
                     _num(raw["within_5"], 1), _num(cor["within_5"], 1),
                     _num(raw["within_10"], 1), _num(cor["within_10"], 1)])
    out += _table(["조합", "적용 편차", "단독 매칭 팁",
                   "중앙값<br>보정 전", "중앙값<br>보정 후",
                   "평균<br>보정 전", "평균<br>보정 후",
                   "P90<br>보정 전", "P90<br>보정 후",
                   "≤5 px %<br>보정 전", "≤5 px %<br>보정 후",
                   "≤10 px %<br>보정 전", "≤10 px %<br>보정 후"],
                  rows, "llr" + "r" * 10)
    out += ["---", ""]


def _section_sessions(out: list[str], combos: list[dict]) -> list[tuple[str, list[str]]]:
    """Per-session tables. Returns appendix blocks for datasets with many sessions."""
    out += [
        "## 10. 세션 / 영상별 지표",
        "",
        "`per_tip.csv`의 `session` 열로 묶은 결과다(평가 스크립트는 프레임 파일명에서 마지막 `_` "
        "앞부분을 세션 ID로 쓴다). `summary.json`의 `per_session`도 팁 수·미탐지 수·평균 거리를 "
        "담고 있으며 값이 일치한다.",
        "",
    ]
    appendix = []
    for dataset in DATASETS:
        group = [c for c in combos if c["dataset"] == dataset]
        if not group:
            continue
        sessions = sorted(group[0]["per_tip"]["per_session"])
        out += [f"### 10.{DATASETS.index(dataset) + 1} `{dataset}` "
                f"({len(sessions)}개 세션)", ""]

        headers = ["세션", "test 팁", "팁 비중"] + [
            f"`{c['target_mode']}`<br>`{c['model_type']}`" for c in group]
        note = ("각 조합 칸은 `Miss율 % / 평균 거리 px / Hit@10 %`다.")
        n_all = group[0]["per_tip"]["n_tips"]

        def session_row(sid):
            base = group[0]["per_tip"]["per_session"][sid]
            cells = [f"`{sid}`", _int(base["n_tips"]),
                     _num(100.0 * base["n_tips"] / n_all, 1, " %")]
            for c in group:
                g = c["per_tip"]["per_session"].get(sid)
                cells.append("-" if not g else
                             f"{g['miss_pct']:.2f} / {g['mean']:.1f} / {g['hit_pct'][10]:.1f}")
            return cells

        align = "lrr" + "r" * len(group)
        if len(sessions) <= SESSION_TABLE_MAX:
            out += [note, ""]
            rows = [session_row(s) for s in sessions]
            rows.append(["**전체**", _int(n_all), "100.0 %"]
                        + [f"{c['per_tip']['miss_pct']:.2f} / {c['per_tip']['mean']:.1f} / "
                           f"{c['per_tip']['hit_pct'][10]:.1f}" for c in group])
            out += _table(headers, rows, align)
            continue

        # Too many sessions for the main body: distribution here, full table in the appendix.
        out += [f"세션이 {len(sessions)}개이므로 분포와 상·하위 "
                f"{SESSION_EXTRACT_N}개만 싣고, 전체 표는 부록에 둔다. {note}", ""]
        rows = []
        for c in group:
            per = c["per_tip"]["per_session"]
            means = [v["mean"] for v in per.values() if v["mean"] is not None]
            misses = [v["miss_pct"] for v in per.values() if v["miss_pct"] is not None]
            rows.append([f"`{c['target_mode']}`/`{c['model_type']}`",
                         _num(min(means), 2), _num(float(np.median(means)), 2), _num(max(means), 2),
                         _num(min(misses), 2), _num(float(np.median(misses)), 2),
                         _num(max(misses), 2)])
        out += _table(["조합", "평균 거리 최소", "중앙값", "최대",
                       "Miss율 최소", "중앙값", "최대"], rows, "lrrrrrr")

        out += [f"**평균 거리 상·하위 {SESSION_EXTRACT_N}개 세션** (px)", ""]
        rows = []
        for c in group:
            per = c["per_tip"]["per_session"]
            ranked = sorted((v["mean"], s) for s, v in per.items() if v["mean"] is not None)
            best = ", ".join(f"`{s}` {m:.2f}" for m, s in ranked[:SESSION_EXTRACT_N])
            worst = ", ".join(f"`{s}` {m:.2f}" for m, s in reversed(ranked[-SESSION_EXTRACT_N:]))
            rows.append([f"`{c['target_mode']}`/`{c['model_type']}`", best, worst])
        out += _table(["조합", f"최우수 {SESSION_EXTRACT_N}", f"최저 {SESSION_EXTRACT_N}"], rows, "lll")

        block = [f"### `{dataset}` 세션별 전체 지표", "", note, ""]
        rows = [session_row(s) for s in sessions]
        rows.append(["**전체**", _int(n_all), "100.0 %"]
                    + [f"{c['per_tip']['miss_pct']:.2f} / {c['per_tip']['mean']:.1f} / "
                       f"{c['per_tip']['hit_pct'][10]:.1f}" for c in group])
        block += _table(headers, rows, align)
        appendix.append((dataset, block))

    out += ["---", ""]
    return appendix


def _section_speed(out: list[str], speed: list[dict]) -> None:
    out += ["## 11. 추론 속도", ""]
    if not speed:
        out += ["`speed-comparison.json`이 없다. `run compare-speed --dataset <이름>`으로 생성한다.",
                "", "---", ""]
        return

    out += ["`compare-speed`는 데이터셋·타겟 방식 단위로 파일을 쓴다(모델 타입별로 나뉘지 않는다). "
            "`forward`는 모델 forward 연산만, `wall`은 데이터 로딩·후처리를 포함한 종단 간 루프 시간이다.",
            ""]
    for entry in speed:
        d = entry["data"]
        out += [f"### `{entry['path']}`", ""]
        out += [f"- 측정 시각: {d.get('timestamp', '-')} / 장치: `{d.get('device', '-')}` / "
                f"스플릿: `{d.get('split', '-')}`",
                f"- 샘플: {_int(d.get('num_samples_used'))}건 "
                f"(seed {d.get('sample_seed', '-')}, batch {d.get('batch_size', '-')}, "
                f"workers {d.get('workers', '-')})",
                ""]
        rows = []
        for r in d.get("results", []):
            rows.append([f"`{r['model_type']}`", _int(r["parameter_count"]),
                         _num(r["forward_time_per_frame_ms"], 2, " ms"),
                         _num(r["wall_time_per_frame_ms"], 2, " ms"),
                         _num(r["forward_fps"], 2), _num(r["wall_fps"], 2)])
        out += _table(["모델", "파라미터 수", "Forward / frame", "Wall / frame",
                       "Forward FPS", "Wall FPS"], rows, "lrrrrr")

        results = d.get("results", [])
        if len(results) == 2:
            a, b = results
            out += [f"- 파라미터 비: `{b['model_type']}` / `{a['model_type']}` = "
                    f"{100.0 * b['parameter_count'] / a['parameter_count']:.1f} %"]
        cmp = d.get("comparison", {})
        for kind, label in (("forward", "forward"), ("wall", "wall-clock")):
            speedup = next((v for k, v in cmp.items() if k.endswith(f"_{kind}_speedup")), None)
            if speedup is None:
                continue
            faster = cmp.get(f"faster_model_by_{kind}_time", "-")
            out += [f"- {label} 기준 속도 향상: **{speedup:.2f}배** (빠른 쪽: `{faster}`)"]
        out += [""]
    out += ["---", ""]


def _section_sources(out: list[str], combos: list[dict], speed: list[dict]) -> None:
    out += ["## 12. 원본 파일과 산출 시각", "", "각 조합이 어떤 파일에서 왔고 그 파일이 언제 "
            "쓰였는지다. `평가 시각`은 `summary.json`에 기록된 실행 시각, 나머지는 파일 수정 시각이다.",
            ""]
    rows = []
    for c in combos:
        rows.append([f"`{c['label']}`",
                     c["status"].get("timestamp", "-"),
                     _file_time(os.path.join(c["model_dir"], "metric.csv")),
                     _eval_time(c["summary"]),
                     _file_time(os.path.join(c["results_dir"], "per_tip.csv"))])
    out += _table(["조합", "학습 완료 (train-status.json)", "metric.csv", "평가 (summary.json)",
                   "per_tip.csv"], rows, "lllll")
    if speed:
        out += _table(["속도 비교 파일", "측정 시각"],
                      [[f"`{e['path']}`", e["data"].get("timestamp", "-")] for e in speed], "ll")


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def build_document(combos: list[dict], incomplete: list[dict], speed: list[dict],
                   models_root: str, results_root: str) -> str:
    out: list[str] = []
    _section_overview(out, combos, incomplete, speed, models_root, results_root)
    if combos:
        _section_training(out, combos)
        _section_convergence(out, combos)
        _section_evaluation(out, combos)
        _section_axes(out, combos)
        _section_distribution(out, combos)
        _section_tip_count(out, combos)
        _section_shared_peaks(out, combos)
        _section_offset(out, combos)
        appendix = _section_sessions(out, combos)
        _section_speed(out, speed)
        _section_sources(out, combos, speed)
        if appendix:
            out += ["", "---", "", "## 부록. 세션별 전체 지표", ""]
            for _, block in appendix:
                out += block
    else:
        out += ["완비된 조합이 없어 집계할 수치가 없다. `run train-model`과 `run eval-model`을 "
                "먼저 실행한다.", ""]
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect experiment numbers from data/models and data/results "
                    "into a markdown summary for docs/final-report.md.")
    parser.add_argument("--models-root",  default="data/models",
                        help="root of trained checkpoints (default: data/models)")
    parser.add_argument("--results-root", default="data/results",
                        help="root of evaluation results (default: data/results)")
    parser.add_argument("--output",       default="docs/results-summary.md",
                        help="markdown file to write (default: docs/results-summary.md)")
    args = parser.parse_args()

    print(f"Scanning {args.models_root} and {args.results_root} ...")
    combos, incomplete = collect(args.models_root, args.results_root)
    speed = collect_speed(args.results_root)

    doc = build_document(combos, incomplete, speed, args.models_root, args.results_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"\n  {len(combos)} complete combination(s), {len(incomplete)} with missing files")
    print(f"  {len(speed)} speed-comparison file(s)")
    print(f"  → {args.output}  ({len(doc.splitlines()):,} lines)")


if __name__ == "__main__":
    main()
