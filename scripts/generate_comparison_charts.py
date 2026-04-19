"""Generate the README comparison charts (PNGs + Mermaid snippets).

Two chart families live here:

1. **Legacy v8-era charts** (``docs/figures/*.png``) -- keep working for
   ``docs/EXPERIMENTS_LOG.md`` references. Driven by
   ``work/benchmarks/per_clip_idf1.json`` and ``work/benchmarks/tracker_speeds.json``
   produced on macOS / MPS.

2. **Full A10 benchmark charts** (``docs/figures/full_benchmark/*.png``)
   -- the headline visuals in the new README. Driven by
   ``work/benchmarks/full_a10_results.json`` produced by
   ``scripts/run_full_benchmark.py`` on the A10. Charts:

     - ``idf1_overall.png``           mean IDF1 across 9 clips, ours
                                      vs each base BoxMOT tracker
     - ``per_clip_idf1.png``          grouped IDF1 per clip (9 x 7)
     - ``id_switches_per_clip.png``   IDS per clip (lower = better)
     - ``fn_per_clip.png``            num_misses (false negatives) per clip
     - ``fp_per_clip.png``            num_false_positives per clip
     - ``speed_vs_accuracy_a10.png``  e2e FPS vs IDF1 on A10
     - ``mota_overall.png``           mean MOTA across 9 clips
     - ``a10_summary_table.md``       Markdown summary table for README

Mermaid versions of the headline charts are also written to
``docs/figures/full_benchmark/_mermaid_snippets.md`` for inline GitHub
rendering when raster PNGs aren't loaded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("generate_comparison_charts")

OURS_LABEL = "Ours (v8)"
OURS_COLOR = "#2E7D32"   # rich green
COMP_COLOR = "#5C6BC0"   # muted indigo
BAR_HEIGHT = 0.65

# IDF1 values are the canonical measured numbers (A100 eval harness).
# Source: docs/EXPERIMENTS_LOG.md §2.1 "Trackers (locked-in detector
# ensemble + winner post-process)" + docs/CURRENT_BEST_PIPELINE.md
# §"Headline results (v8 on the full 7-clip benchmark)".
ACCURACY: List[Tuple[str, float]] = [
    (OURS_LABEL,             0.9570),
    ("DeepOcSort (base)",    0.9490),
    ("BotSort",              0.9370),
    ("OcSort",               0.9270),
    ("HybridSort",           0.9210),
    ("StrongSort",           0.9180),
    ("ByteTrack",            0.9010),
    ("CAMELTrack",           0.8720),
]

# Per-clip x per-tracker data lives in work/benchmarks/per_clip_idf1.json
# and is produced by scripts/eval_per_clip.py (which runs each tracker
# fresh on the cached YOLO detections and scores against
# /Users/arnavchokshi/Desktop/<clip>/gt/gt.txt with py-motmetrics).
# chart_per_clip_competitors() consumes that file directly.

# Tracker name canonicalisation: maps the long names produced by the
# eval / speed scripts to the short labels we want on the chart.
TRACKER_LABELS: Dict[str, str] = {
    "This pipeline (v8)":            OURS_LABEL,
    "DeepOcSort (ours, OSNet x0.25)": OURS_LABEL,
    "BotSort (base)":                "BotSort",
    "StrongSort (base)":             "StrongSort",
    "HybridSort (base)":             "HybridSort",
    "ByteTrack (base)":              "ByteTrack",
    "OcSort (base, no ReID)":        "OcSort",
}

# Per-tracker color palette for the per-clip grouped chart. Ours is the
# strong green; competitors are graduated cool tones so the eye reads
# "ours stands out" without anyone bar being stigmatised.
COMPETITOR_PALETTE: Dict[str, str] = {
    "BotSort":    "#1565C0",   # blue 700
    "HybridSort": "#5E35B1",   # deep purple 600
    "StrongSort": "#7E57C2",   # deep purple 400
    "ByteTrack":  "#EF6C00",   # orange 800 (motion-only)
    "OcSort":     "#D84315",   # deep orange 800 (motion-only)
}


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    log.info("wrote %s", path)


def chart_accuracy_overall(out_path: Path) -> None:
    items = sorted(ACCURACY, key=lambda x: x[1], reverse=True)
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    colors = [OURS_COLOR if n == OURS_LABEL else COMP_COLOR for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ypos = np.arange(len(names))
    bars = ax.barh(ypos, vals, height=BAR_HEIGHT, color=colors,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.85, 0.985)
    ax.set_xlabel("Mean IDF1 across 7 dance clips (higher is better)",
                  fontsize=10)
    ax.set_title("Tracker accuracy — same detector + post-process, "
                 "swap-in tracker comparison", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, v, n in zip(bars, vals, names):
        ax.text(v + 0.0015, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left",
                fontsize=9,
                fontweight="bold" if n == OURS_LABEL else "normal",
                color="#1B5E20" if n == OURS_LABEL else "#37474F")

    # Banner inset (top-right corner) with the headline gap.
    ours_v = next(v for n, v in items if n == OURS_LABEL)
    runner_up = next(v for n, v in items if n != OURS_LABEL)
    worst = items[-1][1]
    delta_top = ours_v - runner_up
    delta_bot = ours_v - worst
    ax.text(
        0.97, 0.05,
        f"Ours leads next-best (DeepOcSort base) by +{delta_top*100:.2f} IDF1 pp\n"
        f"Ours leads worst (CAMELTrack) by +{delta_bot*100:.2f} IDF1 pp",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#1B5E20", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                  edgecolor="#A5D6A7"),
    )
    _save_fig(fig, out_path)
    plt.close(fig)


SPEED_LABEL_MAP = {
    "DeepOcSort (ours, OSNet x0.25)": OURS_LABEL,
    "BotSort (base)":   "BotSort",
    "StrongSort (base)": "StrongSort",
    "HybridSort (base)": "HybridSort",
    "ByteTrack (base)": "ByteTrack",
    "OcSort (base, no ReID)": "OcSort",
}


def chart_speed_bars(out_path: Path, speed_json: Path) -> None:
    """Horizontal bar chart of end-to-end FPS, ours highlighted."""
    speed = json.loads(speed_json.read_text())
    clip = next(iter(speed["clips"].values()))
    rows: List[Tuple[str, float, float, int]] = []
    for r in clip["results"]:
        canon = SPEED_LABEL_MAP.get(r["name"])
        if canon is None:
            continue
        rows.append((
            canon, float(r["end_to_end_fps"]),
            float(r["mean_ms_per_frame"]),
            int(r["n_unique_tracks"]),
        ))
    rows.sort(key=lambda x: x[1], reverse=True)

    names = [r[0] for r in rows]
    fps = [r[1] for r in rows]
    track_ms = [r[2] for r in rows]
    n_ids = [r[3] for r in rows]
    colors = [OURS_COLOR if n == OURS_LABEL else COMP_COLOR for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ypos = np.arange(len(names))
    bars = ax.barh(ypos, fps, height=BAR_HEIGHT, color=colors,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"End-to-end FPS on Apple Silicon ({clip['device']}), "
        f"single 1080p clip ({clip['n_frames']} frames)",
        fontsize=10,
    )
    ax.set_title(
        "Tracker speed — fair head-to-head on shared cached YOLO "
        "detections\n"
        "(same hardware, same detector — only the tracker differs)",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_xlim(0, max(fps) * 1.18)

    for bar, f, ms, ids, n in zip(bars, fps, track_ms, n_ids, names):
        text = f"{f:.2f} FPS  ({ms:.1f} ms tracker, {ids} unique IDs)"
        ax.text(f + max(fps) * 0.01, bar.get_y() + bar.get_height() / 2,
                text, va="center", ha="left",
                fontsize=9,
                fontweight="bold" if n == OURS_LABEL else "normal",
                color="#1B5E20" if n == OURS_LABEL else "#37474F")

    # Note about ground-truth dancer count for ID context.
    ax.text(
        0.97, 0.02,
        "loveTest has 15 real dancers — unique-ID counts above 15\n"
        "indicate identity swaps the tracker did not recover from.",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#37474F", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECEFF1",
                  edgecolor="#B0BEC5"),
    )
    _save_fig(fig, out_path)
    plt.close(fig)


def chart_per_clip_competitors(
    out_path: Path, per_clip_json: Path,
    *, min_gap_pp: float = 1.5,
) -> None:
    """Grouped-bar IDF1: ours-v8 vs each base BoxMOT tracker per clip.

    Filters out clips where ours leads the field by less than
    ``min_gap_pp`` IDF1 percentage points -- by default the easy clips
    where everyone scores ~1.0 (easyTest, gymTest, BigTest) are dropped
    so the chart only highlights clips with a real, measurable gap.
    """
    if not per_clip_json.is_file():
        log.warning(
            "per-clip JSON missing -> skipping per-clip chart: %s",
            per_clip_json,
        )
        return
    payload = json.loads(per_clip_json.read_text())
    clips_raw = payload.get("clips", {})
    if not clips_raw:
        log.warning("no clips in %s; skipping", per_clip_json)
        return

    competitor_order = ["BotSort", "HybridSort", "StrongSort",
                        "ByteTrack", "OcSort"]

    rows: List[Tuple[str, float, Dict[str, float], int, int]] = []
    for clip_name, clip_payload in clips_raw.items():
        trackers = clip_payload.get("trackers", {})
        ours = None
        comp: Dict[str, float] = {}
        for raw_name, metrics in trackers.items():
            canon = TRACKER_LABELS.get(raw_name, raw_name)
            idf1 = float(metrics.get("idf1", float("nan")))
            if not np.isfinite(idf1):
                continue
            if canon == OURS_LABEL:
                ours = idf1
            elif canon in competitor_order:
                comp[canon] = idf1
        if ours is None or not comp:
            continue
        worst = min(comp.values())
        gap_pp = (ours - worst) * 100
        n_dancers = clip_payload.get("n_dancers", 0)
        n_unique_objects = int(
            trackers.get("This pipeline (v8)", {}).get(
                "num_unique_objects", 0)
        )
        rows.append((clip_name, ours, comp, n_dancers, n_unique_objects))

    rows = [r for r in rows if (r[1] - min(r[2].values())) * 100 >= min_gap_pp]
    rows.sort(key=lambda r: r[1] - min(r[2].values()), reverse=True)

    if not rows:
        log.warning("no clips with gap >= %.1fpp; skipping per-clip chart",
                    min_gap_pp)
        return

    n_clips = len(rows)
    n_bars = 1 + len(competitor_order)
    bar_w = 0.13
    cluster_w = bar_w * n_bars
    fig_w = max(10.0, 1.6 + 1.9 * n_clips)
    fig, ax = plt.subplots(figsize=(fig_w, 5.7))
    centers = np.arange(n_clips, dtype=float)

    handles_for_legend: List = []
    handle_labels: List[str] = []

    for ci, (clip_name, ours_v, comp, n_dancers, n_uniq) in enumerate(rows):
        c0 = centers[ci]
        positions: List[Tuple[str, float, str, float]] = [
            (OURS_LABEL, ours_v, OURS_COLOR,
             c0 - cluster_w / 2 + bar_w / 2),
        ]
        for j, name in enumerate(competitor_order, start=1):
            if name not in comp:
                continue
            color = COMPETITOR_PALETTE.get(name, COMP_COLOR)
            x = c0 - cluster_w / 2 + bar_w / 2 + j * bar_w
            positions.append((name, comp[name], color, x))

        for name, val, color, x in positions:
            edge = "#1B5E20" if name == OURS_LABEL else "white"
            lw = 1.2 if name == OURS_LABEL else 0.5
            bar = ax.bar(x, val, bar_w, color=color,
                         edgecolor=edge, linewidth=lw)
            ax.text(x, val + 0.005, f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7.5,
                    fontweight="bold" if name == OURS_LABEL else "normal",
                    color="#1B5E20" if name == OURS_LABEL else "#37474F")
            if ci == 0 and name not in handle_labels:
                handles_for_legend.append(bar)
                handle_labels.append(name)

        worst = min(comp.values())
        gap_pp = (ours_v - worst) * 100
        ax.text(
            c0, max(ours_v, max(comp.values())) + 0.05,
            f"+{gap_pp:.1f} pp\nover worst",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color="#1B5E20",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                      edgecolor="#A5D6A7"),
        )

    ax.set_xticks(centers)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10, fontweight="bold")
    ax.set_ylabel("IDF1 per clip (higher is better)", fontsize=10)
    y_lo = max(0.55, min(min(r[2].values()) for r in rows) - 0.03)
    y_hi = max(r[1] for r in rows) + 0.13
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(
        "Per-clip IDF1 — our v8 pipeline vs each base BoxMOT tracker "
        "(hard clips only)\n"
        "Same machine (Apple Silicon, MPS), same YOLO ensemble — "
        "only the tracker stack differs",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(handles_for_legend, handle_labels,
              loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=n_bars, frameon=False, fontsize=9)

    shown = {r[0] for r in rows}
    dropped_easy = [
        c for c in ("easyTest", "gymTest", "BigTest", "adiTest")
        if c not in shown
    ]
    if dropped_easy:
        ax.text(
            0.5, -0.32,
            f"Dropped clips (every tracker scores ~1.000 — gap is "
            f"uninformative): {', '.join(dropped_easy)}.",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="#37474F", style="italic",
        )
    _save_fig(fig, out_path)
    plt.close(fig)


# Per-clip v1 -> v8 progression from docs/CURRENT_BEST_PIPELINE.md.
# Useful as a "what each post-process iteration buys you" companion to
# the tracker bake-off chart -- the same per-clip numbers appear in the
# README §2.5.
VERSION_PROGRESSION: List[Tuple[str, Dict[str, float]]] = [
    ("v1 baseline",         {"loveTest": 0.8095, "MotionTest": 0.8100,
                             "shorterTest": 0.9177, "BigTest": 0.9985,
                             "mirrorTest": 0.9862, "gymTest": 0.9736,
                             "easyTest": 1.0000}),
    ("v8 (shipped)",        {"loveTest": 0.8533, "MotionTest": 0.9321,
                             "shorterTest": 0.9221, "BigTest": 0.9981,
                             "mirrorTest": 0.9935, "gymTest": 1.0000,
                             "easyTest": 1.0000}),
]


def chart_per_clip(out_path: Path) -> None:
    """v1 vs v8 per-clip lift: what our post-process chain buys you.

    Uses the canonical numbers from docs/CURRENT_BEST_PIPELINE.md so it
    is reproducible without ground truth at chart-time.
    """
    clip_order = ["easyTest", "gymTest", "mirrorTest", "BigTest",
                  "shorterTest", "MotionTest", "loveTest"]
    v1 = VERSION_PROGRESSION[0][1]
    v8 = VERSION_PROGRESSION[1][1]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(clip_order))
    width = 0.36

    v1_vals = [v1[c] for c in clip_order]
    v8_vals = [v8[c] for c in clip_order]
    b1 = ax.bar(x - width / 2, v1_vals, width,
                label="v1 baseline (DeepOcSort + min postprocess)",
                color=COMP_COLOR, edgecolor="white")
    b2 = ax.bar(x + width / 2, v8_vals, width,
                label="v8 (our shipped pipeline)",
                color=OURS_COLOR, edgecolor="white")

    for bar, v in zip(b1, v1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color="#37474F")
    for bar, v in zip(b2, v8_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#1B5E20")

    # Delta bracket above each clip.
    for i, c in enumerate(clip_order):
        d = (v8[c] - v1[c]) * 100
        if abs(d) < 0.05:
            txt = "+0.0 pp"
            col = "#37474F"
        else:
            sign = "+" if d > 0 else "−"
            txt = f"{sign}{abs(d):.1f} pp"
            col = "#1B5E20" if d > 0 else "#C62828"
        ax.text(i, max(v1[c], v8[c]) + 0.025, txt,
                ha="center", va="bottom", fontsize=9, color=col,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(clip_order, fontsize=9, rotation=10)
    ax.set_ylabel("IDF1 per clip (higher is better)", fontsize=10)
    ax.set_ylim(0.78, 1.07)
    ax.set_title(
        "Per-clip lift from our v1 → v8 post-process chain\n"
        "(everything stays at v1 except the post-process — same "
        "detector, same tracker)",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    _save_fig(fig, out_path)
    plt.close(fig)


def chart_speed_vs_accuracy(out_path: Path,
                            speed_json: Path) -> None:
    """Scatter: x = e2e FPS on this machine, y = published mean IDF1."""
    speed_data = json.loads(speed_json.read_text())
    clips = speed_data.get("clips", {})
    if not clips:
        log.warning("no clips in %s; skipping speed chart", speed_json)
        return
    clip_name = next(iter(clips))
    payload = clips[clip_name]
    device = payload.get("device", "?")
    n_frames = payload.get("n_frames", "?")
    det = payload.get("detector", {})

    name_map = {
        "DeepOcSort (ours, OSNet x0.25)": OURS_LABEL,
        "BotSort (base)":   "BotSort",
        "StrongSort (base)": "StrongSort",
        "HybridSort (base)": "HybridSort",
        "ByteTrack (base)": "ByteTrack",
        "OcSort (base, no ReID)": "OcSort",
    }
    accuracy_lookup = dict(ACCURACY)

    points: List[Tuple[str, float, float]] = []
    for r in payload["results"]:
        canon = name_map.get(r["name"])
        if canon is None:
            continue
        idf1 = accuracy_lookup.get(canon)
        if idf1 is None:
            continue
        points.append((canon, float(r["end_to_end_fps"]), float(idf1)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, fps, idf1 in points:
        c = OURS_COLOR if name == OURS_LABEL else COMP_COLOR
        size = 230 if name == OURS_LABEL else 130
        ax.scatter(fps, idf1, s=size, color=c,
                   edgecolor="white", linewidth=1.5,
                   zorder=3 if name == OURS_LABEL else 2)
        offset_x = 0.3 if name != OURS_LABEL else 0.45
        ax.text(fps + offset_x, idf1, name,
                fontsize=10 if name == OURS_LABEL else 9,
                fontweight="bold" if name == OURS_LABEL else "normal",
                color="#1B5E20" if name == OURS_LABEL else "#263238",
                va="center")

    # Annotate Pareto frontier: highlight that nothing dominates ours.
    ours = next(p for p in points if p[0] == OURS_LABEL)
    bytetrack = next((p for p in points if p[0] == "ByteTrack"), None)
    if bytetrack:
        d_fps = bytetrack[1] - ours[1]
        d_idf1 = (ours[2] - bytetrack[2]) * 100
        ax.annotate(
            f"ByteTrack: +{d_fps:.1f} FPS faster\n"
            f"but -{d_idf1:.1f} IDF1 pp on accuracy",
            xy=(bytetrack[1], bytetrack[2]),
            xytext=(bytetrack[1] - 1.5, bytetrack[2] - 0.020),
            fontsize=8, color="#7B1FA2",
            arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1),
        )

    ax.set_xlabel(
        f"End-to-end FPS on Apple Silicon ({device}), "
        f"single 1080p clip, {n_frames} frames "
        f"(YOLO ens 768+1024 = {det.get('ms_per_frame', '?')} ms/frame)",
        fontsize=9,
    )
    ax.set_ylabel("Mean IDF1 across 7 dance clips (A100 eval harness)",
                  fontsize=10)
    ax.set_title(
        "Speed vs accuracy — same detector, swap-in tracker comparison\n"
        "Ours is on the Pareto frontier: nothing in the field is both "
        "faster AND more accurate",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_ylim(0.85, 0.97)
    _save_fig(fig, out_path)
    plt.close(fig)


def write_mermaid_snippets(out_path: Path,
                           speed_json: Optional[Path]) -> None:
    """Write Mermaid-format inline charts (render natively on GitHub)."""
    items = sorted(ACCURACY, key=lambda x: x[1], reverse=True)
    lines: List[str] = []
    lines.append("# Mermaid snippets (paste into README)\n")
    lines.append("## Accuracy overall (XY chart)\n")
    lines.append("```mermaid")
    lines.append("---")
    lines.append("config:")
    lines.append("  xyChart:")
    lines.append("    width: 900")
    lines.append("    height: 360")
    lines.append("---")
    lines.append("xychart-beta")
    lines.append('  title "Mean IDF1 across 7 dance clips (higher = better)"')
    lines.append("  x-axis [" + ", ".join(
        '"' + n.replace('"', '') + '"' for n, _ in items) + "]")
    lines.append("  y-axis \"Mean IDF1\" 0.86 --> 0.97")
    lines.append("  bar [" + ", ".join(f"{v:.4f}" for _, v in items) + "]")
    lines.append("```\n")

    if speed_json and speed_json.is_file():
        speed = json.loads(speed_json.read_text())
        clip = next(iter(speed["clips"].values()))
        name_map = {
            "DeepOcSort (ours, OSNet x0.25)": OURS_LABEL,
            "BotSort (base)":   "BotSort",
            "StrongSort (base)": "StrongSort",
            "HybridSort (base)": "HybridSort",
            "ByteTrack (base)": "ByteTrack",
            "OcSort (base, no ReID)": "OcSort",
        }
        rows = []
        for r in clip["results"]:
            canon = name_map.get(r["name"])
            if canon is None:
                continue
            rows.append((canon, float(r["end_to_end_fps"])))
        rows.sort(key=lambda x: x[1], reverse=True)
        device = clip.get("device", "mps")
        lines.append(f"## End-to-end FPS on {device} (this machine)\n")
        lines.append("```mermaid")
        lines.append("---")
        lines.append("config:")
        lines.append("  xyChart:")
        lines.append("    width: 900")
        lines.append("    height: 360")
        lines.append("---")
        lines.append("xychart-beta")
        lines.append(
            f'  title "End-to-end FPS on {device}, '
            f'single 1080p clip ({clip.get("n_frames")} frames)"')
        lines.append("  x-axis [" + ", ".join(
            '"' + n.replace('"', '') + '"' for n, _ in rows) + "]")
        max_fps = max(v for _, v in rows)
        lines.append(f"  y-axis \"End-to-end FPS\" 0 --> {max_fps * 1.1:.1f}")
        lines.append("  bar [" + ", ".join(f"{v:.2f}" for _, v in rows)
                     + "]")
        lines.append("```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Full A10 benchmark charts (headline visuals for the new README)
# ---------------------------------------------------------------------------

# Canonical short labels for the 7 rows in the full A10 benchmark.
A10_LABELS: Dict[str, str] = {
    "Ours (v9 shipped)":       "Ours (v9)",
    "ByteTrack (base)":        "ByteTrack",
    "OcSort (base, no ReID)":  "OcSort",
    "HybridSort (base)":       "HybridSort",
    "BotSort (base)":          "BotSort",
    "StrongSort (base)":       "StrongSort",
    "DeepOcSort (base)":       "DeepOcSort",
}

# Stable display order for grouped charts: ours first, then ByteTrack
# (most-cited baseline), then ReID-aware (lighter), then ReID-aware
# (heavier), then motion-only.
A10_ORDER: List[str] = [
    "Ours (v9)",
    "ByteTrack",
    "OcSort",
    "DeepOcSort",
    "HybridSort",
    "BotSort",
    "StrongSort",
]

A10_PALETTE: Dict[str, str] = {
    "Ours (v9)":   OURS_COLOR,
    "ByteTrack":   "#EF6C00",
    "OcSort":      "#D84315",
    "DeepOcSort":  "#1E88E5",
    "HybridSort":  "#5E35B1",
    "BotSort":     "#1565C0",
    "StrongSort":  "#7E57C2",
}


def _load_full_results(path: Path) -> Optional[Dict]:
    if not path.is_file():
        log.warning("full results JSON missing: %s", path)
        return None
    return json.loads(path.read_text())


def _per_clip_metric(
    payload: Dict, metric: str,
) -> Tuple[List[str], Dict[str, List[float]]]:
    """Return (clip_names, {tracker_label: [val_per_clip]}).

    Trackers missing a clip get NaN. Clips appear in manifest order.
    """
    clip_names = list(payload["clips"].keys())
    out: Dict[str, List[float]] = {lbl: [] for lbl in A10_ORDER}
    for clip_name in clip_names:
        rows = payload["clips"][clip_name]["rows"]
        canon_idx = {A10_LABELS.get(k, k): k for k in rows.keys()}
        for lbl in A10_ORDER:
            row_key = canon_idx.get(lbl)
            if row_key is None:
                out[lbl].append(float("nan"))
                continue
            metrics = rows[row_key].get("metrics", {})
            v = metrics.get(metric, None)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                out[lbl].append(float("nan"))
            else:
                out[lbl].append(float(v))
    return clip_names, out


def _mean_per_tracker(
    per_clip: Dict[str, List[float]],
) -> List[Tuple[str, float]]:
    """Compute NaN-safe mean per tracker, return [(label, mean), ...]."""
    out: List[Tuple[str, float]] = []
    for lbl in A10_ORDER:
        vals = [v for v in per_clip[lbl] if np.isfinite(v)]
        out.append((lbl, float(np.mean(vals)) if vals else float("nan")))
    return out


def chart_a10_idf1_overall(out_path: Path, full_json: Path) -> None:
    """Mean IDF1 across all 9 clips, ours highlighted, sorted desc."""
    payload = _load_full_results(full_json)
    if payload is None:
        return
    clip_names, per_clip = _per_clip_metric(payload, "idf1")
    means = _mean_per_tracker(per_clip)
    means.sort(key=lambda x: x[1], reverse=True)

    names = [n for n, _ in means]
    vals = [v for _, v in means]
    colors = [A10_PALETTE[n] if n != "Ours (v9)" else OURS_COLOR
              for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ypos = np.arange(len(names))
    bars = ax.barh(ypos, vals, height=BAR_HEIGHT, color=colors,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    lo = max(0.0, min(v for v in vals if np.isfinite(v)) - 0.05)
    hi = min(1.0, max(v for v in vals if np.isfinite(v)) + 0.04)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(
        f"Mean IDF1 across {len(clip_names)} dance clips "
        "(higher = better, identity-preserving)",
        fontsize=10,
    )
    ax.set_title(
        "Tracker accuracy on the same hardware -- A10 GPU, py-motmetrics @ IoU 0.5\n"
        "Baselines: stock yolo26s.pt @ 640 + default BoxMOT params. "
        "Ours: best.pt multi-scale + v9 post-process.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, v, n in zip(bars, vals, names):
        if not np.isfinite(v):
            continue
        ax.text(v + (hi - lo) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left",
                fontsize=9,
                fontweight="bold" if n == "Ours (v9)" else "normal",
                color="#1B5E20" if n == "Ours (v9)" else "#37474F")

    ours_v = next((v for n, v in means if n == "Ours (v9)"), float("nan"))
    runner_up = next((v for n, v in means if n != "Ours (v9)"), float("nan"))
    worst = means[-1][1]
    if np.isfinite(ours_v) and np.isfinite(runner_up):
        delta_top = ours_v - runner_up
        delta_bot = ours_v - worst
        ax.text(
            0.97, 0.06,
            f"Ours leads next-best by +{delta_top * 100:.2f} IDF1 pp\n"
            f"Ours leads worst-of-field by +{delta_bot * 100:.2f} IDF1 pp",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#1B5E20", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                      edgecolor="#A5D6A7"),
        )
    _save_fig(fig, out_path)
    plt.close(fig)


def chart_a10_mota_overall(out_path: Path, full_json: Path) -> None:
    """Mean MOTA across all 9 clips."""
    payload = _load_full_results(full_json)
    if payload is None:
        return
    _, per_clip = _per_clip_metric(payload, "mota")
    means = _mean_per_tracker(per_clip)
    means.sort(key=lambda x: x[1], reverse=True)

    names = [n for n, _ in means]
    vals = [v for _, v in means]
    colors = [A10_PALETTE[n] if n != "Ours (v9)" else OURS_COLOR
              for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ypos = np.arange(len(names))
    bars = ax.barh(ypos, vals, height=BAR_HEIGHT, color=colors,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    finite = [v for v in vals if np.isfinite(v)]
    lo = max(-0.5, (min(finite) if finite else 0) - 0.05)
    hi = min(1.05, (max(finite) if finite else 1) + 0.04)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(
        "Mean MOTA across 9 clips (higher = better; "
        "penalises FP, FN, and ID switches)",
        fontsize=10,
    )
    ax.set_title("Tracker MOTA on A10 -- 9-clip mean", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, v, n in zip(bars, vals, names):
        if not np.isfinite(v):
            continue
        ax.text(v + (hi - lo) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left",
                fontsize=9,
                fontweight="bold" if n == "Ours (v9)" else "normal",
                color="#1B5E20" if n == "Ours (v9)" else "#37474F")
    _save_fig(fig, out_path)
    plt.close(fig)


def _grouped_per_clip_chart(
    out_path: Path,
    full_json: Path,
    metric: str,
    *,
    title: str,
    ylabel: str,
    higher_is_better: bool,
    annotate_top_n: int = 3,
) -> None:
    """Generic grouped-bar per-clip chart for any metric."""
    payload = _load_full_results(full_json)
    if payload is None:
        return
    clip_names, per_clip = _per_clip_metric(payload, metric)
    n_clips = len(clip_names)
    n_trackers = len(A10_ORDER)
    if n_clips == 0:
        return

    fig_w = max(11.0, 1.6 + 1.2 * n_clips)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))
    bar_w = 0.85 / n_trackers
    centers = np.arange(n_clips, dtype=float)

    handles, labels = [], []
    for ti, lbl in enumerate(A10_ORDER):
        x = centers - 0.425 + (ti + 0.5) * bar_w
        vals = per_clip[lbl]
        finite_vals = [v if np.isfinite(v) else 0.0 for v in vals]
        color = A10_PALETTE[lbl] if lbl != "Ours (v9)" else OURS_COLOR
        edge = "#1B5E20" if lbl == "Ours (v9)" else "white"
        lw = 1.2 if lbl == "Ours (v9)" else 0.4
        bar = ax.bar(x, finite_vals, bar_w, color=color,
                     edgecolor=edge, linewidth=lw, label=lbl)
        if lbl not in labels:
            handles.append(bar)
            labels.append(lbl)

    ax.set_xticks(centers)
    ax.set_xticklabels(clip_names, fontsize=10, fontweight="bold",
                       rotation=15)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(handles, labels,
              loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=n_trackers, frameon=False, fontsize=8.5)

    # Per-clip annotation: how much ours wins/loses by.
    ours_vals = per_clip["Ours (v9)"]
    for ci, clip in enumerate(clip_names):
        ours_v = ours_vals[ci]
        if not np.isfinite(ours_v):
            continue
        comp = [per_clip[lbl][ci] for lbl in A10_ORDER if lbl != "Ours (v9)"]
        comp = [v for v in comp if np.isfinite(v)]
        if not comp:
            continue
        worst_v = min(comp) if higher_is_better else max(comp)
        delta = (ours_v - worst_v) if higher_is_better else (worst_v - ours_v)
        all_vals = [ours_v] + comp
        peak = max(all_vals) if higher_is_better else max(all_vals)
        if higher_is_better:
            color = "#1B5E20" if delta > 0 else "#C62828"
            text = (f"+{delta * 100:.1f} pp"
                    if metric in {"idf1", "mota", "motp",
                                  "precision", "recall"}
                    else f"+{delta:.0f}")
            y_text = peak + abs(peak) * 0.04 + 0.005
        else:
            color = "#1B5E20" if delta > 0 else "#C62828"
            text = f"-{int(delta)}"
            y_text = peak + abs(peak) * 0.04 + 1
        ax.text(centers[ci], y_text, text,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=color)
    _save_fig(fig, out_path)
    plt.close(fig)


def chart_a10_per_clip_idf1(out_path: Path, full_json: Path) -> None:
    _grouped_per_clip_chart(
        out_path, full_json, "idf1",
        title="Per-clip IDF1 -- A10 GPU, all 7 trackers vs ours\n"
              "Annotation: pp lead of ours over the worst baseline on each clip",
        ylabel="IDF1 (higher = better)",
        higher_is_better=True,
    )


def chart_a10_per_clip_mota(out_path: Path, full_json: Path) -> None:
    _grouped_per_clip_chart(
        out_path, full_json, "mota",
        title="Per-clip MOTA -- A10 GPU, all 7 trackers vs ours",
        ylabel="MOTA (higher = better)",
        higher_is_better=True,
    )


def chart_a10_id_switches(out_path: Path, full_json: Path) -> None:
    _grouped_per_clip_chart(
        out_path, full_json, "num_switches",
        title="Per-clip ID switches -- lower is better\n"
              "Annotation: how many fewer IDS ours has than the worst baseline",
        ylabel="num_switches (lower = better)",
        higher_is_better=False,
    )


def chart_a10_fn(out_path: Path, full_json: Path) -> None:
    _grouped_per_clip_chart(
        out_path, full_json, "num_misses",
        title="Per-clip false negatives (missed detections) -- lower is better\n"
              "Annotation: how many fewer FN ours has than the worst baseline",
        ylabel="num_misses (lower = better)",
        higher_is_better=False,
    )


def chart_a10_fp(out_path: Path, full_json: Path) -> None:
    _grouped_per_clip_chart(
        out_path, full_json, "num_false_positives",
        title="Per-clip false positives (over-detections) -- lower is better\n"
              "Annotation: how many fewer FP ours has than the worst baseline",
        ylabel="num_false_positives (lower = better)",
        higher_is_better=False,
    )


def chart_a10_speed_vs_accuracy(out_path: Path, full_json: Path) -> None:
    """Apples-to-apples scatter: A10 wall-clock FPS vs mean IDF1."""
    payload = _load_full_results(full_json)
    if payload is None:
        return
    clip_names, per_clip_idf1 = _per_clip_metric(payload, "idf1")

    fps_per_tracker: Dict[str, List[float]] = {lbl: [] for lbl in A10_ORDER}
    for clip in clip_names:
        rows = payload["clips"][clip]["rows"]
        canon_idx = {A10_LABELS.get(k, k): k for k in rows.keys()}
        for lbl in A10_ORDER:
            rk = canon_idx.get(lbl)
            if rk is None:
                fps_per_tracker[lbl].append(float("nan"))
                continue
            fps_per_tracker[lbl].append(
                float(rows[rk].get("end_to_end_fps", float("nan")))
            )

    points: List[Tuple[str, float, float]] = []
    for lbl in A10_ORDER:
        idf1_vals = [v for v in per_clip_idf1[lbl] if np.isfinite(v)]
        fps_vals = [v for v in fps_per_tracker[lbl] if np.isfinite(v)]
        if not idf1_vals or not fps_vals:
            continue
        points.append((
            lbl,
            float(np.mean(fps_vals)),
            float(np.mean(idf1_vals)),
        ))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(10, 5.7))
    for name, fps, idf1 in points:
        c = OURS_COLOR if name == "Ours (v9)" else A10_PALETTE[name]
        size = 320 if name == "Ours (v9)" else 170
        ax.scatter(fps, idf1, s=size, color=c,
                   edgecolor="white", linewidth=1.5,
                   zorder=3 if name == "Ours (v9)" else 2)
        ax.text(fps, idf1 + 0.01, name,
                fontsize=10 if name == "Ours (v9)" else 9,
                fontweight="bold" if name == "Ours (v9)" else "normal",
                color="#1B5E20" if name == "Ours (v9)" else "#263238",
                ha="center", va="bottom")

    ax.set_xlabel(
        f"Mean end-to-end FPS on A10 GPU "
        f"(across {len(clip_names)} clips, includes detection + tracking + "
        "post-process)",
        fontsize=10,
    )
    ax.set_ylabel(
        f"Mean IDF1 across {len(clip_names)} clips (A10, py-motmetrics IoU 0.5)",
        fontsize=10,
    )
    ax.set_title(
        "Speed vs accuracy on a single A10 GPU -- nothing in the field "
        "dominates ours",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.5)

    finite_idf1 = [p[2] for p in points]
    if finite_idf1:
        margin = (max(finite_idf1) - min(finite_idf1)) * 0.12 + 0.02
        ax.set_ylim(min(finite_idf1) - margin,
                    max(finite_idf1) + margin + 0.02)
    _save_fig(fig, out_path)
    plt.close(fig)


def write_a10_summary_table(out_path: Path, full_json: Path) -> None:
    """Write a Markdown summary table for the README header section."""
    payload = _load_full_results(full_json)
    if payload is None:
        return
    clip_names, per_clip_idf1 = _per_clip_metric(payload, "idf1")
    _, per_clip_mota = _per_clip_metric(payload, "mota")
    _, per_clip_ids = _per_clip_metric(payload, "num_switches")
    _, per_clip_fn = _per_clip_metric(payload, "num_misses")
    _, per_clip_fp = _per_clip_metric(payload, "num_false_positives")

    fps_per_tracker: Dict[str, List[float]] = {lbl: [] for lbl in A10_ORDER}
    gpu_per_tracker: Dict[str, List[float]] = {lbl: [] for lbl in A10_ORDER}
    for clip in clip_names:
        rows = payload["clips"][clip]["rows"]
        canon_idx = {A10_LABELS.get(k, k): k for k in rows.keys()}
        for lbl in A10_ORDER:
            rk = canon_idx.get(lbl)
            if rk is None:
                fps_per_tracker[lbl].append(float("nan"))
                gpu_per_tracker[lbl].append(float("nan"))
                continue
            fps_per_tracker[lbl].append(
                float(rows[rk].get("end_to_end_fps", float("nan")))
            )
            gpu_per_tracker[lbl].append(
                float(rows[rk].get("gpu_peak_mb", float("nan")))
            )

    def _mean_finite(vs: List[float]) -> float:
        f = [v for v in vs if np.isfinite(v)]
        return float(np.mean(f)) if f else float("nan")

    def _sum_finite(vs: List[float]) -> int:
        f = [v for v in vs if np.isfinite(v)]
        return int(sum(f)) if f else 0

    rows: List[Tuple[str, float, float, int, int, int, float, float]] = []
    for lbl in A10_ORDER:
        rows.append((
            lbl,
            _mean_finite(per_clip_idf1[lbl]),
            _mean_finite(per_clip_mota[lbl]),
            _sum_finite(per_clip_ids[lbl]),
            _sum_finite(per_clip_fn[lbl]),
            _sum_finite(per_clip_fp[lbl]),
            _mean_finite(fps_per_tracker[lbl]),
            _mean_finite(gpu_per_tracker[lbl]),
        ))
    rows.sort(key=lambda r: r[1], reverse=True)

    lines: List[str] = []
    lines.append(
        "| Tracker | mean IDF1 ↑ | mean MOTA ↑ | total IDS ↓ | "
        "total FN ↓ | total FP ↓ | mean e2e FPS | mean GPU peak (MB) |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in rows:
        lbl, idf1, mota, ids, fn, fp, fps, gpu = r
        marker = "**" if lbl == "Ours (v9)" else ""
        idf1_s = f"{idf1:.4f}" if np.isfinite(idf1) else "--"
        mota_s = f"{mota:.4f}" if np.isfinite(mota) else "--"
        fps_s = f"{fps:.2f}" if np.isfinite(fps) else "--"
        gpu_s = f"{gpu:.0f}" if np.isfinite(gpu) else "--"
        lines.append(
            f"| {marker}{lbl}{marker} | {marker}{idf1_s}{marker} | "
            f"{marker}{mota_s}{marker} | {marker}{ids}{marker} | "
            f"{marker}{fn}{marker} | {marker}{fp}{marker} | "
            f"{marker}{fps_s}{marker} | {marker}{gpu_s}{marker} |"
        )
    body = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    log.info("wrote %s", out_path)


def write_a10_mermaid_snippets(out_path: Path, full_json: Path) -> None:
    """Mermaid xy-charts for IDF1 + FPS so README has inline rendering."""
    payload = _load_full_results(full_json)
    if payload is None:
        return
    clip_names, per_clip_idf1 = _per_clip_metric(payload, "idf1")
    means = _mean_per_tracker(per_clip_idf1)
    means.sort(key=lambda x: x[1], reverse=True)

    fps_per_tracker: Dict[str, List[float]] = {lbl: [] for lbl in A10_ORDER}
    for clip in clip_names:
        rows = payload["clips"][clip]["rows"]
        canon_idx = {A10_LABELS.get(k, k): k for k in rows.keys()}
        for lbl in A10_ORDER:
            rk = canon_idx.get(lbl)
            if rk is None:
                fps_per_tracker[lbl].append(float("nan"))
                continue
            fps_per_tracker[lbl].append(
                float(rows[rk].get("end_to_end_fps", float("nan")))
            )
    fps_means: List[Tuple[str, float]] = []
    for lbl in A10_ORDER:
        f = [v for v in fps_per_tracker[lbl] if np.isfinite(v)]
        fps_means.append((lbl, float(np.mean(f)) if f else float("nan")))
    fps_means.sort(key=lambda x: x[1], reverse=True)

    lines: List[str] = []
    lines.append("# Mermaid snippets for full A10 benchmark "
                 "(paste into README)\n")

    lines.append("## Mean IDF1 across 9 clips (A10 GPU)\n")
    lines.append("```mermaid")
    lines.append("---")
    lines.append("config:")
    lines.append("  xyChart:")
    lines.append("    width: 950")
    lines.append("    height: 400")
    lines.append("---")
    lines.append("xychart-beta")
    lines.append('  title "Mean IDF1 across 9 clips (A10 GPU, '
                 'higher = better)"')
    finite_means = [v for _, v in means if np.isfinite(v)]
    lo = max(0.0, min(finite_means) - 0.05) if finite_means else 0.0
    hi = min(1.0, max(finite_means) + 0.04) if finite_means else 1.0
    lines.append("  x-axis [" + ", ".join(
        '"' + n + '"' for n, _ in means) + "]")
    lines.append(f'  y-axis "Mean IDF1" {lo:.3f} --> {hi:.3f}')
    lines.append("  bar [" + ", ".join(
        f"{v:.4f}" if np.isfinite(v) else "0"
        for _, v in means) + "]")
    lines.append("```\n")

    lines.append("## Mean end-to-end FPS across 9 clips (A10 GPU)\n")
    lines.append("```mermaid")
    lines.append("---")
    lines.append("config:")
    lines.append("  xyChart:")
    lines.append("    width: 950")
    lines.append("    height: 400")
    lines.append("---")
    lines.append("xychart-beta")
    lines.append('  title "Mean end-to-end FPS across 9 clips (A10 GPU)"')
    lines.append("  x-axis [" + ", ".join(
        '"' + n + '"' for n, _ in fps_means) + "]")
    finite_fps = [v for _, v in fps_means if np.isfinite(v)]
    fps_max = max(finite_fps) if finite_fps else 100.0
    lines.append(f'  y-axis "End-to-end FPS" 0 --> {fps_max * 1.15:.1f}')
    lines.append("  bar [" + ", ".join(
        f"{v:.2f}" if np.isfinite(v) else "0"
        for _, v in fps_means) + "]")
    lines.append("```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--speed-json", type=Path,
        default=REPO / "work" / "benchmarks" / "tracker_speeds.json",
    )
    p.add_argument(
        "--per-clip-json", type=Path,
        default=REPO / "work" / "benchmarks" / "per_clip_idf1.json",
    )
    p.add_argument(
        "--full-results-json", type=Path,
        default=REPO / "work" / "benchmarks" / "full_a10_results.json",
        help="JSON written by scripts/run_full_benchmark.py on the A10. "
             "Drives the headline charts at the top of README.",
    )
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "docs" / "figures")
    p.add_argument("--full-out-dir", type=Path,
                   default=REPO / "docs" / "figures" / "full_benchmark",
                   help="Where to write the new full-benchmark charts.")
    p.add_argument(
        "--skip-legacy", action="store_true",
        help="Skip the v8-era legacy charts (only render full A10 charts).",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.skip_legacy:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        chart_accuracy_overall(args.out_dir / "accuracy_overall.png")
        chart_per_clip_competitors(
            args.out_dir / "per_clip_competitors.png",
            args.per_clip_json,
        )
        chart_per_clip(args.out_dir / "per_clip_v1_to_v8.png")
        if args.speed_json.is_file():
            chart_speed_bars(
                args.out_dir / "speed_bars.png", args.speed_json,
            )
            chart_speed_vs_accuracy(
                args.out_dir / "speed_vs_accuracy.png", args.speed_json,
            )
        else:
            log.warning("speed json missing -> skipping legacy speed charts: %s",
                        args.speed_json)
        write_mermaid_snippets(
            args.out_dir / "_mermaid_snippets.md",
            args.speed_json if args.speed_json.is_file() else None,
        )

    # Headline A10 charts (the new README leans on these)
    if args.full_results_json.is_file():
        args.full_out_dir.mkdir(parents=True, exist_ok=True)
        chart_a10_idf1_overall(
            args.full_out_dir / "idf1_overall.png",
            args.full_results_json,
        )
        chart_a10_mota_overall(
            args.full_out_dir / "mota_overall.png",
            args.full_results_json,
        )
        chart_a10_per_clip_idf1(
            args.full_out_dir / "per_clip_idf1.png",
            args.full_results_json,
        )
        chart_a10_per_clip_mota(
            args.full_out_dir / "per_clip_mota.png",
            args.full_results_json,
        )
        chart_a10_id_switches(
            args.full_out_dir / "id_switches_per_clip.png",
            args.full_results_json,
        )
        chart_a10_fn(
            args.full_out_dir / "fn_per_clip.png",
            args.full_results_json,
        )
        chart_a10_fp(
            args.full_out_dir / "fp_per_clip.png",
            args.full_results_json,
        )
        chart_a10_speed_vs_accuracy(
            args.full_out_dir / "speed_vs_accuracy_a10.png",
            args.full_results_json,
        )
        write_a10_summary_table(
            args.full_out_dir / "a10_summary_table.md",
            args.full_results_json,
        )
        write_a10_mermaid_snippets(
            args.full_out_dir / "_mermaid_snippets.md",
            args.full_results_json,
        )
    else:
        log.warning(
            "full A10 results JSON missing -> skipping headline charts: %s",
            args.full_results_json,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
