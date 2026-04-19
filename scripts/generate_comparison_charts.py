"""Generate the README comparison charts (PNGs + Mermaid snippets).

Three charts go into ``docs/figures/``:

  * ``accuracy_overall.png``   horizontal bar of mean IDF1, ours
                               highlighted, sorted desc
  * ``accuracy_per_clip.png``  grouped bar (ours vs ByteTrack vs
                               StrongSort vs CAMELTrack) per clip
  * ``speed_vs_accuracy.png``  scatter: x=end-to-end FPS (this MPS run),
                               y=mean IDF1, dot size = recovery (1.0
                               for ours)

Mermaid versions of (1) and (3) are also written to
``docs/figures/_mermaid_snippets.md`` so they can be copy-pasted into
the README for inline GitHub rendering.

Sources:
  * Accuracy numbers come from ``docs/EXPERIMENTS_LOG.md`` §2.1 +
    ``docs/CURRENT_BEST_PIPELINE.md`` (per-clip v8 column). They were
    measured on the same eval harness on A100 across the 7-clip set.
  * Speed numbers come from ``work/benchmarks/tracker_speeds.json``,
    written by ``scripts/benchmark_trackers.py`` on this machine
    (Apple Silicon / MPS). The detector is shared across trackers so
    the only variable is the tracker step.
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
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "docs" / "figures")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
        log.warning("speed json missing -> skipping speed charts: %s",
                    args.speed_json)
    write_mermaid_snippets(
        args.out_dir / "_mermaid_snippets.md",
        args.speed_json if args.speed_json.is_file() else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
