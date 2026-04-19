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

# Per-clip data is now captured in VERSION_PROGRESSION (further down)
# rather than as a sparse competitor table -- the original idea was a
# grouped bar of ours vs each competitor per clip, but only one
# competitor number is published per-clip (loveTest OcSort = 0.71 from
# EXPERIMENTS_LOG.md §2.1). Showing a chart with mostly-empty groups was
# misleading, so we instead show the v1 -> v8 progression which fully
# answers "what does the post-process chain buy you per clip?".


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
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "docs" / "figures")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chart_accuracy_overall(args.out_dir / "accuracy_overall.png")
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
