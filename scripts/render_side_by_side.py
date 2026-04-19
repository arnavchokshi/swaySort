"""Render side-by-side comparison videos: Ours (v9) vs the worst-performing
baseline tracker on each clip's densest 10-15 s window.

Inputs:

  * ``work/benchmarks/full_a10_results.json`` from
    ``scripts/run_full_benchmark.py`` -- used to identify, per clip,
    which baseline scored worst on IDF1.
  * ``work/benchmarks/full_a10_mot/<clip>/<tracker>.txt`` -- MOT15-format
    output for each baseline tracker (also written by
    ``run_full_benchmark.py``).
  * ``work/results/<clip>/tracks.pkl`` -- Ours tracks.pkl (Track objects).
  * ``<gt-root>/<clip>/gt/gt.txt`` -- ground-truth MOT file (used to
    find the densest GT window for each clip).
  * The original video file from the manifest passed via ``--clips-manifest``.

Outputs (per clip):

  * ``work/comparison_videos/<clip>/<clip>_ours_vs_<worst>.mp4``
    Side-by-side composite with metric banners on each side.
  * ``work/comparison_videos/<clip>/clip_window.json``
    The (start_frame, end_frame, dense_count) used for the render.

Usage::

    python scripts/render_side_by_side.py \\
        --full-results-json work/benchmarks/full_a10_results.json \\
        --mot-out-root work/benchmarks/full_a10_mot \\
        --ours-tracks-root work/results \\
        --clips-manifest configs/clips.json \\
        --gt-root /Users/arnavchokshi/Desktop \\
        --out-root work/comparison_videos \\
        --window-seconds 12 \\
        --top-n 4
"""

from __future__ import annotations

import argparse
import colorsys
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("render_side_by_side")

OURS_LABEL = "Ours (v9 shipped)"
BASELINES = [
    "ByteTrack (base)",
    "OcSort (base, no ReID)",
    "HybridSort (base)",
    "BotSort (base)",
    "StrongSort (base)",
    "DeepOcSort (base)",
]


# ---------------------------------------------------------------------------
# Manifest + MOT parsing
# ---------------------------------------------------------------------------
def _load_manifest(path: Path) -> Dict[str, Path]:
    if not path.is_file():
        raise FileNotFoundError(f"clip manifest not found: {path}")
    data = json.loads(path.read_text())
    out: Dict[str, Path] = {}
    for c in data.get("clips", []):
        out[c["name"]] = Path(os.path.expanduser(c["video"]))
    return out


def _parse_mot_file(path: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """Return {frame_idx (0-based): [(track_id, xyxy), ...]}."""
    out: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            f = int(parts[0]) - 1  # MOT is 1-based
            tid = int(parts[1])
            x = float(parts[2]) - 1.0  # MOT writes x+1
            y = float(parts[3]) - 1.0
            w = float(parts[4])
            h = float(parts[5])
        except ValueError:
            continue
        out[f].append((tid, np.array([x, y, x + w, y + h], dtype=np.float32)))
    return out


def _ours_tracks_to_per_frame(
    tracks_pkl: Path,
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """Read the Ours tracks.pkl and flatten to {frame_idx: [(tid, xyxy)]}."""
    raw = joblib.load(str(tracks_pkl))
    if not isinstance(raw, dict):
        raise TypeError(f"unexpected tracks.pkl type: {type(raw)}")
    out: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for tid, tr in raw.items():
        if not hasattr(tr, "frames"):
            continue
        for f, bb in zip(tr.frames, tr.bboxes):
            out[int(f)].append((int(tid), np.asarray(bb, dtype=np.float32)))
    return out


def _gt_density_per_frame(gt_path: Path) -> Dict[int, int]:
    """Return {frame_idx (0-based): n_gt_objects}."""
    out: Dict[int, int] = defaultdict(int)
    if not gt_path.is_file():
        return out
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            f = int(parts[0]) - 1
            conf = float(parts[6]) if len(parts) >= 7 else 1.0
        except ValueError:
            continue
        if conf < 0.5:
            continue
        out[f] += 1
    return out


# ---------------------------------------------------------------------------
# Window picking + rendering
# ---------------------------------------------------------------------------
def pick_dense_window(
    gt_density: Dict[int, int],
    n_frames_total: int,
    window_frames: int,
) -> Tuple[int, int, float]:
    """Slide a window over GT density; return (start, end, mean_density)."""
    if window_frames >= n_frames_total or n_frames_total <= 0:
        return 0, max(0, n_frames_total - 1), float(
            np.mean([gt_density.get(i, 0) for i in range(n_frames_total)])
            if n_frames_total else 0
        )
    densities = np.array(
        [gt_density.get(i, 0) for i in range(n_frames_total)],
        dtype=np.float64,
    )
    cumsum = np.cumsum(np.insert(densities, 0, 0))
    window_sums = cumsum[window_frames:] - cumsum[:-window_frames]
    best_start = int(np.argmax(window_sums))
    best_end = best_start + window_frames - 1
    return best_start, best_end, float(window_sums[best_start] / window_frames)


def color_for_id(tid: int) -> Tuple[int, int, int]:
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _draw_boxes(
    frame: np.ndarray,
    per_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    frame_idx: int,
) -> int:
    box_thick = max(2, int(round(min(frame.shape[:2]) / 360)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(frame.shape[:2]) / 900)
    text_thick = max(1, int(round(font_scale * 2)))
    n = 0
    for tid, bb in per_frame.get(frame_idx, []):
        x1, y1, x2, y2 = (int(round(v)) for v in bb)
        color = color_for_id(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)
        label = f"id {tid}"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thick)
        ty = max(0, y1 - 6)
        cv2.rectangle(frame, (x1, ty - th - 4),
                      (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(frame, label, (x1 + 3, ty - 2),
                    font, font_scale, (0, 0, 0), text_thick, cv2.LINE_AA)
        n += 1
    return n


def _draw_banner(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    color: Tuple[int, int, int],
    extra_lines: Optional[List[str]] = None,
) -> np.ndarray:
    """Add a banner strip at the top of the frame and return the result."""
    h, w = frame.shape[:2]
    n_lines = 2 + (len(extra_lines) if extra_lines else 0)
    banner_h = int(34 + 22 * (n_lines - 1))
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = color
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(banner, title, (12, 24), font, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, subtitle, (12, 24 + 22), font, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    if extra_lines:
        y = 24 + 22 * 2
        for line in extra_lines:
            cv2.putText(banner, line, (12, y), font, 0.55,
                        (240, 240, 240), 1, cv2.LINE_AA)
            y += 22
    return np.vstack([banner, frame])


def render_side_by_side(
    clip_name: str,
    video: Path,
    ours_per_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    base_per_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    base_label: str,
    out_mp4: Path,
    *,
    start_frame: int,
    end_frame: int,
    ours_metrics: Dict[str, float],
    base_metrics: Dict[str, float],
    target_height: Optional[int] = None,
) -> Dict:
    """Render a single side-by-side comparison clip."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Down-scale for portability (most viewers want <=1080 wide composite).
    if target_height is None:
        target_height = min(720, src_h)
    scale = target_height / src_h
    out_w_per_panel = int(round(src_w * scale))
    out_h_panel = target_height

    # Banner heights are added on top of each panel; figure out total H.
    base_h = out_h_panel + 100  # rough banner allowance
    out_w = out_w_per_panel * 2

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (out_w, base_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (out_w, base_h))

    GREEN = (32, 109, 46)
    RED = (40, 40, 200)

    ours_extra = [
        f"IDF1 {ours_metrics.get('idf1', float('nan')):.4f}  "
        f"MOTA {ours_metrics.get('mota', float('nan')):.4f}",
        f"IDS {int(ours_metrics.get('num_switches', 0))}  "
        f"FN {int(ours_metrics.get('num_misses', 0))}  "
        f"FP {int(ours_metrics.get('num_false_positives', 0))}",
    ]
    base_extra = [
        f"IDF1 {base_metrics.get('idf1', float('nan')):.4f}  "
        f"MOTA {base_metrics.get('mota', float('nan')):.4f}",
        f"IDS {int(base_metrics.get('num_switches', 0))}  "
        f"FN {int(base_metrics.get('num_misses', 0))}  "
        f"FP {int(base_metrics.get('num_false_positives', 0))}",
    ]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    t0 = time.time()
    for fi in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break

        ours_frame = frame.copy()
        base_frame = frame.copy()
        n_ours = _draw_boxes(ours_frame, ours_per_frame, fi)
        n_base = _draw_boxes(base_frame, base_per_frame, fi)

        ours_frame = cv2.resize(ours_frame, (out_w_per_panel, out_h_panel))
        base_frame = cv2.resize(base_frame, (out_w_per_panel, out_h_panel))

        ours_panel = _draw_banner(
            ours_frame,
            "OURS (v9 shipped)  -  best.pt MS + post-process",
            f"{clip_name}  frame {fi+1}  boxes {n_ours}",
            GREEN, extra_lines=ours_extra,
        )
        base_panel = _draw_banner(
            base_frame,
            f"{base_label}  -  stock yolo26s.pt @ 640",
            f"{clip_name}  frame {fi+1}  boxes {n_base}",
            RED, extra_lines=base_extra,
        )

        # Pad shorter panel so heights match.
        max_h = max(ours_panel.shape[0], base_panel.shape[0])
        if ours_panel.shape[0] < max_h:
            pad = np.zeros(
                (max_h - ours_panel.shape[0], ours_panel.shape[1], 3),
                dtype=np.uint8)
            ours_panel = np.vstack([ours_panel, pad])
        if base_panel.shape[0] < max_h:
            pad = np.zeros(
                (max_h - base_panel.shape[0], base_panel.shape[1], 3),
                dtype=np.uint8)
            base_panel = np.vstack([base_panel, pad])

        composite = np.hstack([ours_panel, base_panel])
        if composite.shape[0] != base_h or composite.shape[1] != out_w:
            composite = cv2.resize(composite, (out_w, base_h))
        writer.write(composite)
        written += 1

    cap.release()
    writer.release()
    return {
        "clip": clip_name,
        "out": str(out_mp4),
        "frames_written": written,
        "wall_seconds": round(time.time() - t0, 2),
        "fps": round(fps, 3),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "out_resolution": [out_w, base_h],
    }


# ---------------------------------------------------------------------------
# Top-level: pick worst baseline per clip, render the densest window
# ---------------------------------------------------------------------------
def _pick_worst_baseline(
    rows: Dict[str, Dict],
) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    worst_name = None
    worst_idf1 = float("inf")
    worst_metrics = None
    for tname, payload in rows.items():
        if tname == OURS_LABEL or tname not in BASELINES:
            continue
        m = payload.get("metrics", {})
        idf1 = m.get("idf1", None)
        if idf1 is None or not np.isfinite(float(idf1)):
            continue
        if float(idf1) < worst_idf1:
            worst_idf1 = float(idf1)
            worst_name = tname
            worst_metrics = m
    return worst_name, worst_metrics


def _safe_label(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("/", "_")
    )


def _gap_score(rows: Dict[str, Dict]) -> float:
    """Compute (ours - worst_baseline) IDF1 gap; -inf if either missing."""
    ours = rows.get(OURS_LABEL, {}).get("metrics", {}).get("idf1", None)
    if ours is None or not np.isfinite(float(ours)):
        return -float("inf")
    worst = float("inf")
    for tname in BASELINES:
        v = rows.get(tname, {}).get("metrics", {}).get("idf1", None)
        if v is None or not np.isfinite(float(v)):
            continue
        worst = min(worst, float(v))
    if not np.isfinite(worst):
        return -float("inf")
    return float(ours) - worst


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-results-json", type=Path,
                   default=REPO / "work" / "benchmarks" /
                   "full_a10_results.json")
    p.add_argument("--mot-out-root", type=Path,
                   default=REPO / "work" / "benchmarks" / "full_a10_mot")
    p.add_argument("--ours-tracks-root", type=Path,
                   default=REPO / "work" / "results")
    p.add_argument("--clips-manifest", type=Path,
                   default=REPO / "configs" / "clips.json")
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path,
                   default=REPO / "work" / "comparison_videos")
    p.add_argument("--window-seconds", type=float, default=12.0,
                   help="Length of the dense-window clip in seconds.")
    p.add_argument(
        "--top-n", type=int, default=4,
        help="Render only the top-N clips by (ours - worst-baseline) "
             "IDF1 gap (default: 4). Pass 0 to render every clip.",
    )
    p.add_argument("--clips", nargs="+", default=None,
                   help="Explicit clip names; overrides --top-n.")
    p.add_argument("--target-height", type=int, default=720,
                   help="Each panel is resized to this height before "
                        "concatenation (composite is 2x wider).")
    p.add_argument("--prefer-pkl", action="store_true",
                   help="Read Ours tracks from work/results/<clip>/tracks.pkl "
                        "even when the A10 MOT file is also present. "
                        "Default: prefer the MOT file (it always matches the "
                        "metrics that run_full_benchmark.py wrote into the "
                        "banner).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    clips_map = _load_manifest(args.clips_manifest)
    payload = json.loads(args.full_results_json.read_text())
    clip_payloads = payload.get("clips", {})

    if args.clips:
        chosen = [c for c in args.clips if c in clip_payloads]
        if len(chosen) != len(args.clips):
            missing = sorted(set(args.clips) - set(chosen))
            log.warning("ignored unknown clips: %s", missing)
    else:
        ranked = sorted(
            clip_payloads.items(),
            key=lambda kv: _gap_score(kv[1]["rows"]),
            reverse=True,
        )
        if args.top_n and args.top_n > 0:
            ranked = ranked[: args.top_n]
        chosen = [name for name, _ in ranked]

    log.info("rendering %d clips: %s", len(chosen), chosen)
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary: List[Dict] = []

    for clip_name in chosen:
        clip_data = clip_payloads.get(clip_name)
        if clip_data is None:
            log.warning("no payload for clip %s; skipping", clip_name)
            continue
        rows = clip_data.get("rows", {})
        worst_name, worst_metrics = _pick_worst_baseline(rows)
        if worst_name is None:
            log.warning("no baseline metrics for clip %s; skipping",
                        clip_name)
            continue
        ours_row = rows.get(OURS_LABEL)
        if ours_row is None:
            log.warning("no Ours row for clip %s; skipping", clip_name)
            continue
        ours_metrics = ours_row.get("metrics", {})
        ours_pkl = args.ours_tracks_root / clip_name / "tracks.pkl"
        ours_mot = args.mot_out_root / clip_name / "Ours_v9_shipped.txt"
        baseline_mot = (
            args.mot_out_root / clip_name /
            f"{_safe_label(worst_name)}.txt"
        )
        video_path = clips_map.get(clip_name)
        gt_path = args.gt_root / clip_name / "gt" / "gt.txt"

        # The MOT file is what produced the metrics shown in the
        # banner (run_full_benchmark.py writes both side-by-side),
        # so prefer it over a possibly-stale local tracks.pkl.
        # Fall back to the .pkl if --prefer-pkl is set or the MOT
        # is missing.
        ours_source: Optional[Path] = None
        if args.prefer_pkl and ours_pkl.is_file() and ours_pkl.stat().st_size > 1024:
            ours_source = ours_pkl
        elif ours_mot.is_file():
            ours_source = ours_mot
        elif ours_pkl.is_file() and ours_pkl.stat().st_size > 1024:
            ours_source = ours_pkl

        for label, p in (("ours tracks", ours_source),
                         ("baseline mot", baseline_mot),
                         ("video", video_path),
                         ("gt", gt_path)):
            if p is None or not p.is_file():
                log.warning("[%s] missing %s -> %s; skipping",
                            clip_name, label, p)
                break
        else:
            if ours_source.suffix == ".pkl":
                ours_pf = _ours_tracks_to_per_frame(ours_source)
                log.info("[%s] Ours from pkl: %s", clip_name, ours_source)
            else:
                ours_pf = _parse_mot_file(ours_source)
                log.info("[%s] Ours from A10 MOT: %s",
                         clip_name, ours_source)
            base_pf = _parse_mot_file(baseline_mot)
            cap = cv2.VideoCapture(str(video_path))
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            window_frames = int(round(args.window_seconds * fps))
            density = _gt_density_per_frame(gt_path)
            sf, ef, mean_d = pick_dense_window(
                density, n_total, window_frames)
            log.info(
                "[%s] worst=%s window=[%d..%d] mean_gt_density=%.2f",
                clip_name, worst_name, sf, ef, mean_d,
            )
            out_mp4 = (
                args.out_root / clip_name /
                f"{clip_name}_ours_vs_{_safe_label(worst_name)}.mp4"
            )
            r = render_side_by_side(
                clip_name=clip_name,
                video=video_path,
                ours_per_frame=ours_pf,
                base_per_frame=base_pf,
                base_label=worst_name,
                out_mp4=out_mp4,
                start_frame=sf,
                end_frame=ef,
                ours_metrics=ours_metrics,
                base_metrics=worst_metrics or {},
                target_height=args.target_height,
            )
            r["worst_baseline"] = worst_name
            r["mean_gt_density"] = round(mean_d, 2)
            r["clip_window_seconds"] = round(
                (ef - sf + 1) / max(fps, 1e-6), 2)
            (out_mp4.parent / "clip_window.json").write_text(
                json.dumps(r, indent=2)
            )
            summary.append(r)
            log.info("[%s] wrote %s in %.1fs", clip_name, out_mp4,
                     r["wall_seconds"])

    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("done; summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
