"""Per-clip IDF1 head-to-head: ours-v8 vs each base BoxMOT tracker.

For every clip in the manifest, this script:

1. Builds (or loads) the YOLO multi-scale detection cache *once*
   (the same cache ``scripts/benchmark_trackers.py`` already produces).
2. Replays those cached detections through every BoxMOT tracker in
   their out-of-the-box BoxMOT-default configuration, capturing each
   tracker's per-frame outputs in MOT-Challenge format.
3. Loads our shipped, post-processed tracks for the same clip from
   ``work/results/<clip>/tracks.pkl`` (so the "ours-v8" entry is the
   *real* shipped pipeline output, not a one-shot rerun).
4. Scores every per-clip MOT file against your ground-truth file at
   ``<gt-root>/<clip>/gt/gt.txt`` using py-motmetrics at IoU 0.5
   (CLEAR + ID metrics, the same protocol used in
   ``docs/EXPERIMENTS_LOG.md``).
5. Writes the full per-clip x per-tracker matrix to
   ``work/benchmarks/per_clip_idf1.json``.

Usage::

    python scripts/eval_per_clip.py \\
        --clips loveTest MotionTest shorterTest mirrorTest \\
        --gt-root /Users/arnavchokshi/Desktop \\
        --device mps

The chart generator (``scripts/generate_comparison_charts.py``)
consumes the produced JSON to render the per-clip comparison bar
chart embedded in the README.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse the *exact* benchmark plumbing so the detection cache and the
# tracker construction match the speed bench bit-for-bit. This way the
# accuracy and FPS numbers in the README come from the same code path.
from scripts.benchmark_trackers import (  # noqa: E402
    DEFAULT_REID, DEFAULT_WEIGHTS,
    _build_or_load_detection_cache, _build_tracker,
)
from tracking.deepocsort_runner import install_kalman_jitter_patch  # noqa: E402
from tracking.postprocess import Track, tracks_to_frame_detections  # noqa: E402

log = logging.getLogger("eval_per_clip")


# These trackers are run head-to-head per clip. The "ours" entry is
# special-cased: instead of rerunning, we load the shipped, post-
# processed tracks from work/results/<clip>/tracks.pkl. That makes the
# v8 entry honest -- it includes every post-process stage from
# docs/PIPELINE_SPEC.md, not just DeepOcSort tracking.
COMPETITOR_NAMES = [
    "DeepOcSort (ours, OSNet x0.25)",
    "BotSort (base)",
    "StrongSort (base)",
    "HybridSort (base)",
    "ByteTrack (base)",
    "OcSort (base, no ReID)",
]


def _xyxyc_to_mot_rows(
    boxes_per_frame: List[np.ndarray],
    *, frame_offset: int = 1,
) -> List[str]:
    """Encode per-frame ``(N, >=6)`` arrays to MOT-Challenge rows.

    Input rows are ``[x1, y1, x2, y2, track_id, conf, ...]`` (the
    BoxMOT v18 ``tracker.update`` layout). Output coordinates are
    1-based per the MOT format that ``mm.io.loadtxt`` expects.
    """
    rows: List[str] = []
    for i, arr in enumerate(boxes_per_frame):
        if arr is None or len(arr) == 0:
            continue
        a = np.asarray(arr, dtype=np.float64)
        frame_id = frame_offset + i
        for r in a:
            x1, y1, x2, y2, tid, conf = r[0], r[1], r[2], r[3], int(r[4]), r[5]
            w = float(x2 - x1)
            h = float(y2 - y1)
            X = float(x1 + 1.0)
            Y = float(y1 + 1.0)
            rows.append(
                f"{frame_id},{tid},{X:.6f},{Y:.6f},{w:.6f},{h:.6f},"
                f"{float(conf):.6f},1,-1"
            )
    return rows


def _ours_tracks_to_mot_rows(tracks_pkl: Path) -> List[str]:
    """Convert our shipped post-processed tracks.pkl to MOT rows."""
    raw = joblib.load(str(tracks_pkl))
    if not isinstance(raw, dict):
        raise TypeError(f"unexpected tracks.pkl type: {type(raw)}")
    n_frames = 0
    for t in raw.values():
        if hasattr(t, "frames") and len(t.frames):
            n_frames = max(n_frames, int(t.frames.max()) + 1)
    fds = tracks_to_frame_detections(raw, n_frames)
    rows: List[str] = []
    for i, fd in enumerate(fds):
        if len(fd.tids) == 0:
            continue
        frame_id = i + 1
        for j in range(len(fd.tids)):
            x1, y1, x2, y2 = (float(v) for v in fd.xyxys[j])
            conf = float(fd.confs[j])
            tid = int(fd.tids[j])
            w = x2 - x1
            h = y2 - y1
            rows.append(
                f"{frame_id},{tid},{x1 + 1.0:.6f},{y1 + 1.0:.6f},"
                f"{w:.6f},{h:.6f},{conf:.6f},1,-1"
            )
    return rows


def _score_mot_rows_vs_gt(
    pred_rows: List[str], gt_path: Path,
) -> Dict[str, float]:
    """py-motmetrics CLEAR+ID metrics at IoU 0.5."""
    import motmetrics as mm

    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    text = "\n".join(pred_rows) + ("\n" if pred_rows else "")
    ts = mm.io.loadtxt(io.StringIO(text), fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
    mh = mm.metrics.create()
    names = [
        "idf1", "mota",
        "num_switches", "num_fragmentations",
        "num_misses", "num_false_positives",
        "mostly_tracked", "mostly_lost",
        "num_unique_objects",
    ]
    summary = mh.compute(acc, metrics=names, name="seq")
    out: Dict[str, float] = {}
    for n in names:
        try:
            out[n] = float(summary[n].iloc[0])
        except Exception:
            out[n] = float("nan")
    out["num_frames"] = float(
        len(gt.index.get_level_values(0).unique())
    )
    return out


def _run_one_competitor(
    name: str, frames_cache: List[Dict[str, np.ndarray]],
    *, device: str, reid_weights: str,
) -> List[np.ndarray]:
    """Replay cached detections through one tracker; capture outputs."""
    install_kalman_jitter_patch()
    tracker, _cfg = _build_tracker(
        name, device=device, reid_weights=reid_weights,
    )
    captured: List[np.ndarray] = []
    for fc in frames_cache:
        try:
            out = tracker.update(fc["dets"], fc["bgr"])
        except Exception as exc:
            log.warning("%s.update failed: %s", name, exc)
            out = np.zeros((0, 8), dtype=np.float32)
        if out is None or len(out) == 0:
            captured.append(np.zeros((0, 8), dtype=np.float32))
        else:
            captured.append(np.asarray(out, dtype=np.float32))
    del tracker
    return captured


def _eval_one_clip(
    *, clip_name: str, video: Path, gt_path: Path,
    weights: Path, device: str, reid_weights: str,
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Run all six trackers on one clip and score them vs GT."""
    log.info("=== clip %s ===", clip_name)
    if not video.is_file():
        log.error("video missing: %s", video)
        return {}
    if not gt_path.is_file():
        log.error("gt missing: %s", gt_path)
        return {}

    frames_cache = _build_or_load_detection_cache(
        video=video, weights=weights, device=device, max_frames=None,
    )
    if not frames_cache:
        log.error("no frames produced for %s", clip_name)
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    per_tracker: Dict[str, Dict[str, float]] = {}

    # Ours-v8: load shipped post-processed tracks instead of rerunning.
    ours_pkl = REPO / "work" / "results" / clip_name / "tracks.pkl"
    if ours_pkl.is_file():
        log.info("[ours-v8] loading shipped tracks: %s", ours_pkl)
        ours_rows = _ours_tracks_to_mot_rows(ours_pkl)
        (out_dir / "ours_v8.txt").write_text(
            "\n".join(ours_rows) + ("\n" if ours_rows else "")
        )
        t0 = time.time()
        ours_metrics = _score_mot_rows_vs_gt(ours_rows, gt_path)
        log.info(
            "[ours-v8] IDF1=%.4f MOTA=%.4f IDS=%d (scored in %.1fs)",
            ours_metrics["idf1"], ours_metrics["mota"],
            int(ours_metrics["num_switches"]), time.time() - t0,
        )
        per_tracker["This pipeline (v8)"] = ours_metrics
    else:
        log.warning(
            "[ours-v8] no shipped tracks at %s -- skipping; "
            "run work/run_all_tests.py first", ours_pkl,
        )

    # Competitors: real run on the cached detections, then score.
    for name in COMPETITOR_NAMES:
        if name.startswith("DeepOcSort (ours"):
            continue  # already loaded above as the post-processed v8
        log.info("[%s] running on cached detections", name)
        t0 = time.time()
        try:
            captured = _run_one_competitor(
                name, frames_cache, device=device, reid_weights=reid_weights,
            )
        except Exception as exc:
            log.exception("[%s] failed: %s", name, exc)
            continue
        run_s = time.time() - t0
        rows = _xyxyc_to_mot_rows(captured)
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        (out_dir / f"{safe}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else "")
        )
        try:
            metrics = _score_mot_rows_vs_gt(rows, gt_path)
        except Exception as exc:
            log.exception("[%s] scoring failed: %s", name, exc)
            continue
        log.info(
            "[%s] IDF1=%.4f MOTA=%.4f IDS=%d "
            "FN=%d FP=%d (run %.1fs)",
            name, metrics["idf1"], metrics["mota"],
            int(metrics["num_switches"]),
            int(metrics["num_misses"]), int(metrics["num_false_positives"]),
            run_s,
        )
        per_tracker[name] = metrics

    return per_tracker


def _load_clips_manifest(path: Path) -> List[Tuple[str, Path]]:
    """Return ``[(clip_name, video_path), ...]`` from a clips manifest."""
    data = json.loads(path.read_text())
    out: List[Tuple[str, Path]] = []
    for c in data.get("clips", []):
        name = c["name"]
        video = Path(os.path.expanduser(c["video"]))
        out.append((name, video))
    return out


def _resolve_video(
    clip_name: str, manifest_clips: List[Tuple[str, Path]],
) -> Optional[Path]:
    for n, v in manifest_clips:
        if n == clip_name:
            return v
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clips", nargs="+", required=True,
        help="Clip names to evaluate (must match work/results/<name>/ "
             "and the names in the clips manifest).",
    )
    p.add_argument(
        "--gt-root", type=Path, required=True,
        help="Directory containing <clip>/gt/gt.txt for each clip.",
    )
    p.add_argument(
        "--clips-manifest", type=Path, default=None,
        help="Clips manifest JSON. Defaults to configs/clips.json or "
             "configs/clips.example.json.",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--reid-weights", default=DEFAULT_REID)
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--out", type=Path,
        default=REPO / "work" / "benchmarks" / "per_clip_idf1.json",
    )
    p.add_argument(
        "--mot-out-dir", type=Path,
        default=REPO / "work" / "benchmarks" / "per_clip_mot",
        help="Where to drop per-tracker MOT-format predictions (one .txt "
             "per tracker per clip). Useful for re-scoring with TrackEval.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = args.clips_manifest or (
        REPO / "configs" / "clips.json"
        if (REPO / "configs" / "clips.json").is_file()
        else REPO / "configs" / "clips.example.json"
    )
    manifest_clips = _load_clips_manifest(manifest)
    log.info("loaded %d clips from %s", len(manifest_clips), manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.mot_out_dir.mkdir(parents=True, exist_ok=True)

    if args.out.is_file():
        existing = json.loads(args.out.read_text())
        all_results = existing.get("clips", {}) if isinstance(existing, dict) else {}
    else:
        all_results = {}

    for clip_name in args.clips:
        video = _resolve_video(clip_name, manifest_clips)
        if video is None:
            log.error("clip %s not in manifest", clip_name)
            continue
        gt_path = args.gt_root / clip_name / "gt" / "gt.txt"
        out_dir = args.mot_out_dir / clip_name
        per_tracker = _eval_one_clip(
            clip_name=clip_name, video=video, gt_path=gt_path,
            weights=args.weights, device=args.device,
            reid_weights=args.reid_weights, out_dir=out_dir,
        )
        if per_tracker:
            all_results[clip_name] = {
                "video": str(video),
                "gt": str(gt_path),
                "device": args.device,
                "trackers": per_tracker,
            }
            payload = {
                "device": args.device,
                "scoring": "py-motmetrics, IoU 0.5, mot15-2D",
                "ours_source": "work/results/<clip>/tracks.pkl "
                               "(post-processed v8 pipeline)",
                "competitors_source": "scripts/eval_per_clip.py rerun on "
                                      "the cached YOLO multi-scale detections; "
                                      "default BoxMOT v18 hyperparameters",
                "clips": all_results,
            }
            args.out.write_text(json.dumps(payload, indent=2))
            log.info("wrote %s after clip %s", args.out, clip_name)

    log.info("done. final summary at %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
