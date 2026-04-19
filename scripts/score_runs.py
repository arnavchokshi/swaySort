"""Score every clip's tracks.pkl against ground truth.

Usage::

    python scripts/score_runs.py \\
        --results-dir work/results \\
        --gt-root /home/ubuntu/work/data \\
        --out work/results/scores.json

Reads ``<results-dir>/<clip>/tracks.pkl``, converts to MOT rows, scores
versus ``<gt-root>/<clip>/gt/gt.txt`` using py-motmetrics at IoU 0.5.

Writes a single JSON containing per-clip + summary metrics.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tracking.postprocess import tracks_to_frame_detections  # noqa: E402

log = logging.getLogger("score_runs")


def _tracks_to_mot_rows(tracks_pkl: Path) -> List[str]:
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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory with <clip>/tracks.pkl per clip.")
    p.add_argument("--gt-root", type=Path, required=True,
                   help="Directory containing <clip>/gt/gt.txt per clip.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path.")
    p.add_argument("--clips", nargs="*", default=None,
                   help="Optional subset of clip names. Default = all "
                        "clips found under --results-dir.")
    p.add_argument("--label", default="run",
                   help="Label written into the output JSON.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results_dir = args.results_dir
    gt_root = args.gt_root

    if args.clips:
        clips = list(args.clips)
    else:
        clips = sorted(
            d.name for d in results_dir.iterdir()
            if d.is_dir() and (d / "tracks.pkl").is_file()
        )

    log.info("scoring %d clips: %s", len(clips), clips)

    per_clip: Dict[str, Dict[str, float]] = {}
    for clip in clips:
        tracks_pkl = results_dir / clip / "tracks.pkl"
        gt_path = gt_root / clip / "gt" / "gt.txt"
        if not tracks_pkl.is_file():
            log.warning("missing %s", tracks_pkl)
            continue
        if not gt_path.is_file():
            log.warning("missing GT for %s at %s", clip, gt_path)
            continue
        try:
            rows = _tracks_to_mot_rows(tracks_pkl)
            metrics = _score_mot_rows_vs_gt(rows, gt_path)
        except Exception as exc:
            log.exception("scoring %s failed: %s", clip, exc)
            continue
        per_clip[clip] = metrics
        log.info(
            "%s  IDF1=%.4f MOTA=%.4f IDS=%d FN=%d FP=%d "
            "MT=%d ML=%d num_obj=%d",
            clip, metrics["idf1"], metrics["mota"],
            int(metrics["num_switches"]),
            int(metrics["num_misses"]),
            int(metrics["num_false_positives"]),
            int(metrics["mostly_tracked"]),
            int(metrics["mostly_lost"]),
            int(metrics["num_unique_objects"]),
        )

    if per_clip:
        idf1s = np.asarray([v["idf1"] for v in per_clip.values()])
        motas = np.asarray([v["mota"] for v in per_clip.values()])
        idss = np.asarray([v["num_switches"] for v in per_clip.values()])
        fns = np.asarray([v["num_misses"] for v in per_clip.values()])
        fps = np.asarray([v["num_false_positives"] for v in per_clip.values()])
        summary = {
            "n_clips": int(len(per_clip)),
            "mean_idf1": float(idf1s.mean()),
            "median_idf1": float(np.median(idf1s)),
            "min_idf1": float(idf1s.min()),
            "mean_mota": float(motas.mean()),
            "total_switches": int(idss.sum()),
            "total_misses": int(fns.sum()),
            "total_false_positives": int(fps.sum()),
        }
        log.info(
            "SUMMARY  mean IDF1=%.4f  min=%.4f  total IDS=%d FN=%d FP=%d",
            summary["mean_idf1"], summary["min_idf1"],
            summary["total_switches"], summary["total_misses"],
            summary["total_false_positives"],
        )
    else:
        summary = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "label": args.label,
        "scoring": "py-motmetrics, IoU 0.5, mot15-2D",
        "summary": summary,
        "per_clip": per_clip,
    }, indent=2))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
