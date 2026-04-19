"""Regression + speed harness for pipeline optimization work.

Drives ``tracking.run_pipeline.run_pipeline_on_video`` twice (baseline +
candidate), captures per-stage wall-clock timings for each run, and
diffs the resulting ``FrameDetections`` caches and ``Track`` outputs for
bit-exact (or tolerance-bounded) equivalence.

Use this after every optimization to verify the strict no-regression rule
in ``docs/PIPELINE_SPEC.md`` still holds. Each Tier-A optimization is
gated by an env var (see ``BEST_ID_*`` flags below), so the same script
binary runs both legs.

Typical workflow::

    # Baseline (current/old code path)
    python scripts/regression_check.py \
        --video /Users/.../loveTest/IMG_9265.mov \
        --out work/regression/baseline --max-frames 100 --device mps

    # Candidate (new code path)
    BEST_ID_GPU_NMS=1 python scripts/regression_check.py \
        --video /Users/.../loveTest/IMG_9265.mov \
        --out work/regression/gpunms --max-frames 100 --device mps

    # Diff
    python scripts/regression_check.py diff \
        --a work/regression/baseline --b work/regression/gpunms

The diff is bit-exact by default. Pass ``--tol-box 1.0 --tol-conf 1e-3``
for tolerance-bounded comparison (used when comparing TensorRT FP32
against PyTorch FP32).

Output layout (per leg)::

    <out>/cache.pkl     # list[FrameDetections]
    <out>/tracks.pkl    # dict[int, Track]
    <out>/timings.json  # per-stage wall clock + opt flags + env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("regression_check")


OPT_FLAGS = (
    "BEST_ID_GPU_NMS",
    "BEST_ID_PREFETCH",
    "BEST_ID_PIPELINE_PARALLEL",
    "BEST_ID_TRT_ENGINE",
)


def _capture_opt_env() -> Dict[str, str]:
    return {k: os.environ.get(k, "") for k in OPT_FLAGS}


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Run the pipeline once, save cache + tracks + per-stage timings."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.pkl"
    tracks_path = out_dir / "tracks.pkl"
    timings_path = out_dir / "timings.json"

    from tracking.run_pipeline import run_pipeline_on_video

    if cache_path.exists() and not args.force:
        log.info("cache exists, skipping detect+track: %s "
                 "(pass --force to rebuild)", cache_path)
    else:
        if cache_path.exists():
            cache_path.unlink()
        if tracks_path.exists():
            tracks_path.unlink()

    t0 = time.perf_counter()
    run_pipeline_on_video(
        video=Path(args.video),
        out=tracks_path,
        device=args.device,
        max_frames=args.max_frames,
        cache_path=cache_path,
        force=args.force,
    )
    wall = time.perf_counter() - t0

    cache = joblib.load(str(cache_path))
    tracks = joblib.load(str(tracks_path))

    payload: Dict[str, Any] = {
        "video": str(args.video),
        "device": args.device,
        "max_frames": args.max_frames,
        "wall_seconds": round(wall, 4),
        "n_frames": len(cache),
        "n_tracks": len(tracks),
        "fps": round(len(cache) / max(wall, 1e-9), 3),
        "ms_per_frame": round(1000.0 * wall / max(len(cache), 1), 3),
        "opts": _capture_opt_env(),
    }
    timings_path.write_text(json.dumps(payload, indent=2))
    log.info("wall=%.2fs (%.1f ms/frame) n_frames=%d n_tracks=%d -> %s",
             wall, payload["ms_per_frame"], len(cache), len(tracks),
             timings_path)
    return 0


# ---------------------------------------------------------------------------
# DIFF
# ---------------------------------------------------------------------------


def _frame_diff(
    fa, fb, *, tol_box: float, tol_conf: float, idx: int,
) -> List[str]:
    diffs: List[str] = []
    if fa.xyxys.shape != fb.xyxys.shape:
        diffs.append(
            f"frame {idx}: xyxys shape {tuple(fa.xyxys.shape)} != "
            f"{tuple(fb.xyxys.shape)}"
        )
        return diffs
    if fa.xyxys.size == 0:
        return diffs

    box_err = float(np.max(np.abs(
        fa.xyxys.astype(np.float64) - fb.xyxys.astype(np.float64),
    )))
    conf_err = float(np.max(np.abs(
        fa.confs.astype(np.float64) - fb.confs.astype(np.float64),
    )))
    if box_err > tol_box:
        diffs.append(f"frame {idx}: bbox L_inf={box_err:.4f} > {tol_box}")
    if conf_err > tol_conf:
        diffs.append(f"frame {idx}: conf L_inf={conf_err:.6f} > {tol_conf}")
    if not np.array_equal(fa.tids, fb.tids):
        diff_count = int(np.sum(fa.tids != fb.tids))
        diffs.append(
            f"frame {idx}: track-id mismatch on {diff_count}/{len(fa.tids)} dets"
        )
    return diffs


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (M,4) and (N,4) xyxy boxes -> (M,N)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    # Intersection
    x1 = np.maximum(a64[:, None, 0], b64[None, :, 0])
    y1 = np.maximum(a64[:, None, 1], b64[None, :, 1])
    x2 = np.minimum(a64[:, None, 2], b64[None, :, 2])
    y2 = np.minimum(a64[:, None, 3], b64[None, :, 3])
    iw = np.clip(x2 - x1, 0.0, None)
    ih = np.clip(y2 - y1, 0.0, None)
    inter = iw * ih
    area_a = np.clip(a64[:, 2] - a64[:, 0], 0.0, None) * np.clip(
        a64[:, 3] - a64[:, 1], 0.0, None,
    )
    area_b = np.clip(b64[:, 2] - b64[:, 0], 0.0, None) * np.clip(
        b64[:, 3] - b64[:, 1], 0.0, None,
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _frame_diff_iou(
    fa, fb, *, iou_thresh: float, tol_box: float, tol_conf: float, idx: int,
) -> List[str]:
    """IoU-matched diff: tolerates detection reordering between A/B.

    Greedy assignment by descending IoU. Reports unmatched dets (count
    mismatch or low-IoU pairing) and max per-pair box/conf delta.
    """
    diffs: List[str] = []
    na, nb = fa.xyxys.shape[0], fb.xyxys.shape[0]
    if na == 0 and nb == 0:
        return diffs
    if na != nb:
        diffs.append(f"frame {idx}: det count {na} != {nb}")
    iou = _iou_xyxy(fa.xyxys, fb.xyxys)
    matched_a: List[int] = []
    matched_b: List[int] = []
    pairs: List[Tuple[int, int, float]] = []
    used_b = set()
    # Greedy: walk A in descending confidence, take best free B.
    order_a = np.argsort(-fa.confs.astype(np.float64))
    for ai in order_a:
        if iou.shape[1] == 0:
            break
        row = iou[ai].copy()
        for bj in used_b:
            row[bj] = -1.0
        bj = int(np.argmax(row))
        if row[bj] < iou_thresh:
            continue
        used_b.add(bj)
        matched_a.append(int(ai))
        matched_b.append(bj)
        pairs.append((int(ai), bj, float(row[bj])))
    if len(matched_a) < min(na, nb):
        diffs.append(
            f"frame {idx}: only {len(matched_a)}/{min(na, nb)} dets matched "
            f"(IoU>={iou_thresh})"
        )
    if pairs:
        max_box = 0.0
        max_conf = 0.0
        tid_mismatch = 0
        for ai, bj, _ in pairs:
            box_err = float(np.max(np.abs(
                fa.xyxys[ai].astype(np.float64)
                - fb.xyxys[bj].astype(np.float64),
            )))
            conf_err = float(abs(
                float(fa.confs[ai]) - float(fb.confs[bj]),
            ))
            if box_err > max_box:
                max_box = box_err
            if conf_err > max_conf:
                max_conf = conf_err
            if int(fa.tids[ai]) != int(fb.tids[bj]):
                tid_mismatch += 1
        if max_box > tol_box:
            diffs.append(
                f"frame {idx}: matched bbox L_inf={max_box:.4f} > {tol_box}"
            )
        if max_conf > tol_conf:
            diffs.append(
                f"frame {idx}: matched conf L_inf={max_conf:.6f} > {tol_conf}"
            )
        if tid_mismatch:
            diffs.append(
                f"frame {idx}: matched track-id mismatch on "
                f"{tid_mismatch}/{len(pairs)} dets"
            )
    return diffs


def _diff_caches(
    a: List[Any], b: List[Any], *, tol_box: float, tol_conf: float,
    max_report: int, iou_match: float = 0.0,
) -> List[str]:
    if len(a) != len(b):
        return [f"frame count: {len(a)} != {len(b)}"]
    out: List[str] = []
    for i, (fa, fb) in enumerate(zip(a, b)):
        if iou_match > 0.0:
            out.extend(_frame_diff_iou(
                fa, fb, iou_thresh=iou_match,
                tol_box=tol_box, tol_conf=tol_conf, idx=i,
            ))
        else:
            out.extend(_frame_diff(
                fa, fb, tol_box=tol_box, tol_conf=tol_conf, idx=i,
            ))
        if len(out) >= max_report:
            out.append(f"... (suppressed; first {max_report} reported)")
            return out
    return out


def _diff_tracks(
    a: Dict[int, Any], b: Dict[int, Any], *, tol_box: float, max_report: int,
) -> List[str]:
    out: List[str] = []
    if set(a.keys()) != set(b.keys()):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        if only_a:
            out.append(f"track ids only in A: {only_a[:20]}"
                       + ("..." if len(only_a) > 20 else ""))
        if only_b:
            out.append(f"track ids only in B: {only_b[:20]}"
                       + ("..." if len(only_b) > 20 else ""))
        return out
    for tid in sorted(a):
        ta, tb = a[tid], b[tid]
        if not np.array_equal(ta.frames, tb.frames):
            out.append(
                f"tid {tid}: frame array differs (lens "
                f"{len(ta.frames)} vs {len(tb.frames)})"
            )
            continue
        if ta.bboxes.shape != tb.bboxes.shape:
            out.append(
                f"tid {tid}: bbox shape {tuple(ta.bboxes.shape)} != "
                f"{tuple(tb.bboxes.shape)}"
            )
            continue
        if ta.bboxes.size:
            err = float(np.max(np.abs(
                ta.bboxes.astype(np.float64) - tb.bboxes.astype(np.float64),
            )))
            if err > tol_box:
                out.append(f"tid {tid}: bbox L_inf={err:.4f} > {tol_box}")
        if len(out) >= max_report:
            out.append(f"... (suppressed; first {max_report} reported)")
            return out
    return out


def cmd_diff(args: argparse.Namespace) -> int:
    a_dir = Path(args.a)
    b_dir = Path(args.b)
    cache_a = joblib.load(str(a_dir / "cache.pkl"))
    cache_b = joblib.load(str(b_dir / "cache.pkl"))
    tracks_a = joblib.load(str(a_dir / "tracks.pkl"))
    tracks_b = joblib.load(str(b_dir / "tracks.pkl"))
    timings_a = json.loads((a_dir / "timings.json").read_text())
    timings_b = json.loads((b_dir / "timings.json").read_text())

    print(f"A: {a_dir}")
    print(f"  wall={timings_a['wall_seconds']}s  "
          f"ms/frame={timings_a['ms_per_frame']}  "
          f"n_frames={timings_a['n_frames']}  "
          f"n_tracks={timings_a['n_tracks']}")
    print(f"  opts: {timings_a['opts']}")
    print(f"B: {b_dir}")
    print(f"  wall={timings_b['wall_seconds']}s  "
          f"ms/frame={timings_b['ms_per_frame']}  "
          f"n_frames={timings_b['n_frames']}  "
          f"n_tracks={timings_b['n_tracks']}")
    print(f"  opts: {timings_b['opts']}")

    speedup = timings_a["wall_seconds"] / max(timings_b["wall_seconds"], 1e-9)
    delta_pct = 100.0 * (1.0 - timings_b["wall_seconds"]
                         / max(timings_a["wall_seconds"], 1e-9))
    print(f"\nSpeed: B is {speedup:.3f}x faster ({delta_pct:+.1f}% wall)")

    print("\n--- cache diff ---")
    cache_diffs = _diff_caches(
        cache_a, cache_b,
        tol_box=args.tol_box, tol_conf=args.tol_conf,
        max_report=args.max_report,
        iou_match=args.iou_match,
    )
    if not cache_diffs:
        print("  cache identical within tolerance")
    else:
        for d in cache_diffs:
            print(f"  {d}")

    print("\n--- tracks diff ---")
    track_diffs = _diff_tracks(
        tracks_a, tracks_b,
        tol_box=args.tol_box, max_report=args.max_report,
    )
    if not track_diffs:
        print("  tracks identical within tolerance")
    else:
        for d in track_diffs:
            print(f"  {d}")

    if cache_diffs or track_diffs:
        return 1
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run pipeline once, save cache+tracks+timings")
    pr.add_argument("--video", required=True)
    pr.add_argument("--out", required=True, help="output directory")
    pr.add_argument("--device", default="mps")
    pr.add_argument("--max-frames", type=int, default=None)
    pr.add_argument("--force", action="store_true",
                    help="rebuild cache even if it exists")

    pd = sub.add_parser("diff", help="diff two run/ directories for equivalence")
    pd.add_argument("--a", required=True, help="baseline run directory")
    pd.add_argument("--b", required=True, help="candidate run directory")
    pd.add_argument("--tol-box", type=float, default=0.0,
                    help="absolute bbox tolerance in pixels (0=bit-exact)")
    pd.add_argument("--tol-conf", type=float, default=0.0,
                    help="absolute confidence tolerance")
    pd.add_argument("--max-report", type=int, default=20,
                    help="cap diff lines printed")
    pd.add_argument("--iou-match", type=float, default=0.0,
                    help="if >0, IoU-match dets across A/B before "
                         "comparing (use for TRT FP32 vs PyTorch FP32 "
                         "where NMS sort order can flip; try 0.5)")

    # Default subcommand to run if first arg looks like a flag (back-compat).
    parsed = p.parse_args(argv)
    return parsed


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "run":
        return cmd_run(args)
    if args.command == "diff":
        return cmd_diff(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
