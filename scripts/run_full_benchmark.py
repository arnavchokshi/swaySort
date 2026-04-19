"""Full A10 benchmark: ours v9 shipped vs stock yolo26s.pt + base BoxMOT trackers.

For each clip in --clips-manifest:
  1. Build (or load) a STOCK-YOLO26s single-scale-640 detection cache,
     stored next to the video as <stem>.base_det_cache.pkl. This is the
     "out-of-the-box" detector a new user gets.
  2. Replay that cache through every base BoxMOT tracker (default
     constructor params), capture per-frame outputs in MOT format,
     score against <clip>/gt/gt.txt with py-motmetrics, time each
     tracker.update() call.
  3. Run the shipped tracking.run_pipeline.run_pipeline_on_video for
     the SAME clip end-to-end (with our weights/best.pt + multi-scale
     + post-process chain), score the result against the same gt, time
     the wall-clock.
  4. Roll all rows into a single per-clip dict and dump
     work/benchmarks/full_results.json after every clip (so a crash
     doesn't lose work).

Usage::

    # On the A10 (production):
    python scripts/run_full_benchmark.py \\
        --device cuda:0 \\
        --clips-manifest configs/clips.remote.json \\
        --base-yolo-weights weights/yolo26s.pt \\
        --our-yolo-weights  weights/best.pt \\
        --reid-weights      weights/osnet_x0_25_msmt17.pt \\
        --gt-root           /home/ubuntu/clips

    # Local smoke test (50 frames of one clip):
    python scripts/run_full_benchmark.py \\
        --device cpu --clips loveTest --max-frames 50 \\
        --base-yolo-weights weights/yolo26s.pt \\
        --our-yolo-weights  weights/best.pt \\
        --gt-root           /Users/arnavchokshi/Desktop
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("run_full_benchmark")

BASELINE_TRACKERS = [
    "ByteTrack (base)",
    "OcSort (base, no ReID)",
    "HybridSort (base)",
    "BotSort (base)",
    "StrongSort (base)",
    "DeepOcSort (base)",
]
OURS_LABEL = "Ours (v9 shipped)"
ALL_ROWS = [OURS_LABEL, *BASELINE_TRACKERS]


@dataclass
class TrackerRun:
    """Per-(clip, row) row in the final JSON."""
    row: str
    detector: str
    tracker_config: str
    n_frames: int
    wall_seconds: float
    det_ms_per_frame_mean: float
    tracker_ms_per_frame_mean: float
    tracker_ms_per_frame_median: float
    tracker_ms_per_frame_p95: float
    end_to_end_fps: float
    gpu_peak_mb: float
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detection cache (stock yolo26s.pt @ single-scale 640, COCO defaults)
# ---------------------------------------------------------------------------
def _build_or_load_base_det_cache(
    *, video: Path, weights: Path, device: str,
    max_frames: Optional[int],
    conf: float = 0.25, iou: float = 0.7, imgsz: int = 640,
) -> Tuple[List[Dict[str, np.ndarray]], float]:
    """Run stock YOLO26s once at single-scale 640 (the out-of-the-box
    Ultralytics quickstart config). Cache result next to the video so
    subsequent tracker replays are free.

    Returns (frames_cache, mean_detector_ms_per_frame).
    """
    cache_path = video.with_suffix(".base_det_cache.pkl")
    if cache_path.is_file() and max_frames is None:
        log.info("base detector cache hit: %s", cache_path)
        cached = joblib.load(str(cache_path))
        return cached["frames"], float(cached.get("mean_det_ms", 0.0))

    from ultralytics import YOLO
    log.info("running stock YOLO26s @ %d on %s (cache miss or max_frames cap)",
             imgsz, video)
    model = YOLO(str(weights))

    import cv2
    cap = cv2.VideoCapture(str(video))
    frames_out: List[Dict[str, np.ndarray]] = []
    det_ms: List[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break
        t0 = time.time()
        res = model.predict(
            frame, imgsz=imgsz, conf=conf, iou=iou,
            device=device, verbose=False, classes=[0],
        )
        det_ms.append(1000.0 * (time.time() - t0))
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            b = res[0].boxes
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            cf = b.conf.cpu().numpy().astype(np.float32)
            cl = b.cls.cpu().numpy().astype(np.float32)
            dets = np.concatenate(
                [xyxy, cf[:, None], cl[:, None]], axis=1,
            ).astype(np.float32)
        else:
            dets = np.zeros((0, 6), dtype=np.float32)
        frames_out.append({"bgr": frame, "dets": dets})
        idx += 1
    cap.release()

    mean_ms = float(np.mean(det_ms)) if det_ms else 0.0
    log.info(
        "stock-yolo26s detector: %d frames, mean=%.1f ms/frame "
        "(%.1f FPS detect-only)", len(frames_out), mean_ms,
        1000.0 / max(mean_ms, 1e-6),
    )
    if max_frames is None:
        joblib.dump({"frames": frames_out, "mean_det_ms": mean_ms},
                    str(cache_path))
        log.info("wrote base-det cache: %s", cache_path)
    return frames_out, mean_ms


# ---------------------------------------------------------------------------
# Base BoxMOT tracker construction (default constructor args only)
# ---------------------------------------------------------------------------
def _build_baseline_tracker(name: str, *, device: str, reid_weights: Path):
    """Construct a base BoxMOT tracker with default constructor args.

    For ReID-using trackers we pass OSNet x0.25 weights (the ones BoxMOT
    auto-downloads anyway -- same head our shipped pipeline uses, so the
    only variable across the table is post-process + detector). For
    motion-only trackers (ByteTrack, OcSort) we pass nothing extra.
    """
    import torch
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
    from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
    from boxmot.trackers.hybridsort.hybridsort import HybridSort
    from boxmot.trackers.ocsort.ocsort import OcSort
    from boxmot.trackers.strongsort.strongsort import StrongSort

    if device.startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(device)
    elif device == "mps" and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")

    if name == "ByteTrack (base)":
        return ByteTrack(), "BoxMOT defaults; no ReID"
    if name == "OcSort (base, no ReID)":
        return OcSort(), "BoxMOT defaults; no ReID"
    if name == "HybridSort (base)":
        return HybridSort(
            reid_weights=reid_weights, device=torch_device,
            half=False, with_reid=True,
        ), "BoxMOT defaults; OSNet x0.25 ReID"
    if name == "BotSort (base)":
        return BotSort(
            reid_weights=reid_weights, device=torch_device, half=False,
        ), "BoxMOT defaults; OSNet x0.25 ReID"
    if name == "StrongSort (base)":
        return StrongSort(
            reid_weights=reid_weights, device=torch_device, half=False,
        ), "BoxMOT defaults; OSNet x0.25 ReID"
    if name == "DeepOcSort (base)":
        return DeepOcSort(
            reid_weights=reid_weights, device=torch_device, half=False,
        ), "BoxMOT defaults; OSNet x0.25 ReID; NO Kalman jitter patch"
    raise ValueError(f"unknown baseline tracker: {name}")


# ---------------------------------------------------------------------------
# MOT-format encoders + py-motmetrics scoring
# ---------------------------------------------------------------------------
def _xyxy_to_mot_rows(per_frame: List[np.ndarray]) -> List[str]:
    """Encode list of (N, >=6) tracker outputs to MOT15 rows (1-based)."""
    rows: List[str] = []
    for i, arr in enumerate(per_frame):
        if arr is None or len(arr) == 0:
            continue
        a = np.asarray(arr, dtype=np.float64)
        for r in a:
            x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
            tid = int(r[4])
            cf = float(r[5])
            w, h = float(x2 - x1), float(y2 - y1)
            rows.append(
                f"{i + 1},{tid},{x1 + 1.0:.6f},{y1 + 1.0:.6f},"
                f"{w:.6f},{h:.6f},{cf:.6f},1,-1"
            )
    return rows


def _ours_tracks_to_mot_rows(tracks_pkl: Path) -> List[str]:
    from tracking.postprocess import tracks_to_frame_detections
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
        for j in range(len(fd.tids)):
            x1, y1, x2, y2 = (float(v) for v in fd.xyxys[j])
            cf = float(fd.confs[j])
            tid = int(fd.tids[j])
            w, h = x2 - x1, y2 - y1
            rows.append(
                f"{i + 1},{tid},{x1 + 1.0:.6f},{y1 + 1.0:.6f},"
                f"{w:.6f},{h:.6f},{cf:.6f},1,-1"
            )
    return rows


def _scrub_nan(d: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Replace NaN/+/-Inf with None so json.dumps emits valid JSON."""
    out: Dict[str, Optional[float]] = {}
    for k, v in d.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            out[k] = None
            continue
        if math.isnan(fv) or math.isinf(fv):
            out[k] = None
        else:
            out[k] = fv
    return out


def _score(pred_rows: List[str], gt_path: Path) -> Dict[str, float]:
    """py-motmetrics CLEAR + ID at IoU 0.5, plus a few derived metrics.

    `num_predicted_unique_objects` is computed manually from the MOT
    rows (py-motmetrics 1.4.0 doesn't expose it as a metric name; the
    closest exposed value is `num_predictions`, which is the total
    number of bbox rows, not unique track IDs).
    """
    import io
    import motmetrics as mm

    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    text = "\n".join(pred_rows) + ("\n" if pred_rows else "")
    ts = mm.io.loadtxt(io.StringIO(text), fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
    mh = mm.metrics.create()
    names = [
        "idf1", "idp", "idr", "mota", "motp",
        "precision", "recall",
        "num_switches", "num_fragmentations",
        "num_misses", "num_false_positives",
        "mostly_tracked", "partially_tracked", "mostly_lost",
        "num_unique_objects", "num_predictions",
        "num_frames",
    ]
    summary = mh.compute(acc, metrics=names, name="seq")
    out: Dict[str, float] = {}
    for n in names:
        try:
            out[n] = float(summary[n].iloc[0])
        except Exception:
            out[n] = float("nan")

    # Count unique predicted track IDs from the raw MOT rows.
    pred_tids = set()
    for line in pred_rows:
        if not line.strip():
            continue
        try:
            pred_tids.add(int(line.split(",", 2)[1]))
        except (IndexError, ValueError):
            continue
    out["num_predicted_unique_objects"] = float(len(pred_tids))
    return _scrub_nan(out)


# ---------------------------------------------------------------------------
# Per-clip orchestration
# ---------------------------------------------------------------------------
def _run_one_baseline(
    name: str, frames_cache: List[Dict[str, np.ndarray]],
    *, device: str, reid_weights: Path,
) -> Tuple[List[np.ndarray], List[float], str]:
    tracker, cfg_str = _build_baseline_tracker(
        name, device=device, reid_weights=reid_weights,
    )
    captured: List[np.ndarray] = []
    track_ms: List[float] = []
    for fc in frames_cache:
        t0 = time.time()
        try:
            out = tracker.update(fc["dets"], fc["bgr"])
        except Exception as exc:
            log.warning("%s.update failed at frame %d: %s",
                        name, len(captured), exc)
            out = np.zeros((0, 8), dtype=np.float32)
        track_ms.append(1000.0 * (time.time() - t0))
        if out is None or len(out) == 0:
            captured.append(np.zeros((0, 8), dtype=np.float32))
        else:
            captured.append(np.asarray(out, dtype=np.float32))
    del tracker
    return captured, track_ms, cfg_str


def _peak_gpu_mb(device: str) -> float:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    return 0.0


def _reset_gpu_peak(device: str) -> None:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _run_one_clip(
    *, clip_name: str, video: Path, gt_path: Path,
    base_weights: Path, our_weights: Path,
    reid_weights: Path, device: str, max_frames: Optional[int],
    out_pkl_dir: Path,
) -> Dict[str, TrackerRun]:
    log.info("=" * 80)
    log.info("CLIP %s  video=%s", clip_name, video)
    if not video.is_file():
        log.error("video missing: %s -- skipping clip", video)
        return {}
    if not gt_path.is_file():
        log.error("gt missing: %s -- skipping clip", gt_path)
        return {}

    out_pkl_dir.mkdir(parents=True, exist_ok=True)

    # ---- Baseline path -----------------------------------------------------
    frames_cache, det_ms_mean = _build_or_load_base_det_cache(
        video=video, weights=base_weights, device=device,
        max_frames=max_frames,
    )

    rows: Dict[str, TrackerRun] = {}
    for tname in BASELINE_TRACKERS:
        log.info("--- [%s] %s ---", clip_name, tname)
        _reset_gpu_peak(device)
        t0 = time.time()
        try:
            captured, track_ms, cfg_str = _run_one_baseline(
                tname, frames_cache, device=device,
                reid_weights=reid_weights,
            )
        except Exception as exc:
            log.exception("[%s] %s FAILED: %s", clip_name, tname, exc)
            continue
        wall = time.time() - t0
        gpu_mb = _peak_gpu_mb(device)
        mot_rows = _xyxy_to_mot_rows(captured)
        try:
            metrics = _score(mot_rows, gt_path)
        except Exception as exc:
            log.exception("[%s] %s scoring failed: %s", clip_name, tname, exc)
            continue
        track_arr = np.asarray(track_ms, dtype=np.float64)
        run = TrackerRun(
            row=tname,
            detector="stock yolo26s.pt @ imgsz=640, conf=0.25, iou=0.7, classes=[0]",
            tracker_config=cfg_str,
            n_frames=len(frames_cache),
            wall_seconds=round(wall, 3),
            det_ms_per_frame_mean=round(det_ms_mean, 2),
            tracker_ms_per_frame_mean=round(float(track_arr.mean()), 2),
            tracker_ms_per_frame_median=round(float(np.median(track_arr)), 2),
            tracker_ms_per_frame_p95=round(float(np.percentile(track_arr, 95)), 2),
            end_to_end_fps=round(
                len(frames_cache) / max(wall, 1e-6), 2),
            gpu_peak_mb=round(gpu_mb, 1),
            metrics=metrics,
        )
        rows[tname] = run
        safe = tname.replace(" ", "_").replace("(", "").replace(")", "")\
            .replace(",", "")
        (out_pkl_dir / f"{safe}.txt").write_text(
            "\n".join(mot_rows) + ("\n" if mot_rows else "")
        )
        log.info(
            "[%s] %s  IDF1=%.4f MOTA=%.4f IDS=%d FN=%d FP=%d  "
            "wall=%.1fs e2e_fps=%.2f tracker=%.1fms gpu_peak=%.0fMB",
            clip_name, tname, metrics["idf1"], metrics["mota"],
            int(metrics["num_switches"]), int(metrics["num_misses"]),
            int(metrics["num_false_positives"]),
            wall, run.end_to_end_fps,
            run.tracker_ms_per_frame_mean, gpu_mb,
        )

    # Pipeline + baseline trackers see the same frame range (both capped
    # by --max-frames in smoke mode, both full-video in production), so
    # reuse the count from the cache for the Ours wall-clock FPS calc.
    n_frames_processed = len(frames_cache)

    # Free cached frames (each is ~6 GB for 1080p; let GC run before
    # we spin up the shipped pipeline which loads its own frames).
    del frames_cache

    # ---- Ours path ---------------------------------------------------------
    log.info("--- [%s] %s ---", clip_name, OURS_LABEL)
    from tracking.run_pipeline import run_pipeline_on_video
    os.environ.setdefault("BEST_ID_DARK_PROFILE", "v9")
    out_dir = REPO / "work" / "results" / clip_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = out_dir / "tracks.pkl"
    _reset_gpu_peak(device)
    t0 = time.time()
    try:
        tracks = run_pipeline_on_video(
            video=video, out=out_pkl, device=device,
            weights=our_weights, force=True,
            max_frames=max_frames,
        )
    except Exception as exc:
        log.exception("[%s] OURS FAILED: %s", clip_name, exc)
        return rows
    wall = time.time() - t0
    gpu_mb = _peak_gpu_mb(device)

    ours_rows_mot = _ours_tracks_to_mot_rows(out_pkl)
    safe = OURS_LABEL.replace(" ", "_").replace("(", "").replace(")", "")
    (out_pkl_dir / f"{safe}.txt").write_text(
        "\n".join(ours_rows_mot) + ("\n" if ours_rows_mot else "")
    )
    metrics = _score(ours_rows_mot, gt_path)
    rows[OURS_LABEL] = TrackerRun(
        row=OURS_LABEL,
        detector=("weights/best.pt multi-scale {768,1024} ensemble, "
                  "conf=0.34, ensemble_iou=0.6, dark-recovery v9"),
        tracker_config=("DeepOcSort + OSNet x0.25 + Kalman jitter "
                        "patch + full v9 post-process chain"),
        n_frames=n_frames_processed,
        wall_seconds=round(wall, 3),
        det_ms_per_frame_mean=0.0,  # not separately measured for shipped path
        tracker_ms_per_frame_mean=0.0,
        tracker_ms_per_frame_median=0.0,
        tracker_ms_per_frame_p95=0.0,
        end_to_end_fps=round(n_frames_processed / max(wall, 1e-6), 2),
        gpu_peak_mb=round(gpu_mb, 1),
        metrics=metrics,
    )
    log.info(
        "[%s] %s  IDF1=%.4f MOTA=%.4f IDS=%d FN=%d FP=%d  "
        "wall=%.1fs e2e_fps=%.2f gpu_peak=%.0fMB n_tracks=%d",
        clip_name, OURS_LABEL, metrics["idf1"], metrics["mota"],
        int(metrics["num_switches"]), int(metrics["num_misses"]),
        int(metrics["num_false_positives"]),
        wall, rows[OURS_LABEL].end_to_end_fps, gpu_mb, len(tracks),
    )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clips-manifest", type=Path,
        default=REPO / "configs" / "clips.json",
    )
    p.add_argument(
        "--clips", nargs="+", default=None,
        help="Optional subset of clip names to run (default: all in manifest).",
    )
    p.add_argument(
        "--gt-root", type=Path, required=True,
        help="Directory containing <clip>/gt/gt.txt for every clip.",
    )
    p.add_argument(
        "--base-yolo-weights", type=Path,
        default=REPO / "weights" / "yolo26s.pt",
    )
    p.add_argument(
        "--our-yolo-weights", type=Path,
        default=REPO / "weights" / "best.pt",
    )
    p.add_argument(
        "--reid-weights", type=Path,
        default=REPO / "weights" / "osnet_x0_25_msmt17.pt",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames (smoke testing only).")
    p.add_argument(
        "--out-json", type=Path,
        default=REPO / "work" / "benchmarks" / "full_results.json",
    )
    p.add_argument(
        "--mot-out-root", type=Path,
        default=REPO / "work" / "benchmarks" / "full_mot",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from work.run_all_tests import _load_manifest
    clips = _load_manifest(args.clips_manifest)
    if args.clips:
        clips = [(n, v) for n, v in clips if n in set(args.clips)]
    if not clips:
        log.error("no clips selected")
        return 1
    log.info("running %d clips: %s",
             len(clips), [n for n, _ in clips])

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.mot_out_root.mkdir(parents=True, exist_ok=True)

    if args.out_json.is_file():
        all_results = json.loads(args.out_json.read_text())
        all_results.setdefault("clips", {})
    else:
        all_results = {
            "device": args.device,
            "scoring": "py-motmetrics, IoU 0.5, mot15-2D",
            "ours_detector": "weights/best.pt multi-scale {768,1024}",
            "baseline_detector": "stock yolo26s.pt single-scale 640, conf=0.25, iou=0.7",
            "n_baseline_trackers": len(BASELINE_TRACKERS),
            "clips": {},
        }

    overall_t0 = time.time()
    for clip_name, video in clips:
        gt_path = args.gt_root / clip_name / "gt" / "gt.txt"
        out_pkl_dir = args.mot_out_root / clip_name
        try:
            rows = _run_one_clip(
                clip_name=clip_name, video=video, gt_path=gt_path,
                base_weights=args.base_yolo_weights,
                our_weights=args.our_yolo_weights,
                reid_weights=args.reid_weights,
                device=args.device, max_frames=args.max_frames,
                out_pkl_dir=out_pkl_dir,
            )
        except Exception as exc:
            log.exception("clip %s crashed: %s", clip_name, exc)
            continue
        if not rows:
            continue
        all_results["clips"][clip_name] = {
            "video": str(video),
            "gt": str(gt_path),
            "rows": {n: asdict(r) for n, r in rows.items()},
        }
        args.out_json.write_text(json.dumps(all_results, indent=2))
        log.info("wrote %s after clip %s", args.out_json, clip_name)

    overall = time.time() - overall_t0
    all_results["total_wall_seconds"] = round(overall, 3)
    args.out_json.write_text(json.dumps(all_results, indent=2))
    log.info("ALL DONE in %.1f min  ->  %s",
             overall / 60.0, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
