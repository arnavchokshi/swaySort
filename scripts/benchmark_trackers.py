"""Fair head-to-head speed benchmark across BoxMOT trackers.

Caches YOLO multi-scale detections once per clip, then replays the *same*
frames + detections through each tracker so the only variable is the
tracker itself. Reports per-tracker latency, FPS, and bbox/track counts
to ``work/benchmarks/tracker_speeds.json``.

We deliberately use **base / default** competitor configs (the
out-of-the-box BoxMOT constructor defaults) so the comparison is the
"swap-in baseline" a user would actually get if they followed BoxMOT's
quickstart -- not the per-tracker oracle. The shipped DeepOcSort entry
matches what ``tracking/run_pipeline.py`` ships in production
(OSNet x0.25 ReID, default hyperparameters, cholesky-jitter Kalman
patch installed).

Usage::

    python scripts/benchmark_trackers.py \\
        --video /Users/.../loveTest/IMG_9265.mov \\
        --clip-name loveTest \\
        --device mps

The detector cache lives next to the video as
``<video_stem>.det_cache.pkl`` so successive tracker runs only pay
detection cost once.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tracking.deepocsort_runner import (  # noqa: E402
    install_kalman_jitter_patch, iter_video_frames, make_tracker,
)
from tracking.multi_scale_detector import make_multi_scale_detector  # noqa: E402

log = logging.getLogger("benchmark_trackers")

DEFAULT_WEIGHTS = REPO / "weights" / "best.pt"
DETECTOR_IMGSZ = (768, 1024)
DETECTOR_ENSEMBLE_IOU = 0.6
DETECTOR_CONF = 0.34
DETECTOR_IOU = 0.70
PERSON_CLASS_ID = 0
DEFAULT_REID = "osnet_x0_25_msmt17.pt"


@dataclass
class TrackerResult:
    """Per-tracker speed + output statistics."""
    name: str
    config: str
    n_frames: int
    total_track_seconds: float
    mean_ms_per_frame: float
    median_ms_per_frame: float
    p95_ms_per_frame: float
    tracker_fps: float
    end_to_end_ms_per_frame: float
    end_to_end_fps: float
    n_unique_tracks: int
    mean_active_per_frame: float


def _build_tracker(name: str, *, device: str, reid_weights):
    """Construct a BoxMOT tracker with default / base hyperparameters.

    For ReID-using trackers (BotSort, DeepOcSort, StrongSort, HybridSort)
    we pass the OSNet x0.25 weights -- the same head our shipped
    pipeline uses. For motion-only trackers (ByteTrack, OcSort) ReID is
    not constructed.
    """
    import torch
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
    from boxmot.trackers.hybridsort.hybridsort import HybridSort
    from boxmot.trackers.ocsort.ocsort import OcSort
    from boxmot.trackers.strongsort.strongsort import StrongSort

    if device == "mps" and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")

    if name == "DeepOcSort (ours, OSNet x0.25)":
        # The shipped configuration: same call path as run_pipeline.
        return make_tracker(
            reid_weights=Path(reid_weights), device=device, half=False,
        ), "OSNet x0.25 ReID, max_age=30, BoxMOT defaults + cholesky patch"
    if name == "BotSort (base)":
        return BotSort(
            reid_weights=Path(reid_weights), device=torch_device, half=False,
        ), "OSNet x0.25 ReID, BoxMOT defaults"
    if name == "ByteTrack (base)":
        return ByteTrack(), "BoxMOT defaults, no ReID, no appearance"
    if name == "OcSort (base, no ReID)":
        return OcSort(), "BoxMOT defaults, no ReID"
    if name == "StrongSort (base)":
        return StrongSort(
            reid_weights=Path(reid_weights), device=torch_device, half=False,
        ), "OSNet x0.25 ReID, BoxMOT defaults"
    if name == "HybridSort (base)":
        return HybridSort(
            reid_weights=Path(reid_weights), device=torch_device, half=False,
            with_reid=True,
        ), "OSNet x0.25 ReID, BoxMOT defaults"
    raise ValueError(f"unknown tracker: {name}")


def _build_or_load_detection_cache(
    *, video: Path, weights: Path, device: str,
    max_frames: Optional[int],
) -> List[Dict[str, np.ndarray]]:
    """Run YOLO once, return a per-frame list of {bgr, dets}.

    Frames are kept in memory (not just dets) because every BoxMOT
    tracker.update(dets, frame_bgr) takes the BGR frame as the second
    arg (BotSort/StrongSort/DeepOcSort use it for ReID + CMC; ByteTrack
    + OcSort accept it for API uniformity). For loveTest @ 820 frames
    of 1080p that's ~6 GB which is fine.
    """
    cache_path = video.with_suffix(".det_cache.pkl")
    if cache_path.is_file():
        log.info("detection cache hit: %s", cache_path)
        cached = joblib.load(str(cache_path))
        if max_frames is not None:
            cached = cached[:max_frames]
        return cached

    log.info("running YOLO multi-scale detection (one-time): %s", video)
    detect = make_multi_scale_detector(
        weights=weights, imgsz_list=list(DETECTOR_IMGSZ),
        conf=DETECTOR_CONF, iou=DETECTOR_IOU, device=device,
        ensemble_iou=DETECTOR_ENSEMBLE_IOU, classes=[PERSON_CLASS_ID],
        tta_flip=False,
    )

    frames_out: List[Dict[str, np.ndarray]] = []
    det_total_t = 0.0
    for idx, frame in iter_video_frames(video):
        if max_frames is not None and idx >= max_frames:
            break
        t0 = time.time()
        dets = detect(frame).astype(np.float32)
        det_total_t += time.time() - t0
        if dets.size == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        frames_out.append({"bgr": frame, "dets": dets})

    log.info(
        "detector: %d frames in %.1fs (%.2f FPS, %.1f ms/frame)",
        len(frames_out), det_total_t,
        len(frames_out) / max(det_total_t, 1e-6),
        1000 * det_total_t / max(len(frames_out), 1),
    )
    joblib.dump(frames_out, str(cache_path))
    log.info("wrote detection cache: %s", cache_path)
    return frames_out


def _measure_tracker(
    tracker, name: str, config: str, frames_cache: List[Dict[str, np.ndarray]],
    *, det_ms_per_frame: float,
) -> TrackerResult:
    """Replay cached detections through ``tracker``, time each update()."""
    install_kalman_jitter_patch()  # safe no-op for non-DeepOc trackers
    per_frame_ms: List[float] = []
    seen_ids: set[int] = set()
    active_counts: List[int] = []

    for fc in frames_cache:
        dets = fc["dets"]
        frame = fc["bgr"]
        t0 = time.time()
        try:
            out = tracker.update(dets, frame)
        except Exception as exc:
            log.warning("%s.update failed: %s", name, exc)
            out = np.zeros((0, 7), dtype=np.float32)
        per_frame_ms.append(1000 * (time.time() - t0))

        if out is not None and len(out) > 0:
            arr = np.asarray(out, dtype=np.float32)
            if arr.shape[1] >= 5:
                ids = arr[:, 4].astype(int).tolist()
                seen_ids.update(ids)
                active_counts.append(len(ids))
            else:
                active_counts.append(0)
        else:
            active_counts.append(0)

    arr_ms = np.asarray(per_frame_ms, dtype=np.float64)
    total_s = float(arr_ms.sum() / 1000.0)
    mean_ms = float(arr_ms.mean()) if len(arr_ms) else 0.0
    median_ms = float(np.median(arr_ms)) if len(arr_ms) else 0.0
    p95_ms = float(np.percentile(arr_ms, 95)) if len(arr_ms) else 0.0
    fps_track = len(arr_ms) / max(total_s, 1e-6)
    e2e_ms = mean_ms + det_ms_per_frame
    e2e_fps = 1000.0 / max(e2e_ms, 1e-6)
    return TrackerResult(
        name=name, config=config, n_frames=len(arr_ms),
        total_track_seconds=round(total_s, 3),
        mean_ms_per_frame=round(mean_ms, 2),
        median_ms_per_frame=round(median_ms, 2),
        p95_ms_per_frame=round(p95_ms, 2),
        tracker_fps=round(fps_track, 2),
        end_to_end_ms_per_frame=round(e2e_ms, 2),
        end_to_end_fps=round(e2e_fps, 2),
        n_unique_tracks=len(seen_ids),
        mean_active_per_frame=round(
            float(np.mean(active_counts)) if active_counts else 0.0, 2,
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--clip-name", type=str, required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="mps")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap frames (e.g. 200) for a quick smoke-test")
    p.add_argument("--out", type=Path,
                   default=REPO / "work" / "benchmarks" / "tracker_speeds.json")
    p.add_argument("--reid-weights", default=DEFAULT_REID)
    p.add_argument("--trackers", nargs="*", default=None,
                   help="optional subset; default = all 6")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    frames_cache = _build_or_load_detection_cache(
        video=args.video, weights=args.weights, device=args.device,
        max_frames=args.max_frames,
    )
    if not frames_cache:
        log.error("no frames produced from %s", args.video)
        return 1

    # Re-derive a clean detector-only timing from the cache run by
    # measuring the wall-clock cost of just one fresh forward pass per
    # frame. Doing the full clip twice would double the benchmark, so
    # we instead time a calibration sample of 50 frames.
    detect = make_multi_scale_detector(
        weights=args.weights, imgsz_list=list(DETECTOR_IMGSZ),
        conf=DETECTOR_CONF, iou=DETECTOR_IOU, device=args.device,
        ensemble_iou=DETECTOR_ENSEMBLE_IOU, classes=[PERSON_CLASS_ID],
        tta_flip=False,
    )
    sample = frames_cache[: min(50, len(frames_cache))]
    t0 = time.time()
    for fc in sample:
        _ = detect(fc["bgr"])
    det_ms = 1000 * (time.time() - t0) / len(sample)
    log.info("calibrated detector latency: %.1f ms/frame on %s",
             det_ms, args.device)

    candidates = [
        "DeepOcSort (ours, OSNet x0.25)",
        "BotSort (base)",
        "StrongSort (base)",
        "HybridSort (base)",
        "ByteTrack (base)",
        "OcSort (base, no ReID)",
    ]
    if args.trackers:
        candidates = [c for c in candidates if c in args.trackers]

    results: List[TrackerResult] = []
    for name in candidates:
        log.info("=== benchmarking %s ===", name)
        try:
            tracker, cfg_str = _build_tracker(
                name, device=args.device, reid_weights=args.reid_weights,
            )
        except Exception as exc:
            log.exception("could not construct %s: %s", name, exc)
            continue
        try:
            res = _measure_tracker(
                tracker, name, cfg_str, frames_cache,
                det_ms_per_frame=det_ms,
            )
        except Exception as exc:
            log.exception("benchmark for %s failed: %s", name, exc)
            continue
        log.info(
            "  -> mean=%.1f ms median=%.1f p95=%.1f tracker_fps=%.1f "
            "e2e_fps=%.2f unique_ids=%d active/frame=%.1f",
            res.mean_ms_per_frame, res.median_ms_per_frame,
            res.p95_ms_per_frame, res.tracker_fps, res.end_to_end_fps,
            res.n_unique_tracks, res.mean_active_per_frame,
        )
        results.append(res)
        del tracker

    payload = {
        "clip": args.clip_name,
        "video": str(args.video),
        "device": args.device,
        "n_frames": len(frames_cache),
        "detector": {
            "weights": str(args.weights),
            "imgsz_ensemble": list(DETECTOR_IMGSZ),
            "conf": DETECTOR_CONF, "iou": DETECTOR_IOU,
            "ensemble_iou": DETECTOR_ENSEMBLE_IOU,
            "ms_per_frame": round(det_ms, 2),
            "fps": round(1000.0 / max(det_ms, 1e-6), 2),
        },
        "results": [asdict(r) for r in results],
    }

    if args.out.is_file():
        existing = json.loads(args.out.read_text())
        if isinstance(existing, dict) and "clips" in existing:
            existing["clips"][args.clip_name] = payload
            out_payload = existing
        else:
            out_payload = {"clips": {args.clip_name: payload}}
    else:
        out_payload = {"clips": {args.clip_name: payload}}
    args.out.write_text(json.dumps(out_payload, indent=2))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
