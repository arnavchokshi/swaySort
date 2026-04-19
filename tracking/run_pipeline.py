"""End-to-end best-pipeline driver: video -> tracks.pkl.

This is the **single** entry point for running the production 2D
person-tracking + ID-assignment pipeline on a new video.

Pipeline stages (one tracker, one detector, one ReID, one
post-process chain -- see ``docs/PIPELINE_SPEC.md``):

  1. Multi-scale YOLO detection ensemble
        weights = ``weights/best.pt`` (dance-fine-tuned YOLO26s)
        imgsz   = (768, 1024), NMS-union at ensemble_iou=0.6
        conf    = 0.34 (DET_CONF)
        iou     = 0.70, classes = [person]

  2. DeepOcSort + OSNet x0.25 ReID
        max_age=30 (boxmot defaults; do NOT raise)
        cholesky-jitter Kalman patch installed at startup

  3. Post-process chain (``tracking.best_pipeline.build_tracks``)
        - postprocess_tracks (relaxed pre-merge + widened ID merge)
        - filter_tracks_post_merge (length AND mean_conf AND p90_conf)
        - bbox_continuity_stitch (loose: gap=400, jump=2000, size=4.0)
        - size_smooth_cv_gated (CV-gated size smoother)
        - smooth_centers_median (per-track center median)

  4. Output: ``dict[track_id -> Track]`` joblib-pickled to ``tracks.pkl``.

Usage::

    # CLI
    python -m tracking.run_pipeline \\
        --video path/to/dance.mp4 \\
        --out   work/dance_tracks.pkl

    # Programmatic
    from pathlib import Path
    from tracking.run_pipeline import run_pipeline_on_video

    tracks = run_pipeline_on_video(
        video=Path("dance.mp4"),
        out=Path("work/dance_tracks.pkl"),
        device="cuda:0",
    )

The intermediate FrameDetections cache is also dumped (next to the
output ``tracks.pkl``) so repeated post-process experimentation does
not have to re-run YOLO + DeepOcSort.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prune_tracks import FrameDetections  # noqa: E402

from tracking.best_pipeline import DET_CONF, build_tracks
from tracking.deepocsort_runner import (
    install_kalman_jitter_patch,
    iter_video_frames,
    iter_video_frames_prefetched,
    make_tracker,
)
from tracking.multi_scale_detector import make_multi_scale_detector
from tracking.postprocess import Track


log = logging.getLogger("tracking.run_pipeline")


def _resolve_prefetch_depth() -> int:
    """Read BEST_ID_PREFETCH and return the queue depth (0 = sync)."""
    raw = os.environ.get("BEST_ID_PREFETCH", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError:
        log.warning("BEST_ID_PREFETCH=%r is not an int; falling back to 0", raw)
        return 0
    return max(0, n)


def _resolve_pipeline_parallel() -> bool:
    """Read BEST_ID_PIPELINE_PARALLEL flag (truthy values: 1/true/yes/on)."""
    raw = os.environ.get("BEST_ID_PIPELINE_PARALLEL", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _frame_iter(video: Path) -> Iterator[Tuple[int, np.ndarray]]:
    """Choose sync vs prefetched frame decoder based on env var."""
    depth = _resolve_prefetch_depth()
    if depth <= 0:
        return iter_video_frames(video)
    log.info("frame prefetch enabled: queue_depth=%d", depth)
    return iter_video_frames_prefetched(video, queue_size=depth)


# Production constants. Every value here was selected by sweeping
# across a 7-clip benchmark under a strict no-regression rule. Full
# spec in docs/PIPELINE_SPEC.md, sweep tables in docs/EXPERIMENTS_LOG.md.

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "best.pt"
DEFAULT_CFG = REPO_ROOT / "configs" / "best_pipeline.json"
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"

def _resolve_imgsz_ensemble() -> Tuple[int, ...]:
    """Read BEST_ID_IMGSZ_ENSEMBLE if set (comma-separated ints), else
    return the v8 default (768, 1024). Set unset == v8 byte-identical.
    """
    raw = os.environ.get("BEST_ID_IMGSZ_ENSEMBLE", "").strip()
    if not raw:
        return (768, 1024)
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            log.warning("BEST_ID_IMGSZ_ENSEMBLE token %r not int; ignoring",
                        tok)
    if not out:
        return (768, 1024)
    return tuple(sorted(set(out)))


def _resolve_ensemble_iou() -> float:
    raw = os.environ.get("BEST_ID_ENSEMBLE_IOU", "").strip()
    if not raw:
        return 0.6
    try:
        return float(raw)
    except ValueError:
        log.warning("BEST_ID_ENSEMBLE_IOU=%r not float; using 0.6", raw)
        return 0.6


def _resolve_tta_flip() -> bool:
    raw = os.environ.get("BEST_ID_TTA_FLIP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


DETECTOR_IMGSZ_ENSEMBLE = (768, 1024)
DETECTOR_ENSEMBLE_IOU = 0.6
DETECTOR_CONF = float(DET_CONF)  # 0.34, the plateau centre
DETECTOR_IOU = 0.70
PERSON_CLASS_ID = 0


def _record_tracker_output(
    out_frames: List[FrameDetections],
    tracks_per_frame: Optional[np.ndarray],
) -> None:
    """Append a single tracker frame to ``out_frames`` -- shared between
    the serial and pipelined drivers so they emit byte-identical caches.
    """
    if tracks_per_frame is None or len(tracks_per_frame) == 0:
        out_frames.append(FrameDetections(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ))
        return
    arr = np.asarray(tracks_per_frame, dtype=np.float32)
    xyxys = arr[:, 0:4].astype(np.float32)
    tids = arr[:, 4].astype(np.float32)
    if arr.shape[1] > 5:
        confs = arr[:, 5].astype(np.float32)
    else:
        confs = np.ones(len(arr), dtype=np.float32)
    out_frames.append(FrameDetections(xyxys, confs, tids))


def _safe_detect(detect, frame_bgr: np.ndarray) -> np.ndarray:
    dets = detect(frame_bgr)
    if dets.size == 0:
        dets = np.zeros((0, 6), dtype=np.float32)
    return dets


def _safe_tracker_update(tracker, dets, frame_bgr, idx) -> np.ndarray:
    try:
        tracks_per_frame = tracker.update(dets, frame_bgr)
    except Exception as exc:
        log.exception("DeepOcSort.update failed at frame %d: %s", idx, exc)
        tracks_per_frame = np.zeros((0, 7), dtype=np.float32)
    return tracks_per_frame


def _detect_and_track_serial(
    *, video: Path, detect, tracker, max_frames: Optional[int],
) -> List[FrameDetections]:
    out_frames: List[FrameDetections] = []
    n_processed = 0
    t0 = time.time()
    for idx, frame_bgr in _frame_iter(Path(video)):
        if max_frames is not None and idx >= max_frames:
            break
        dets = _safe_detect(detect, frame_bgr)
        tracks_per_frame = _safe_tracker_update(tracker, dets, frame_bgr, idx)
        _record_tracker_output(out_frames, tracks_per_frame)
        n_processed += 1
    dt = time.time() - t0
    log.info(
        "%d frames in %.1fs (%.2f FPS)", n_processed, dt,
        n_processed / max(dt, 1e-6),
    )
    return out_frames


def _detect_and_track_pipelined(
    *, video: Path, detect, tracker, max_frames: Optional[int],
) -> List[FrameDetections]:
    """One-frame look-ahead: while tracker runs on frame N, detector is
    already running on frame N+1 in a worker thread.

    Order of tracker.update() calls is preserved exactly, so DeepOcSort's
    Kalman + ReID-gallery state evolves identically to the serial path.
    The only thing that changes is *when* detector forwards happen.

    On CUDA the detector and tracker run on separate streams so there's
    real GPU concurrency on top of the CPU/GPU overlap. On MPS the GPU
    serialises but we still hide the tracker's CPU work (Kalman,
    association, ReID gallery maintenance) behind the next detector
    forward.
    """
    from concurrent.futures import ThreadPoolExecutor

    out_frames: List[FrameDetections] = []
    n_processed = 0
    log.info("detect/track pipeline parallelism enabled (1-frame lookahead)")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="det") as pool:
        prev_idx: Optional[int] = None
        prev_frame: Optional[np.ndarray] = None
        det_future = None

        for idx, frame_bgr in _frame_iter(Path(video)):
            if max_frames is not None and idx >= max_frames:
                break

            new_future = pool.submit(_safe_detect, detect, frame_bgr)

            if det_future is not None:
                prev_dets = det_future.result()
                tracks_per_frame = _safe_tracker_update(
                    tracker, prev_dets, prev_frame, prev_idx,
                )
                _record_tracker_output(out_frames, tracks_per_frame)
                n_processed += 1

            det_future = new_future
            prev_frame = frame_bgr
            prev_idx = idx

        if det_future is not None:
            prev_dets = det_future.result()
            tracks_per_frame = _safe_tracker_update(
                tracker, prev_dets, prev_frame, prev_idx,
            )
            _record_tracker_output(out_frames, tracks_per_frame)
            n_processed += 1

    dt = time.time() - t0
    log.info(
        "%d frames in %.1fs (%.2f FPS) [pipelined]", n_processed, dt,
        n_processed / max(dt, 1e-6),
    )
    return out_frames


def _detect_and_track(
    *,
    video: Path,
    weights: Path,
    reid_weights: Path,
    device: str,
    max_frames: Optional[int],
) -> List[FrameDetections]:
    """Run multi-scale YOLO + DeepOcSort over ``video`` and return
    a list of ``FrameDetections`` (one per processed frame).

    The output schema matches the cache format consumed by
    ``tracking.best_pipeline.build_tracks``.
    """
    install_kalman_jitter_patch()

    detect = make_multi_scale_detector(
        weights=weights,
        imgsz_list=list(_resolve_imgsz_ensemble()),
        conf=DETECTOR_CONF,
        iou=DETECTOR_IOU,
        device=device,
        ensemble_iou=_resolve_ensemble_iou(),
        classes=[PERSON_CLASS_ID],
        tta_flip=_resolve_tta_flip(),
    )
    tracker = make_tracker(
        reid_weights=reid_weights, device=device, half=False,
    )

    if _resolve_pipeline_parallel():
        return _detect_and_track_pipelined(
            video=video, detect=detect, tracker=tracker,
            max_frames=max_frames,
        )
    return _detect_and_track_serial(
        video=video, detect=detect, tracker=tracker, max_frames=max_frames,
    )


def run_pipeline_on_video(
    *,
    video: Path,
    out: Path,
    weights: Path = DEFAULT_WEIGHTS,
    cfg: Path = DEFAULT_CFG,
    reid_weights: Path = Path(DEFAULT_REID_WEIGHTS),
    device: str = "cuda:0",
    max_frames: Optional[int] = None,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> Dict[int, Track]:
    """Run the full v8 best pipeline on ``video`` and dump tracks.pkl.

    Args:
        video: Input video file (any container OpenCV / imageio can
            read, e.g. .mp4, .mov) OR a directory of ordered frame
            images.
        out:   Output path for the tracks pickle (joblib-saved
            ``dict[int, Track]``). The parent directory is created if
            missing.
        weights: YOLO weights. Defaults to ``weights/best.pt`` (the
            dance-fine-tuned YOLO26s).
        cfg: Path to ``configs/best_pipeline.json`` (post-process
            knobs). Defaults to the bundled production config.
        reid_weights: Path or name of the ReID checkpoint. BoxMOT will
            auto-download the canonical OSNet x0.25 checkpoint if a
            registered name is given (default).
        device: Torch device string ("cuda:0" / "mps" / "cpu").
        max_frames: Optional cap on the number of input frames (for
            testing).
        cache_path: Optional explicit path for the intermediate
            FrameDetections cache. Defaults to ``<out>.cache.pkl``
            beside the output. Kept on disk so post-process tweaks
            don't need to re-run YOLO.
        force: When True, re-build the cache even if one is on disk.

    Returns:
        ``dict[int -> Track]`` -- the v8 final tracks. Same dict is
        joblib-pickled to ``out``.
    """
    video = Path(video)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = out.with_suffix(out.suffix + ".cache.pkl")
    cache_path = Path(cache_path)

    if cache_path.is_file() and not force:
        log.info("cache hit: %s", cache_path)
    else:
        log.info("running detector + tracker on %s -> %s", video, cache_path)
        frames = _detect_and_track(
            video=video, weights=weights, reid_weights=reid_weights,
            device=device, max_frames=max_frames,
        )
        # Optional FN-recovery pass (env-gated; no-op when disabled).
        from tracking import fn_recovery
        if fn_recovery.is_enabled():
            n_added = fn_recovery.recover_missing_detections(frames)
            log.info("fn-recovery added %d synthetic detections", n_added)
        # Optional SAM 2.1 per-bbox verifier (env-gated; no-op when
        # disabled). Drops phantom detections by checking that SAM's
        # foreground mask actually fills the bbox. Image predictor only
        # -- never video predictor -- so the past mask-propagation
        # identity-fusion failure mode cannot occur.
        from tracking import sam2_verifier
        if sam2_verifier.is_enabled():
            n_dropped = sam2_verifier.verify_cache(frames, video=video)
            log.info("sam-verify dropped %d phantom detections", n_dropped)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(frames, str(cache_path))
        log.info("wrote cache: %s (%d frames)", cache_path, len(frames))
        sidecar = cache_path.with_suffix(cache_path.suffix + ".video.json")
        try:
            sidecar.write_text(json.dumps({
                "video": str(Path(video).resolve()),
                "frames": int(len(frames)),
            }))
        except OSError as exc:
            log.warning("could not write video sidecar %s: %s", sidecar, exc)

    log.info("building tracks from cache: %s", cache_path)
    tracks = build_tracks(
        cache_path=cache_path, cfg_path=Path(cfg), save_to=out,
    )
    log.info("wrote %s (%d tracks)", out, len(tracks))
    return tracks


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", type=Path, required=True,
                   help="Input video file or directory of frames.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output tracks.pkl path "
                        "(joblib-pickled dict[int, Track]).")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help=f"YOLO weights (default {DEFAULT_WEIGHTS}).")
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG,
                   help=f"Post-process config JSON "
                        f"(default {DEFAULT_CFG}).")
    p.add_argument("--reid-weights", type=Path,
                   default=Path(DEFAULT_REID_WEIGHTS),
                   help="ReID checkpoint name or path. BoxMOT "
                        "auto-downloads canonical OSNet weights.")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device (cuda:0 / mps / cpu).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional cap on input frames (for testing).")
    p.add_argument("--cache", type=Path, default=None,
                   help="Optional explicit cache path; default is "
                        "<out>.cache.pkl.")
    p.add_argument("--force", action="store_true",
                   help="Re-run YOLO + DeepOcSort even if a cache "
                        "exists on disk.")
    p.add_argument("--log-level", default="INFO",
                   help="Python logging level (default INFO).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_pipeline_on_video(
        video=args.video,
        out=args.out,
        weights=args.weights,
        cfg=args.cfg,
        reid_weights=args.reid_weights,
        device=args.device,
        max_frames=args.max_frames,
        cache_path=args.cache,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
