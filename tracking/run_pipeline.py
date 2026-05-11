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


def _timing_add(timing: Optional[dict], key: str, elapsed_s: float) -> None:
    if timing is None:
        return
    timers = timing.setdefault("timers_s", {})
    timers[key] = float(timers.get(key, 0.0)) + float(elapsed_s)


def _timing_inc(timing: Optional[dict], key: str, n: int = 1) -> None:
    if timing is None:
        return
    counts = timing.setdefault("counts", {})
    counts[key] = int(counts.get(key, 0)) + int(n)


def _write_timing_json(path: Path, timing: dict) -> None:
    payload = dict(timing)
    payload["timers_s"] = {
        str(k): round(float(v), 6)
        for k, v in sorted((payload.get("timers_s") or {}).items())
    }
    payload["counts"] = {
        str(k): int(v)
        for k, v in sorted((payload.get("counts") or {}).items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


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
    """Read BEST_ID_PIPELINE_PARALLEL.

    Default OFF: the H100 Phase 2 slice showed a speedup, but later
    end-to-end reruns found no current wall-clock win and showed YOLO
    detector and DeepOcSort/ReID timings stretching each other when they
    shared one CUDA device. Keep the overlap path opt-in until it is
    revalidated on the production image/hardware.
    """
    raw = os.environ.get("BEST_ID_PIPELINE_PARALLEL", "").strip().lower()
    if not raw:
        return False
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    log.warning("BEST_ID_PIPELINE_PARALLEL=%r not recognized; defaulting to OFF", raw)
    return False


def _resolve_reid_half() -> bool:
    raw = os.environ.get("BEST_ID_REID_HALF", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _frame_iter(video: Path, *, frame_stride: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
    """Choose sync vs prefetched frame decoder based on env var."""
    depth = _resolve_prefetch_depth()
    if depth <= 0:
        return iter_video_frames(video, frame_stride=frame_stride)
    log.info("frame prefetch enabled: queue_depth=%d", depth)
    return iter_video_frames_prefetched(video, queue_size=depth, frame_stride=frame_stride)


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    log.warning("%s=%r not recognized; using default %s", name, raw, default)
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r not float; using default %.3f", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r not int; using default %d", name, raw, default)
        return default


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


def _record_detector_output(out_frames: List[FrameDetections], detections: np.ndarray) -> None:
    if detections.size == 0:
        out_frames.append(FrameDetections(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ))
        return
    arr = np.asarray(detections, dtype=np.float32)
    out_frames.append(FrameDetections(
        arr[:, 0:4].astype(np.float32),
        arr[:, 4].astype(np.float32) if arr.shape[1] > 4 else np.ones(len(arr), dtype=np.float32),
        arr[:, 5].astype(np.float32) if arr.shape[1] > 5 else np.zeros(len(arr), dtype=np.float32),
    ))


def _box_area_xyxy(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _intersection_area_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _suppress_group_boxes_for_tracker(
    dets: np.ndarray,
    *,
    timing: Optional[dict] = None,
) -> np.ndarray:
    """Drop oversized detections that contain multiple smaller people boxes.

    This is an opt-in experiment for gym-style crowded choreography where YOLO
    sometimes emits one high-confidence "group" box over adjacent dancers.  It
    is intentionally conservative and disabled unless BEST_ID_GROUP_SUPPRESS=1.
    """
    if not _env_bool("BEST_ID_GROUP_SUPPRESS", False):
        return dets
    arr = np.asarray(dets, dtype=np.float32)
    if arr.ndim != 2 or len(arr) < 3 or arr.shape[1] < 4:
        return arr
    cover = _env_float("BEST_ID_GROUP_SUPPRESS_COVER", 0.65)
    area_ratio = _env_float("BEST_ID_GROUP_SUPPRESS_AREA_RATIO", 1.8)
    min_children = max(1, _env_int("BEST_ID_GROUP_SUPPRESS_MIN_CHILDREN", 2))

    boxes = arr[:, 0:4]
    areas = np.asarray([_box_area_xyxy(b) for b in boxes], dtype=np.float64)
    keep = np.ones((len(arr),), dtype=bool)
    for i, box in enumerate(boxes):
        children = 0
        for j, other in enumerate(boxes):
            if i == j:
                continue
            if areas[i] < areas[j] * area_ratio:
                continue
            if _intersection_area_xyxy(box, other) / max(1e-9, areas[j]) >= cover:
                children += 1
        if children >= min_children:
            keep[i] = False
    dropped = int(len(arr) - int(keep.sum()))
    if dropped:
        _timing_inc(timing, "tracker_group_boxes_suppressed", dropped)
    return arr[keep]


def _safe_detect(detect, frame_bgr: np.ndarray) -> np.ndarray:
    dets = detect(frame_bgr)
    if dets.size == 0:
        dets = np.zeros((0, 6), dtype=np.float32)
    return dets


def _sanitize_detections_for_tracker(
    dets: np.ndarray,
    frame_bgr: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Clip detector boxes to the image and drop invalid tracker crops.

    Some detector/backend combinations can emit boxes that extend just outside
    the frame, or degenerate boxes after numerical conversion. BoxMOT crops
    those boxes before ReID; an empty crop raises inside ``cv2.resize`` and
    drops the whole tracker update for that frame. Dropping only the invalid
    rows keeps the valid detections alive.
    """
    arr = np.asarray(dets, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 6), dtype=np.float32), 0
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.zeros((0, 6), dtype=np.float32), int(len(arr))

    out = arr.copy()
    h, w = frame_bgr.shape[:2]
    out[:, 0] = np.clip(out[:, 0], 0, max(0, w - 1))
    out[:, 2] = np.clip(out[:, 2], 0, max(0, w - 1))
    out[:, 1] = np.clip(out[:, 1], 0, max(0, h - 1))
    out[:, 3] = np.clip(out[:, 3], 0, max(0, h - 1))

    finite = np.isfinite(out[:, 0:4]).all(axis=1)
    positive = (out[:, 2] > out[:, 0] + 1.0) & (out[:, 3] > out[:, 1] + 1.0)
    keep = finite & positive
    dropped = int(len(out) - int(keep.sum()))
    out = out[keep]
    if out.shape[1] == 4:
        out = np.concatenate(
            [
                out,
                np.ones((len(out), 1), dtype=np.float32),
                np.zeros((len(out), 1), dtype=np.float32),
            ],
            axis=1,
        )
    return out.astype(np.float32, copy=False), dropped


def _safe_tracker_update(tracker, dets, frame_bgr, idx, timing: Optional[dict] = None) -> np.ndarray:
    dets = _suppress_group_boxes_for_tracker(dets, timing=timing)
    dets, n_dropped = _sanitize_detections_for_tracker(dets, frame_bgr)
    if n_dropped:
        _timing_inc(timing, "tracker_input_dets_dropped", n_dropped)
        log.debug("dropped %d invalid tracker detections at frame %d", n_dropped, idx)
    try:
        tracks_per_frame = tracker.update(dets, frame_bgr)
    except Exception as exc:
        _timing_inc(timing, "tracker_update_failures")
        log.exception("DeepOcSort.update failed at frame %d: %s", idx, exc)
        tracks_per_frame = np.zeros((0, 7), dtype=np.float32)
    return tracks_per_frame


def _detect_and_track_serial(
    *, video: Path, detect, tracker, max_frames: Optional[int], detector_frames: Optional[List[FrameDetections]] = None,
    frame_stride: int = 1, timing: Optional[dict] = None,
) -> List[FrameDetections]:
    out_frames: List[FrameDetections] = []
    n_processed = 0
    t0 = time.time()
    frame_iter = iter(_frame_iter(Path(video), frame_stride=frame_stride))
    while True:
        decode_t0 = time.perf_counter()
        try:
            idx, frame_bgr = next(frame_iter)
        except StopIteration:
            _timing_add(timing, "tracking_decode_iter_s", time.perf_counter() - decode_t0)
            break
        _timing_add(timing, "tracking_decode_iter_s", time.perf_counter() - decode_t0)
        _timing_inc(timing, "tracking_frames_decoded")
        if max_frames is not None and idx >= max_frames:
            break
        detect_t0 = time.perf_counter()
        dets = _safe_detect(detect, frame_bgr)
        _timing_add(timing, "detector_calls_s", time.perf_counter() - detect_t0)
        _timing_inc(timing, "detector_calls")
        if detector_frames is not None:
            _record_detector_output(detector_frames, dets)
        tracker_t0 = time.perf_counter()
        tracks_per_frame = _safe_tracker_update(tracker, dets, frame_bgr, idx, timing)
        _timing_add(timing, "tracker_update_calls_s", time.perf_counter() - tracker_t0)
        _timing_inc(timing, "tracker_update_calls")
        _record_tracker_output(out_frames, tracks_per_frame)
        n_processed += 1
    dt = time.time() - t0
    log.info(
        "%d frames in %.1fs (%.2f FPS)", n_processed, dt,
        n_processed / max(dt, 1e-6),
    )
    return out_frames


def _detect_and_track_pipelined(
    *, video: Path, detect, tracker, max_frames: Optional[int], detector_frames: Optional[List[FrameDetections]] = None,
    frame_stride: int = 1, timing: Optional[dict] = None,
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

    def timed_detect(frame_bgr):
        start = time.perf_counter()
        try:
            return _safe_detect(detect, frame_bgr)
        finally:
            _timing_add(timing, "detector_calls_s", time.perf_counter() - start)
            _timing_inc(timing, "detector_calls")

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="det") as pool:
        prev_idx: Optional[int] = None
        prev_frame: Optional[np.ndarray] = None
        det_future = None

        frame_iter = iter(_frame_iter(Path(video), frame_stride=frame_stride))
        while True:
            decode_t0 = time.perf_counter()
            try:
                idx, frame_bgr = next(frame_iter)
            except StopIteration:
                _timing_add(timing, "tracking_decode_iter_s", time.perf_counter() - decode_t0)
                break
            _timing_add(timing, "tracking_decode_iter_s", time.perf_counter() - decode_t0)
            _timing_inc(timing, "tracking_frames_decoded")
            if max_frames is not None and idx >= max_frames:
                break

            new_future = pool.submit(timed_detect, frame_bgr)

            if det_future is not None:
                prev_dets = det_future.result()
                if detector_frames is not None:
                    _record_detector_output(detector_frames, prev_dets)
                tracker_t0 = time.perf_counter()
                tracks_per_frame = _safe_tracker_update(
                    tracker, prev_dets, prev_frame, prev_idx, timing,
                )
                _timing_add(timing, "tracker_update_calls_s", time.perf_counter() - tracker_t0)
                _timing_inc(timing, "tracker_update_calls")
                _record_tracker_output(out_frames, tracks_per_frame)
                n_processed += 1

            det_future = new_future
            prev_frame = frame_bgr
            prev_idx = idx

        if det_future is not None:
            prev_dets = det_future.result()
            if detector_frames is not None:
                _record_detector_output(detector_frames, prev_dets)
            tracker_t0 = time.perf_counter()
            tracks_per_frame = _safe_tracker_update(
                tracker, prev_dets, prev_frame, prev_idx, timing,
            )
            _timing_add(timing, "tracker_update_calls_s", time.perf_counter() - tracker_t0)
            _timing_inc(timing, "tracker_update_calls")
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
    detections_cache_path: Optional[Path] = None,
    frame_stride: int = 1,
    timing: Optional[dict] = None,
) -> List[FrameDetections]:
    """Run multi-scale YOLO + DeepOcSort over ``video`` and return
    a list of ``FrameDetections`` (one per processed frame).

    The output schema matches the cache format consumed by
    ``tracking.best_pipeline.build_tracks``.
    """
    patch_t0 = time.perf_counter()
    install_kalman_jitter_patch()
    _timing_add(timing, "kalman_patch_s", time.perf_counter() - patch_t0)

    detector_t0 = time.perf_counter()
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
    _timing_add(timing, "detector_init_s", time.perf_counter() - detector_t0)
    tracker_t0 = time.perf_counter()
    tracker = make_tracker(
        reid_weights=reid_weights, device=device, half=_resolve_reid_half(),
    )
    _timing_add(timing, "tracker_init_s", time.perf_counter() - tracker_t0)

    detector_frames: Optional[List[FrameDetections]] = [] if detections_cache_path is not None else None
    loop_t0 = time.perf_counter()
    if _resolve_pipeline_parallel():
        frames = _detect_and_track_pipelined(
            video=video, detect=detect, tracker=tracker,
            max_frames=max_frames, detector_frames=detector_frames,
            frame_stride=frame_stride, timing=timing,
        )
    else:
        frames = _detect_and_track_serial(
            video=video, detect=detect, tracker=tracker, max_frames=max_frames,
            detector_frames=detector_frames,
            frame_stride=frame_stride, timing=timing,
        )
    _timing_add(timing, "detect_track_loop_s", time.perf_counter() - loop_t0)
    if detections_cache_path is not None and detector_frames is not None:
        detections_cache_path.parent.mkdir(parents=True, exist_ok=True)
        dump_t0 = time.perf_counter()
        joblib.dump(detector_frames, str(detections_cache_path))
        _timing_add(timing, "detections_cache_dump_s", time.perf_counter() - dump_t0)
        log.info("wrote detector cache: %s (%d frames)", detections_cache_path, len(detector_frames))
    return frames


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
    detections_cache_path: Optional[Path] = None,
    force: bool = False,
    frame_stride: int = 1,
    timing_path: Optional[Path] = None,
    expected_total_performers: Optional[int] = None,
    presence_plan_path: Optional[Path] = None,
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
    timing: Optional[dict] = None
    if timing_path is not None:
        timing = {
            "video": str(video),
            "out": str(out),
            "device": device,
            "frame_stride": int(frame_stride),
            "expected_total_performers": (
                int(expected_total_performers)
                if expected_total_performers is not None
                else None
            ),
            "cache_hit": False,
            "pipeline_parallel": _resolve_pipeline_parallel(),
            "prefetch_depth": _resolve_prefetch_depth(),
            "detector": {
                "weights": str(weights),
                "imgsz_ensemble": list(_resolve_imgsz_ensemble()),
                "ensemble_iou": _resolve_ensemble_iou(),
                "tta_flip": _resolve_tta_flip(),
                "conf": DETECTOR_CONF,
                "iou": DETECTOR_IOU,
                "yolo_half": _env_bool("BEST_ID_YOLO_HALF", False),
            },
            "tracker": {
                "reid_half": _resolve_reid_half(),
                "env_overrides": {
                    k: v for k, v in sorted(os.environ.items())
                    if k.startswith("BEST_ID_DEEPOCSORT_")
                    or k.startswith("BEST_ID_TRACKER_")
                },
            },
        }
    total_t0 = time.perf_counter()
    if cache_path is None:
        cache_path = out.with_suffix(out.suffix + ".cache.pkl")
    cache_path = Path(cache_path)

    if cache_path.is_file() and not force:
        log.info("cache hit: %s", cache_path)
        if timing is not None:
            timing["cache_hit"] = True
    else:
        log.info("running detector + tracker on %s -> %s", video, cache_path)
        frames = _detect_and_track(
            video=video, weights=weights, reid_weights=reid_weights,
            device=device, max_frames=max_frames,
            detections_cache_path=detections_cache_path,
            frame_stride=frame_stride,
            timing=timing,
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
        cache_dump_t0 = time.perf_counter()
        joblib.dump(frames, str(cache_path))
        _timing_add(timing, "tracking_cache_dump_s", time.perf_counter() - cache_dump_t0)
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
    post_t0 = time.perf_counter()
    tracks = build_tracks(
        cache_path=cache_path,
        cfg_path=Path(cfg),
        save_to=out,
        expected_total_performers=expected_total_performers,
        presence_plan_path=presence_plan_path,
    )
    _timing_add(timing, "postprocess_build_tracks_s", time.perf_counter() - post_t0)
    log.info("wrote %s (%d tracks)", out, len(tracks))
    if timing is not None:
        timing["n_tracks"] = len(tracks)
        timing["cache_path"] = str(cache_path)
        timing["total_run_pipeline_s"] = round(time.perf_counter() - total_t0, 6)
        _write_timing_json(Path(timing_path), timing)
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
    p.add_argument("--detections-cache", type=Path, default=None,
                   help="Optional path for raw detector boxes before "
                        "DeepOcSort association.")
    p.add_argument("--force", action="store_true",
                   help="Re-run YOLO + DeepOcSort even if a cache "
                        "exists on disk.")
    p.add_argument("--timing-json", type=Path, default=None,
                   help="Write detector/tracker/postprocess timing JSON.")
    p.add_argument("--expected-total-performers", type=int, default=None,
                   help="Optional total unique target performer count prior.")
    p.add_argument("--presence-plan", type=Path, default=None,
                   help="Optional path for the expected-count presence plan JSON.")
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
        detections_cache_path=args.detections_cache,
        force=args.force,
        timing_path=args.timing_json,
        expected_total_performers=args.expected_total_performers,
        presence_plan_path=args.presence_plan,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
