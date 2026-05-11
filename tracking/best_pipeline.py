"""Production best-pipeline post-process chain.

Consumes a ``FrameDetections`` cache (raw YOLO + DeepOcSort output) and
returns the final ``dict[int -> Track]``. This is stage 3+ of the
pipeline; ``tracking.run_pipeline`` produces the cache via stages 1-2.

Five post-process stages, in order::

    1. postprocess_tracks(min_total_frames=20, id_merge_max_gap=48,
                          id_merge_iou_thresh=0.10, ...)
    2. filter_tracks_post_merge(len>=60 AND mean_conf>=0.55
                                AND p90_conf>=0.84)
    3. bbox_continuity_stitch(gap=400, jump=2000, size_ratio=4.0)
    4. size_smooth_cv_gated(cv_thresh=0.20, fallback_window=21)
    5. smooth_centers_median(window=21)

Every constant in this module was selected by sweeping over a 7-clip
benchmark and accepting it only if the **strict no-regression on any
clip** condition was met. Provenance lives in
``docs/EXPERIMENTS_LOG.md``; the full reproduction spec is in
``docs/PIPELINE_SPEC.md``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from scipy.ndimage import median_filter

from tracking.bbox_stitch import bbox_continuity_stitch
from tracking.postprocess import (
    Track,
    frame_detections_to_raw_tracks,
    postprocess_tracks,
)
from tracking.expected_count import enforce_expected_count, write_presence_plan


log = logging.getLogger(__name__)


__all__ = [
    "DET_CONF",
    "PRE_MIN_TOTAL_FRAMES",
    "POST_MIN_LEN",
    "POST_MIN_CONF",
    "POST_MIN_P90_CONF",
    "ID_MERGE_MAX_GAP",
    "ID_MERGE_IOU_THRESH",
    "BBOX_STITCH_KWARGS",
    "SIZE_SMOOTHER_KWARGS",
    "CENTER_SMOOTHER_KWARGS",
    "filter_tracks_post_merge",
    "size_smooth_cv_gated",
    "smooth_centers_median",
    "build_tracks",
]


# Detector knob (used by tracking.run_pipeline at cache-build time).
# Sweep was {0.30..0.36} across all 7 clips; plateau is 0.33-0.345 with
# +0.0043 mean IDF1 vs the historical 0.31. Centre of the plateau.
DET_CONF: float = 0.34

# Stage 1 -- postprocess_tracks pre-merge length filter.
# Lower = recovers MotionTest fragments that the tracker briefly lost;
# 20 captures the full effect (tested 10..60).
PRE_MIN_TOTAL_FRAMES: int = 20

# Stage 1 -- postprocess_tracks ID-merge spatiotemporal gates.
# 2-D grid sweep (gap in {16..96} x iou in {0.10..0.50}) on the v7 cache
# peaks at gap=48, iou=0.10 (mean IDF1 0.9570 vs 0.9556 baseline).
# Plateau: gap in [48, 64] x iou in [0.10, 0.20]; values inside differ
# by <0.0002. We pick the corner of the plateau (shortest gap, loosest
# IoU) so we *consider* the most candidate merges. The OSNet cosine gate
# (configs/best_pipeline.json::pp_id_merge_osnet_cos_thresh = 0.7) still
# filters wrong merges, so loosening the spatial gates is safe.
ID_MERGE_MAX_GAP: int = 48
ID_MERGE_IOU_THRESH: float = 0.10

# Stage 2 -- post-merge AND-gate. Drops phantom tracks the relaxed
# pre-filter would otherwise let through.
# - len  >= 60   : sweep {60..200}, 60-70 tie at the best mean.
# - mean >= 0.55 : sweep {0.50..0.75}, 0.55-0.65 tie; below loses
#                  mirror reflection (mean_conf=0.49); 0.75 starts
#                  killing real tracks (-0.05 MotionTest).
# - p90  >= 0.84 : sweep {0.80..0.88}, 0.84 is sweet-spot. 0.85 starts
#                  removing real tracks; 0.86+ regresses BigTest/gymTest
#                  (real-track p90 floor is 0.86; phantom p90 = 0.835).
POST_MIN_LEN: int = 60
POST_MIN_CONF: float = 0.55
POST_MIN_P90_CONF: float = 0.84

# Stage 3 -- bbox_continuity_stitch. Intentionally permissive: the
# OSNet-gated id_merge and the pose-merge already handle short gaps;
# this stitch only fires on long off-frame walkouts (5 stitches total
# across the whole 7-clip benchmark, all real re-entries).
# - max_gap_frames        sweep 100..2000, plateau >= 400.
# - max_position_jump_px  sweep 200..5000, plateau >= 500.
# - max_size_ratio        sweep 1.4..6.0, 4.0 captures the win.
BBOX_STITCH_KWARGS: Dict[str, object] = {
    "max_gap_frames": 400,
    "max_position_jump_px": 2000.0,
    "max_size_ratio": 4.0,
}

# Stage 4 -- per-track CV-gated size smoother. Constant size when no
# depth motion (cv_w + cv_h <= 0.20), 21-frame median filter otherwise.
# Window sweep {7..51}, 21 had the cleanest no-regression profile.
SIZE_SMOOTHER_KWARGS: Dict[str, object] = {
    "cv_thresh": 0.20,
    "fallback_window": 21,
}

# Stage 5 -- per-track center median filter. Window sweep {11..71},
# 21 is the strict-no-regression sweet spot (>= 31 starts hurting
# easyTest by ~0.002).
CENTER_SMOOTHER_KWARGS: Dict[str, object] = {
    "window": 21,
}

# Default bbox calibration chosen on the A40 CVAT priority set after excluding
# MotionTest, whose annotations are not valid for acceptance.  The transform is
# intentionally simple and global: it keeps the detector's vertical extent,
# trims slight horizontal overreach, and preserves the left shift that aligns
# the wide-angle phone footage better than raw YOLO boxes.
BOX_CALIBRATE_DEFAULT: bool = True
BOX_CALIBRATE_SCALE_X: float = 0.90
BOX_CALIBRATE_SCALE_Y: float = 1.00
BOX_CALIBRATE_SHIFT_X: float = -0.10
BOX_CALIBRATE_SHIFT_Y: float = 0.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r invalid; using default %d", name, raw, default)
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r invalid; using default %.3f", name, raw, default)
        return float(default)


def _resolve_pose_max_center_dist(cfg_value: float) -> float:
    """Resolve the optional long-gap proximity merge gate.

    The default preserves the historical config.  The env switch exists so
    CVAT cache ablations can disable this merge without editing JSON.
    """
    if not _env_bool("BEST_ID_POSE_PROX_MERGE", True):
        return float("inf")
    raw = os.environ.get("BEST_ID_POSE_MAX_CENTER_DIST", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            log.warning("BEST_ID_POSE_MAX_CENTER_DIST=%r invalid; using config value", raw)
    return float(cfg_value)


def _resolve_pre_min_total_frames() -> int:
    return _env_int("BEST_ID_PRE_MIN_TOTAL_FRAMES", PRE_MIN_TOTAL_FRAMES)


def _resolve_id_merge_max_gap() -> int:
    return _env_int("BEST_ID_ID_MERGE_MAX_GAP", ID_MERGE_MAX_GAP)


def _resolve_id_merge_iou_thresh() -> float:
    return _env_float("BEST_ID_ID_MERGE_IOU_THRESH", ID_MERGE_IOU_THRESH)


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _box_center_wh(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(box, dtype=np.float64)
    center = np.asarray(
        [(arr[0] + arr[2]) * 0.5, (arr[1] + arr[3]) * 0.5],
        dtype=np.float64,
    )
    wh = np.asarray(
        [max(1.0, arr[2] - arr[0]), max(1.0, arr[3] - arr[1])],
        dtype=np.float64,
    )
    return center, wh


def _size_ratio(a_wh: np.ndarray, b_wh: np.ndarray) -> float:
    ratios = np.asarray(
        [
            float(a_wh[0]) / max(1.0, float(b_wh[0])),
            float(b_wh[0]) / max(1.0, float(a_wh[0])),
            float(a_wh[1]) / max(1.0, float(b_wh[1])),
            float(b_wh[1]) / max(1.0, float(a_wh[1])),
        ],
        dtype=np.float64,
    )
    return float(np.max(ratios))


def _tracks_by_frame(tracks: Dict[int, Track]) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for tid, tr in tracks.items():
        frames = np.asarray(tr.frames, dtype=np.int64)
        boxes = np.asarray(tr.bboxes, dtype=np.float32)
        for i, frame in enumerate(frames):
            if i >= len(boxes):
                continue
            by_frame.setdefault(int(frame), []).append((int(tid), boxes[i]))
    return by_frame


def _iter_detector_rows(fd: Any) -> Iterable[Tuple[np.ndarray, float]]:
    boxes = np.asarray(getattr(fd, "xyxys", []), dtype=np.float32)
    confs = np.asarray(getattr(fd, "confs", []), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        return
    for i, box in enumerate(boxes[:, :4]):
        if not np.isfinite(box).all():
            continue
        if box[2] <= box[0] + 1.0 or box[3] <= box[1] + 1.0:
            continue
        conf = float(confs[i]) if i < len(confs) else 1.0
        yield box.astype(np.float32, copy=False), conf


def _best_detector_edge_candidate(
    detector_frames: List[Any],
    frame: int,
    *,
    expected_center: np.ndarray,
    expected_wh: np.ndarray,
    occupied: List[Tuple[int, np.ndarray]],
    own_tid: int,
    min_conf: float,
    max_center_dist: float,
    max_size_ratio: float,
    occupied_iou: float,
) -> Optional[Tuple[np.ndarray, float]]:
    if frame < 0 or frame >= len(detector_frames):
        return None
    best: Optional[Tuple[float, np.ndarray, float]] = None
    for box, conf in _iter_detector_rows(detector_frames[frame]):
        if conf < min_conf:
            continue
        if any(
            tid != own_tid and _box_iou_xyxy(box, occ_box) >= occupied_iou
            for tid, occ_box in occupied
        ):
            continue
        center, wh = _box_center_wh(box)
        dist = float(np.linalg.norm(center - expected_center))
        if dist > max_center_dist:
            continue
        ratio = _size_ratio(wh, expected_wh)
        if ratio > max_size_ratio:
            continue
        score = (dist / max(1.0, max_center_dist)) + (0.25 * (ratio - 1.0)) - (0.05 * conf)
        if best is None or score < best[0]:
            best = (score, box, conf)
    if best is None:
        return None
    return best[1], best[2]


def _extend_track_with_rows(
    tr: Track,
    rows: List[Tuple[int, np.ndarray, float]],
) -> Track:
    if not rows:
        return tr
    frames = np.concatenate(
        [np.asarray(tr.frames, dtype=np.int64), np.asarray([r[0] for r in rows], dtype=np.int64)]
    )
    bboxes = np.concatenate(
        [np.asarray(tr.bboxes, dtype=np.float32), np.stack([r[1] for r in rows]).astype(np.float32)]
    )
    confs = np.concatenate(
        [np.asarray(tr.confs, dtype=np.float32), np.asarray([r[2] for r in rows], dtype=np.float32)]
    )
    detected = np.asarray(getattr(tr, "detected", []), dtype=bool)
    if len(detected) != len(tr.frames):
        detected = np.ones((len(tr.frames),), dtype=bool)
    detected = np.concatenate([detected, np.zeros((len(rows),), dtype=bool)])
    order = np.argsort(frames, kind="mergesort")
    return type(tr)(
        track_id=tr.track_id,
        frames=frames[order],
        bboxes=bboxes[order],
        confs=confs[order],
        masks=None,
        detected=detected[order],
    )


def _infer_modal_frame_count(tracks: Dict[int, Track], num_frames: int) -> int:
    if num_frames <= 0:
        return 0
    counts = np.zeros((num_frames,), dtype=np.int64)
    for tr in tracks.values():
        for frame in np.asarray(tr.frames, dtype=np.int64):
            if 0 <= int(frame) < num_frames:
                counts[int(frame)] += 1
    positive = counts[counts > 0]
    if len(positive) == 0:
        return 0
    bins = np.bincount(positive.astype(np.int64))
    # Pick the most common count. Ties choose the lower count so one-off false
    # positives do not inflate the inferred cast size.
    return int(np.flatnonzero(bins == bins.max())[0])


def _frame_counts(tracks: Dict[int, Track], num_frames: int) -> np.ndarray:
    counts = np.zeros((num_frames,), dtype=np.int64)
    for tr in tracks.values():
        for frame in np.asarray(tr.frames, dtype=np.int64):
            if 0 <= int(frame) < num_frames:
                counts[int(frame)] += 1
    return counts


def _extrapolate_box(first: np.ndarray, second: Optional[np.ndarray], step: int, *, backward: bool) -> np.ndarray:
    if second is None:
        delta = np.zeros((4,), dtype=np.float32)
    elif backward:
        delta = np.asarray(second, dtype=np.float32) - np.asarray(first, dtype=np.float32)
    else:
        delta = np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)
    return np.asarray(first, dtype=np.float32) - (float(step) * delta)


def _valid_box(box: np.ndarray) -> bool:
    return (
        np.isfinite(box).all()
        and float(box[2]) > float(box[0]) + 1.0
        and float(box[3]) > float(box[1]) + 1.0
    )


def _edge_extrapolate_to_modal_count(
    tracks: Dict[int, Track],
    *,
    num_frames: int,
    target_count: int,
    max_frames: int,
) -> Tuple[Dict[int, Track], int]:
    """Fill short edge gaps when a performance clip is below cast count.

    This uses only a very small extrapolation from the first/last boxes of an
    existing long track. It is meant for near-perfect clips where a dancer is
    temporarily absent from tracker output but the scene/cast count proves they
    did not disappear.
    """
    if target_count <= 0 or max_frames <= 0 or num_frames <= 0:
        return tracks, 0
    counts = _frame_counts(tracks, num_frames)
    out: Dict[int, Track] = {}
    added = 0
    for tid, tr in sorted(tracks.items(), key=lambda kv: int(kv[0])):
        frames = np.asarray(tr.frames, dtype=np.int64)
        boxes = np.asarray(tr.bboxes, dtype=np.float32)
        confs = np.asarray(tr.confs, dtype=np.float32)
        if len(frames) == 0 or len(boxes) == 0:
            out[tid] = tr
            continue
        order = np.argsort(frames, kind="mergesort")
        frames = frames[order]
        boxes = boxes[order]
        confs = confs[order] if len(confs) == len(order) else np.ones((len(order),), dtype=np.float32)

        rows: List[Tuple[int, np.ndarray, float]] = []
        second = boxes[1] if len(boxes) > 1 else None
        for step in range(1, max_frames + 1):
            frame = int(frames[0]) - step
            if frame < 0 or counts[frame] >= target_count:
                break
            box = _extrapolate_box(boxes[0], second, step, backward=True)
            if not _valid_box(box):
                break
            rows.append((frame, box.astype(np.float32), float(confs[0]) * 0.5))
            counts[frame] += 1
            added += 1

        prev = boxes[-2] if len(boxes) > 1 else None
        for step in range(1, max_frames + 1):
            frame = int(frames[-1]) + step
            if frame >= num_frames or counts[frame] >= target_count:
                break
            box = _extrapolate_box(boxes[-1], prev, step, backward=False)
            if not _valid_box(box):
                break
            rows.append((frame, box.astype(np.float32), float(confs[-1]) * 0.5))
            counts[frame] += 1
            added += 1

        out[tid] = _extend_track_with_rows(tr, rows)
    return out, added


def _maybe_edge_extrapolate_to_modal_count(
    tracks: Dict[int, Track],
    *,
    num_frames: int,
) -> Dict[int, Track]:
    if not _env_bool("BEST_ID_EDGE_EXTRAPOLATE", False):
        return tracks
    target_count = _env_int("BEST_ID_TARGET_COUNT", 0)
    if target_count <= 0:
        target_count = _infer_modal_frame_count(tracks, num_frames)
    out, added = _edge_extrapolate_to_modal_count(
        tracks,
        num_frames=num_frames,
        target_count=target_count,
        max_frames=max(0, _env_int("BEST_ID_EDGE_EXTRAP_MAX_FRAMES", 3)),
    )
    if added:
        log.info("edge extrapolation added %d boxes toward target count %d", added, target_count)
    return out


def _cap_frames_to_modal_count(
    tracks: Dict[int, Track],
    *,
    num_frames: int,
    target_count: int,
) -> Tuple[Dict[int, Track], int]:
    if target_count <= 0 or num_frames <= 0:
        return tracks, 0
    rows_by_frame: Dict[int, List[Tuple[int, int, float, bool, int]]] = {}
    lengths = {int(tid): len(tr.frames) for tid, tr in tracks.items()}
    for tid, tr in tracks.items():
        frames = np.asarray(tr.frames, dtype=np.int64)
        confs = np.asarray(tr.confs, dtype=np.float32)
        detected = np.asarray(getattr(tr, "detected", []), dtype=bool)
        if len(detected) != len(frames):
            detected = np.ones((len(frames),), dtype=bool)
        for idx, frame in enumerate(frames):
            if 0 <= int(frame) < num_frames:
                conf = float(confs[idx]) if idx < len(confs) else 1.0
                rows_by_frame.setdefault(int(frame), []).append(
                    (int(tid), int(idx), conf, bool(detected[idx]), lengths[int(tid)])
                )
    drop: Dict[int, set[int]] = {}
    dropped = 0
    for frame, rows in rows_by_frame.items():
        extra = len(rows) - target_count
        if extra <= 0:
            continue
        # Drop backfilled/interpolated samples first, then low confidence,
        # then shorter tracks.
        rows_sorted = sorted(rows, key=lambda r: (r[3], r[2], r[4]))
        for tid, idx, _conf, _detected, _length in rows_sorted[:extra]:
            drop.setdefault(tid, set()).add(idx)
            dropped += 1
    if not drop:
        return tracks, 0

    out: Dict[int, Track] = {}
    for tid, tr in tracks.items():
        idxs = drop.get(int(tid))
        if not idxs:
            out[tid] = tr
            continue
        keep = np.ones((len(tr.frames),), dtype=bool)
        keep[list(idxs)] = False
        out[tid] = type(tr)(
            track_id=tr.track_id,
            frames=np.asarray(tr.frames)[keep],
            bboxes=np.asarray(tr.bboxes)[keep],
            confs=np.asarray(tr.confs)[keep],
            masks=None,
            detected=(
                np.asarray(tr.detected)[keep]
                if getattr(tr, "detected", None) is not None and len(tr.detected) == len(keep)
                else np.ones((int(keep.sum()),), dtype=bool)
            ),
        )
    return out, dropped


def _maybe_cap_frames_to_modal_count(
    tracks: Dict[int, Track],
    *,
    num_frames: int,
) -> Dict[int, Track]:
    if not _env_bool("BEST_ID_FRAME_COUNT_CAP", False):
        return tracks
    target_count = _env_int("BEST_ID_TARGET_COUNT", 0)
    if target_count <= 0:
        target_count = _infer_modal_frame_count(tracks, num_frames)
    out, dropped = _cap_frames_to_modal_count(
        tracks,
        num_frames=num_frames,
        target_count=target_count,
    )
    if dropped:
        log.info("frame count cap dropped %d boxes above target count %d", dropped, target_count)
    return out


def _estimate_frame_bounds(tracks: Dict[int, Track]) -> Tuple[float, float]:
    max_x = 0.0
    max_y = 0.0
    for tr in tracks.values():
        boxes = np.asarray(tr.bboxes, dtype=np.float32)
        if boxes.ndim == 2 and boxes.shape[1] >= 4 and len(boxes):
            max_x = max(max_x, float(np.nanmax(boxes[:, 2])))
            max_y = max(max_y, float(np.nanmax(boxes[:, 3])))
    return max_x, max_y


def _touches_horizontal_edge(box: np.ndarray, frame_w: float, margin: float) -> bool:
    if frame_w <= 0:
        return float(box[0]) <= margin
    return float(box[0]) <= margin or float(box[2]) >= frame_w - margin


def _inside_stage(box: np.ndarray, frame_w: float, margin: float) -> bool:
    if frame_w <= 0:
        return float(box[0]) > margin
    return float(box[0]) > margin and float(box[2]) < frame_w - margin


def _trim_stage_edge_samples(
    tracks: Dict[int, Track],
    *,
    frame_w: float,
    edge_margin: float,
    entry_visible_width: float,
    synthetic_conf_thresh: float,
    max_trim: int,
) -> Tuple[Dict[int, Track], int]:
    """Trim track-edge samples that look like early/late stage entries.

    The rule is local to each track edge. It does not enforce a global cast
    count, which proved too blunt for videos where the number of visible
    dancers changes during entrances.
    """
    out: Dict[int, Track] = {}
    dropped = 0
    for tid, tr in tracks.items():
        frames = np.asarray(tr.frames, dtype=np.int64)
        boxes = np.asarray(tr.bboxes, dtype=np.float32)
        confs = np.asarray(tr.confs, dtype=np.float32)
        detected = np.asarray(getattr(tr, "detected", []), dtype=bool)
        if len(frames) == 0 or len(boxes) != len(frames):
            out[tid] = tr
            continue
        if len(confs) != len(frames):
            confs = np.ones((len(frames),), dtype=np.float32)
        if len(detected) != len(frames):
            detected = np.ones((len(frames),), dtype=bool)

        order = np.argsort(frames, kind="mergesort")
        frames = frames[order]
        boxes = boxes[order]
        confs = confs[order]
        detected = detected[order]

        keep = np.ones((len(frames),), dtype=bool)

        for idx in range(min(max_trim, len(frames))):
            box = boxes[idx]
            width = float(box[2] - box[0])
            edge = _touches_horizontal_edge(box, frame_w, edge_margin)
            synthetic = not bool(detected[idx])
            if synthetic and (float(confs[idx]) < synthetic_conf_thresh or edge):
                keep[idx] = False
                dropped += 1
                continue
            if edge and width <= entry_visible_width:
                keep[idx] = False
                dropped += 1
                continue
            break

        for idx in range(len(frames) - 1, max(-1, len(frames) - max_trim - 1), -1):
            if idx < 0:
                break
            box = boxes[idx]
            width = float(box[2] - box[0])
            edge = _touches_horizontal_edge(box, frame_w, edge_margin)
            synthetic = not bool(detected[idx])
            if synthetic and (float(confs[idx]) < synthetic_conf_thresh or edge):
                keep[idx] = False
                dropped += 1
                continue
            if edge and width <= entry_visible_width:
                keep[idx] = False
                dropped += 1
                continue
            break

        out[tid] = type(tr)(
            track_id=tr.track_id,
            frames=frames[keep],
            bboxes=boxes[keep],
            confs=confs[keep],
            masks=None,
            detected=detected[keep],
        )
    return out, dropped


def _stage_prior_extrapolate_edges(
    tracks: Dict[int, Track],
    *,
    num_frames: int,
    frame_w: float,
    edge_margin: float,
    max_frames: int,
    synthetic_run_min: int,
) -> Tuple[Dict[int, Track], int]:
    """Use performance continuity to fill very short track-edge gaps."""
    if num_frames <= 0 or max_frames <= 0:
        return tracks, 0
    out: Dict[int, Track] = {}
    added = 0
    for tid, tr in tracks.items():
        frames = np.asarray(tr.frames, dtype=np.int64)
        boxes = np.asarray(tr.bboxes, dtype=np.float32)
        confs = np.asarray(tr.confs, dtype=np.float32)
        detected = np.asarray(getattr(tr, "detected", []), dtype=bool)
        if len(frames) == 0 or len(boxes) != len(frames):
            out[tid] = tr
            continue
        if len(confs) != len(frames):
            confs = np.ones((len(frames),), dtype=np.float32)
        if len(detected) != len(frames):
            detected = np.ones((len(frames),), dtype=bool)

        order = np.argsort(frames, kind="mergesort")
        frames = frames[order]
        boxes = boxes[order]
        confs = confs[order]
        detected = detected[order]
        sorted_tr = type(tr)(
            track_id=tr.track_id,
            frames=frames,
            bboxes=boxes,
            confs=confs,
            masks=None,
            detected=detected,
        )

        rows: List[Tuple[int, np.ndarray, float]] = []

        prefix_len = 0
        while prefix_len < len(detected) and not bool(detected[prefix_len]):
            prefix_len += 1
        if (
            prefix_len >= synthetic_run_min
            and int(frames[0]) > 0
            and _inside_stage(boxes[0], frame_w, edge_margin)
        ):
            second = boxes[1] if len(boxes) > 1 else None
            for step in range(1, max_frames + 1):
                frame = int(frames[0]) - step
                if frame < 0:
                    break
                box = _extrapolate_box(boxes[0], second, step, backward=True)
                if not _valid_box(box) or not _inside_stage(box, frame_w, edge_margin):
                    break
                rows.append((frame, box.astype(np.float32), max(0.05, float(confs[0]) * 0.5)))

        if (
            int(frames[-1]) < num_frames - 1
            and int(frames[-1]) >= num_frames - 1 - max_frames
            and _inside_stage(boxes[-1], frame_w, edge_margin)
        ):
            prev = boxes[-2] if len(boxes) > 1 else None
            for step in range(1, max_frames + 1):
                frame = int(frames[-1]) + step
                if frame >= num_frames:
                    break
                box = _extrapolate_box(boxes[-1], prev, step, backward=False)
                if not _valid_box(box) or not _inside_stage(box, frame_w, edge_margin):
                    break
                rows.append((frame, box.astype(np.float32), max(0.05, float(confs[-1]) * 0.5)))

        added += len(rows)
        out[tid] = _extend_track_with_rows(sorted_tr, rows)
    return out, added


def _maybe_stage_prior_repair(
    tracks: Dict[int, Track],
    *,
    num_frames: int,
) -> Dict[int, Track]:
    if not _env_bool("BEST_ID_STAGE_PRIOR_REPAIR", False):
        return tracks
    frame_w, _frame_h = _estimate_frame_bounds(tracks)
    edge_margin = _env_float("BEST_ID_STAGE_EDGE_MARGIN", 8.0)
    entry_visible_width = _env_float("BEST_ID_STAGE_ENTRY_VISIBLE_WIDTH", 50.0)
    synthetic_conf_thresh = _env_float("BEST_ID_STAGE_SYNTH_CONF", 0.55)
    max_trim = max(0, _env_int("BEST_ID_STAGE_MAX_TRIM", 3))
    repaired, dropped = _trim_stage_edge_samples(
        tracks,
        frame_w=frame_w,
        edge_margin=edge_margin,
        entry_visible_width=entry_visible_width,
        synthetic_conf_thresh=synthetic_conf_thresh,
        max_trim=max_trim,
    )
    repaired, added = _stage_prior_extrapolate_edges(
        repaired,
        num_frames=num_frames,
        frame_w=frame_w,
        edge_margin=edge_margin,
        max_frames=max(0, _env_int("BEST_ID_STAGE_EXTRAP_MAX_FRAMES", 2)),
        synthetic_run_min=max(1, _env_int("BEST_ID_STAGE_SYNTH_RUN_MIN", 3)),
    )
    if dropped or added:
        log.info("stage prior repair dropped %d edge boxes and added %d continuity boxes", dropped, added)
    return repaired


def calibrate_track_boxes(
    tracks: Dict[int, Track],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> Dict[int, Track]:
    """Apply a simple center-relative bbox calibration.

    ``shift_x`` and ``shift_y`` are fractions of the original box width/height.
    This is env-gated for CVAT experiments where detector boxes are consistent
    but too wide/tall or slightly shifted.
    """
    out: Dict[int, Track] = {}
    for tid, tr in tracks.items():
        bb = np.asarray(tr.bboxes, dtype=np.float64)
        if len(bb) == 0:
            out[tid] = tr
            continue
        w = bb[:, 2] - bb[:, 0]
        h = bb[:, 3] - bb[:, 1]
        cx = (bb[:, 0] + bb[:, 2]) * 0.5 + float(shift_x) * w
        cy = (bb[:, 1] + bb[:, 3]) * 0.5 + float(shift_y) * h
        nw = np.maximum(1.0, w * float(scale_x))
        nh = np.maximum(1.0, h * float(scale_y))
        new_bb = np.stack(
            [cx - nw * 0.5, cy - nh * 0.5, cx + nw * 0.5, cy + nh * 0.5],
            axis=1,
        ).astype(np.float32)
        out[tid] = type(tr)(
            track_id=tr.track_id,
            frames=np.asarray(tr.frames),
            bboxes=new_bb,
            confs=np.asarray(tr.confs, dtype=np.float32),
            masks=getattr(tr, "masks", None),
            detected=getattr(tr, "detected", None),
        )
    return out


def _maybe_calibrate_track_boxes(
    tracks: Dict[int, Track],
    *,
    expected_count: Optional[int] = None,
) -> Dict[int, Track]:
    if not _env_bool("BEST_ID_BOX_CALIBRATE", BOX_CALIBRATE_DEFAULT):
        return tracks
    sx = _env_float("BEST_ID_BOX_SCALE_X", BOX_CALIBRATE_SCALE_X)
    sy = _env_float("BEST_ID_BOX_SCALE_Y", BOX_CALIBRATE_SCALE_Y)
    dx = _env_float("BEST_ID_BOX_SHIFT_X", BOX_CALIBRATE_SHIFT_X)
    dy = _env_float("BEST_ID_BOX_SHIFT_Y", BOX_CALIBRATE_SHIFT_Y)
    manual_box_env = any(
        os.environ.get(name, "").strip()
        for name in (
            "BEST_ID_BOX_SCALE_X",
            "BEST_ID_BOX_SCALE_Y",
            "BEST_ID_BOX_SHIFT_X",
            "BEST_ID_BOX_SHIFT_Y",
        )
    )
    if not manual_box_env and expected_count is not None and int(expected_count) >= 12:
        # Dense ensemble clips have many upper-body boxes packed side-by-side.
        # On the CVAT priority set, a slightly shorter box avoids vertical
        # overlap swaps in loveTest while preserving BigTest's zero-switch
        # behavior.  Lower-count clips keep the wider global preset because it
        # protects mirrorTest and shorterTest.
        sx = 0.85
        sy = 0.92
        dx = 0.0
    log.info("bbox calibration scale=(%.3f, %.3f) shift=(%.3f, %.3f)", sx, sy, dx, dy)
    return calibrate_track_boxes(
        tracks,
        scale_x=sx,
        scale_y=sy,
        shift_x=dx,
        shift_y=dy,
    )


def _detector_edge_backfill(
    tracks: Dict[int, Track],
    detector_frames: List[Any],
    *,
    max_frames: int,
    min_run: int,
    min_conf: float,
    max_center_dist: float,
    max_size_ratio: float,
    occupied_iou: float,
) -> Tuple[Dict[int, Track], int]:
    """Prepend/append unmatched detector boxes near track edges.

    This is intentionally edge-only. It targets delayed track starts and early
    track ends where YOLO already has a clean person box but the online tracker
    has not created or retained an ID.
    """
    if not tracks or not detector_frames or max_frames <= 0:
        return tracks, 0

    occupied_by_frame = _tracks_by_frame(tracks)
    out: Dict[int, Track] = {}
    added = 0
    for tid, tr in sorted(tracks.items(), key=lambda kv: int(kv[0])):
        frames = np.asarray(tr.frames, dtype=np.int64)
        boxes = np.asarray(tr.bboxes, dtype=np.float32)
        confs = np.asarray(tr.confs, dtype=np.float32)
        if len(frames) == 0 or len(boxes) == 0:
            out[tid] = tr
            continue
        order = np.argsort(frames, kind="mergesort")
        frames = frames[order]
        boxes = boxes[order]
        confs = confs[order] if len(confs) == len(order) else np.ones((len(order),), dtype=np.float32)

        rows: List[Tuple[int, np.ndarray, float]] = []
        back_rows: List[Tuple[int, np.ndarray, float]] = []

        first_center, first_wh = _box_center_wh(boxes[0])
        if len(boxes) >= 2:
            second_center, _ = _box_center_wh(boxes[1])
            back_step = first_center - second_center
        else:
            back_step = np.zeros((2,), dtype=np.float64)
        current_center = first_center
        expected_wh = first_wh
        for step in range(1, max_frames + 1):
            frame = int(frames[0]) - step
            expected_center = current_center + back_step
            candidate = _best_detector_edge_candidate(
                detector_frames,
                frame,
                expected_center=expected_center,
                expected_wh=expected_wh,
                occupied=occupied_by_frame.get(frame, []),
                own_tid=int(tid),
                min_conf=min_conf,
                max_center_dist=max_center_dist,
                max_size_ratio=max_size_ratio,
                occupied_iou=occupied_iou,
            )
            if candidate is None:
                break
            box, conf = candidate
            back_rows.append((frame, box, conf))
            current_center, expected_wh = _box_center_wh(box)
        if len(back_rows) >= min_run:
            rows.extend(back_rows)
            for frame, box, _conf in back_rows:
                occupied_by_frame.setdefault(frame, []).append((int(tid), box))
            added += len(back_rows)

        last_center, last_wh = _box_center_wh(boxes[-1])
        if len(boxes) >= 2:
            prev_center, _ = _box_center_wh(boxes[-2])
            forward_step = last_center - prev_center
        else:
            forward_step = np.zeros((2,), dtype=np.float64)
        current_center = last_center
        expected_wh = last_wh
        forward_rows: List[Tuple[int, np.ndarray, float]] = []
        for step in range(1, max_frames + 1):
            frame = int(frames[-1]) + step
            expected_center = current_center + forward_step
            candidate = _best_detector_edge_candidate(
                detector_frames,
                frame,
                expected_center=expected_center,
                expected_wh=expected_wh,
                occupied=occupied_by_frame.get(frame, []),
                own_tid=int(tid),
                min_conf=min_conf,
                max_center_dist=max_center_dist,
                max_size_ratio=max_size_ratio,
                occupied_iou=occupied_iou,
            )
            if candidate is None:
                break
            box, conf = candidate
            forward_rows.append((frame, box, conf))
            current_center, expected_wh = _box_center_wh(box)
        if len(forward_rows) >= min_run:
            rows.extend(forward_rows)
            for frame, box, _conf in forward_rows:
                occupied_by_frame.setdefault(frame, []).append((int(tid), box))
            added += len(forward_rows)

        out[tid] = _extend_track_with_rows(tr, rows)
    return out, added


def _maybe_detector_edge_backfill(
    tracks: Dict[int, Track],
    *,
    cache_path: Path,
) -> Dict[int, Track]:
    if not _env_bool("BEST_ID_DETECTOR_BACKFILL", False):
        return tracks
    detector_cache = os.environ.get("BEST_ID_DETECTIONS_CACHE", "").strip()
    detector_path = Path(detector_cache) if detector_cache else Path(cache_path).parent / "detections.pkl"
    if not detector_path.is_file():
        log.warning("BEST_ID_DETECTOR_BACKFILL=1 but detector cache is missing: %s", detector_path)
        return tracks
    detector_frames = joblib.load(str(detector_path))
    if not isinstance(detector_frames, list):
        log.warning("detector cache did not contain a frame list: %s", detector_path)
        return tracks
    out, added = _detector_edge_backfill(
        tracks,
        detector_frames,
        max_frames=max(0, _env_int("BEST_ID_BACKFILL_MAX_FRAMES", 60)),
        min_run=max(1, _env_int("BEST_ID_BACKFILL_MIN_RUN", 1)),
        min_conf=_env_float("BEST_ID_BACKFILL_MIN_CONF", 0.34),
        max_center_dist=_env_float("BEST_ID_BACKFILL_MAX_CENTER_DIST", 80.0),
        max_size_ratio=_env_float("BEST_ID_BACKFILL_MAX_SIZE_RATIO", 1.8),
        occupied_iou=_env_float("BEST_ID_BACKFILL_OCCUPIED_IOU", 0.35),
    )
    if added:
        log.info("detector edge backfill added %d boxes from %s", added, detector_path)
    return out


def filter_tracks_post_merge(
    tracks: Dict[int, Track],
    *,
    min_len: int = POST_MIN_LEN,
    min_conf: float = POST_MIN_CONF,
    min_p90_conf: float = POST_MIN_P90_CONF,
) -> Dict[int, Track]:
    """Post-merge AND-gate: keep a track only if all three hold.

    - ``len(frames) >= min_len``
    - ``mean(confs) >= min_conf``
    - ``percentile(confs, 90) >= min_p90_conf``
    """
    min_len = _env_int("BEST_ID_POST_MIN_LEN", int(min_len))
    min_conf = _env_float("BEST_ID_POST_MIN_CONF", float(min_conf))
    min_p90_conf = _env_float("BEST_ID_POST_MIN_P90_CONF", float(min_p90_conf))
    out: Dict[int, Track] = {}
    for tid, tr in tracks.items():
        confs = np.asarray(tr.confs)
        if len(tr.frames) < min_len:
            continue
        if float(confs.mean()) < min_conf:
            continue
        if float(np.percentile(confs, 90)) < min_p90_conf:
            continue
        out[tid] = tr
    return out


def size_smooth_cv_gated(
    tracks: Dict[int, Track],
    *,
    cv_thresh: float = 0.20,
    fallback_window: int = 21,
) -> Dict[int, Track]:
    """Per-track CV-gated size smoother. Centers are preserved.

    For each track compute ``CV(w) + CV(h)``. If <= ``cv_thresh`` the
    dancer's apparent size barely changes (no real depth motion), so
    replace per-frame ``(w, h)`` with the per-track median. Otherwise
    apply a ``fallback_window``-length median filter to ``(w, h)`` so
    real depth changes are kept but noise is dampened.
    """
    out: Dict[int, Track] = {}
    for tid, tr in tracks.items():
        bbox = np.asarray(tr.bboxes, dtype=np.float64)
        if len(bbox) == 0:
            out[tid] = tr
            continue
        cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
        cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
        w = bbox[:, 2] - bbox[:, 0]
        h = bbox[:, 3] - bbox[:, 1]

        cv_w = float(np.std(w) / max(1.0, float(np.mean(w))))
        cv_h = float(np.std(h) / max(1.0, float(np.mean(h))))

        if cv_w + cv_h <= cv_thresh:
            w_use = np.full_like(w, float(np.median(w)))
            h_use = np.full_like(h, float(np.median(h)))
        else:
            eff = min(int(fallback_window), len(w))
            if eff % 2 == 0:
                eff = max(1, eff - 1)
            if eff > 1:
                w_use = median_filter(w, size=eff, mode="nearest")
                h_use = median_filter(h, size=eff, mode="nearest")
            else:
                w_use, h_use = w, h

        new_bbox = np.stack(
            [cx - w_use / 2, cy - h_use / 2, cx + w_use / 2, cy + h_use / 2],
            axis=1,
        )
        out[tid] = type(tr)(
            track_id=tr.track_id,
            frames=np.asarray(tr.frames),
            bboxes=new_bbox,
            confs=np.asarray(tr.confs, dtype=np.float64),
            masks=getattr(tr, "masks", None),
            detected=getattr(tr, "detected", None),
        )
    return out


def smooth_centers_median(
    tracks: Dict[int, Track],
    *,
    window: int = 21,
) -> Dict[int, Track]:
    """Per-track median filter on bbox CENTERS. Sizes are preserved.

    1-D ``window``-length median filter on ``(cx, cy)`` per track with
    ``mode="nearest"`` so track edges aren't biased by zero-padding.
    Bbox is rebuilt from the smoothed centers and the unchanged
    ``(w, h)`` (which the size smoother already cleaned).
    """
    out: Dict[int, Track] = {}
    for tid, tr in tracks.items():
        bb = np.asarray(tr.bboxes, dtype=np.float64)
        if len(bb) == 0:
            out[tid] = tr
            continue
        cx = (bb[:, 0] + bb[:, 2]) / 2.0
        cy = (bb[:, 1] + bb[:, 3]) / 2.0
        w = bb[:, 2] - bb[:, 0]
        h = bb[:, 3] - bb[:, 1]

        eff = min(int(window), len(cx))
        if eff % 2 == 0:
            eff = max(1, eff - 1)
        if eff > 1:
            cx = median_filter(cx, size=eff, mode="nearest")
            cy = median_filter(cy, size=eff, mode="nearest")

        new_bb = np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            axis=1,
        )
        out[tid] = type(tr)(
            track_id=tr.track_id,
            frames=np.asarray(tr.frames),
            bboxes=new_bb,
            confs=np.asarray(tr.confs, dtype=np.float64),
            masks=getattr(tr, "masks", None),
            detected=getattr(tr, "detected", None),
        )
    return out


def _make_frame_loader_for_cache(cache_path: Path):
    """Resolve a frame_loader(int) -> ndarray for the video associated
    with ``cache_path``.

    Resolution order:
      1. Sidecar JSON ``<cache>.video.json`` (written by
         :func:`tracking.run_pipeline.run_pipeline_on_video`) with key
         ``"video"`` holding the absolute video path. This is the
         authoritative source -- every cache produced by the pipeline
         since v9 has one.
      2. Sibling guesses: any ``.mov`` / ``.mp4`` / ``.avi`` / ``.mkv``
         next to the cache (for legacy work/results layouts).
      3. No-op loader (returns ``None`` for every frame); the RTMW
         pose-merge gate then silently skips the lookup.
    """
    cache_path = Path(cache_path).resolve()
    cache_dir = cache_path.parent
    sidecar = cache_path.with_suffix(cache_path.suffix + ".video.json")
    video_path: Optional[Path] = None
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text())
            cand = Path(str(meta.get("video", ""))).expanduser()
            if cand.is_file():
                video_path = cand
        except (OSError, ValueError) as exc:
            log.debug("video sidecar %s unreadable: %s", sidecar, exc)
    if video_path is None:
        candidates = []
        for ext in (".mov", ".mp4", ".avi", ".mkv"):
            for stem in ("video", cache_dir.name):
                candidates.append(cache_dir / f"{stem}{ext}")
        candidates.extend(sorted(cache_dir.glob("*.mov")))
        candidates.extend(sorted(cache_dir.glob("*.mp4")))
        video_path = next((p for p in candidates if p.is_file()), None)
    if video_path is None:
        log.warning(
            "frame_loader: could not locate source video for %s "
            "(sidecar absent and no sibling match); RTMW pose-merge will "
            "silently skip cosine gating",
            cache_path,
        )
        return lambda _idx: None

    import cv2
    _state = {"cap": None, "video": video_path}

    def _loader(idx: int):
        cap = _state["cap"]
        if cap is None:
            cap = cv2.VideoCapture(str(_state["video"]))
            _state["cap"] = cap
            if not cap.isOpened():
                return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        return frame if ok else None

    return _loader


def build_tracks(
    cache_path: Path,
    cfg_path: Path,
    *,
    save_to: Optional[Path] = None,
    expected_total_performers: Optional[int] = None,
    presence_plan_path: Optional[Path] = None,
) -> Dict[int, Track]:
    """Build the production tracks from a FrameDetections cache.

    Args:
        cache_path: joblib pickle of ``list[FrameDetections]`` produced
            by ``tracking.run_pipeline._detect_and_track`` (multi-scale
            YOLO + DeepOcSort, detector ``conf=DET_CONF``).
        cfg_path: ``configs/best_pipeline.json`` (post-process knobs
            consumed by ``postprocess_tracks``).
        save_to: optional path to joblib-dump the resulting
            ``dict[int -> Track]`` to.
        expected_total_performers: optional user-provided total unique target
            performer count for the clip.  This is a global identity-count
            prior, not a per-frame visible-count cap.
        presence_plan_path: optional JSON artifact describing final target
            identity presence and any expected-count decisions.

    Returns:
        ``dict[int -> Track]`` -- the final per-track output.
    """
    cfg = json.loads(Path(cfg_path).read_text())["best_pipeline_cfg"]
    fd = joblib.load(str(cache_path))
    raw = frame_detections_to_raw_tracks(fd)

    # Optional RTMW pose-aware ID merge gate (env-gated; no-op when
    # disabled). When enabled, the AND-gate adds wholebody pose
    # similarity (body+hands+face) on top of the existing IoU/proximity
    # gate, vetoing merges between visually-similar but biomechanically
    # distinct dancers. video_loader is required so postprocess can
    # decode the bbox crop frames.
    pose_extractor = None
    pose_cos_thresh = 0.0
    frame_loader = None
    try:
        from tracking import rtmw_pose_merge
        if rtmw_pose_merge.is_enabled():
            pose_extractor = rtmw_pose_merge.make_extractor()
            pose_cos_thresh = rtmw_pose_merge.get_pose_cos_thresh()
            frame_loader = _make_frame_loader_for_cache(cache_path)
    except ImportError:
        pass

    expected_count = (
        int(expected_total_performers)
        if expected_total_performers is not None and int(expected_total_performers) > 0
        else None
    )
    expected_reports: List[Dict[str, Any]] = []

    # Stage 1: prune + interpolate + ID merge (ReID-gated).
    stage1 = postprocess_tracks(
        raw,
        min_box_w=10, min_box_h=10,
        min_total_frames=_resolve_pre_min_total_frames(),
        min_conf=cfg["pp_min_conf"],
        max_gap_interp=cfg["pp_max_gap_interp"],
        id_merge_max_gap=_resolve_id_merge_max_gap(),
        id_merge_iou_thresh=_resolve_id_merge_iou_thresh(),
        id_merge_osnet_cos_thresh=cfg["pp_id_merge_osnet_cos_thresh"],
        medfilt_window=cfg["pp_medfilt_window"],
        gaussian_sigma=cfg["pp_gaussian_sigma"],
        num_max_people=cfg["pp_num_max_people"],
        overlap_merge_iou_thresh=cfg["pp_overlap_merge_iou_thresh"],
        overlap_merge_min_frames=cfg["pp_overlap_merge_min_frames"],
        edge_trim_conf_thresh=cfg["pp_edge_trim_conf_thresh"],
        edge_trim_max_frames=cfg["pp_edge_trim_max_frames"],
        pose_extractor=pose_extractor,
        pose_cos_thresh=pose_cos_thresh,
        pose_max_gap=cfg["pp_pose_max_gap"],
        pose_min_iou_for_pair=cfg["pp_pose_min_iou_for_pair"],
        pose_max_center_dist=_resolve_pose_max_center_dist(cfg["pp_pose_max_center_dist"]),
        frame_loader=frame_loader,
    )
    stage1_candidates = stage1
    if expected_count is not None and (
        _env_bool("BEST_ID_EXPECTED_RECALL_ALWAYS", True)
        or len(stage1_candidates) < expected_count
    ):
        recall_min_frames = max(1, _env_int("BEST_ID_EXPECTED_RECALL_MIN_FRAMES", 5))
        recall_min_conf = _env_float(
            "BEST_ID_EXPECTED_RECALL_MIN_CONF",
            max(0.20, float(cfg["pp_min_conf"]) - 0.08),
        )
        log.info(
            "expected-count recall pool: stage1 has %d tracks for target=%d; rerunning relaxed pool "
            "(min_frames=%d min_conf=%.3f)",
            len(stage1_candidates),
            expected_count,
            recall_min_frames,
            recall_min_conf,
        )
        recall_stage1 = postprocess_tracks(
            raw,
            min_box_w=10, min_box_h=10,
            min_total_frames=recall_min_frames,
            min_conf=recall_min_conf,
            max_gap_interp=cfg["pp_max_gap_interp"],
            id_merge_max_gap=_resolve_id_merge_max_gap(),
            id_merge_iou_thresh=_resolve_id_merge_iou_thresh(),
            id_merge_osnet_cos_thresh=cfg["pp_id_merge_osnet_cos_thresh"],
            medfilt_window=cfg["pp_medfilt_window"],
            gaussian_sigma=cfg["pp_gaussian_sigma"],
            num_max_people=max(cfg["pp_num_max_people"], expected_count * 4),
            overlap_merge_iou_thresh=cfg["pp_overlap_merge_iou_thresh"],
            overlap_merge_min_frames=cfg["pp_overlap_merge_min_frames"],
            edge_trim_conf_thresh=cfg["pp_edge_trim_conf_thresh"],
            edge_trim_max_frames=cfg["pp_edge_trim_max_frames"],
            pose_extractor=pose_extractor,
            pose_cos_thresh=pose_cos_thresh,
            pose_max_gap=cfg["pp_pose_max_gap"],
            pose_min_iou_for_pair=cfg["pp_pose_min_iou_for_pair"],
            pose_max_center_dist=_resolve_pose_max_center_dist(cfg["pp_pose_max_center_dist"]),
            frame_loader=frame_loader,
        )
        if len(recall_stage1) > len(stage1_candidates):
            stage1_candidates = {**recall_stage1, **stage1}
            log.info(
                "expected-count recall pool expanded candidates to %d tracks",
                len(stage1_candidates),
            )

    # Stage 2: post-merge AND-gate (length / mean / p90).
    stage2 = filter_tracks_post_merge(stage1)

    # Stage 3: long-gap bbox continuity stitch.
    stage3, _ = bbox_continuity_stitch(stage2, **BBOX_STITCH_KWARGS)

    # Optional detector-backed edge backfill. This uses the raw detector cache
    # only to extend track starts/ends; it is disabled by default.
    stage3b = _maybe_detector_edge_backfill(stage3, cache_path=cache_path)
    stage3c = _maybe_stage_prior_repair(stage3b, num_frames=len(fd))
    stage3d = _maybe_edge_extrapolate_to_modal_count(stage3c, num_frames=len(fd))

    # Stage 4: per-track size smoother.
    stage4 = size_smooth_cv_gated(stage3d, **SIZE_SMOOTHER_KWARGS)

    # Stage 5: per-track center median.
    final = smooth_centers_median(stage4, **CENTER_SMOOTHER_KWARGS)
    final = _maybe_calibrate_track_boxes(final, expected_count=expected_count)
    final = _maybe_cap_frames_to_modal_count(final, num_frames=len(fd))

    if expected_count is not None:
        candidate_pool = size_smooth_cv_gated(stage1_candidates, **SIZE_SMOOTHER_KWARGS)
        candidate_pool = smooth_centers_median(candidate_pool, **CENTER_SMOOTHER_KWARGS)
        candidate_pool = _maybe_calibrate_track_boxes(candidate_pool, expected_count=expected_count)
        final, report = enforce_expected_count(
            final,
            expected_count=expected_count,
            candidate_tracks=candidate_pool,
            num_frames=len(fd),
            phase="final",
        )
        expected_reports.append(report)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, str(save_to))
    if presence_plan_path is not None:
        write_presence_plan(
            Path(presence_plan_path),
            final,
            expected_count=expected_count,
            num_frames=len(fd),
            selection_report={
                "expected_count_reports": expected_reports,
            },
        )

    return final


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True,
                    help="FrameDetections joblib pickle "
                         "(produced by tracking.run_pipeline).")
    ap.add_argument("--cfg", type=Path,
                    default=Path("configs/best_pipeline.json"),
                    help="best_pipeline.json path "
                         "(default: configs/best_pipeline.json).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional tracks.pkl save path.")
    args = ap.parse_args()

    tracks = build_tracks(args.cache, args.cfg, save_to=args.out)
    print(f"Built {len(tracks)} tracks")
    if args.out:
        print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
