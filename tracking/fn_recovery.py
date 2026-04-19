"""Cardinality-voting FN recovery via Kalman/linear extrapolation.

Implements the top-priority loveTest FN-recovery strategy from
``work/research/novel_strategies.md`` (§2.1) -- the only recovery path
that operates on the post-tracker FrameDetections cache without needing
a new model.

The single entry point is :func:`recover_missing_detections`. It walks
the cache, finds frames where the per-frame dancer count drops below
the running median (a "missing-dancer frame"), and for each track that
was active in the immediate past but absent at frame t it linearly
extrapolates the bbox from the track's last 3 observations and inserts
a synthetic detection iff:

  * The extrapolated bbox does not overlap any existing detection at t
    (IoU < ``iou_overlap_thresh``, default 0.30).
  * The track had at least ``min_history`` observations in [t-N, t-1]
    (default N=20, min_history=8), so the velocity estimate is reliable.

Synthetic detections are tagged with the ``Track``-id of the missing
track, given conf = ``synthetic_conf`` (default 0.36, just above the
detector gate at 0.34), and inserted into the cache in-place.

Env-var gating (env-unset == byte-identical pass-through):

  * ``BEST_ID_FN_RECOVERY``           - "1"/"true" enables the pass.
  * ``BEST_ID_FN_RECOVERY_DROP``      - cardinality drop threshold
                                        (frames with N(t) <= median - DROP
                                        are candidates). Default 1.
  * ``BEST_ID_FN_RECOVERY_WINDOW``    - running-median window in frames.
                                        Default 30.
  * ``BEST_ID_FN_RECOVERY_LOOKBACK``  - how many frames back to search
                                        for "recently active" tracks.
                                        Default 5.
  * ``BEST_ID_FN_RECOVERY_IOU``       - ``iou_overlap_thresh``. Default 0.3.
  * ``BEST_ID_FN_RECOVERY_CONF``      - synthetic conf. Default 0.36.
  * ``BEST_ID_FN_RECOVERY_MIN_HIST``  - minimum past observations for
                                        a track to be eligible. Default 8.
  * ``BEST_ID_FN_RECOVERY_MAX_DISP``  - maximum px displacement of the
                                        extrapolated bbox center from the
                                        last-known center (drops crazy
                                        extrapolations). Default 200.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("env %s=%r not int; using %d", key, raw, default)
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("env %s=%r not float; using %.4f", key, raw, default)
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    return _env_bool("BEST_ID_FN_RECOVERY", False)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    bb = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def _linear_extrapolate(
    frames: np.ndarray, boxes: np.ndarray, target_t: int,
) -> np.ndarray:
    """Linear regression on the last K frames to predict bbox at target_t.
    Falls back to last-known box on degenerate inputs.
    """
    if len(frames) < 2:
        return boxes[-1].astype(np.float32)
    # Use last 5 obs for regression; weighted toward most recent.
    K = min(5, len(frames))
    f = frames[-K:].astype(np.float64)
    b = boxes[-K:].astype(np.float64)
    pred = np.empty(4, dtype=np.float32)
    for c in range(4):
        try:
            slope, intercept = np.polyfit(f, b[:, c], 1)
        except (np.linalg.LinAlgError, ValueError):
            return boxes[-1].astype(np.float32)
        pred[c] = float(slope * target_t + intercept)
    return pred


def _build_recent_history(
    cache: List, lookback: int, min_history: int,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Map tid -> { t : bbox } for last `lookback` frames, with at least
    `min_history` observations across the cache. Returned dict only
    includes tids that are eligible for FN recovery somewhere.
    """
    per_tid_obs: Dict[int, Dict[int, np.ndarray]] = {}
    for t, fd in enumerate(cache):
        if not hasattr(fd, "tids") or len(fd.tids) == 0:
            continue
        for k in range(len(fd.tids)):
            tid = int(fd.tids[k])
            if tid <= 0:
                continue
            per_tid_obs.setdefault(tid, {})[t] = fd.xyxys[k].astype(np.float32)
    return {
        tid: hist for tid, hist in per_tid_obs.items()
        if len(hist) >= min_history
    }


def _present_in_window(
    obs: Dict[int, np.ndarray], t: int, lookback: int,
) -> Optional[np.ndarray]:
    """Return the (frames, boxes) pair from obs in [t-lookback, t-1].
    Returns None if there are fewer than 2 observations in the window.
    """
    keys = sorted(k for k in obs.keys() if (t - lookback) <= k < t)
    if len(keys) < 2:
        return None
    f = np.asarray(keys, dtype=np.int64)
    b = np.stack([obs[k] for k in keys], axis=0)
    return f, b


def recover_missing_detections(cache: List) -> int:
    """Mutate `cache` in-place by inserting synthetic detections for
    "missing-dancer frames". Returns the number of inserted detections.

    Caller is responsible for env-gating; if the gate is off this function
    is a no-op (returns 0). When called the function reads its tunables
    from env vars on every invocation.
    """
    if not is_enabled():
        return 0
    if not cache:
        return 0

    drop_thresh = _env_int("BEST_ID_FN_RECOVERY_DROP", 1)
    window = _env_int("BEST_ID_FN_RECOVERY_WINDOW", 30)
    lookback = _env_int("BEST_ID_FN_RECOVERY_LOOKBACK", 5)
    iou_thresh = _env_float("BEST_ID_FN_RECOVERY_IOU", 0.30)
    syn_conf = _env_float("BEST_ID_FN_RECOVERY_CONF", 0.36)
    min_history = _env_int("BEST_ID_FN_RECOVERY_MIN_HIST", 8)
    max_disp = _env_float("BEST_ID_FN_RECOVERY_MAX_DISP", 200.0)

    counts = np.array(
        [len(getattr(fd, "tids", [])) for fd in cache], dtype=np.float64,
    )
    n_frames = len(cache)
    half_w = max(1, window // 2)
    running_median = np.empty(n_frames, dtype=np.float64)
    for t in range(n_frames):
        a = max(0, t - half_w)
        b = min(n_frames, t + half_w + 1)
        running_median[t] = float(np.median(counts[a:b]))

    obs_by_tid = _build_recent_history(cache, lookback, min_history)
    if not obs_by_tid:
        log.info("fn-recovery: no eligible tracks (min_history=%d)",
                 min_history)
        return 0

    n_inserted = 0
    n_drop_frames = 0
    for t in range(n_frames):
        if counts[t] > running_median[t] - drop_thresh:
            continue
        n_drop_frames += 1
        fd = cache[t]
        present_tids = (
            {int(x) for x in fd.tids} if len(fd.tids) > 0 else set()
        )
        new_dets: List[tuple] = []
        for tid, hist in obs_by_tid.items():
            if tid in present_tids:
                continue
            past = _present_in_window(hist, t, lookback)
            if past is None:
                continue
            past_frames, past_boxes = past
            pred = _linear_extrapolate(past_frames, past_boxes, t)
            # Sanity bounds: positive width/height.
            if pred[2] - pred[0] <= 1 or pred[3] - pred[1] <= 1:
                continue
            # Displacement gate.
            last_cx = (past_boxes[-1, 0] + past_boxes[-1, 2]) * 0.5
            last_cy = (past_boxes[-1, 1] + past_boxes[-1, 3]) * 0.5
            pred_cx = (pred[0] + pred[2]) * 0.5
            pred_cy = (pred[1] + pred[3]) * 0.5
            disp = float(
                np.hypot(pred_cx - last_cx, pred_cy - last_cy)
            )
            if disp > max_disp:
                continue
            # IoU gate against all existing detections at t.
            overlap = False
            for k in range(len(fd.tids)):
                if _iou_xyxy(pred, fd.xyxys[k]) > iou_thresh:
                    overlap = True
                    break
            if overlap:
                continue
            new_dets.append((pred, syn_conf, float(tid)))

        if not new_dets:
            continue
        from prune_tracks import FrameDetections
        xyxys = np.concatenate([
            fd.xyxys,
            np.stack([d[0] for d in new_dets]).astype(np.float32),
        ])
        confs = np.concatenate([
            fd.confs,
            np.array([d[1] for d in new_dets], dtype=np.float32),
        ])
        tids = np.concatenate([
            fd.tids,
            np.array([d[2] for d in new_dets], dtype=np.float32),
        ])
        cache[t] = FrameDetections(xyxys, confs, tids)
        n_inserted += len(new_dets)

    log.info(
        "fn-recovery: inserted %d synthetic dets across %d drop-frames "
        "(eligible tids=%d, drop_thresh=%d, lookback=%d, iou<=%.2f, "
        "min_history=%d, max_disp=%.0fpx)",
        n_inserted, n_drop_frames, len(obs_by_tid),
        drop_thresh, lookback, iou_thresh, min_history, max_disp,
    )
    return n_inserted
