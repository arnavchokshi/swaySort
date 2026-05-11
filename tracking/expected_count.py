"""Expected-performer-count prior for Stage A track selection.

The user-provided count is the total number of target performers in the clip,
not the number visible in any particular frame.  This module therefore works at
the tracklet/identity level: it ranks real tracker proposals globally, keeps or
recovers identities until the expected total is reached, and records why each
identity survived or was suppressed.  It never fabricates a track without
detector/tracker evidence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


log = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _track_arrays(track: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray(getattr(track, "frames", []), dtype=np.int64)
    boxes = np.asarray(getattr(track, "bboxes", []), dtype=np.float32)
    confs = np.asarray(getattr(track, "confs", []), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        boxes = np.empty((0, 4), dtype=np.float32)
    if len(confs) != len(frames):
        confs = np.ones((len(frames),), dtype=np.float32)
    n = min(len(frames), len(boxes), len(confs))
    return frames[:n], boxes[:n, :4], confs[:n]


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
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


def _mean_overlap_iou(a: Any, b: Any) -> Tuple[int, float]:
    af, ab, _ac = _track_arrays(a)
    bf, bb, _bc = _track_arrays(b)
    if len(af) == 0 or len(bf) == 0:
        return 0, 0.0
    ai = {int(frame): idx for idx, frame in enumerate(af)}
    bi = {int(frame): idx for idx, frame in enumerate(bf)}
    common = sorted(set(ai) & set(bi))
    if not common:
        return 0, 0.0
    ious = [_box_iou(ab[ai[frame]], bb[bi[frame]]) for frame in common]
    return len(common), float(np.mean(ious)) if ious else 0.0


def _bbox_area_stats(boxes: np.ndarray) -> Tuple[float, float]:
    if len(boxes) == 0:
        return 0.0, 0.0
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    areas = widths * heights
    return float(np.median(areas)), float(np.percentile(areas, 90))


def _estimate_frame_width(tracks: Dict[int, Any]) -> float:
    max_x = 0.0
    for tr in tracks.values():
        _frames, boxes, _confs = _track_arrays(tr)
        if len(boxes):
            max_x = max(max_x, float(np.nanmax(boxes[:, 2])))
    return max_x


def _edge_fraction(boxes: np.ndarray, frame_w: float, margin: float = 8.0) -> float:
    if len(boxes) == 0:
        return 1.0
    if frame_w <= 0:
        touch = boxes[:, 0] <= margin
    else:
        touch = (boxes[:, 0] <= margin) | (boxes[:, 2] >= frame_w - margin)
    return float(np.mean(touch.astype(np.float32)))


def track_summary(track: Any, *, num_frames: Optional[int] = None, frame_w: float = 0.0) -> Dict[str, Any]:
    frames, boxes, confs = _track_arrays(track)
    if len(frames) == 0:
        return {
            "n_frames": 0,
            "first_frame": None,
            "last_frame": None,
            "span_frames": 0,
            "coverage_in_span": 0.0,
            "coverage_total": 0.0,
            "confidence_mean": 0.0,
            "confidence_p90": 0.0,
            "median_area": 0.0,
            "edge_fraction": 1.0,
        }
    first = int(np.min(frames))
    last = int(np.max(frames))
    span = max(1, last - first + 1)
    median_area, p90_area = _bbox_area_stats(boxes)
    return {
        "n_frames": int(len(frames)),
        "first_frame": first,
        "last_frame": last,
        "span_frames": int(span),
        "coverage_in_span": float(len(np.unique(frames)) / span),
        "coverage_total": float(len(np.unique(frames)) / max(1, int(num_frames or 0)))
        if num_frames
        else 0.0,
        "confidence_mean": float(np.mean(confs)) if len(confs) else 0.0,
        "confidence_p90": float(np.percentile(confs, 90)) if len(confs) else 0.0,
        "median_area": median_area,
        "p90_area": p90_area,
        "edge_fraction": _edge_fraction(boxes, frame_w),
    }


def _quality_score(summary: Dict[str, Any]) -> float:
    n_frames = float(summary["n_frames"])
    span = float(summary["span_frames"])
    mean_conf = float(summary["confidence_mean"])
    p90_conf = float(summary["confidence_p90"])
    coverage = float(summary["coverage_in_span"])
    edge = float(summary["edge_fraction"])
    area_term = min(40.0, np.sqrt(max(0.0, float(summary["median_area"]))) * 0.12)
    return (
        n_frames
        + 0.05 * span
        + 35.0 * mean_conf
        + 20.0 * p90_conf
        + 20.0 * coverage
        + area_term
        - 12.0 * edge
    )


def _clone_with_track_id(track: Any, tid: int) -> Any:
    frames, boxes, confs = _track_arrays(track)
    detected = getattr(track, "detected", None)
    if detected is not None:
        detected_arr = np.asarray(detected, dtype=bool)
        if len(detected_arr) != len(frames):
            detected_arr = np.ones((len(frames),), dtype=bool)
    else:
        detected_arr = np.ones((len(frames),), dtype=bool)
    try:
        return type(track)(
            track_id=int(tid),
            frames=frames.copy(),
            bboxes=boxes.copy(),
            confs=confs.copy(),
            masks=getattr(track, "masks", None),
            detected=detected_arr.copy(),
        )
    except TypeError:
        return track


def _center_wh(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(box, dtype=np.float64)
    center = np.asarray([(arr[0] + arr[2]) * 0.5, (arr[1] + arr[3]) * 0.5], dtype=np.float64)
    wh = np.asarray([max(1.0, arr[2] - arr[0]), max(1.0, arr[3] - arr[1])], dtype=np.float64)
    return center, wh


def _size_ratio(a_wh: np.ndarray, b_wh: np.ndarray) -> float:
    return float(
        max(
            a_wh[0] / max(1.0, b_wh[0]),
            b_wh[0] / max(1.0, a_wh[0]),
            a_wh[1] / max(1.0, b_wh[1]),
            b_wh[1] / max(1.0, a_wh[1]),
        )
    )


def _merge_tracks(a: Any, b: Any) -> Any:
    af, ab, ac = _track_arrays(a)
    bf, bb, bc = _track_arrays(b)
    ad = np.asarray(getattr(a, "detected", np.ones((len(af),), dtype=bool)), dtype=bool)
    bd = np.asarray(getattr(b, "detected", np.ones((len(bf),), dtype=bool)), dtype=bool)
    if len(ad) != len(af):
        ad = np.ones((len(af),), dtype=bool)
    if len(bd) != len(bf):
        bd = np.ones((len(bf),), dtype=bool)

    rows: Dict[int, Tuple[np.ndarray, float, bool]] = {}
    for frame, box, conf, detected in zip(af, ab, ac, ad):
        rows[int(frame)] = (np.asarray(box, dtype=np.float32), float(conf), bool(detected))
    for frame, box, conf, detected in zip(bf, bb, bc, bd):
        old = rows.get(int(frame))
        if old is None or float(conf) >= old[1]:
            rows[int(frame)] = (np.asarray(box, dtype=np.float32), float(conf), bool(detected))

    frames = np.asarray(sorted(rows), dtype=np.int64)
    boxes = np.stack([rows[int(frame)][0] for frame in frames]).astype(np.float32)
    confs = np.asarray([rows[int(frame)][1] for frame in frames], dtype=np.float32)
    detected = np.asarray([rows[int(frame)][2] for frame in frames], dtype=bool)
    tid = min(int(getattr(a, "track_id", 0)), int(getattr(b, "track_id", 0)))
    return type(a)(
        track_id=int(tid),
        frames=frames,
        bboxes=boxes,
        confs=confs,
        masks=None,
        detected=detected,
    )


def _slice_track(track: Any, mask: np.ndarray, *, tid: Optional[int] = None) -> Optional[Any]:
    frames, boxes, confs = _track_arrays(track)
    if len(frames) == 0:
        return None
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(frames) or not bool(np.any(mask)):
        return None

    detected = getattr(track, "detected", None)
    if detected is not None:
        detected_arr = np.asarray(detected, dtype=bool)
        if len(detected_arr) != len(frames):
            detected_arr = np.ones((len(frames),), dtype=bool)
    else:
        detected_arr = np.ones((len(frames),), dtype=bool)

    masks = getattr(track, "masks", None)
    sliced_masks = None
    if masks is not None:
        masks_arr = np.asarray(masks)
        if len(masks_arr) == len(frames):
            sliced_masks = masks_arr[mask].copy()

    return type(track)(
        track_id=int(tid if tid is not None else getattr(track, "track_id", 0)),
        frames=frames[mask].copy(),
        bboxes=boxes[mask].copy(),
        confs=confs[mask].copy(),
        masks=sliced_masks,
        detected=detected_arr[mask].copy(),
    )


def _is_generated_split_id(tid: int) -> bool:
    # tracking.postprocess._interpolate assigns later pieces as
    # base_track_id * 1000 + piece_idx.
    return int(tid) >= 1000 and int(tid) % 1000 != 0


def _initial_overlap_stats(
    carrier: Any,
    split: Any,
    *,
    window: int = 20,
) -> Optional[Dict[str, float]]:
    cf, cb, _cc = _track_arrays(carrier)
    sf, sb, _sc = _track_arrays(split)
    if len(cf) == 0 or len(sf) == 0:
        return None
    cidx = {int(frame): idx for idx, frame in enumerate(cf)}
    sidx = {int(frame): idx for idx, frame in enumerate(sf)}
    common = sorted(set(cidx) & set(sidx))
    if not common:
        return None
    common = common[: max(1, int(window))]
    ious: List[float] = []
    norm_dists: List[float] = []
    size_ratios: List[float] = []
    for frame in common:
        cbox = cb[cidx[frame]]
        sbox = sb[sidx[frame]]
        c_center, c_wh = _center_wh(cbox)
        s_center, s_wh = _center_wh(sbox)
        diag = float(np.linalg.norm((c_wh + s_wh) * 0.5))
        ious.append(_box_iou(cbox, sbox))
        norm_dists.append(float(np.linalg.norm(c_center - s_center)) / max(1.0, diag))
        size_ratios.append(_size_ratio(c_wh, s_wh))
    return {
        "overlap_frames": float(len(common)),
        "first_frame": float(common[0]),
        "last_frame": float(common[-1]),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_norm_center_dist": float(np.mean(norm_dists)) if norm_dists else 999.0,
        "mean_size_ratio": float(np.mean(size_ratios)) if size_ratios else 999.0,
    }


def _segment_containing_or_after(frames: np.ndarray, start_frame: int) -> Optional[Tuple[int, int]]:
    vals = sorted(int(v) for v in frames if int(v) >= int(start_frame))
    if not vals:
        return None
    seg_start = vals[0]
    prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
            continue
        return seg_start, prev
    return seg_start, prev


def _next_segment_start(frames: np.ndarray, after_frame: int) -> Optional[int]:
    vals = sorted(int(v) for v in frames if int(v) > int(after_frame))
    if not vals:
        return None
    return int(vals[0])


def _repair_generated_split_handoffs(
    tracks: Dict[int, Any],
    *,
    min_prefix_frames: int = 3,
    min_suffix_frames: int = 5,
) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
    """Split a carrier track when an interpolated high-id piece takes over.

    DeepOcSort can follow one person with a low id, then later
    ``postprocess._interpolate`` can emit a high id such as ``14001`` after a
    long internal gap.  When the low-id track also continues as a different
    nearby performer, a plain total-count clamp keeps both tracks and leaves an
    ID switch.  This repair turns that pattern into:

    * carrier prefix + generated split id => one target identity
    * carrier suffix after the handoff => a separate target identity
    """
    out = {int(k): v for k, v in tracks.items()}
    repairs: List[Dict[str, Any]] = []
    next_tid = max(out.keys(), default=0) + 1

    split_ids = sorted(
        (tid for tid in out if _is_generated_split_id(tid)),
        key=lambda tid: int(_track_arrays(out[tid])[0][0]) if len(_track_arrays(out[tid])[0]) else 10**9,
    )
    for split_tid in split_ids:
        split = out.get(int(split_tid))
        if split is None:
            continue
        sf, _sb, _sc = _track_arrays(split)
        if len(sf) == 0:
            continue
        split_start = int(sf[0])

        best: Optional[Tuple[float, int, Dict[str, float]]] = None
        for carrier_tid, carrier in list(out.items()):
            carrier_tid = int(carrier_tid)
            if carrier_tid == int(split_tid):
                continue
            cf, _cb, _cc = _track_arrays(carrier)
            if len(cf) == 0:
                continue
            if int(cf[0]) >= split_start:
                continue
            prefix_count = int(np.sum(cf < split_start))
            suffix_count = int(np.sum(cf >= split_start))
            if prefix_count < min_prefix_frames or suffix_count < min_suffix_frames:
                continue
            stats = _initial_overlap_stats(carrier, split)
            if stats is None or int(stats["overlap_frames"]) < 3:
                continue
            mean_iou = float(stats["mean_iou"])
            norm_dist = float(stats["mean_norm_center_dist"])
            size_ratio = float(stats["mean_size_ratio"])
            if size_ratio > 2.4:
                continue
            if mean_iou < 0.35:
                continue
            score = mean_iou - (0.20 * norm_dist) - (0.05 * max(0.0, size_ratio - 1.0))
            if best is None or score > best[0]:
                best = (float(score), carrier_tid, stats)

        if best is None:
            continue

        _score, carrier_tid, stats = best
        carrier = out.get(carrier_tid)
        split = out.get(int(split_tid))
        if carrier is None or split is None:
            continue
        cf, _cb, _cc = _track_arrays(carrier)

        prefix = _slice_track(carrier, cf < split_start, tid=carrier_tid)
        if prefix is None:
            continue

        suffix_start = split_start
        first_segment = _segment_containing_or_after(cf, split_start)
        # If the carrier and generated split overlap strongly for a short
        # segment, that segment is the duplicated continuation.  Keep the next
        # carrier segment as the other performer instead of preserving the
        # duplicate frames.
        if (
            first_segment is not None
            and float(stats["mean_iou"]) >= 0.35
            and int(first_segment[1] - first_segment[0] + 1) <= 30
        ):
            next_start = _next_segment_start(cf, int(first_segment[1]))
            suffix_start = int(next_start) if next_start is not None else int(first_segment[1]) + 1

        suffix = _slice_track(carrier, cf >= suffix_start, tid=next_tid)
        merged = _merge_tracks(prefix, split)
        merged_tid = min(int(carrier_tid), int(split_tid))
        merged = _clone_with_track_id(merged, merged_tid)

        del out[carrier_tid]
        if int(split_tid) in out:
            del out[int(split_tid)]
        out[int(merged_tid)] = merged

        suffix_tid: Optional[int] = None
        if suffix is not None and len(_track_arrays(suffix)[0]) >= min_suffix_frames:
            while int(next_tid) in out:
                next_tid += 1
            suffix_tid = int(next_tid)
            out[suffix_tid] = _clone_with_track_id(suffix, suffix_tid)
            next_tid += 1

        repairs.append(
            {
                "carrier_track_id": int(carrier_tid),
                "generated_split_track_id": int(split_tid),
                "merged_track_id": int(merged_tid),
                "suffix_track_id": suffix_tid,
                "split_start_frame": int(split_start),
                "suffix_start_frame": int(suffix_start) if suffix_tid is not None else None,
                "overlap_frames": int(stats["overlap_frames"]),
                "mean_initial_iou": float(stats["mean_iou"]),
                "mean_initial_norm_center_dist": float(stats["mean_norm_center_dist"]),
                "mean_initial_size_ratio": float(stats["mean_size_ratio"]),
            }
        )

    return {int(tid): out[tid] for tid in sorted(out)}, repairs


def _active_counts_by_frame(tracks: Dict[int, Any], num_frames: int) -> np.ndarray:
    counts = np.zeros((max(0, int(num_frames)),), dtype=np.int32)
    for tr in tracks.values():
        frames, _boxes, _confs = _track_arrays(tr)
        for frame in np.unique(frames.astype(np.int64)):
            if 0 <= int(frame) < len(counts):
                counts[int(frame)] += 1
    return counts


def _backfill_track_to_start(track: Any, start_frame: int = 0) -> Optional[Any]:
    frames, boxes, confs = _track_arrays(track)
    if len(frames) == 0:
        return None
    first = int(frames[0])
    if first <= int(start_frame):
        return track
    add_frames = np.arange(int(start_frame), first, dtype=np.int64)
    if len(add_frames) == 0:
        return track

    detected = getattr(track, "detected", None)
    if detected is not None:
        detected_arr = np.asarray(detected, dtype=bool)
        if len(detected_arr) != len(frames):
            detected_arr = np.ones((len(frames),), dtype=bool)
    else:
        detected_arr = np.ones((len(frames),), dtype=bool)

    add_boxes = np.repeat(boxes[:1], len(add_frames), axis=0).astype(np.float32)
    add_confs = np.repeat(confs[:1] * 0.75, len(add_frames)).astype(np.float32)
    masks = getattr(track, "masks", None)
    return type(track)(
        track_id=int(getattr(track, "track_id", 0)),
        frames=np.concatenate([add_frames, frames]).astype(np.int64),
        bboxes=np.concatenate([add_boxes, boxes]).astype(np.float32),
        confs=np.concatenate([add_confs, confs]).astype(np.float32),
        masks=masks,
        detected=np.concatenate([np.zeros((len(add_frames),), dtype=bool), detected_arr]),
    )


def _backfill_early_interior_starts(
    tracks: Dict[int, Any],
    *,
    expected_count: int,
    num_frames: Optional[int],
    frame_w: float,
    max_start_frame: int = 90,
    min_track_frames: int = 60,
) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
    if num_frames is None or int(num_frames) <= 0 or expected_count <= 0:
        return tracks, []
    out = {int(k): v for k, v in tracks.items()}
    repairs: List[Dict[str, Any]] = []
    margin = max(20.0, 0.02 * float(frame_w))

    for tid, tr in sorted(list(out.items()), key=lambda kv: track_summary(kv[1], num_frames=num_frames, frame_w=frame_w)["first_frame"] or 0):
        frames, boxes, _confs = _track_arrays(tr)
        if len(frames) == 0:
            continue
        if len(frames) < int(min_track_frames):
            continue
        first = int(frames[0])
        if first <= 0 or first > int(max_start_frame):
            continue
        first_box = boxes[0]
        if frame_w > 0 and (float(first_box[0]) <= margin or float(first_box[2]) >= frame_w - margin):
            continue
        counts = _active_counts_by_frame(out, int(num_frames))
        deficit = counts[:first] < int(expected_count)
        if len(deficit) == 0 or float(np.mean(deficit.astype(np.float32))) < 0.60:
            continue
        backfilled = _backfill_track_to_start(tr, 0)
        if backfilled is None:
            continue
        out[int(tid)] = backfilled
        repairs.append(
            {
                "track_id": int(tid),
                "original_first_frame": int(first),
                "backfilled_to_frame": 0,
                "reason": "early_interior_start_with_expected_count_deficit",
            }
        )

    return {int(tid): out[tid] for tid in sorted(out)}, repairs


def _continuity_merge_candidate(a: Any, b: Any) -> Optional[Tuple[float, Tuple[int, int]]]:
    af, ab, _ac = _track_arrays(a)
    bf, bb, _bc = _track_arrays(b)
    if len(af) == 0 or len(bf) == 0:
        return None

    a_id = int(getattr(a, "track_id", 0))
    b_id = int(getattr(b, "track_id", 0))
    if int(af[0]) <= int(bf[0]):
        first_id, second_id = a_id, b_id
        first_f, first_b = af, ab
        second_f, second_b = bf, bb
    else:
        first_id, second_id = b_id, a_id
        first_f, first_b = bf, bb
        second_f, second_b = af, ab

    overlap = len(set(int(v) for v in first_f) & set(int(v) for v in second_f))
    # This pass is for split identities, not two long simultaneous tracks.
    if overlap > 3:
        return None

    gap = int(second_f[0]) - int(first_f[-1])
    if gap < -3 or gap > 90:
        return None

    c1, wh1 = _center_wh(first_b[-1])
    c2, wh2 = _center_wh(second_b[0])
    dist = float(np.linalg.norm(c2 - c1))
    ratio = _size_ratio(wh1, wh2)
    diag = float(np.linalg.norm((wh1 + wh2) * 0.5))
    max_dist = max(90.0, 0.60 * diag)
    if dist > max_dist or ratio > 2.2:
        return None

    # Lower is better. Prefer tiny gaps/overlaps, close centers, similar size.
    score = (dist / max(1.0, max_dist)) + (0.02 * max(0, gap)) + (0.35 * (ratio - 1.0))
    return float(score), (first_id, second_id)


def _consolidate_split_tracklets(
    tracks: Dict[int, Any],
    *,
    target_count: int,
) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
    out = {int(k): v for k, v in tracks.items()}
    merges: List[Dict[str, Any]] = []
    while len(out) > target_count:
        best: Optional[Tuple[float, int, int]] = None
        tids = sorted(out)
        for i, tid_a in enumerate(tids):
            for tid_b in tids[i + 1:]:
                cand = _continuity_merge_candidate(out[tid_a], out[tid_b])
                if cand is None:
                    continue
                score, ordered = cand
                if best is None or score < best[0]:
                    best = (score, int(ordered[0]), int(ordered[1]))
        if best is None:
            break
        score, tid_a, tid_b = best
        merged = _merge_tracks(out[tid_a], out[tid_b])
        new_tid = int(getattr(merged, "track_id", min(tid_a, tid_b)))
        del out[tid_a]
        del out[tid_b]
        out[new_tid] = merged
        merges.append({
            "from_track_ids": [int(tid_a), int(tid_b)],
            "to_track_id": int(new_tid),
            "score": float(score),
        })
    return {int(tid): out[tid] for tid in sorted(out)}, merges


def _is_duplicate_candidate(track: Any, selected: Dict[int, Any]) -> bool:
    for existing in selected.values():
        overlap, mean_iou = _mean_overlap_iou(track, existing)
        if overlap >= 3 and mean_iou >= 0.65:
            return True
    return False


def _ordered_candidates(
    tracks: Dict[int, Any],
    *,
    num_frames: Optional[int],
    frame_w: float,
) -> List[Tuple[int, Any, Dict[str, Any], float]]:
    rows: List[Tuple[int, Any, Dict[str, Any], float]] = []
    for tid, tr in tracks.items():
        summary = track_summary(tr, num_frames=num_frames, frame_w=frame_w)
        if int(summary["n_frames"]) <= 0:
            continue
        rows.append((int(tid), tr, summary, _quality_score(summary)))
    rows.sort(key=lambda row: (-row[3], int(row[0])))
    return rows


def _compact_track_ids(tracks: Dict[int, Any]) -> Tuple[Dict[int, Any], Dict[int, int]]:
    sorted_tids = sorted(int(tid) for tid in tracks)
    expected = list(range(1, len(sorted_tids) + 1))
    if sorted_tids == expected:
        return ({int(tid): tracks[int(tid)] for tid in sorted_tids}, {})

    remap: Dict[int, int] = {}
    out: Dict[int, Any] = {}
    for new_tid, old_tid in enumerate(sorted_tids, start=1):
        remap[int(old_tid)] = int(new_tid)
        out[int(new_tid)] = _clone_with_track_id(tracks[int(old_tid)], int(new_tid))
    return out, remap


def enforce_expected_count(
    tracks: Dict[int, Any],
    *,
    expected_count: Optional[int],
    candidate_tracks: Optional[Dict[int, Any]] = None,
    num_frames: Optional[int] = None,
    phase: str = "final",
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """Select final target identities using a total-count prior.

    ``tracks`` is the current best set. ``candidate_tracks`` may include weaker
    tracklets from a high-recall earlier stage.  When the current set has too
    many identities, low-quality extras are suppressed.  When it has too few,
    the highest-quality non-duplicate candidates are recovered.
    """
    report: Dict[str, Any] = {
        "phase": phase,
        "expected_performer_count": int(expected_count) if expected_count else None,
        "input_track_count": int(len(tracks)),
        "candidate_track_count": int(len(candidate_tracks or {})),
        "output_track_count": int(len(tracks)),
        "suppressed_track_ids": [],
        "recovered_track_ids": [],
        "merged_tracklets": [],
        "split_handoff_repairs": [],
        "early_start_backfills": [],
        "track_id_remap": {},
        "reason_codes": [],
        "track_scores": {},
    }
    if expected_count is None or int(expected_count) <= 0:
        return tracks, report

    target = int(expected_count)
    all_for_width: Dict[int, Any] = {}
    all_for_width.update({int(k): v for k, v in (candidate_tracks or {}).items()})
    all_for_width.update({int(k): v for k, v in tracks.items()})
    frame_w = _estimate_frame_width(all_for_width)

    primary_rows = _ordered_candidates(tracks, num_frames=num_frames, frame_w=frame_w)
    report["track_scores"].update(
        {
            str(tid): {
                **summary,
                "score": float(score),
                "source": "primary",
            }
            for tid, _tr, summary, score in primary_rows
        }
    )

    selected: Dict[int, Any] = {tid: tr for tid, tr, _summary, _score in primary_rows}

    selected, early_backfills = _backfill_early_interior_starts(
        selected,
        expected_count=target,
        num_frames=num_frames,
        frame_w=frame_w,
    )
    if early_backfills:
        report["early_start_backfills"] = early_backfills
        report["reason_codes"].append("early_occlusion_backfill")

    selected, split_repairs = _repair_generated_split_handoffs(selected)
    if split_repairs:
        report["split_handoff_repairs"] = split_repairs
        report["reason_codes"].append("generated_split_handoff_repaired")

    if len(selected) > target:
        selected, merges = _consolidate_split_tracklets(selected, target_count=target)
        if merges:
            report["merged_tracklets"] = merges
            report["reason_codes"].append("tracklet_consolidation")

    if len(selected) > target:
        primary_rows = _ordered_candidates(selected, num_frames=num_frames, frame_w=frame_w)
        keep_ids = {tid for tid, _tr, _summary, _score in primary_rows[:target]}
        suppressed = [tid for tid in sorted(selected) if tid not in keep_ids]
        selected = {tid: tr for tid, tr in selected.items() if tid in keep_ids}
        report["suppressed_track_ids"] = suppressed
        if suppressed:
            report["reason_codes"].append("duplicate_suppression")
            report["reason_codes"].append("over_expected_count_suppressed")

    if len(selected) < target and candidate_tracks:
        already = set(selected)
        candidate_rows = _ordered_candidates(
            {int(k): v for k, v in candidate_tracks.items() if int(k) not in already},
            num_frames=num_frames,
            frame_w=frame_w,
        )
        deferred_duplicates: List[Tuple[int, Any, Dict[str, Any], float]] = []
        for tid, tr, summary, score in candidate_rows:
            report["track_scores"][str(tid)] = {
                **summary,
                "score": float(score),
                "source": "candidate",
            }
            if len(selected) >= target:
                break
            if _is_duplicate_candidate(tr, selected):
                deferred_duplicates.append((tid, tr, summary, score))
                continue
            selected[tid] = tr
            report["recovered_track_ids"].append(int(tid))

        for tid, tr, _summary, _score in deferred_duplicates:
            if len(selected) >= target:
                break
            selected[tid] = tr
            report["recovered_track_ids"].append(int(tid))
            report["reason_codes"].append("recovered_duplicate_like_candidate")

    if len(selected) <= target:
        selected, recovered_backfills = _backfill_early_interior_starts(
            selected,
            expected_count=target,
            num_frames=num_frames,
            frame_w=frame_w,
        )
        if recovered_backfills:
            report["early_start_backfills"].extend(recovered_backfills)
            if "early_occlusion_backfill" not in report["reason_codes"]:
                report["reason_codes"].append("early_occlusion_backfill")

    if report["recovered_track_ids"]:
        report["reason_codes"].append("recovered_track")
    if len(selected) == target:
        report["reason_codes"].append("expected_total_count_matched")
    else:
        report["reason_codes"].append("expected_total_count_unresolved")
        log.warning(
            "expected-count prior phase=%s could not reach target=%d from %d candidates (output=%d)",
            phase,
            target,
            len(candidate_tracks or {}),
            len(selected),
        )

    selected, remap = _compact_track_ids(selected)
    if remap:
        report["track_id_remap"] = {str(int(old)): int(new) for old, new in remap.items()}
        report["reason_codes"].append("dense_id_remap")

    # Sorting makes the pickle deterministic.
    out = {int(tid): selected[tid] for tid in sorted(selected)}
    report["output_track_count"] = int(len(out))
    return out, report


def _ranges(values: Iterable[int]) -> List[List[int]]:
    vals = sorted(set(int(v) for v in values))
    if not vals:
        return []
    out: List[List[int]] = []
    start = prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append([start, prev])
        start = prev = value
    out.append([start, prev])
    return out


def build_presence_plan(
    tracks: Dict[int, Any],
    *,
    expected_count: Optional[int],
    num_frames: Optional[int],
    selection_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    frame_w = _estimate_frame_width({int(k): v for k, v in tracks.items()})
    frame_ids: List[int]
    if num_frames is not None and int(num_frames) > 0:
        frame_ids = list(range(int(num_frames)))
    else:
        max_frame = -1
        for tr in tracks.values():
            frames, _boxes, _confs = _track_arrays(tr)
            if len(frames):
                max_frame = max(max_frame, int(np.max(frames)))
        frame_ids = list(range(max_frame + 1))

    active_by_frame: Dict[int, List[int]] = {frame: [] for frame in frame_ids}
    track_rows: List[Dict[str, Any]] = []
    for tid, tr in sorted(tracks.items(), key=lambda kv: int(kv[0])):
        frames, _boxes, _confs = _track_arrays(tr)
        summary = track_summary(tr, num_frames=len(frame_ids), frame_w=frame_w)
        for frame in sorted(set(int(f) for f in frames if int(f) in active_by_frame)):
            active_by_frame[frame].append(int(tid))
        first = summary["first_frame"]
        last = summary["last_frame"]
        track_rows.append(
            {
                "target_id": int(tid),
                "visible_ranges": _ranges(int(f) for f in frames),
                "first_frame": first,
                "last_frame": last,
                "n_visible_frames": int(summary["n_frames"]),
                "span_frames": int(summary["span_frames"]),
                "coverage_in_span": float(summary["coverage_in_span"]),
                "coverage_total": float(summary["coverage_total"]),
                "confidence_mean": float(summary["confidence_mean"]),
                "confidence_p90": float(summary["confidence_p90"]),
                "late_entry": bool(first is not None and int(first) > 0),
                "early_exit": bool(
                    last is not None and frame_ids and int(last) < int(frame_ids[-1])
                ),
            }
        )

    review_frames: List[Dict[str, Any]] = []
    expected = int(expected_count) if expected_count else None
    for frame in frame_ids:
        ids = sorted(active_by_frame.get(frame, []))
        reasons: List[str] = []
        if expected is not None and len(ids) > expected:
            reasons.append("active_count_exceeds_expected_total")
        if expected is not None and expected > 0 and len(ids) == 0:
            reasons.append("no_active_target")
        if reasons:
            review_frames.append(
                {
                    "frame": int(frame),
                    "predicted_active_count": int(len(ids)),
                    "predicted_active_target_ids": ids,
                    "reason_codes": reasons,
                }
            )

    return {
        "expected_performer_count": expected,
        "predicted_total_target_identities": int(len(tracks)),
        "total_count_match": bool(expected is not None and len(tracks) == expected),
        "per_frame_predicted_active_target_ids": [
            {
                "frame": int(frame),
                "target_ids": sorted(active_by_frame.get(frame, [])),
            }
            for frame in frame_ids
        ],
        "tracks": track_rows,
        "frames_requiring_review": review_frames,
        "selection_report": selection_report or {},
    }


def write_presence_plan(
    path: Path,
    tracks: Dict[int, Any],
    *,
    expected_count: Optional[int],
    num_frames: Optional[int],
    selection_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = build_presence_plan(
        tracks,
        expected_count=expected_count,
        num_frames=num_frames,
        selection_report=selection_report,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return payload
