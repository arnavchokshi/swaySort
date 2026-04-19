"""Tracker-agnostic pruning + straightening pass.

Applied uniformly to Trackers A, B, and C *before* SAM 2.1 mask propagation,
keypoint estimation, and PromptHMR. Mirrors the official PromptHMR pipeline's
post-track preprocessing (interpolation + medfilt(11) + Gaussian sigma=3) but
adds dancer-specific:
  - per-frame box w/h floor
  - confidence + length pruning (PRUNE_MIN_CONF / PRUNE_MIN_TOTAL_FRAMES)
  - tail-of-A to head-of-B ID merge with optional ReID gate

The output of `postprocess_tracks` is a `dict[track_id -> Track]` matching the
shape consumed by the downstream PromptHMR data path:
    {
        track_id: int,
        frames:   np.ndarray (T,)      int frame indices, contiguous
        bboxes:   np.ndarray (T, 4)    xyxy
        confs:    np.ndarray (T,)      float (interpolated for filled gaps)
        masks:    np.ndarray (T, H, W) bool (optional; padded with False for filled gaps)
        detected: np.ndarray (T,)      bool, True for real frames, False for interp
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


log = logging.getLogger(__name__)


@dataclass
class RawTrack:
    """Pre-pruning tracker output, one per `tracker_id`."""

    track_id: int
    frames: np.ndarray   # (n,) int, may be non-contiguous
    bboxes: np.ndarray   # (n, 4) float xyxy
    confs:  np.ndarray   # (n,)
    masks: Optional[np.ndarray] = None  # (n, H, W) bool
    embeds: Optional[np.ndarray] = None  # (n, D) ReID embedding (optional)


@dataclass
class Track:
    """Post-pruning track with straightened/interpolated frames."""

    track_id: int
    frames: np.ndarray
    bboxes: np.ndarray
    confs: np.ndarray
    masks: Optional[np.ndarray] = None
    detected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    b_area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def _filter_min_size(t: RawTrack, min_w: int, min_h: int) -> RawTrack:
    w = t.bboxes[:, 2] - t.bboxes[:, 0]
    h = t.bboxes[:, 3] - t.bboxes[:, 1]
    keep = (w > min_w) & (h > min_h)
    if keep.all():
        return t
    return RawTrack(
        track_id=t.track_id,
        frames=t.frames[keep],
        bboxes=t.bboxes[keep],
        confs=t.confs[keep],
        masks=(t.masks[keep] if t.masks is not None else None),
        embeds=(t.embeds[keep] if t.embeds is not None else None),
    )


def _interpolate(t: RawTrack, max_gap: int) -> List[RawTrack]:
    """Linear-interpolate gaps <= max_gap. Longer gaps split the track."""
    if len(t.frames) <= 1:
        return [t]

    order = np.argsort(t.frames)
    frames = t.frames[order]
    bboxes = t.bboxes[order]
    confs = t.confs[order]
    masks = t.masks[order] if t.masks is not None else None

    splits: List[List[int]] = [[0]]
    for i in range(1, len(frames)):
        gap = int(frames[i] - frames[i - 1])
        if gap > max_gap:
            splits.append([i])
        else:
            splits[-1].append(i)

    out: List[RawTrack] = []
    for piece_idx, idxs in enumerate(splits):
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        if len(idxs_arr) < 2:
            f = frames[idxs_arr]
            b = bboxes[idxs_arr]
            c = confs[idxs_arr]
            m = masks[idxs_arr] if masks is not None else None
            new_id = t.track_id if piece_idx == 0 else int(t.track_id * 1000 + piece_idx)
            out.append(RawTrack(new_id, f.astype(np.int64), b, c, m))
            continue

        sub_frames = frames[idxs_arr]
        sub_bboxes = bboxes[idxs_arr]
        sub_confs = confs[idxs_arr]
        sub_masks = masks[idxs_arr] if masks is not None else None

        all_frames = np.arange(int(sub_frames[0]), int(sub_frames[-1]) + 1, dtype=np.int64)
        bbox_interp = interp1d(sub_frames, sub_bboxes, axis=0, kind="linear")(all_frames)
        conf_interp = interp1d(sub_frames, sub_confs, axis=0, kind="linear")(all_frames)

        if sub_masks is not None:
            full_masks = np.zeros((len(all_frames), *sub_masks.shape[1:]), dtype=bool)
            mapping = sub_frames - all_frames[0]
            full_masks[mapping] = sub_masks
        else:
            full_masks = None

        new_id = t.track_id if piece_idx == 0 else int(t.track_id * 1000 + piece_idx)
        out.append(RawTrack(new_id, all_frames, bbox_interp.astype(np.float32),
                            conf_interp.astype(np.float32), full_masks))

    return out


def _smooth_boxes(boxes: np.ndarray, medfilt_window: int, sigma: float) -> np.ndarray:
    if len(boxes) < max(medfilt_window, 5):
        return boxes
    smoothed = np.empty_like(boxes)
    for c in range(boxes.shape[1]):
        s = medfilt(boxes[:, c], kernel_size=medfilt_window)
        s = gaussian_filter1d(s, sigma=sigma)
        smoothed[:, c] = s
    return smoothed


def _track_extrapolate_box(t: Track, frame_idx: int) -> np.ndarray:
    """Linear-extrapolate the last 3 frames forward, or first 3 backward."""
    if frame_idx >= int(t.frames[-1]):
        anchor = t.frames[-3:].astype(np.float64)
        boxes = t.bboxes[-3:].astype(np.float64)
    else:
        anchor = t.frames[:3].astype(np.float64)
        boxes = t.bboxes[:3].astype(np.float64)

    if len(anchor) < 2:
        return t.bboxes[-1] if frame_idx >= int(t.frames[-1]) else t.bboxes[0]

    coeffs = []
    for c in range(4):
        slope, intercept = np.polyfit(anchor, boxes[:, c], 1)
        coeffs.append((slope, intercept))
    return np.array([s * frame_idx + i for s, i in coeffs], dtype=np.float32)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def postprocess_tracks(
    raw: Sequence[RawTrack],
    *,
    min_box_w: int = 10,
    min_box_h: int = 10,
    min_total_frames: int = 10,
    min_conf: float = 0.38,
    max_gap_interp: int = 12,
    id_merge_max_gap: int = 8,
    id_merge_iou_thresh: float = 0.5,
    id_merge_osnet_cos_thresh: float = 0.7,
    medfilt_window: int = 11,
    gaussian_sigma: float = 3.0,
    num_max_people: int = 100,
    overlap_merge_iou_thresh: float = 0.7,
    overlap_merge_min_frames: int = 5,
    edge_trim_conf_thresh: float = 0.0,
    edge_trim_max_frames: int = 0,
    # Phase 1 pose-cosine ID merge (disabled by default).
    pose_extractor: Optional[object] = None,
    pose_cos_thresh: float = 0.0,
    pose_max_gap: int = 40,
    pose_min_iou_for_pair: float = 0.0,
    pose_max_center_dist: float = float("inf"),
    frame_loader: Optional[Callable[[int], np.ndarray]] = None,
) -> Dict[int, Track]:
    """Apply the canonical pruning/straightening to a list of raw tracks.

    The pipeline is:
      1) per-frame box-size floor
      2) per-track confidence + length pruning
      3) split on long gaps + interpolate short gaps
      4) sequential id-merge (tail-of-A → head-of-B, non-overlapping)
      5) overlapping id-merge (same-person id-swaps, e.g. partner-work
         occlusions; controlled by ``overlap_merge_iou_thresh`` /
         ``overlap_merge_min_frames``)
      6) box smoothing
      7) top-K by total frame coverage
    """

    # 1. per-frame box w/h floor
    cleaned: List[RawTrack] = [_filter_min_size(t, min_box_w, min_box_h) for t in raw]
    cleaned = [t for t in cleaned if len(t.frames) > 0]

    # 2. confidence + length pruning
    # NOTE: tracks with a negative track_id are *placeholders* synthesised by
    # `tracking_collect.py` for frames where YOLO produced detections but the
    # tracker hadn't assigned IDs yet (typical: BoT-SORT/ByteTrack frame 0
    # warmup). We deliberately exempt them from the length prune so the
    # subsequent _id_merge pass can splice them into a real track. They will
    # be re-pruned after the merge step (step 4b).
    pre = [t for t in cleaned
           if int(t.track_id) < 0
              or (len(t.frames) >= min_total_frames and float(np.mean(t.confs)) >= min_conf)]

    # 3. trim leading/trailing zeros (already enforced by filter), then split on big gaps
    interpolated: List[RawTrack] = []
    for t in pre:
        interpolated.extend(_interpolate(t, max_gap_interp))

    # length pruning again after split (placeholder negative-id tracks are
    # exempt; the subsequent _id_merge pass needs them so it can splice
    # warmup-frame detections into the corresponding real track)
    interpolated = [t for t in interpolated
                    if int(t.track_id) < 0 or len(t.frames) >= min_total_frames]

    # 4. ID-merge (greedy, non-overlapping tail→head)
    merged = _id_merge(
        interpolated,
        max_gap=id_merge_max_gap,
        iou_thresh=id_merge_iou_thresh,
        osnet_cos_thresh=id_merge_osnet_cos_thresh,
        pose_extractor=pose_extractor,
        pose_cos_thresh=pose_cos_thresh,
        pose_max_gap=pose_max_gap,
        pose_min_iou_for_pair=pose_min_iou_for_pair,
        pose_max_center_dist=pose_max_center_dist,
        frame_loader=frame_loader,
    )

    # 4b. Drop any placeholder track that didn't get spliced into a real
    # track during the merge (still has a negative id, still too short).
    # Anything with a real track_id >= 0 is kept regardless.
    merged = [
        t for t in merged
        if int(t.track_id) >= 0
           or (len(t.frames) >= min_total_frames and float(np.mean(t.confs)) >= min_conf)
    ]

    # 5. Overlapping-track ID-swap merge (same person, two ids)
    if overlap_merge_iou_thresh > 0:
        merged = _overlap_id_merge(
            merged,
            iou_thresh=overlap_merge_iou_thresh,
            min_overlap_frames=overlap_merge_min_frames,
        )

    # 5b. Low-confidence edge trim
    #
    # Many YOLO detections "warm up" at track entry and "cool down" at
    # track exit -- the first few frames where a dancer becomes visible
    # (or the last few before they leave) tend to have noticeably lower
    # confidence than the mid-section. The tracker keeps them anyway,
    # which causes a small per-frame count overshoot vs human GT (the
    # annotator typically marks entry/exit a few frames LATER/EARLIER
    # than the tracker does). Trim the leading and trailing samples
    # whose confidence is below ``edge_trim_conf_thresh`` -- but cap the
    # total trim by ``edge_trim_max_frames`` so we never delete the
    # majority of a track. Disabled when either knob is 0.
    if edge_trim_conf_thresh > 0 and edge_trim_max_frames > 0:
        merged = _trim_low_conf_edges(
            merged,
            conf_thresh=edge_trim_conf_thresh,
            max_trim=edge_trim_max_frames,
            min_total_frames=min_total_frames,
        )

    # 6. box smoothing
    out: Dict[int, Track] = {}
    for t in merged:
        smoothed = _smooth_boxes(t.bboxes, medfilt_window, gaussian_sigma)
        detected = np.ones(len(t.frames), dtype=bool)
        out[int(t.track_id)] = Track(
            track_id=int(t.track_id),
            frames=t.frames.astype(np.int64),
            bboxes=smoothed.astype(np.float32),
            confs=t.confs.astype(np.float32),
            masks=t.masks,
            detected=detected,
        )

    # 7. top-K by total frames
    if len(out) > num_max_people:
        ordered = sorted(out.items(), key=lambda kv: len(kv[1].frames), reverse=True)
        out = dict(ordered[:num_max_people])

    return out


# ---------------------------------------------------------------------------
# id merging
# ---------------------------------------------------------------------------


def _id_merge(
    tracks: List[RawTrack],
    *,
    max_gap: int,
    iou_thresh: float,
    osnet_cos_thresh: float,
    pose_extractor: Optional[object] = None,
    pose_cos_thresh: float = 0.0,
    pose_max_gap: int = 40,
    pose_min_iou_for_pair: float = 0.0,
    pose_max_center_dist: float = float("inf"),
    frame_loader: Optional[Callable[[int], np.ndarray]] = None,
) -> List[RawTrack]:
    """Greedily merge non-overlapping tracks A,B with small frame gaps + good extrap-IoU + (if available) ReID similarity.

    Two passes:

    1) **IoU + (optional) OSNet pass** — unchanged from production.
       Bridges short-gap (``gap <= max_gap``) splits where the
       extrapolated tail of A overlaps the head of B in pixel space.

    2) **Pose-cosine pass (Phase 1, NEW)** — runs only when
       ``pose_extractor is not None`` AND ``pose_cos_thresh > 0`` AND
       ``frame_loader is not None``. Targets long-gap splits the IoU
       pass cannot bridge: pairs with ``max_gap < gap <= pose_max_gap``.
       For each such candidate (A, B), extracts a 17-COCO-keypoint
       feature for A's last frame box and B's first frame box, then
       merges if the bbox-normalised cosine similarity is at least
       ``pose_cos_thresh``. Greedy by descending cosine, fail-closed
       (``cosine == -1.0`` means "not enough jointly-visible joints to
       decide", which never matches).
    """

    if not tracks:
        return tracks

    # Convert to a working list keyed by stable index
    work: List[RawTrack] = list(tracks)

    # ------------------------------------------------------------------
    # Pass 1: existing IoU + (optional) OSNet merge.
    # ------------------------------------------------------------------
    while True:
        merged_any = False
        # Sort by start frame, then end frame
        work.sort(key=lambda t: (int(t.frames[0]), int(t.frames[-1])))

        for i in range(len(work)):
            ti = work[i]
            best: Optional[Tuple[int, float]] = None
            for j in range(len(work)):
                if i == j:
                    continue
                tj = work[j]
                # tj must start strictly after ti ends
                gap = int(tj.frames[0]) - int(ti.frames[-1])
                if not (1 <= gap <= max_gap):
                    continue

                # Predict ti's box at tj's first frame; compare to tj's first box
                target_frame = int(tj.frames[0])
                fake_track = Track(
                    track_id=ti.track_id,
                    frames=ti.frames,
                    bboxes=ti.bboxes,
                    confs=ti.confs,
                    masks=None,
                    detected=np.ones(len(ti.frames), dtype=bool),
                )
                pred_box = _track_extrapolate_box(fake_track, target_frame)
                iou = _box_iou(pred_box, tj.bboxes[0])
                if iou < iou_thresh:
                    continue

                cos_sim = 1.0
                if ti.embeds is not None and tj.embeds is not None and len(ti.embeds) > 0 and len(tj.embeds) > 0:
                    a = ti.embeds[-1] / (np.linalg.norm(ti.embeds[-1]) + 1e-9)
                    b = tj.embeds[0] / (np.linalg.norm(tj.embeds[0]) + 1e-9)
                    cos_sim = float(np.dot(a, b))
                    if cos_sim < osnet_cos_thresh:
                        continue

                score = iou + 0.5 * cos_sim
                if best is None or score > best[1]:
                    best = (j, score)

            if best is not None:
                j, _ = best
                tj = work[j]
                merged = _concatenate_with_gap(ti, tj)
                # Replace i with merged, drop j
                work[i] = merged
                del work[j]
                merged_any = True
                break

        if not merged_any:
            break

    # ------------------------------------------------------------------
    # Pass 2: NEW pose-cosine pass for longer gaps.
    # ------------------------------------------------------------------
    # The pose pass triggers when ANY of its primary signals is enabled:
    #   - proximity gating (pose_max_center_dist < inf), OR
    #   - pose-cosine gating (pose_cos_thresh > 0 and pose_extractor + frame_loader given)
    # Both signals can be combined; both default to "off" (inf and 0.0
    # respectively) so unmodified callers see no behavioural change.
    proximity_enabled = pose_max_center_dist < float("inf")
    cosine_enabled = (
        pose_extractor is not None
        and pose_cos_thresh > 0
        and frame_loader is not None
    )
    if (proximity_enabled or cosine_enabled) and pose_max_gap > max_gap and len(work) > 1:
        work = _id_merge_pose_pass(
            work,
            max_gap=max_gap,
            pose_max_gap=pose_max_gap,
            pose_cos_thresh=pose_cos_thresh,
            pose_min_iou_for_pair=pose_min_iou_for_pair,
            pose_max_center_dist=pose_max_center_dist,
            pose_extractor=pose_extractor,
            frame_loader=frame_loader,
        )

    return work


def _id_merge_pose_pass(
    tracks: List[RawTrack],
    *,
    max_gap: int,
    pose_max_gap: int,
    pose_cos_thresh: float,
    pose_min_iou_for_pair: float,
    pose_max_center_dist: float,
    pose_extractor: Optional[object],
    frame_loader: Optional[Callable[[int], np.ndarray]],
) -> List[RawTrack]:
    """Phase 1 second-pass long-gap merge.

    Two complementary signals (either may be the primary, both can stack):

      * **Spatial proximity** — center-distance between A's last bbox and
        B's first bbox must be ``<= pose_max_center_dist`` (px). When
        ``pose_max_center_dist == inf`` this gate is disabled.
        BigTest evidence shows this is the *strongest* signal for
        post-occlusion ID resumption: same-dancer pairs sit at 25–35 px
        while different-dancer pairs sit at 100+ px in the synchronous
        choreography case.
      * **Pose cosine** — bbox-normalised 17-COCO-keypoint cosine, must
        be ``>= pose_cos_thresh`` when ``pose_cos_thresh > 0`` and a
        ``pose_extractor`` + ``frame_loader`` are provided. Pose alone
        was insufficient on BigTest (false positives ranked higher than
        true matches), so by default it is paired with proximity as a
        safety check.
      * **Optional cheap IoU pre-gate** via track extrapolation, kept
        for backward compatibility (``pose_min_iou_for_pair``).

    Ranking: smallest center-distance wins (then highest cosine as
    tie-break). When proximity is disabled, ranking falls back to
    highest cosine.
    """
    if len(tracks) < 2:
        return tracks
    work: List[RawTrack] = list(tracks)
    proximity_enabled = pose_max_center_dist < float("inf")
    cosine_enabled = (
        pose_extractor is not None
        and pose_cos_thresh > 0
        and frame_loader is not None
    )
    if not (proximity_enabled or cosine_enabled):
        return work

    # Cache extracted pose features (only used when cosine_enabled).
    feat_cache: Dict[Tuple[int, int, str], np.ndarray] = {}

    def _feat_for(t: RawTrack, which: str) -> Optional[np.ndarray]:
        frame_id = int(t.frames[-1] if which == "tail" else t.frames[0])
        bbox = t.bboxes[-1] if which == "tail" else t.bboxes[0]
        key = (int(t.track_id), frame_id, which)
        if key in feat_cache:
            return feat_cache[key]
        try:
            frame_bgr = frame_loader(frame_id)  # type: ignore[misc]
        except Exception as exc:
            log.debug("pose-merge: frame_loader(%d) raised %s", frame_id, exc)
            feat_cache[key] = None
            return None
        if frame_bgr is None:
            feat_cache[key] = None
            return None
        feat = pose_extractor.extract(frame_bgr, np.asarray(bbox, dtype=np.float32))  # type: ignore[union-attr]
        feat_cache[key] = feat
        return feat

    def _center(box: np.ndarray) -> Tuple[float, float]:
        return ((float(box[0]) + float(box[2])) * 0.5,
                (float(box[1]) + float(box[3])) * 0.5)

    # Repeat-merge until quiescent. Each merge can enable a new candidate
    # pair (e.g. A-B-C all on the same dancer split twice by occlusion).
    while True:
        work.sort(key=lambda t: (int(t.frames[0]), int(t.frames[-1])))
        # candidate tuple: (rank_key, dist, neg_cos, i, j)
        # rank_key is set such that smaller is better:
        #   - if proximity enabled: rank_key = (dist, -cos)
        #   - else:                 rank_key = (-cos, dist)
        candidates: List[Tuple[Tuple[float, float], int, int]] = []
        for i in range(len(work)):
            ti = work[i]
            for j in range(len(work)):
                if i == j:
                    continue
                tj = work[j]
                gap = int(tj.frames[0]) - int(ti.frames[-1])
                if not (max_gap < gap <= pose_max_gap):
                    continue

                # 1) Proximity gate (center distance between tail/head boxes).
                cx_a, cy_a = _center(ti.bboxes[-1])
                cx_b, cy_b = _center(tj.bboxes[0])
                dist = float(np.hypot(cx_a - cx_b, cy_a - cy_b))
                if proximity_enabled and dist > pose_max_center_dist:
                    continue

                # 2) Optional cheap IoU pre-gate (kept for BC).
                if pose_min_iou_for_pair > 0:
                    target_frame = int(tj.frames[0])
                    fake_track = Track(
                        track_id=ti.track_id,
                        frames=ti.frames,
                        bboxes=ti.bboxes,
                        confs=ti.confs,
                        masks=None,
                        detected=np.ones(len(ti.frames), dtype=bool),
                    )
                    pred_box = _track_extrapolate_box(fake_track, target_frame)
                    if _box_iou(pred_box, tj.bboxes[0]) < pose_min_iou_for_pair:
                        continue

                # 3) Pose cosine gate (only when caller provided one).
                cos = 0.0
                if cosine_enabled:
                    feat_a = _feat_for(ti, "tail")
                    feat_b = _feat_for(tj, "head")
                    cos = float(type(pose_extractor).cosine(feat_a, feat_b))  # type: ignore[union-attr]
                    if cos < pose_cos_thresh:
                        continue

                if proximity_enabled:
                    rank_key = (dist, -cos)
                else:
                    rank_key = (-cos, dist)
                candidates.append((rank_key, i, j))

        if not candidates:
            break

        candidates.sort(key=lambda c: c[0])
        rank_key, i, j = candidates[0]
        ti = work[i]; tj = work[j]
        # Recompute distance & cos for the log.
        cx_a, cy_a = _center(ti.bboxes[-1])
        cx_b, cy_b = _center(tj.bboxes[0])
        log_dist = float(np.hypot(cx_a - cx_b, cy_a - cy_b))
        log_cos: float
        if cosine_enabled:
            feat_a = _feat_for(ti, "tail")
            feat_b = _feat_for(tj, "head")
            log_cos = float(type(pose_extractor).cosine(feat_a, feat_b))  # type: ignore[union-attr]
        else:
            log_cos = float("nan")
        log.info("pose-merge: combining ids %d (frames %d-%d) + %d "
                 "(frames %d-%d) dist=%.1fpx cos=%.4f gap=%d",
                 int(ti.track_id), int(ti.frames[0]), int(ti.frames[-1]),
                 int(tj.track_id), int(tj.frames[0]), int(tj.frames[-1]),
                 log_dist, log_cos, int(tj.frames[0]) - int(ti.frames[-1]))
        merged = _concatenate_with_gap(ti, tj)
        work[i] = merged
        feat_cache.pop((int(ti.track_id), int(ti.frames[-1]), "tail"), None)
        feat_cache.pop((int(tj.track_id), int(tj.frames[0]), "head"), None)
        del work[j]

    return work


def _trim_low_conf_edges(
    tracks: List[RawTrack],
    *,
    conf_thresh: float,
    max_trim: int,
    min_total_frames: int,
) -> List[RawTrack]:
    """Trim leading and trailing samples whose confidence is below
    ``conf_thresh``, capped at ``max_trim`` per side.

    A track is dropped entirely if trimming pushes it below
    ``min_total_frames``. This is intentional: a track whose only
    high-confidence samples form a sub-``min_total_frames`` window is
    almost always noise."""
    out: List[RawTrack] = []
    for t in tracks:
        n = len(t.frames)
        if n == 0:
            continue
        # leading
        head = 0
        while head < min(max_trim, n) and float(t.confs[head]) < conf_thresh:
            head += 1
        # trailing
        tail = 0
        while tail < min(max_trim, n - head) and float(t.confs[n - 1 - tail]) < conf_thresh:
            tail += 1
        new_n = n - head - tail
        if new_n < min_total_frames:
            # don't keep a track that becomes too short -- it was almost
            # certainly noise to begin with
            continue
        if head == 0 and tail == 0:
            out.append(t)
            continue
        sl = slice(head, n - tail)
        out.append(RawTrack(
            track_id=int(t.track_id),
            frames=np.asarray(t.frames[sl], dtype=np.int64),
            bboxes=np.asarray(t.bboxes[sl], dtype=np.float32),
            confs=np.asarray(t.confs[sl], dtype=np.float32),
            masks=None if t.masks is None else t.masks[sl],
            embeds=None if t.embeds is None else t.embeds[sl],
        ))
    return out


def _overlap_id_merge(
    tracks: List[RawTrack],
    *,
    iou_thresh: float,
    min_overlap_frames: int,
) -> List[RawTrack]:
    """Merge two tracks that overlap in time and have HIGH spatial IoU.

    This catches the canonical "id-swap during partner work" failure mode
    where the tracker assigns a new id to the same person after a brief
    occlusion. Symptom in the postprocessed track set: track A and track
    B share several frames, both bounding boxes are essentially the same
    person, and B continues after A ends (or vice versa).

    Algorithm (greedy, repeats until no merges remain):
      * For every pair (i, j) compute the overlapping-frame IoU.
      * Pair (i, j) qualifies if ``len(overlap) >= min_overlap_frames``
        AND ``mean(IoU over overlap) >= iou_thresh``.
      * Pick the highest-mean-IoU qualifying pair, merge them, repeat.

    Merge rule:
      * For the union of frames, prefer the box from whichever track has
        higher confidence at that frame; for unique-to-one frames, take
        that track's box. The merged track's ``track_id`` becomes the
        smaller of (ti.track_id, tj.track_id).

    Conservatism: ``iou_thresh`` defaults to 0.7. Two real different
    dancers occasionally cross paths but their average IoU over a multi-
    frame window stays well below 0.5; setting the bar at 0.7 has yielded
    zero false-positive merges on easyTest / gymTest / adiTest /
    mirrorTest in our regression. Keep a feed of these in the eval log
    so any future regression is caught.
    """
    if len(tracks) < 2:
        return tracks
    work: List[RawTrack] = list(tracks)

    while True:
        best: Optional[Tuple[int, int, float]] = None
        for i in range(len(work)):
            ti = work[i]
            fi = {int(f): k for k, f in enumerate(ti.frames)}
            for j in range(i + 1, len(work)):
                tj = work[j]
                fj = {int(f): k for k, f in enumerate(tj.frames)}
                common = sorted(set(fi.keys()) & set(fj.keys()))
                if len(common) < min_overlap_frames:
                    continue
                ious: List[float] = []
                for f in common:
                    ai = fi[f]; bi = fj[f]
                    ious.append(_box_iou(ti.bboxes[ai], tj.bboxes[bi]))
                mean_iou = float(np.mean(ious))
                if mean_iou < iou_thresh:
                    continue
                if best is None or mean_iou > best[2]:
                    best = (i, j, mean_iou)

        if best is None:
            break

        i, j, _score = best
        ti = work[i]; tj = work[j]
        merged = _merge_overlapping(ti, tj)
        # Replace i with merged, drop j (j > i so the list stays consistent).
        work[i] = merged
        del work[j]
    return work


def _merge_overlapping(ti: RawTrack, tj: RawTrack) -> RawTrack:
    """Union the frames of ti and tj.

    For overlapping frames, keep the higher-confidence sample (this also
    avoids any risk of dragging a smoothed/interpolated box into a
    region where the other track had a real high-confidence detection).
    Keep the smaller of the two ids as the canonical merged id so the
    output is deterministic across runs.
    """
    fi = {int(f): k for k, f in enumerate(ti.frames)}
    fj = {int(f): k for k, f in enumerate(tj.frames)}
    all_frames = sorted(set(fi.keys()) | set(fj.keys()))
    boxes = np.empty((len(all_frames), 4), dtype=np.float32)
    confs = np.empty((len(all_frames),), dtype=np.float32)
    for k, f in enumerate(all_frames):
        in_i = f in fi; in_j = f in fj
        if in_i and in_j:
            ci = float(ti.confs[fi[f]]); cj = float(tj.confs[fj[f]])
            if ci >= cj:
                boxes[k] = ti.bboxes[fi[f]]; confs[k] = ci
            else:
                boxes[k] = tj.bboxes[fj[f]]; confs[k] = cj
        elif in_i:
            boxes[k] = ti.bboxes[fi[f]]; confs[k] = float(ti.confs[fi[f]])
        else:
            boxes[k] = tj.bboxes[fj[f]]; confs[k] = float(tj.confs[fj[f]])
    return RawTrack(
        track_id=int(min(ti.track_id, tj.track_id)),
        frames=np.asarray(all_frames, dtype=np.int64),
        bboxes=boxes,
        confs=confs,
        masks=None,
        embeds=None,
    )


def _concatenate_with_gap(ti: RawTrack, tj: RawTrack) -> RawTrack:
    """Linear-interpolate the inter-track gap, then concat."""
    gap_start = int(ti.frames[-1])
    gap_end = int(tj.frames[0])
    if gap_end - gap_start <= 1:
        new_frames = np.concatenate([ti.frames, tj.frames])
        new_boxes = np.concatenate([ti.bboxes, tj.bboxes], axis=0)
        new_confs = np.concatenate([ti.confs, tj.confs])
    else:
        gap_frames = np.arange(gap_start + 1, gap_end, dtype=np.int64)
        anchor = np.array([gap_start, gap_end], dtype=np.float64)
        boxes_anchor = np.stack([ti.bboxes[-1], tj.bboxes[0]])
        confs_anchor = np.array([ti.confs[-1], tj.confs[0]])
        bbox_interp = interp1d(anchor, boxes_anchor, axis=0, kind="linear")(gap_frames).astype(np.float32)
        conf_interp = interp1d(anchor, confs_anchor, axis=0, kind="linear")(gap_frames).astype(np.float32)
        new_frames = np.concatenate([ti.frames, gap_frames, tj.frames])
        new_boxes = np.concatenate([ti.bboxes, bbox_interp, tj.bboxes], axis=0)
        new_confs = np.concatenate([ti.confs, conf_interp, tj.confs])

    # Prefer a real (non-negative) track id over a placeholder negative id
    # so warmup-frame splices inherit the real tracker id.
    out_id = ti.track_id if int(ti.track_id) >= 0 else tj.track_id
    return RawTrack(
        track_id=out_id,
        frames=new_frames,
        bboxes=new_boxes,
        confs=new_confs,
        masks=None,
        embeds=None,
    )


# ---------------------------------------------------------------------------
# adapters from FrameDetections -> RawTrack and back
# ---------------------------------------------------------------------------


def frame_detections_to_raw_tracks(frames) -> List[RawTrack]:
    """Convert the legacy `prune_tracks.FrameDetections` list to RawTrack."""
    by_id: Dict[int, Dict[str, list]] = {}
    for frame_idx, fd in enumerate(frames):
        if len(fd.tids) == 0:
            continue
        for k in range(len(fd.tids)):
            tid = int(fd.tids[k])
            d = by_id.setdefault(tid, {"frames": [], "bboxes": [], "confs": []})
            d["frames"].append(frame_idx)
            d["bboxes"].append(fd.xyxys[k])
            d["confs"].append(float(fd.confs[k]))

    raw: List[RawTrack] = []
    for tid, d in by_id.items():
        raw.append(
            RawTrack(
                track_id=tid,
                frames=np.asarray(d["frames"], dtype=np.int64),
                bboxes=np.asarray(d["bboxes"], dtype=np.float32),
                confs=np.asarray(d["confs"], dtype=np.float32),
            )
        )
    return raw


def tracks_to_frame_detections(tracks: Dict[int, Track], num_frames: int):
    """Inverse of `frame_detections_to_raw_tracks` for MOT-style eval."""
    from prune_tracks import FrameDetections

    out = []
    for i in range(num_frames):
        rows: List[Tuple[np.ndarray, float, int]] = []
        for tid, t in tracks.items():
            mask = t.frames == i
            if not mask.any():
                continue
            box = t.bboxes[mask][0]
            conf = float(t.confs[mask][0])
            rows.append((box, conf, tid))
        if not rows:
            out.append(FrameDetections(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            ))
        else:
            xyxys = np.stack([r[0] for r in rows]).astype(np.float32)
            confs = np.asarray([r[1] for r in rows], dtype=np.float32)
            tids = np.asarray([r[2] for r in rows], dtype=np.float32)
            out.append(FrameDetections(xyxys, confs, tids))
    return out
