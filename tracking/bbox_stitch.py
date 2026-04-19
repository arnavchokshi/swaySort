"""Loose bbox-continuity stitch for re-attaching dropped tracklets.

This module contains the single function ``bbox_continuity_stitch`` that
the v2-v8 best pipelines use as their long-gap track-stitching stage,
and the small private helper ``_merge_tracks`` that physically merges
track records once a stitch is accepted.

The function was extracted from a much larger ``biodance.py`` module
(the rest of which contained pose/SAM-based experimental tracker
post-processing that is not part of the shipped ID pipeline). All other
biodance code was deleted during the project deep-clean; only this
stitch survives because it is the load-bearing recovery mechanism for
dancers who walk completely off-frame and re-enter (e.g. the 4
re-entries on MotionTest, 1 late-clip stitch on loveTest).

Algorithm (per-pair):
  1. For each candidate (tail-track A, head-track B) where A's last
     frame is strictly before B's first and the gap is within
     ``max_gap_frames``:
     a. Estimate A's tail velocity from the last ``velocity_window``
        frames as the median per-frame center delta.
     b. Linearly extrapolate A's center to B's first frame; cap the
        extrapolated jump at ``velocity_extrapolate_cap_px`` so a
        glitchy velocity doesn't sail off.
     c. Compute size-match: ``max(w)/min(w)`` and ``max(h)/min(h)``
        between A's tail box and B's head box must both be
        ``<= max_size_ratio``.
     d. Accept iff ``predicted-vs-actual position diff <=
        max_position_jump_px`` AND size-match passes.
  2. Resolve conflicts greedy by smallest position diff (one tail per
     head, one head per tail).
  3. Union-find connected components, then merge each component with
     ``_merge_tracks``.

Defaults match v8 (``max_gap_frames=400, max_position_jump_px=2000.0,
max_size_ratio=4.0``); the function is intentionally permissive because
the OSNet ReID gate inside ``postprocess_tracks`` already handles the
short-gap, sub-second re-entries.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from tracking.postprocess import Track


__all__ = ["bbox_continuity_stitch"]


def bbox_continuity_stitch(
    tracks: Dict[int, Track],
    *,
    max_gap_frames: int = 10,
    max_position_jump_px: float = 80.0,
    max_size_ratio: float = 1.4,
    velocity_window: int = 5,
    velocity_extrapolate_cap_px: float = 200.0,
) -> Tuple[Dict[int, Track], List[dict]]:
    """Stitch consecutive non-overlapping tracks based on bbox continuity.

    See module docstring for the algorithm. Returns a new dict of merged
    tracks (smallest tid in each connected component is canonical) plus a
    log of every candidate stitch attempt with diagnostics.

    Args:
        tracks: ``{tid: Track}`` to stitch.
        max_gap_frames: maximum frame gap between A's last and B's first.
        max_position_jump_px: maximum predicted-vs-actual center distance.
        max_size_ratio: maximum bbox h/w ratio between tail and head.
        velocity_window: how many frames of A's tail to estimate velocity.
        velocity_extrapolate_cap_px: cap on extrapolated jump magnitude.

    Returns:
        ``(out_tracks, log)``. Log records each stitch attempt with
        diagnostics so it can be diffed across configs.
    """
    out: Dict[int, Track] = dict(tracks)
    log: List[dict] = []
    if len(tracks) <= 1:
        return out, log

    tids = sorted(tracks.keys())
    info: Dict[int, dict] = {}
    for tid in tids:
        t = tracks[tid]
        if len(t.frames) == 0:
            continue
        b = t.bboxes
        cx = (b[:, 0] + b[:, 2]) * 0.5
        cy = (b[:, 1] + b[:, 3]) * 0.5
        w = b[:, 2] - b[:, 0]
        h = b[:, 3] - b[:, 1]
        info[tid] = {
            "first_frame": int(t.frames[0]),
            "last_frame": int(t.frames[-1]),
            "first_center": np.array([cx[0], cy[0]], np.float32),
            "last_center": np.array([cx[-1], cy[-1]], np.float32),
            "first_w": float(w[0]),
            "first_h": float(h[0]),
            "last_w": float(w[-1]),
            "last_h": float(h[-1]),
            "tail_centers": np.stack([cx[-velocity_window:],
                                       cy[-velocity_window:]], axis=1),
            "tail_frames": t.frames[-velocity_window:],
        }

    candidates: List[Tuple[float, int, int, dict]] = []
    for ai in tids:
        if ai not in info:
            continue
        a = info[ai]
        for bi in tids:
            if bi == ai or bi not in info:
                continue
            b = info[bi]
            if b["first_frame"] <= a["last_frame"]:
                continue
            gap = b["first_frame"] - a["last_frame"] - 1
            if gap < 0 or gap > max_gap_frames:
                continue

            tc = a["tail_centers"]
            tf = a["tail_frames"]
            if len(tc) >= 2 and tf[-1] > tf[0]:
                deltas = np.diff(tc, axis=0)
                df = np.diff(tf).astype(np.float32)
                df = np.where(df > 0, df, 1.0)
                per_frame = deltas / df[:, None]
                vx = float(np.median(per_frame[:, 0]))
                vy = float(np.median(per_frame[:, 1]))
            else:
                vx = vy = 0.0

            steps = float(gap + 1)
            jump_x = vx * steps
            jump_y = vy * steps
            jump_norm = float(np.hypot(jump_x, jump_y))
            if jump_norm > velocity_extrapolate_cap_px:
                jump_x *= velocity_extrapolate_cap_px / jump_norm
                jump_y *= velocity_extrapolate_cap_px / jump_norm
            predicted = a["last_center"] + np.array(
                [jump_x, jump_y], np.float32,
            )
            pos_diff = float(np.linalg.norm(predicted - b["first_center"]))

            wr = max(a["last_w"], b["first_w"]) / max(
                min(a["last_w"], b["first_w"]), 1.0,
            )
            hr = max(a["last_h"], b["first_h"]) / max(
                min(a["last_h"], b["first_h"]), 1.0,
            )
            size_ok = wr <= max_size_ratio and hr <= max_size_ratio
            pos_ok = pos_diff <= max_position_jump_px

            rec = {
                "pair": (ai, bi),
                "gap": gap,
                "predicted": predicted.tolist(),
                "actual": b["first_center"].tolist(),
                "pos_diff": pos_diff,
                "size_w_ratio": wr,
                "size_h_ratio": hr,
                "size_ok": size_ok,
                "pos_ok": pos_ok,
                "stitched": False,
            }
            if pos_ok and size_ok:
                candidates.append((pos_diff, ai, bi, rec))
            else:
                log.append(rec)

    candidates.sort(key=lambda c: c[0])
    used_tail = set()
    used_head = set()
    accepted: List[Tuple[int, int, dict]] = []
    for pos_diff, ai, bi, rec in candidates:
        if ai in used_tail or bi in used_head:
            log.append({**rec, "stitched": False, "skipped": "conflict"})
            continue
        used_tail.add(ai)
        used_head.add(bi)
        rec["stitched"] = True
        accepted.append((ai, bi, rec))
        log.append(rec)

    if not accepted:
        return out, log

    parent: Dict[int, int] = {tid: tid for tid in tracks}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for ai, bi, _ in accepted:
        union(ai, bi)

    components: Dict[int, List[int]] = {}
    for tid in tracks:
        components.setdefault(find(tid), []).append(tid)

    new_out: Dict[int, Track] = {}
    for canonical, members in components.items():
        members = sorted(members)
        if len(members) == 1:
            new_out[canonical] = tracks[canonical]
            continue
        new_out[canonical] = _merge_tracks(
            [tracks[m] for m in members], canonical_tid=canonical,
        )
    return new_out, log


def _merge_tracks(parts: Sequence[Track], canonical_tid: int) -> Track:
    """Concatenate frames/bboxes/confs across multiple Track objects,
    sort by frame, dedupe (keeping highest-conf if duplicates)."""
    all_frames = np.concatenate([p.frames for p in parts])
    all_bboxes = np.concatenate([p.bboxes for p in parts], axis=0)
    all_confs = np.concatenate([p.confs for p in parts])
    detected_parts = []
    for p in parts:
        if p.detected is not None and len(p.detected) == len(p.frames):
            detected_parts.append(p.detected)
        else:
            detected_parts.append(np.ones(len(p.frames), dtype=bool))
    all_detected = np.concatenate(detected_parts)

    order = np.argsort(all_frames, kind="stable")
    fs = all_frames[order]
    bs = all_bboxes[order]
    cs = all_confs[order]
    ds = all_detected[order]

    if len(fs) > 1:
        keep = np.ones(len(fs), dtype=bool)
        i = 0
        while i < len(fs):
            j = i + 1
            while j < len(fs) and fs[j] == fs[i]:
                j += 1
            if j - i > 1:
                local = cs[i:j]
                keep_idx = i + int(np.argmax(local))
                for k in range(i, j):
                    if k != keep_idx:
                        keep[k] = False
            i = j
        fs = fs[keep]
        bs = bs[keep]
        cs = cs[keep]
        ds = ds[keep]

    return Track(
        track_id=canonical_tid,
        frames=fs.astype(np.int64),
        bboxes=bs.astype(np.float32),
        confs=cs.astype(np.float32),
        masks=None,
        detected=ds.astype(bool),
    )
