"""
Hard post-track pruning: drop low-confidence boxes and track IDs that appear
in too few frames. Optional per-frame cap (e.g. 2 for two-person clips).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FrameDetections:
    xyxys: np.ndarray  # (N, 4) float
    confs: np.ndarray  # (N,)
    tids: np.ndarray  # (N,)


def count_tid_occurrences(frames: list[FrameDetections]) -> dict[int, int]:
    out: dict[int, int] = {}
    for fd in frames:
        if len(fd.tids) == 0:
            continue
        for tid in np.asarray(fd.tids).astype(int).tolist():
            out[tid] = out.get(tid, 0) + 1
    return out


def prune_detections(
    frames: list[FrameDetections],
    *,
    min_total_frames: int,
    min_conf: float,
    max_tracks_per_frame: Optional[int] = None,
) -> list[FrameDetections]:
    """
    - Remove boxes with conf < min_conf.
    - Remove any track ID whose total number of frame-occurrences < min_total_frames.
    - If max_tracks_per_frame is set, keep only the highest-confidence boxes up to that limit.
    """
    if min_total_frames < 1:
        raise ValueError("min_total_frames must be >= 1")
    counts = count_tid_occurrences(frames)
    valid_tids = {tid for tid, c in counts.items() if c >= min_total_frames}

    pruned: list[FrameDetections] = []
    for fd in frames:
        if len(fd.tids) == 0:
            pruned.append(
                FrameDetections(
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                )
            )
            continue
        tids = np.asarray(fd.tids)
        confs = np.asarray(fd.confs)
        xyxys = np.asarray(fd.xyxys)
        mask = np.array(
            [
                (int(tids[i]) in valid_tids) and (float(confs[i]) >= float(min_conf))
                for i in range(len(tids))
            ],
            dtype=bool,
        )
        xyxys = xyxys[mask]
        confs = confs[mask]
        tids = tids[mask]
        if max_tracks_per_frame is not None and len(tids) > int(max_tracks_per_frame):
            k = int(max_tracks_per_frame)
            order = np.argsort(-confs)[:k]
            xyxys = xyxys[order]
            confs = confs[order]
            tids = tids[order]
        pruned.append(FrameDetections(xyxys, confs, tids))
    return pruned


def max_boxes_per_frame(frames: list[FrameDetections]) -> int:
    return max((len(f.tids) for f in frames), default=0)
