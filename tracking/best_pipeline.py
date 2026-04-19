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
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from scipy.ndimage import median_filter

from tracking.bbox_stitch import bbox_continuity_stitch
from tracking.postprocess import (
    Track,
    frame_detections_to_raw_tracks,
    postprocess_tracks,
)


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

    # Stage 1: prune + interpolate + ID merge (ReID-gated).
    stage1 = postprocess_tracks(
        raw,
        min_box_w=10, min_box_h=10,
        min_total_frames=PRE_MIN_TOTAL_FRAMES,
        min_conf=cfg["pp_min_conf"],
        max_gap_interp=cfg["pp_max_gap_interp"],
        id_merge_max_gap=ID_MERGE_MAX_GAP,
        id_merge_iou_thresh=ID_MERGE_IOU_THRESH,
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
        pose_max_center_dist=cfg["pp_pose_max_center_dist"],
        frame_loader=frame_loader,
    )

    # Stage 2: post-merge AND-gate (length / mean / p90).
    stage2 = filter_tracks_post_merge(stage1)

    # Stage 3: long-gap bbox continuity stitch.
    stage3, _ = bbox_continuity_stitch(stage2, **BBOX_STITCH_KWARGS)

    # Stage 4: per-track size smoother.
    stage4 = size_smooth_cv_gated(stage3, **SIZE_SMOOTHER_KWARGS)

    # Stage 5: per-track center median.
    final = smooth_centers_median(stage4, **CENTER_SMOOTHER_KWARGS)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, str(save_to))

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
