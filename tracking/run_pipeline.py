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
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
    make_tracker,
)
from tracking.multi_scale_detector import make_multi_scale_detector
from tracking.postprocess import Track


log = logging.getLogger("tracking.run_pipeline")


# Production constants. Every value here was selected by sweeping
# across a 7-clip benchmark under a strict no-regression rule. Full
# spec in docs/PIPELINE_SPEC.md, sweep tables in docs/EXPERIMENTS_LOG.md.

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "best.pt"
DEFAULT_CFG = REPO_ROOT / "configs" / "best_pipeline.json"
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"

DETECTOR_IMGSZ_ENSEMBLE = (768, 1024)
DETECTOR_ENSEMBLE_IOU = 0.6
DETECTOR_CONF = float(DET_CONF)  # 0.34, the plateau centre
DETECTOR_IOU = 0.70
PERSON_CLASS_ID = 0


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
        imgsz_list=list(DETECTOR_IMGSZ_ENSEMBLE),
        conf=DETECTOR_CONF,
        iou=DETECTOR_IOU,
        device=device,
        ensemble_iou=DETECTOR_ENSEMBLE_IOU,
        classes=[PERSON_CLASS_ID],
        tta_flip=False,
    )
    tracker = make_tracker(
        reid_weights=reid_weights, device=device, half=False,
    )

    out_frames: List[FrameDetections] = []
    n_processed = 0
    t0 = time.time()
    for idx, frame_bgr in iter_video_frames(Path(video)):
        if max_frames is not None and idx >= max_frames:
            break

        dets = detect(frame_bgr)  # (N, 6) [x1, y1, x2, y2, conf, cls]
        if dets.size == 0:
            dets = np.zeros((0, 6), dtype=np.float32)

        try:
            tracks_per_frame = tracker.update(dets, frame_bgr)
        except Exception as exc:
            log.exception(
                "DeepOcSort.update failed at frame %d: %s", idx, exc,
            )
            tracks_per_frame = np.zeros((0, 7), dtype=np.float32)

        if tracks_per_frame is None or len(tracks_per_frame) == 0:
            out_frames.append(FrameDetections(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            ))
        else:
            tracks_per_frame = np.asarray(tracks_per_frame, dtype=np.float32)
            xyxys = tracks_per_frame[:, 0:4].astype(np.float32)
            tids = tracks_per_frame[:, 4].astype(np.float32)
            if tracks_per_frame.shape[1] > 5:
                confs = tracks_per_frame[:, 5].astype(np.float32)
            else:
                confs = np.ones(len(tracks_per_frame), dtype=np.float32)
            out_frames.append(FrameDetections(xyxys, confs, tids))
        n_processed += 1

    dt = time.time() - t0
    log.info(
        "%d frames in %.1fs (%.2f FPS)", n_processed, dt,
        n_processed / max(dt, 1e-6),
    )
    return out_frames


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
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(frames, str(cache_path))
        log.info("wrote cache: %s (%d frames)", cache_path, len(frames))

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
