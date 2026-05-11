"""Burn per-frame bboxes + IDs + confidence onto a video for visual QA.

Consumes a v9 ``tracks.pkl`` (joblib-pickled
``Dict[int, tracking.postprocess.Track]``) and the source video,
and writes an MP4 with bounding boxes + ID labels + per-frame conf
overlaid. The output is the canonical "did the IDing actually
work?" artefact a developer eyeballs after running the pipeline.

Each track gets a deterministic color (seeded on ``track_id``) so a
given person stays the same color across frames -- ID stability is
the property we care about visually.

Usage::

    python scripts/render_tracks_overlay.py \\
        --video /Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov \\
        --tracks-pkl work/adiTest/tracks.pkl \\
        --out work/adiTest/overlay.mp4

The output codec is ``mp4v`` (broadly supported by QuickTime /
VLC). Frames where a track has ``detected[t] == False`` (filled by
post-process interpolation) are drawn with a dashed outline so the
operator can tell real detections apart from interpolated ones.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tracking.postprocess import Track  # noqa: E402

log = logging.getLogger("render_tracks_overlay")


def _color_for_id(tid: int) -> Tuple[int, int, int]:
    """Deterministic, vivid BGR color per track id.

    ``np.random.default_rng(tid)`` keeps ID->color stable across runs
    so the same person is the same color across the test clips.
    Hard floor of 60 on each channel so we never get near-black.
    """
    rng = np.random.default_rng(int(tid) + 1)
    rgb = rng.integers(60, 256, size=3).tolist()
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash: int = 8,
) -> None:
    """Manual dashed rectangle (cv2 has no native dashed-line API)."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _index_tracks_by_frame(
    tracks: Dict[int, Track],
) -> Dict[int, List[Tuple[int, np.ndarray, float, bool]]]:
    """Pivot ``Dict[tid, Track]`` -> ``Dict[frame_idx, [(tid, bbox, conf, detected), ...]]``.

    O(total entries). Iterating the video once and looking up
    ``per_frame[i]`` for each frame is O(total entries + n_frames),
    much cheaper than scanning every track on every frame.
    """
    per_frame: Dict[int, List[Tuple[int, np.ndarray, float, bool]]] = {}
    for tid, t in tracks.items():
        frames = np.asarray(t.frames, dtype=np.int64)
        bboxes = np.asarray(t.bboxes, dtype=np.float32)
        confs = np.asarray(t.confs, dtype=np.float32)
        detected_arr = getattr(t, "detected", None)
        if detected_arr is None or len(detected_arr) != len(frames):
            detected_arr = np.ones(len(frames), dtype=bool)
        else:
            detected_arr = np.asarray(detected_arr, dtype=bool)
        for i in range(len(frames)):
            f = int(frames[i])
            per_frame.setdefault(f, []).append(
                (int(tid), bboxes[i], float(confs[i]), bool(detected_arr[i]))
            )
    return per_frame


def _overlay_one_frame(
    frame: np.ndarray,
    items: List[Tuple[int, np.ndarray, float, bool]],
    *,
    frame_idx: int,
) -> np.ndarray:
    """Draw bbox + 'id=N c=0.84' label per item; dashed if interp."""
    h, w = frame.shape[:2]
    out = frame.copy()
    for tid, bbox, conf, detected in items:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _color_for_id(tid)
        if detected:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        else:
            _draw_dashed_rect(out, (x1, y1), (x2, y2), color, thickness=2)

        label = f"id={tid} c={conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
        )
        ly = y1 - 4 if y1 - th - baseline - 4 >= 0 else y1 + th + 4
        bg_top = ly - th - baseline
        bg_bot = ly + baseline
        cv2.rectangle(
            out,
            (x1, max(0, bg_top - 2)),
            (min(w - 1, x1 + tw + 4), min(h - 1, bg_bot + 2)),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            out, label, (x1 + 2, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out, f"frame {frame_idx}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        out, f"frame {frame_idx}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def render_overlay(
    *,
    video: Path,
    tracks_pkl: Path,
    out: Path,
    max_frames: Optional[int] = None,
    fps_override: Optional[float] = None,
) -> int:
    """Burn the tracks.pkl overlay onto the video and write to ``out``.

    Returns the number of frames written. Raises if the source video
    can't be opened or the writer fails to initialize (most common
    cause: no ``mp4v`` codec, e.g. headless OpenCV without ffmpeg).
    """
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if not tracks_pkl.is_file():
        raise FileNotFoundError(f"tracks.pkl not found: {tracks_pkl}")

    payload = joblib.load(str(tracks_pkl))
    if not isinstance(payload, dict):
        raise ValueError(
            f"unexpected tracks.pkl payload: {type(payload).__name__} "
            f"(expected Dict[int, Track])"
        )
    tracks: Dict[int, Track] = payload
    log.info("loaded %d tracks from %s", len(tracks), tracks_pkl)
    per_frame = _index_tracks_by_frame(tracks)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for reading: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps_override) if fps_override else float(src_fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError(
            f"video reports invalid frame size {w}x{h} -- "
            f"bad container or unsupported codec?"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"could not open video writer at {out} "
            f"({w}x{h} @ {fps:.2f}fps, mp4v) -- "
            f"is ffmpeg / mp4v support installed?"
        )

    cap_limit = (
        max_frames if (max_frames is not None and max_frames > 0) else None
    )

    n_written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cap_limit is not None and frame_idx >= cap_limit:
            break
        items = per_frame.get(frame_idx, [])
        out_frame = _overlay_one_frame(frame, items, frame_idx=frame_idx)
        writer.write(out_frame)
        n_written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    log.info(
        "wrote %d frames to %s (%dx%d @ %.2ffps)",
        n_written, out, w, h, fps,
    )
    return n_written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True,
                   help="Source video (the same one the IDing pipeline ran on).")
    p.add_argument("--tracks-pkl", type=Path, required=True,
                   help="joblib-pickled Dict[int, Track] from the v9 pipeline.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output MP4 path. Parent dir is created if missing.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap the overlay at the first N frames "
                        "(default: render every frame in the video).")
    p.add_argument("--fps", type=float, default=None,
                   help="Override output FPS (default: copy source FPS).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    n = render_overlay(
        video=args.video,
        tracks_pkl=args.tracks_pkl,
        out=args.out,
        max_frames=args.max_frames,
        fps_override=args.fps,
    )
    print(f"wrote {n} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
