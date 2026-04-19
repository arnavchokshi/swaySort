"""Render bbox + track-id overlays on each test clip and open them."""
from __future__ import annotations

import sys
import time
from pathlib import Path
import colorsys

import cv2
import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


CLIPS = [
    ("BigTest",     Path("/Users/arnavchokshi/Desktop/BigTest/BigTest.mov")),
    ("mirrorTest",  Path("/Users/arnavchokshi/Desktop/mirrorTest/IMG_2946.MP4")),
    ("2pplTest",    Path("/Users/arnavchokshi/Desktop/2pplTest/2pplTest.mov")),
    ("adiTest",     Path("/Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov")),
    ("easyTest",    Path("/Users/arnavchokshi/Desktop/easyTest/IMG_2082.mov")),
    ("gymTest",     Path("/Users/arnavchokshi/Desktop/gymTest/IMG_8309.mov")),
    ("loveTest",    Path("/Users/arnavchokshi/Desktop/loveTest/IMG_9265.mov")),
    ("shorterTest", Path("/Users/arnavchokshi/Desktop/shorterTest/TestVideo.mov")),
    ("MotionTest",  Path("/Users/arnavchokshi/Desktop/MotionTest/IMG_4716.mov")),
]

OUT_ROOT = REPO / "work" / "results"


def color_for_id(tid: int) -> tuple[int, int, int]:
    h = (tid * 0.61803398875) % 1.0  # golden-ratio hue stepping
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def build_per_frame(tracks: dict) -> dict[int, list[tuple[int, np.ndarray, float]]]:
    """frame_idx -> list of (tid, xyxy, conf) entries."""
    per_frame: dict[int, list] = {}
    for tid, tr in tracks.items():
        for f, bb, c in zip(tr.frames, tr.bboxes, tr.confs):
            per_frame.setdefault(int(f), []).append((int(tid), bb, float(c)))
    return per_frame


def render_clip(name: str, video: Path, tracks_pkl: Path, out_mp4: Path) -> dict:
    tracks = joblib.load(str(tracks_pkl))
    per_frame = build_per_frame(tracks)

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))

    box_thick = max(2, int(round(min(w, h) / 360)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 900)
    text_thick = max(1, int(round(font_scale * 2)))

    t0 = time.time()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, bb, conf in per_frame.get(idx, []):
            x1, y1, x2, y2 = [int(round(v)) for v in bb]
            color = color_for_id(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)
            label = f"id {tid}"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thick)
            ty = max(0, y1 - 6)
            cv2.rectangle(
                frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1,
            )
            cv2.putText(
                frame, label, (x1 + 3, ty - 2), font, font_scale,
                (0, 0, 0), text_thick, cv2.LINE_AA,
            )

        cv2.putText(
            frame, f"{name}  frame {idx+1}/{n}  tracks {len(tracks)}",
            (10, 28), font, 0.7, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"{name}  frame {idx+1}/{n}  tracks {len(tracks)}",
            (10, 28), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
        )

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    return {
        "clip": name, "out": str(out_mp4),
        "frames": idx, "wall_seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    results = []
    for i, (name, video) in enumerate(CLIPS, 1):
        pkl = OUT_ROOT / name / "tracks.pkl"
        if not pkl.is_file() or not video.is_file():
            print(f"[{i}/{len(CLIPS)}] SKIP {name}: missing pkl or video", flush=True)
            continue
        out_mp4 = OUT_ROOT / name / f"{name}_overlay.mp4"
        print(f"[{i}/{len(CLIPS)}] rendering {name} -> {out_mp4}", flush=True)
        r = render_clip(name, video, pkl, out_mp4)
        results.append(r)
        print(f"   done {name}: {r['frames']} frames in {r['wall_seconds']}s",
              flush=True)
    print("\nALL OVERLAYS DONE.")
    for r in results:
        print(f"  {r['clip']:<13} -> {r['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
