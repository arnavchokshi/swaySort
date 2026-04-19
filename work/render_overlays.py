"""Render bbox + track-id overlays on each clip listed in the manifest.

Reads the clip list from a manifest JSON (default ``configs/clips.json``,
fall back to ``configs/clips.example.json``); see
``configs/clips.example.json`` for the schema. Override with
``--clips-manifest path/to/your.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import colorsys

import cv2
import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_MANIFEST = REPO / "configs" / "clips.json"
EXAMPLE_MANIFEST = REPO / "configs" / "clips.example.json"
OUT_ROOT = REPO / "work" / "results"


def _load_manifest(path: Path) -> List[Tuple[str, Path]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Clip manifest not found: {path}. Copy "
            f"{EXAMPLE_MANIFEST} -> configs/clips.json and edit it, or "
            f"pass --clips-manifest path/to/your.json."
        )
    data = json.loads(path.read_text())
    return [(c["name"], Path(os.path.expanduser(c["video"])))
            for c in data.get("clips", [])]


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips-manifest", type=Path, default=None,
                   help=f"Clip manifest JSON. Default: {DEFAULT_MANIFEST}, "
                        f"or {EXAMPLE_MANIFEST} if that's missing.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = args.clips_manifest or (
        DEFAULT_MANIFEST if DEFAULT_MANIFEST.is_file() else EXAMPLE_MANIFEST
    )
    clips = _load_manifest(manifest)
    print(f"loaded {len(clips)} clips from {manifest}", flush=True)

    results = []
    for i, (name, video) in enumerate(clips, 1):
        pkl = OUT_ROOT / name / "tracks.pkl"
        if not pkl.is_file() or not video.is_file():
            print(f"[{i}/{len(clips)}] SKIP {name}: missing pkl or video",
                  flush=True)
            continue
        out_mp4 = OUT_ROOT / name / f"{name}_overlay.mp4"
        print(f"[{i}/{len(clips)}] rendering {name} -> {out_mp4}", flush=True)
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
