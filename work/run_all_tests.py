"""Run the v8 pipeline on every test clip and record per-clip timing.

Writes per-clip results to ``work/results/<clip>/tracks.pkl`` (and a
``.cache.pkl`` next to it), and dumps a single ``work/results/timings.json``
with end-to-end wall-clock + per-clip track counts so the canvas can
visualize the run.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

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
TIMINGS_PATH = OUT_ROOT / "timings.json"
LOG_PATH = OUT_ROOT / "run.log"

DEVICE = os.environ.get("PIPE_DEVICE", "mps")


def _video_meta(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"frames": n, "fps": fps, "width": w, "height": h}


def _save_timings(results: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TIMINGS_PATH.write_text(json.dumps({
        "device": DEVICE,
        "results": results,
    }, indent=2))


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    from tracking.run_pipeline import run_pipeline_on_video

    results: list[dict] = []
    overall_t0 = time.time()

    for i, (name, video) in enumerate(CLIPS, start=1):
        if not video.is_file():
            print(f"[{i}/{len(CLIPS)}] SKIP {name}: missing {video}", flush=True)
            results.append({
                "clip": name, "status": "missing_video",
                "video": str(video),
            })
            _save_timings(results)
            continue

        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pkl = out_dir / "tracks.pkl"

        meta = _video_meta(video)
        print(
            f"[{i}/{len(CLIPS)}] START {name}  frames={meta['frames']} "
            f"fps={meta['fps']:.1f} res={meta['width']}x{meta['height']} "
            f"video={video}",
            flush=True,
        )

        t0 = time.time()
        status = "ok"
        n_tracks = 0
        err = None
        try:
            tracks = run_pipeline_on_video(
                video=video,
                out=out_pkl,
                device=DEVICE,
                force=True,
            )
            n_tracks = len(tracks)
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            print(f"[{i}/{len(CLIPS)}] ERROR {name}: {err}", flush=True)
        wall = time.time() - t0

        rec = {
            "clip": name,
            "status": status,
            "video": str(video),
            "frames": meta["frames"],
            "fps_video": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "wall_seconds": round(wall, 3),
            "fps_pipeline": round(meta["frames"] / max(wall, 1e-6), 3),
            "num_tracks": n_tracks,
            "out_pkl": str(out_pkl),
        }
        if err:
            rec["error"] = err
        results.append(rec)

        print(
            f"[{i}/{len(CLIPS)}] DONE  {name}  status={status} "
            f"wall={wall:.1f}s tracks={n_tracks} "
            f"pipe_fps={rec['fps_pipeline']:.2f}",
            flush=True,
        )

        _save_timings(results)

    overall = time.time() - overall_t0
    print(f"\nALL DONE.  total wall = {overall:.1f}s "
          f"({overall/60:.1f} min)", flush=True)

    summary = {
        "device": DEVICE,
        "total_wall_seconds": round(overall, 3),
        "results": results,
    }
    TIMINGS_PATH.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
