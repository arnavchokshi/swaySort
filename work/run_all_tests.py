"""Run the v8 pipeline on every test clip and record per-clip timing.

Writes per-clip results to ``work/results/<clip>/tracks.pkl`` (and a
``.cache.pkl`` next to it), and dumps a single ``work/results/timings.json``
with end-to-end wall-clock + per-clip track counts so the canvas can
visualize the run.

Clip list is read from a manifest JSON (default ``configs/clips.json``,
fall back to ``configs/clips.example.json``). See
``configs/clips.example.json`` for the schema. Override with
``--clips-manifest path/to/your.json`` and the device with
``--device cuda:0`` (or env ``PIPE_DEVICE``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_MANIFEST = REPO / "configs" / "clips.json"
EXAMPLE_MANIFEST = REPO / "configs" / "clips.example.json"


def _load_manifest(path: Path) -> List[Tuple[str, Path]]:
    """Load ``[(clip_name, video_path), ...]`` from a clip manifest JSON."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Clip manifest not found: {path}. Copy "
            f"{EXAMPLE_MANIFEST} -> configs/clips.json and edit it, or "
            f"pass --clips-manifest path/to/your.json."
        )
    data = json.loads(path.read_text())
    clips = data.get("clips", [])
    out: List[Tuple[str, Path]] = []
    for c in clips:
        name = c["name"]
        video = Path(os.path.expanduser(c["video"]))
        out.append((name, video))
    return out


OUT_ROOT = REPO / "work" / "results"
TIMINGS_PATH = OUT_ROOT / "timings.json"
LOG_PATH = OUT_ROOT / "run.log"


def _video_meta(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"frames": n, "fps": fps, "width": w, "height": h}


def _save_timings(results: list[dict], device: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TIMINGS_PATH.write_text(json.dumps({
        "device": device,
        "results": results,
    }, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips-manifest", type=Path, default=None,
                   help=f"Clip manifest JSON. Default: {DEFAULT_MANIFEST}, "
                        f"or {EXAMPLE_MANIFEST} if that's missing.")
    p.add_argument("--device", default=os.environ.get("PIPE_DEVICE", "mps"),
                   help="Torch device (cuda:0 / mps / cpu). Defaults to "
                        "$PIPE_DEVICE or 'mps'.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = args.device
    manifest = args.clips_manifest or (
        DEFAULT_MANIFEST if DEFAULT_MANIFEST.is_file() else EXAMPLE_MANIFEST
    )
    clips = _load_manifest(manifest)
    print(f"loaded {len(clips)} clips from {manifest}", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Default to the v9 dark-recovery profile (CLAHE + gamma=auto), which
    # won the dark_recovery_finalists sweep with +0.0055 mean IDF1 and
    # +0.0501 IDF1 on darkTest, and is byte-identical on every well-lit
    # clip thanks to luma-gating in tracking.dark_recovery. Users can
    # opt out via `BEST_ID_DARK_PROFILE=` (empty) or override individual
    # knobs via BEST_ID_DARK_GAMMA / BEST_ID_DARK_CLAHE.
    os.environ.setdefault("BEST_ID_DARK_PROFILE", "v9")
    print(
        f"BEST_ID_DARK_PROFILE={os.environ['BEST_ID_DARK_PROFILE']!r} "
        f"(dark-recovery preprocessing for low-light frames)",
        flush=True,
    )

    from tracking.run_pipeline import run_pipeline_on_video

    results: list[dict] = []
    overall_t0 = time.time()

    for i, (name, video) in enumerate(clips, start=1):
        if not video.is_file():
            print(f"[{i}/{len(clips)}] SKIP {name}: missing {video}", flush=True)
            results.append({
                "clip": name, "status": "missing_video",
                "video": str(video),
            })
            _save_timings(results, device)
            continue

        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pkl = out_dir / "tracks.pkl"

        meta = _video_meta(video)
        print(
            f"[{i}/{len(clips)}] START {name}  frames={meta['frames']} "
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
                device=device,
                force=True,
            )
            n_tracks = len(tracks)
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            print(f"[{i}/{len(clips)}] ERROR {name}: {err}", flush=True)
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
            f"[{i}/{len(clips)}] DONE  {name}  status={status} "
            f"wall={wall:.1f}s tracks={n_tracks} "
            f"pipe_fps={rec['fps_pipeline']:.2f}",
            flush=True,
        )

        _save_timings(results, device)

    overall = time.time() - overall_t0
    print(f"\nALL DONE.  total wall = {overall:.1f}s "
          f"({overall/60:.1f} min)", flush=True)

    summary = {
        "device": device,
        "total_wall_seconds": round(overall, 3),
        "results": results,
    }
    TIMINGS_PATH.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
