"""Verify a fresh clone is wired up correctly.

Loads every required dependency, instantiates the YOLO detector + the
DeepOcSort tracker, runs them on ~30 frames of synthetic noise, and
exercises the full post-process chain. Exits 0 on success.

Run after ``pip install -r requirements.txt``::

    python scripts/smoke_test.py --device mps      # Apple Silicon
    python scripts/smoke_test.py --device cuda:0   # NVIDIA
    python scripts/smoke_test.py --device cpu      # any platform

Uses ``weights/best.pt`` -- the dance-fine-tuned YOLO26s checkpoint
shipped with the repo. Will warn if the checkpoint is missing.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _make_synthetic_video(path: Path, n_frames: int = 30,
                          w: int = 640, h: int = 360) -> None:
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open synthetic video writer for {path}")
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _check(label: str, fn) -> None:
    print(f"[ ] {label} ... ", end="", flush=True)
    t0 = time.time()
    try:
        fn()
    except Exception as exc:
        print(f"FAIL ({time.time() - t0:.2f}s)\n     {type(exc).__name__}: {exc}")
        raise
    print(f"OK ({time.time() - t0:.2f}s)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu",
                   help="Torch device (cpu / mps / cuda:0). Default cpu.")
    args = p.parse_args(argv)

    _check("import torch + report device",
           lambda: __import__("torch"))
    import torch
    print(f"     torch {torch.__version__}, requested device={args.device}, "
          f"cuda_available={torch.cuda.is_available()}, "
          f"mps_available={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")

    _check("import ultralytics", lambda: __import__("ultralytics"))
    _check("import boxmot", lambda: __import__("boxmot"))
    _check("import cv2 + numpy + scipy + joblib",
           lambda: [__import__(m) for m in ("cv2", "numpy", "scipy", "joblib")])

    _check("import tracking.run_pipeline (full module graph)",
           lambda: __import__("tracking.run_pipeline", fromlist=["*"]))

    weights = REPO / "weights" / "best.pt"
    if not weights.is_file():
        print(f"\n!! weights/best.pt missing at {weights}")
        print("   The repo ships this file; re-clone or `git lfs pull` if "
              "you used a partial clone. Skipping pipeline run.")
        return 1

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        video = tmp / "smoke.mp4"
        out = tmp / "smoke_tracks.pkl"

        _check("write 30-frame synthetic video",
               lambda: _make_synthetic_video(video))

        from tracking.run_pipeline import run_pipeline_on_video

        def _run():
            run_pipeline_on_video(
                video=video, out=out, device=args.device,
                max_frames=30, force=True,
            )
        _check(f"run full v8 pipeline on synthetic video (device={args.device})",
               _run)

        _check("load tracks.pkl back",
               lambda: __import__("joblib").load(str(out)))

    print("\nAll checks passed -- the clone is wired up correctly.")
    print(f"Now run the production pipeline on your own video:")
    print(f"  python -m tracking.run_pipeline --video <your.mp4> "
          f"--out work/yours/tracks.pkl --device {args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
