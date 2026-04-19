"""Export the production YOLO weights to TensorRT engines (one per imgsz).

The shipped multi-scale ensemble runs at imgsz=768 AND imgsz=1024, so we
need TWO engines (TRT engines are baked to a fixed input shape unless
you use dynamic-shape mode, which gives back ~half the speedup).

Output layout (matches the loader in
``tracking.multi_scale_detector._resolve_trt_engines``)::

    <out-dir>/<weights_stem>_768.engine
    <out-dir>/<weights_stem>_1024.engine

Run on an NVIDIA box (TensorRT is CUDA-only). Build is slow (a few
minutes per engine) but only needs to happen once per (weights, imgsz,
GPU arch, TRT version) combo.

Usage::

    python scripts/export_yolo_trt.py \
        --weights weights/best.pt \
        --out-dir weights/ \
        --imgsz 768 1024 \
        --device cuda:0

After the engines are built, launch the pipeline with::

    BEST_ID_TRT_ENGINE_DIR=weights/ python -m tracking.run_pipeline \
        --video <your.mp4> --out work/<your>/tracks.pkl --device cuda:0

The detector will pick up ``best_768.engine`` and ``best_1024.engine``
automatically. Falls back to ``best.pt`` if either is missing, with a
loud warning.

Notes:
  * We export FP32 (``half=False``) by design. ``docs/EXPERIMENTS_LOG.md``
    records that FP16 was net-negative on A100 for our workload because
    the per-call kernel-launch overhead and accuracy guard rails
    cancelled out the throughput win. TRT FP32 is mathematically
    equivalent to PyTorch FP32 modulo cudnn algorithm choice.
  * INT8 quantisation is intentionally NOT enabled. It requires a
    representative calibration set and risks regressing IDF1 -- explicit
    Tier-C reject in ``canvases/pipeline-speedup-analysis.canvas.tsx``.
  * ``simplify=True`` runs onnxsim before TRT build; reduces engine
    size + speeds up the build. ``workspace`` is the TRT scratch space
    in GB (default 4 is fine on A100 40GB).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


log = logging.getLogger("export_yolo_trt")


def export_one(
    weights: Path, *, imgsz: int, out_dir: Path, device: str,
    workspace_gb: int, simplify: bool,
) -> Path:
    """Export a single TRT engine for one imgsz; return the engine path."""
    from ultralytics import YOLO

    target_name = f"{weights.stem}_{imgsz}.engine"
    target_path = out_dir / target_name

    log.info("exporting engine: weights=%s imgsz=%d device=%s -> %s",
             weights, imgsz, device, target_path)
    t0 = time.time()
    model = YOLO(str(weights))
    # Ultralytics writes the engine next to the .pt with name
    # "<stem>.engine", overwriting between exports. Move it after.
    produced = model.export(
        format="engine",
        imgsz=int(imgsz),
        half=False,           # FP32 -- see module docstring
        int8=False,           # INT8 explicitly OFF -- see module docstring
        dynamic=False,        # static shapes are ~1.5-2x faster than dynamic
        simplify=bool(simplify),
        workspace=int(workspace_gb),
        device=device,
        verbose=True,
    )
    produced_path = Path(produced)
    out_dir.mkdir(parents=True, exist_ok=True)
    if produced_path.resolve() != target_path.resolve():
        produced_path.replace(target_path)
    log.info("  built %s (%.1f MB) in %.1fs",
             target_path, target_path.stat().st_size / (1024 * 1024),
             time.time() - t0)
    return target_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", type=Path,
                   default=REPO / "weights" / "best.pt",
                   help="YOLO .pt weights to export (default weights/best.pt)")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "weights",
                   help="directory to drop the .engine files (default weights/)")
    p.add_argument("--imgsz", type=int, nargs="+", default=[768, 1024],
                   help="imgsz values to build engines for (default 768 1024)")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device for the build (default cuda:0)")
    p.add_argument("--workspace", type=int, default=4,
                   help="TRT workspace size in GB (default 4)")
    p.add_argument("--no-simplify", action="store_true",
                   help="disable onnxsim during the ONNX intermediate "
                        "(simplify is on by default)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.weights.is_file():
        log.error("weights not found: %s", args.weights)
        return 1

    try:
        import torch
        if not torch.cuda.is_available():
            log.error("CUDA not available; TensorRT export requires "
                      "an NVIDIA GPU. Run this on the A100 box.")
            return 2
    except ImportError:
        log.error("torch not installed in this env")
        return 2

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        log.error(
            "tensorrt python package missing. On the A100 box: "
            "`pip install tensorrt` (matched to the system TRT/CUDA version)."
        )
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    built: List[Path] = []
    for imgsz in args.imgsz:
        try:
            built.append(export_one(
                args.weights, imgsz=int(imgsz), out_dir=args.out_dir,
                device=args.device, workspace_gb=args.workspace,
                simplify=not args.no_simplify,
            ))
        except Exception:
            log.exception("export failed for imgsz=%s", imgsz)
            return 4

    log.info("done. built %d engines -> %s", len(built), args.out_dir)
    log.info("now run: BEST_ID_TRT_ENGINE_DIR=%s python -m "
             "tracking.run_pipeline --video <X.mp4> --out <Y> --device %s",
             args.out_dir, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
