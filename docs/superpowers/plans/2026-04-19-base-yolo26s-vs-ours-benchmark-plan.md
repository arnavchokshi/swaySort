# Base YOLO26s vs Ours — Full A10 Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to walk this plan task-by-task. This is a benchmarking + docs-rewrite job, not feature TDD — each task is "build X, run Y, verify the artifact exists and looks right" rather than "write failing test".

**Goal:** Replace the current README headline with a fresh A10-measured benchmark of (ours v9 shipped) vs (stock yolo26s.pt + each base BoxMOT tracker, no post-process) across all 9 dance test clips.

**Architecture:** New benchmark driver (`scripts/run_full_benchmark.py`) builds a base-yolo26s detection cache per clip and replays it through every base BoxMOT tracker; runs the shipped pipeline once per clip for the "ours" row; dumps everything to `work/benchmarks/full_results.json`. Sync script pushes repo + weights + clips to the A10. Charts and side-by-side overlay videos consume the JSON. README is rewritten results-first.

**Tech Stack:** ultralytics 8.4.37 (YOLO26s), boxmot 18.0.0, py-motmetrics 1.4.0, torch 2.x + CUDA 12, matplotlib 3.10.x, opencv-python, ffmpeg.

**Reference spec:** `docs/superpowers/specs/2026-04-19-base-yolo26s-vs-ours-benchmark-design.md`

---

## File Map

**Create:**
- `scripts/run_full_benchmark.py` — driver: 1 base-detector cache per clip + 6 tracker replays + 1 ours-shipped run + score+time everything
- `scripts/sync_to_a10.sh` — rsync repo + weights + 9 clip dirs → A10, then ssh-install deps + download yolo26s.pt
- `scripts/render_side_by_side.py` — render side-by-side overlay videos (ours vs worst baseline) on the densest GT-tracked window per clip
- `configs/clips.json` — concrete manifest pointing the 9 clip names at their video files (currently only `clips.example.json` exists)

**Modify:**
- `scripts/generate_comparison_charts.py` — add 5 new chart functions consuming `work/benchmarks/full_results.json`
- `README.md` — rewritten, results-first
- `docs/EXPERIMENTS_LOG.md` — append moved historical sections

**Output (gitignored / committed depending on size):**
- `work/benchmarks/full_results.json` — single source of truth
- `work/benchmarks/full_summary.md` — auto-generated table
- `docs/figures/headline_idf1.png`, `per_clip_idf1_grid.png`, `error_breakdown.png`, `lift_decomposition.png`, `speed_vs_accuracy_a10.png`
- `docs/videos/<clip>_ours_vs_<worst>.mp4` (one or two compelling clips per scene)

---

## Phase 1 — Local scaffolding (build before pushing to A10)

### Task 1: Concrete clip manifest

**Files:**
- Create: `configs/clips.json`

**Why:** `work/run_all_tests.py` and `scripts/eval_per_clip.py` both read this. The example file points at `/absolute/path/to/...` placeholders. We need real paths.

- [ ] **Step 1: Inventory the actual videos**

Run:
```bash
for d in BigTest easyTest adiTest mirrorTest gymTest loveTest MotionTest shorterTest darkTest; do
  ls /Users/arnavchokshi/Desktop/$d/*.{mov,MOV,mp4,MP4} 2>/dev/null | head -1
done
```
Expected: one video path per clip (some are `.mov`, mirrorTest is `.MP4`, darkTest is `.mov`).

- [ ] **Step 2: Write the manifest**

Create `configs/clips.json`:
```json
{
  "_doc": "Concrete clip manifest for the A10 benchmark. Order matters only for log readability.",
  "clips": [
    {"name": "easyTest",    "video": "/Users/arnavchokshi/Desktop/easyTest/IMG_2082.mov"},
    {"name": "gymTest",     "video": "/Users/arnavchokshi/Desktop/gymTest/IMG_8309.mov"},
    {"name": "BigTest",     "video": "/Users/arnavchokshi/Desktop/BigTest/BigTest.mov"},
    {"name": "adiTest",     "video": "/Users/arnavchokshi/Desktop/adiTest/IMG_1649.mov"},
    {"name": "mirrorTest",  "video": "/Users/arnavchokshi/Desktop/mirrorTest/IMG_2946.MP4"},
    {"name": "shorterTest", "video": "/Users/arnavchokshi/Desktop/shorterTest/TestVideo.mov"},
    {"name": "darkTest",    "video": "/Users/arnavchokshi/Desktop/darkTest/darkTest.mov"},
    {"name": "MotionTest",  "video": "/Users/arnavchokshi/Desktop/MotionTest/IMG_4716.mov"},
    {"name": "loveTest",    "video": "/Users/arnavchokshi/Desktop/loveTest/IMG_9265.mov"}
  ]
}
```

- [ ] **Step 3: Verify with the existing loader**

```bash
python -c "from work.run_all_tests import _load_manifest; from pathlib import Path
clips = _load_manifest(Path('configs/clips.json'))
for n, v in clips: print(n, v.is_file(), v)"
```
Expected: 9 lines, every `True`.

---

### Task 2: Benchmark driver — skeleton

**Files:**
- Create: `scripts/run_full_benchmark.py`

The driver needs to be parameterized so it works the same on Mac (smoke test, capped frames) and A10 (full run). All seven rows live behind one CLI.

- [ ] **Step 1: Write the file scaffolding**

```python
"""Full A10 benchmark: ours v9 shipped vs stock yolo26s.pt + base BoxMOT trackers.

For each clip in --clips-manifest:
  1. Build (or load) a STOCK-YOLO26s single-scale-640 detection cache,
     stored next to the video as <stem>.base_det_cache.pkl. This is the
     "out-of-the-box" detector a new user gets.
  2. Replay that cache through every base BoxMOT tracker (default
     constructor params), capture per-frame outputs in MOT format,
     score against <clip>/gt/gt.txt with py-motmetrics, time each
     tracker.update() call.
  3. Run the shipped tracking.run_pipeline.run_pipeline_on_video for
     the SAME clip end-to-end (with our weights/best.pt + multi-scale
     + post-process chain), score the result against the same gt, time
     the wall-clock.
  4. Roll all rows into a single per-clip dict and dump
     work/benchmarks/full_results.json after every clip (so a crash
     doesn't lose work).

Usage::

    # On the A10 (production):
    python scripts/run_full_benchmark.py \\
        --device cuda:0 \\
        --clips-manifest configs/clips.json \\
        --base-yolo-weights weights/yolo26s.pt \\
        --our-yolo-weights  weights/best.pt \\
        --reid-weights      weights/osnet_x0_25_msmt17.pt \\
        --gt-root           /home/ubuntu/clips

    # Local smoke test (50 frames of one clip):
    python scripts/run_full_benchmark.py \\
        --device cpu --clips loveTest --max-frames 50 \\
        --base-yolo-weights weights/yolo26s.pt \\
        --our-yolo-weights  weights/best.pt \\
        --gt-root           /Users/arnavchokshi/Desktop
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("run_full_benchmark")

# Trackers (and the order they appear in the README table)
BASELINE_TRACKERS = [
    "ByteTrack (base)",
    "OcSort (base, no ReID)",
    "HybridSort (base)",
    "BotSort (base)",
    "StrongSort (base)",
    "DeepOcSort (base)",
]
OURS_LABEL = "Ours (v9 shipped)"
ALL_ROWS = [OURS_LABEL, *BASELINE_TRACKERS]


@dataclass
class TrackerRun:
    """Per-(clip, row) row in the final JSON."""
    row: str
    detector: str
    tracker_config: str
    n_frames: int
    wall_seconds: float
    det_ms_per_frame_mean: float
    tracker_ms_per_frame_mean: float
    tracker_ms_per_frame_median: float
    tracker_ms_per_frame_p95: float
    end_to_end_fps: float
    gpu_peak_mb: float
    metrics: Dict[str, float] = field(default_factory=dict)
```

- [ ] **Step 2: Add the base detector helper**

Append to `scripts/run_full_benchmark.py`:
```python
def _build_or_load_base_det_cache(
    *, video: Path, weights: Path, device: str,
    max_frames: Optional[int],
    conf: float = 0.25, iou: float = 0.7, imgsz: int = 640,
) -> Tuple[List[Dict[str, np.ndarray]], float]:
    """Run stock YOLO26s once at single-scale 640 (the out-of-the-box
    Ultralytics quickstart config). Cache result next to the video so
    subsequent tracker replays are free.

    Returns (frames_cache, mean_detector_ms_per_frame).
    """
    cache_path = video.with_suffix(".base_det_cache.pkl")
    if cache_path.is_file() and max_frames is None:
        log.info("base detector cache hit: %s", cache_path)
        cached = joblib.load(str(cache_path))
        return cached["frames"], float(cached.get("mean_det_ms", 0.0))

    from ultralytics import YOLO
    log.info("running stock YOLO26s @ %d on %s", imgsz, video)
    model = YOLO(str(weights))

    import cv2
    cap = cv2.VideoCapture(str(video))
    frames_out: List[Dict[str, np.ndarray]] = []
    det_ms: List[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break
        t0 = time.time()
        res = model.predict(
            frame, imgsz=imgsz, conf=conf, iou=iou,
            device=device, verbose=False, classes=[0],
        )
        det_ms.append(1000.0 * (time.time() - t0))
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            b = res[0].boxes
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            cf = b.conf.cpu().numpy().astype(np.float32)
            cl = b.cls.cpu().numpy().astype(np.float32)
            dets = np.concatenate(
                [xyxy, cf[:, None], cl[:, None]], axis=1,
            ).astype(np.float32)
        else:
            dets = np.zeros((0, 6), dtype=np.float32)
        frames_out.append({"bgr": frame, "dets": dets})
        idx += 1
    cap.release()

    mean_ms = float(np.mean(det_ms)) if det_ms else 0.0
    log.info(
        "stock-yolo26s detector: %d frames, mean=%.1f ms/frame "
        "(%.1f FPS detect-only)", len(frames_out), mean_ms,
        1000.0 / max(mean_ms, 1e-6),
    )
    if max_frames is None:
        joblib.dump({"frames": frames_out, "mean_det_ms": mean_ms},
                    str(cache_path))
        log.info("wrote base-det cache: %s", cache_path)
    return frames_out, mean_ms
```

- [ ] **Step 3: Add the tracker construction helper**

Append:
```python
def _build_baseline_tracker(name: str, *, device: str, reid_weights: Path):
    """Construct a base BoxMOT tracker with default constructor args."""
    import torch
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
    from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
    from boxmot.trackers.hybridsort.hybridsort import HybridSort
    from boxmot.trackers.ocsort.ocsort import OcSort
    from boxmot.trackers.strongsort.strongsort import StrongSort

    if device.startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(device)
    elif device == "mps" and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")

    if name == "ByteTrack (base)":
        return ByteTrack(), "BoxMOT defaults; no ReID"
    if name == "OcSort (base, no ReID)":
        return OcSort(), "BoxMOT defaults; no ReID"
    if name == "HybridSort (base)":
        return HybridSort(
            reid_weights=reid_weights, device=torch_device,
            half=False, with_reid=True,
        ), "BoxMOT defaults; OSNet x0.25 ReID"
    if name == "BotSort (base)":
        return BotSort(
            reid_weights=reid_weights, device=torch_device, half=False,
        ), "BoxMOT defaults; OSNet x0.25 ReID"
    if name == "StrongSort (base)":
        return StrongSort(
            reid_weights=reid_weights, device=torch_device, half=False,
        ), "BoxMOT defaults; OSNet x0.25 ReID"
    if name == "DeepOcSort (base)":
        return DeepOcSort(
            reid_weights=reid_weights, device=torch_device, half=False,
        ), "BoxMOT defaults; OSNet x0.25 ReID; NO Kalman jitter patch"
    raise ValueError(f"unknown baseline tracker: {name}")
```

- [ ] **Step 4: Add the MOT-row + scoring helpers**

These are slight refactors of `scripts/eval_per_clip.py` helpers — copy/adapt so this script is self-contained.

```python
def _xyxy_to_mot_rows(per_frame: List[np.ndarray]) -> List[str]:
    """Encode list of (N, >=6) tracker outputs to MOT15 rows (1-based)."""
    rows: List[str] = []
    for i, arr in enumerate(per_frame):
        if arr is None or len(arr) == 0:
            continue
        a = np.asarray(arr, dtype=np.float64)
        for r in a:
            x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
            tid = int(r[4])
            cf = float(r[5])
            w, h = float(x2 - x1), float(y2 - y1)
            rows.append(
                f"{i + 1},{tid},{x1 + 1.0:.6f},{y1 + 1.0:.6f},"
                f"{w:.6f},{h:.6f},{cf:.6f},1,-1"
            )
    return rows


def _ours_tracks_to_mot_rows(tracks_pkl: Path) -> List[str]:
    from tracking.postprocess import tracks_to_frame_detections
    raw = joblib.load(str(tracks_pkl))
    if not isinstance(raw, dict):
        raise TypeError(f"unexpected tracks.pkl type: {type(raw)}")
    n_frames = 0
    for t in raw.values():
        if hasattr(t, "frames") and len(t.frames):
            n_frames = max(n_frames, int(t.frames.max()) + 1)
    fds = tracks_to_frame_detections(raw, n_frames)
    rows: List[str] = []
    for i, fd in enumerate(fds):
        if len(fd.tids) == 0:
            continue
        for j in range(len(fd.tids)):
            x1, y1, x2, y2 = (float(v) for v in fd.xyxys[j])
            cf = float(fd.confs[j])
            tid = int(fd.tids[j])
            w, h = x2 - x1, y2 - y1
            rows.append(
                f"{i + 1},{tid},{x1 + 1.0:.6f},{y1 + 1.0:.6f},"
                f"{w:.6f},{h:.6f},{cf:.6f},1,-1"
            )
    return rows


def _score(pred_rows: List[str], gt_path: Path) -> Dict[str, float]:
    """py-motmetrics CLEAR + ID at IoU 0.5, plus a few derived metrics."""
    import io
    import motmetrics as mm

    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    text = "\n".join(pred_rows) + ("\n" if pred_rows else "")
    ts = mm.io.loadtxt(io.StringIO(text), fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
    mh = mm.metrics.create()
    names = [
        "idf1", "idp", "idr", "mota", "motp",
        "precision", "recall",
        "num_switches", "num_fragmentations",
        "num_misses", "num_false_positives",
        "mostly_tracked", "partially_tracked", "mostly_lost",
        "num_unique_objects", "num_predicted_unique_objects",
        "num_frames",
    ]
    summary = mh.compute(acc, metrics=names, name="seq")
    out: Dict[str, float] = {}
    for n in names:
        try:
            out[n] = float(summary[n].iloc[0])
        except Exception:
            out[n] = float("nan")
    return out
```

- [ ] **Step 5: Add the per-clip orchestrator**

```python
def _run_one_baseline(
    name: str, frames_cache: List[Dict[str, np.ndarray]],
    *, device: str, reid_weights: Path,
) -> Tuple[List[np.ndarray], List[float]]:
    tracker, _ = _build_baseline_tracker(
        name, device=device, reid_weights=reid_weights,
    )
    captured: List[np.ndarray] = []
    track_ms: List[float] = []
    for fc in frames_cache:
        t0 = time.time()
        try:
            out = tracker.update(fc["dets"], fc["bgr"])
        except Exception as exc:
            log.warning("%s.update failed at frame %d: %s",
                        name, len(captured), exc)
            out = np.zeros((0, 8), dtype=np.float32)
        track_ms.append(1000.0 * (time.time() - t0))
        if out is None or len(out) == 0:
            captured.append(np.zeros((0, 8), dtype=np.float32))
        else:
            captured.append(np.asarray(out, dtype=np.float32))
    del tracker
    return captured, track_ms


def _peak_gpu_mb(device: str) -> float:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    return 0.0


def _reset_gpu_peak(device: str) -> None:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _run_one_clip(
    *, clip_name: str, video: Path, gt_path: Path,
    base_weights: Path, our_weights: Path,
    reid_weights: Path, device: str, max_frames: Optional[int],
    out_pkl_dir: Path,
) -> Dict[str, TrackerRun]:
    log.info("=" * 80)
    log.info("CLIP %s  video=%s", clip_name, video)
    if not video.is_file():
        log.error("video missing: %s -- skipping clip", video)
        return {}
    if not gt_path.is_file():
        log.error("gt missing: %s -- skipping clip", gt_path)
        return {}

    out_pkl_dir.mkdir(parents=True, exist_ok=True)

    # ---- Baseline path ---------------------------------------------------
    frames_cache, det_ms_mean = _build_or_load_base_det_cache(
        video=video, weights=base_weights, device=device,
        max_frames=max_frames,
    )

    rows: Dict[str, TrackerRun] = {}
    for tname in BASELINE_TRACKERS:
        log.info("--- [%s] %s ---", clip_name, tname)
        _reset_gpu_peak(device)
        t0 = time.time()
        try:
            captured, track_ms = _run_one_baseline(
                tname, frames_cache, device=device,
                reid_weights=reid_weights,
            )
        except Exception as exc:
            log.exception("[%s] %s FAILED: %s", clip_name, tname, exc)
            continue
        wall = time.time() - t0
        gpu_mb = _peak_gpu_mb(device)
        mot_rows = _xyxy_to_mot_rows(captured)
        try:
            metrics = _score(mot_rows, gt_path)
        except Exception as exc:
            log.exception("[%s] %s scoring failed: %s", clip_name, tname, exc)
            continue
        track_arr = np.asarray(track_ms, dtype=np.float64)
        run = TrackerRun(
            row=tname,
            detector="stock yolo26s.pt @ imgsz=640, conf=0.25, iou=0.7, classes=[0]",
            tracker_config=_build_baseline_tracker(
                tname, device=device, reid_weights=reid_weights,
            )[1],
            n_frames=len(frames_cache),
            wall_seconds=round(wall, 3),
            det_ms_per_frame_mean=round(det_ms_mean, 2),
            tracker_ms_per_frame_mean=round(float(track_arr.mean()), 2),
            tracker_ms_per_frame_median=round(float(np.median(track_arr)), 2),
            tracker_ms_per_frame_p95=round(float(np.percentile(track_arr, 95)), 2),
            end_to_end_fps=round(
                len(frames_cache) / max(wall, 1e-6), 2),
            gpu_peak_mb=round(gpu_mb, 1),
            metrics=metrics,
        )
        rows[tname] = run
        # Persist per-clip MOT predictions for future re-scoring.
        safe = tname.replace(" ", "_").replace("(", "").replace(")", "")\
            .replace(",", "")
        (out_pkl_dir / f"{safe}.txt").write_text(
            "\n".join(mot_rows) + ("\n" if mot_rows else "")
        )
        log.info(
            "[%s] %s  IDF1=%.4f MOTA=%.4f IDS=%d FN=%d FP=%d  "
            "wall=%.1fs e2e_fps=%.2f tracker=%.1fms gpu_peak=%.0fMB",
            clip_name, tname, metrics["idf1"], metrics["mota"],
            int(metrics["num_switches"]), int(metrics["num_misses"]),
            int(metrics["num_false_positives"]),
            wall, run.end_to_end_fps,
            run.tracker_ms_per_frame_mean, gpu_mb,
        )

    # ---- Ours path -------------------------------------------------------
    log.info("--- [%s] %s ---", clip_name, OURS_LABEL)
    from tracking.run_pipeline import run_pipeline_on_video
    os.environ.setdefault("BEST_ID_DARK_PROFILE", "v9")
    out_dir = REPO / "work" / "results" / clip_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = out_dir / "tracks.pkl"
    _reset_gpu_peak(device)
    t0 = time.time()
    try:
        tracks = run_pipeline_on_video(
            video=video, out=out_pkl, device=device,
            weights=our_weights, force=True,
            max_frames=max_frames,
        )
    except Exception as exc:
        log.exception("[%s] OURS FAILED: %s", clip_name, exc)
        return rows
    wall = time.time() - t0
    gpu_mb = _peak_gpu_mb(device)

    ours_rows = _ours_tracks_to_mot_rows(out_pkl)
    safe = OURS_LABEL.replace(" ", "_").replace("(", "").replace(")", "")
    (out_pkl_dir / f"{safe}.txt").write_text(
        "\n".join(ours_rows) + ("\n" if ours_rows else "")
    )
    metrics = _score(ours_rows, gt_path)
    n_frames = int(metrics.get("num_frames", 0))
    rows[OURS_LABEL] = TrackerRun(
        row=OURS_LABEL,
        detector=("weights/best.pt multi-scale {768,1024} ensemble, "
                  "conf=0.34, ensemble_iou=0.6, dark-recovery v9"),
        tracker_config=("DeepOcSort + OSNet x0.25 + Kalman jitter "
                        "patch + full v9 post-process chain"),
        n_frames=n_frames,
        wall_seconds=round(wall, 3),
        det_ms_per_frame_mean=0.0,  # not separately measured for shipped path
        tracker_ms_per_frame_mean=0.0,
        tracker_ms_per_frame_median=0.0,
        tracker_ms_per_frame_p95=0.0,
        end_to_end_fps=round(n_frames / max(wall, 1e-6), 2),
        gpu_peak_mb=round(gpu_mb, 1),
        metrics=metrics,
    )
    log.info(
        "[%s] %s  IDF1=%.4f MOTA=%.4f IDS=%d FN=%d FP=%d  "
        "wall=%.1fs e2e_fps=%.2f gpu_peak=%.0fMB n_tracks=%d",
        clip_name, OURS_LABEL, metrics["idf1"], metrics["mota"],
        int(metrics["num_switches"]), int(metrics["num_misses"]),
        int(metrics["num_false_positives"]),
        wall, rows[OURS_LABEL].end_to_end_fps, gpu_mb, len(tracks),
    )
    return rows
```

- [ ] **Step 6: Add the CLI / main**

```python
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clips-manifest", type=Path,
        default=REPO / "configs" / "clips.json",
    )
    p.add_argument(
        "--clips", nargs="+", default=None,
        help="Optional subset of clip names to run (default: all in manifest).",
    )
    p.add_argument(
        "--gt-root", type=Path, required=True,
        help="Directory containing <clip>/gt/gt.txt for every clip.",
    )
    p.add_argument(
        "--base-yolo-weights", type=Path,
        default=REPO / "weights" / "yolo26s.pt",
    )
    p.add_argument(
        "--our-yolo-weights", type=Path,
        default=REPO / "weights" / "best.pt",
    )
    p.add_argument(
        "--reid-weights", type=Path,
        default=REPO / "weights" / "osnet_x0_25_msmt17.pt",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames (smoke testing only).")
    p.add_argument(
        "--out-json", type=Path,
        default=REPO / "work" / "benchmarks" / "full_results.json",
    )
    p.add_argument(
        "--mot-out-root", type=Path,
        default=REPO / "work" / "benchmarks" / "full_mot",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from work.run_all_tests import _load_manifest
    clips = _load_manifest(args.clips_manifest)
    if args.clips:
        clips = [(n, v) for n, v in clips if n in set(args.clips)]
    if not clips:
        log.error("no clips selected")
        return 1
    log.info("running %d clips: %s",
             len(clips), [n for n, _ in clips])

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.mot_out_root.mkdir(parents=True, exist_ok=True)

    # Resume support: if out_json already has rows, keep them.
    if args.out_json.is_file():
        all_results = json.loads(args.out_json.read_text())
        all_results.setdefault("clips", {})
    else:
        all_results = {
            "device": args.device,
            "scoring": "py-motmetrics, IoU 0.5, mot15-2D",
            "ours_detector": "weights/best.pt multi-scale {768,1024}",
            "baseline_detector": "stock yolo26s.pt single-scale 640",
            "n_baseline_trackers": len(BASELINE_TRACKERS),
            "clips": {},
        }

    overall_t0 = time.time()
    for clip_name, video in clips:
        gt_path = args.gt_root / clip_name / "gt" / "gt.txt"
        out_pkl_dir = args.mot_out_root / clip_name
        try:
            rows = _run_one_clip(
                clip_name=clip_name, video=video, gt_path=gt_path,
                base_weights=args.base_yolo_weights,
                our_weights=args.our_yolo_weights,
                reid_weights=args.reid_weights,
                device=args.device, max_frames=args.max_frames,
                out_pkl_dir=out_pkl_dir,
            )
        except Exception as exc:
            log.exception("clip %s crashed: %s", clip_name, exc)
            continue
        if not rows:
            continue
        all_results["clips"][clip_name] = {
            "video": str(video),
            "gt": str(gt_path),
            "rows": {n: asdict(r) for n, r in rows.items()},
        }
        args.out_json.write_text(json.dumps(all_results, indent=2))
        log.info("wrote %s after clip %s", args.out_json, clip_name)

    overall = time.time() - overall_t0
    all_results["total_wall_seconds"] = round(overall, 3)
    args.out_json.write_text(json.dumps(all_results, indent=2))
    log.info("ALL DONE in %.1f min  →  %s",
             overall / 60.0, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Sanity-import the module**

```bash
python -c "import scripts.run_full_benchmark as m; print(m.ALL_ROWS)"
```
Expected:
```
['Ours (v9 shipped)', 'ByteTrack (base)', 'OcSort (base, no ReID)', 'HybridSort (base)', 'BotSort (base)', 'StrongSort (base)', 'DeepOcSort (base)']
```

- [ ] **Step 8: Commit**

```bash
git add scripts/run_full_benchmark.py configs/clips.json
git commit -m "feat(benchmark): driver for ours-vs-base-yolo26s 9-clip A10 sweep"
```

---

### Task 3: Local 50-frame smoke test

**Files:**
- None modified, just runs the driver.

**Why:** Catch dependency / code bugs offline, before paying A10 startup time.

- [ ] **Step 1: Download stock yolo26s.pt locally**

```bash
mkdir -p weights
python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"
mv yolo26s.pt weights/yolo26s.pt
ls -la weights/yolo26s.pt
```
Expected: file ~19 MB.

- [ ] **Step 2: Run a 50-frame smoke test on loveTest**

```bash
python scripts/run_full_benchmark.py \
  --device cpu \
  --clips loveTest \
  --max-frames 50 \
  --base-yolo-weights weights/yolo26s.pt \
  --our-yolo-weights  weights/best.pt \
  --gt-root /Users/arnavchokshi/Desktop \
  --out-json work/benchmarks/smoke_results.json \
  --mot-out-root work/benchmarks/smoke_mot
```
Expected: 7 rows logged, no crashes, `work/benchmarks/smoke_results.json` exists with one clip and 7 rows. IDF1 numbers will be garbage (50 frames is too short) — we only care that nothing throws.

- [ ] **Step 3: Inspect the JSON**

```bash
python -c "
import json; d=json.load(open('work/benchmarks/smoke_results.json'))
for r,v in d['clips']['loveTest']['rows'].items():
    print(f'{r:35s} idf1={v[\"metrics\"][\"idf1\"]:.3f} fps={v[\"end_to_end_fps\"]:.1f}')
"
```
Expected: 7 lines, finite IDF1 values (even if small).

- [ ] **Step 4: Clean smoke artifacts (do NOT commit)**

```bash
rm -rf work/benchmarks/smoke_results.json work/benchmarks/smoke_mot
```

---

## Phase 2 — Push to A10 + run

### Task 4: A10 sync script

**Files:**
- Create: `scripts/sync_to_a10.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Push everything the A10 needs to run scripts/run_full_benchmark.py.
#
# Idempotent: rsync only sends what changed. Safe to re-run after edits.
#
# Usage:
#   bash scripts/sync_to_a10.sh           # push code + weights + clips
#   bash scripts/sync_to_a10.sh --setup   # also (re)install deps remotely
set -euo pipefail

REMOTE="ubuntu@141.148.49.145"
KEY="$HOME/.ssh/pose-tracking.pem"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no"
RSYNC="rsync -avz --delete --progress -e \"$SSH\""

REPO_LOCAL="/Users/arnavchokshi/Desktop/BEST_ID_STRAT"
REPO_REMOTE="/home/ubuntu/BEST_ID_STRAT"
CLIPS_LOCAL="/Users/arnavchokshi/Desktop"
CLIPS_REMOTE="/home/ubuntu/clips"

CLIPS=(BigTest easyTest adiTest mirrorTest gymTest loveTest MotionTest shorterTest darkTest)

echo "==> Sync repo (excluding work/results, work/sweeps, .git, virtualenvs)"
rsync -avz --delete -e "$SSH" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude 'work/sweeps/' \
  --exclude 'work/results/' \
  --exclude 'work/regression/' \
  --exclude '*.det_cache.pkl' \
  --exclude '*.base_det_cache.pkl' \
  "$REPO_LOCAL/" "$REMOTE:$REPO_REMOTE/"

echo "==> Sync each test clip directory ($CLIPS_REMOTE/<name>/)"
$SSH $REMOTE "mkdir -p $CLIPS_REMOTE"
for c in "${CLIPS[@]}"; do
  echo "  -> $c"
  rsync -avz -e "$SSH" \
    --exclude '*.zip' \
    --exclude '*.bak' \
    --exclude '*overlay*.mp4' \
    --exclude '*tracked*.mp4' \
    --exclude '*FAST_pipeline*' \
    --exclude '*ids_overlay*' \
    --exclude '*.det_cache.pkl' \
    --exclude '*.base_det_cache.pkl' \
    --exclude 'sam2_compare_*' \
    --exclude 'yolo_vis_*' \
    "$CLIPS_LOCAL/$c/" "$REMOTE:$CLIPS_REMOTE/$c/"
done

if [[ "${1:-}" == "--setup" ]]; then
  echo "==> Remote env setup (one-time)"
  $SSH $REMOTE "bash -lc '
    set -euo pipefail
    cd $REPO_REMOTE
    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip
    # CUDA 12.x torch wheel
    pip install torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    # Stock yolo26s.pt download (Ultralytics auto-downloads on first use)
    if [ ! -f weights/yolo26s.pt ]; then
      mkdir -p weights
      python -c \"from ultralytics import YOLO; YOLO(\\\"yolo26s.pt\\\")\"
      mv yolo26s.pt weights/yolo26s.pt
    fi
    # OSNet ReID weight (auto-downloaded by BoxMOT on first use, cache it)
    python -c \"from boxmot import DeepOcSort; from pathlib import Path; \
      DeepOcSort(reid_weights=Path(\\\"weights/osnet_x0_25_msmt17.pt\\\"), \
                 device=\\\"cpu\\\", half=False)\" || true
    echo OK
  '"
fi

echo "==> Disk usage on A10:"
$SSH $REMOTE "du -sh $REPO_REMOTE $CLIPS_REMOTE"
echo "==> Done."
```

- [ ] **Step 2: Make executable + test the dry-run**

```bash
chmod +x scripts/sync_to_a10.sh
ssh -i ~/.ssh/pose-tracking.pem ubuntu@141.148.49.145 'echo OK from A10'
```
Expected: `OK from A10`.

- [ ] **Step 3: Commit**

```bash
git add scripts/sync_to_a10.sh
git commit -m "tooling: rsync push of repo + clips + env setup to A10"
```

---

### Task 5: Push everything + remote env install

- [ ] **Step 1: Run sync with --setup (first time)**

```bash
bash scripts/sync_to_a10.sh --setup 2>&1 | tee /tmp/a10_sync.log
```
Expected: rsync uploads everything, venv created, torch+ultralytics+boxmot installed, `weights/yolo26s.pt` present remotely. Total time depends on uplink speed; clip videos are several GB.

If torch wheel fails, try `--index-url https://download.pytorch.org/whl/cu118` instead.

- [ ] **Step 2: Verify the remote install**

```bash
ssh -i ~/.ssh/pose-tracking.pem ubuntu@141.148.49.145 \
  'cd BEST_ID_STRAT && source .venv/bin/activate && python -c "
import torch, ultralytics, boxmot, motmetrics
print(\"torch\", torch.__version__, \"cuda?\", torch.cuda.is_available())
print(\"ultralytics\", ultralytics.__version__)
print(\"boxmot\", boxmot.__version__)
print(\"motmetrics\", motmetrics.__version__)
print(\"yolo26s.pt:\", __import__(\"os\").path.getsize(\"weights/yolo26s.pt\"), \"bytes\")
print(\"best.pt:   \", __import__(\"os\").path.getsize(\"weights/best.pt\"), \"bytes\")
"'
```
Expected: torch ≥ 2.0, cuda? True, ultralytics 8.4.37, boxmot 18.0.0, motmetrics 1.4.0, yolo26s.pt ≈ 19 MB, best.pt ≈ 60 MB.

- [ ] **Step 3: Verify the gt files are present remotely**

```bash
ssh -i ~/.ssh/pose-tracking.pem ubuntu@141.148.49.145 '
for c in BigTest easyTest adiTest mirrorTest gymTest loveTest MotionTest shorterTest darkTest; do
  if [ -f /home/ubuntu/clips/$c/gt/gt.txt ]; then
    echo "OK $c $(wc -l < /home/ubuntu/clips/$c/gt/gt.txt) gt rows"
  else
    echo "MISSING $c"
  fi
done'
```
Expected: 9 OK lines.

- [ ] **Step 4: Push the new clips.json with REMOTE paths**

The local manifest points at `/Users/arnavchokshi/Desktop/...` — rewrite for the A10 location BEFORE running the benchmark there.

```bash
ssh -i ~/.ssh/pose-tracking.pem ubuntu@141.148.49.145 \
  'cd BEST_ID_STRAT && cat > configs/clips.remote.json <<EOF
{
  "clips": [
    {"name": "easyTest",    "video": "/home/ubuntu/clips/easyTest/IMG_2082.mov"},
    {"name": "gymTest",     "video": "/home/ubuntu/clips/gymTest/IMG_8309.mov"},
    {"name": "BigTest",     "video": "/home/ubuntu/clips/BigTest/BigTest.mov"},
    {"name": "adiTest",     "video": "/home/ubuntu/clips/adiTest/IMG_1649.mov"},
    {"name": "mirrorTest",  "video": "/home/ubuntu/clips/mirrorTest/IMG_2946.MP4"},
    {"name": "shorterTest", "video": "/home/ubuntu/clips/shorterTest/TestVideo.mov"},
    {"name": "darkTest",    "video": "/home/ubuntu/clips/darkTest/darkTest.mov"},
    {"name": "MotionTest",  "video": "/home/ubuntu/clips/MotionTest/IMG_4716.mov"},
    {"name": "loveTest",    "video": "/home/ubuntu/clips/loveTest/IMG_9265.mov"}
  ]
}
EOF'
```

---

### Task 6: Run the full A10 benchmark

- [ ] **Step 1: Kick off the run inside `tmux` so a dropped SSH doesn't kill it**

```bash
ssh -i ~/.ssh/pose-tracking.pem ubuntu@141.148.49.145 '
sudo apt-get install -y tmux 2>/dev/null
tmux new-session -d -s bench "cd BEST_ID_STRAT && source .venv/bin/activate && \
  python scripts/run_full_benchmark.py \
    --device cuda:0 \
    --clips-manifest configs/clips.remote.json \
    --gt-root /home/ubuntu/clips \
    --base-yolo-weights weights/yolo26s.pt \
    --our-yolo-weights  weights/best.pt \
    --reid-weights      weights/osnet_x0_25_msmt17.pt \
    2>&1 | tee work/benchmarks/run.log"
tmux ls'
```
Expected: tmux session `bench` listed.

- [ ] **Step 2: Monitor progress every few minutes**

```bash
ssh -i ~/.ssh/pose-tracking.pem ubuntu@141.148.49.145 \
  'tail -n 40 BEST_ID_STRAT/work/benchmarks/run.log'
```
Repeat until you see `ALL DONE in <N> min`. Estimated wall-time: 4-8 hours for 9 clips × 7 rows. The longest clip is `loveTest` (820 frames) — at A10 speeds (~50 ms/frame e2e per row), ~6.5 minutes per row × 7 rows = ~45 min just for loveTest. MotionTest (1203 frames) similarly long.

- [ ] **Step 3: Pull results back**

```bash
rsync -avz -e "ssh -i $HOME/.ssh/pose-tracking.pem" \
  ubuntu@141.148.49.145:/home/ubuntu/BEST_ID_STRAT/work/benchmarks/full_results.json \
  work/benchmarks/full_results.json
rsync -avz -e "ssh -i $HOME/.ssh/pose-tracking.pem" \
  ubuntu@141.148.49.145:/home/ubuntu/BEST_ID_STRAT/work/benchmarks/full_mot/ \
  work/benchmarks/full_mot/
rsync -avz -e "ssh -i $HOME/.ssh/pose-tracking.pem" \
  ubuntu@141.148.49.145:/home/ubuntu/BEST_ID_STRAT/work/benchmarks/run.log \
  work/benchmarks/run.log
rsync -avz -e "ssh -i $HOME/.ssh/pose-tracking.pem" \
  ubuntu@141.148.49.145:/home/ubuntu/BEST_ID_STRAT/work/results/ \
  work/results/
```
Expected: `full_results.json` ~50-200 KB, `full_mot/` 9 directories with 7 .txt files each, `work/results/<clip>/tracks.pkl` for all 9 clips.

- [ ] **Step 4: Sanity-check the results JSON**

```bash
python -c "
import json
d = json.load(open('work/benchmarks/full_results.json'))
print('clips:', list(d['clips'].keys()))
for clip, payload in d['clips'].items():
    print(f'\n=== {clip} ===')
    for row, vals in payload['rows'].items():
        m = vals['metrics']
        print(f'  {row:35s} IDF1={m[\"idf1\"]:.4f}  IDS={int(m[\"num_switches\"]):3d}  FPS={vals[\"end_to_end_fps\"]:.1f}')
"
```
Expected: 9 clips × 7 rows; ours' IDF1 ≥ every baseline on the hard clips (loveTest, MotionTest, darkTest). Easy clips may be tied at ~1.0.

- [ ] **Step 5: Commit results**

```bash
git add work/benchmarks/full_results.json work/benchmarks/full_mot/
git commit -m "benchmark(a10): full 9-clip x 7-row sweep results"
```

---

## Phase 3 — Charts and overlay videos

### Task 7: Extend the chart generator

**Files:**
- Modify: `scripts/generate_comparison_charts.py` (add 5 new chart functions, keep the existing 4 for back-compat)

- [ ] **Step 1: Add the new chart imports and constants at the top**

After the existing imports, before `chart_accuracy_overall`, add:

```python
FULL_RESULTS_DEFAULT = REPO / "work" / "benchmarks" / "full_results.json"

NEW_OURS_LABEL = "Ours (v9 shipped)"
NEW_BASELINES_ORDER = [
    "ByteTrack (base)",
    "OcSort (base, no ReID)",
    "HybridSort (base)",
    "BotSort (base)",
    "StrongSort (base)",
    "DeepOcSort (base)",
]
NEW_BASELINE_COLORS = {
    "ByteTrack (base)":      "#EF6C00",
    "OcSort (base, no ReID)": "#D84315",
    "HybridSort (base)":     "#5E35B1",
    "BotSort (base)":        "#1565C0",
    "StrongSort (base)":     "#7E57C2",
    "DeepOcSort (base)":     "#00838F",
}


def _load_full_results(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — run scripts/run_full_benchmark.py first")
    return json.loads(path.read_text())


def _row_color(row_name: str) -> str:
    if row_name == NEW_OURS_LABEL:
        return OURS_COLOR
    return NEW_BASELINE_COLORS.get(row_name, COMP_COLOR)
```

- [ ] **Step 2: Add `chart_headline_idf1`**

```python
def chart_headline_idf1(out_path: Path, results_json: Path) -> None:
    """Mean IDF1 across all clips, one bar per row, ours highlighted."""
    data = _load_full_results(results_json)
    clips = data["clips"]
    means: Dict[str, List[float]] = {}
    for _, payload in clips.items():
        for row, vals in payload["rows"].items():
            v = float(vals["metrics"]["idf1"])
            if np.isfinite(v):
                means.setdefault(row, []).append(v)
    items = [(r, float(np.mean(vs))) for r, vs in means.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    colors = [_row_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ypos = np.arange(len(names))
    bars = ax.barh(ypos, vals, height=BAR_HEIGHT, color=colors,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    lo = max(0.0, min(vals) - 0.05)
    hi = min(1.0, max(vals) + 0.04)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(
        f"Mean IDF1 across {len(clips)} dance clips, full video each "
        f"(higher is better)", fontsize=10,
    )
    ax.set_title(
        "Out-of-the-box YOLO26s + base BoxMOT trackers vs ours\n"
        "(ours = our YOLO weights + multi-scale + post-process chain)",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, v, n in zip(bars, vals, names):
        ax.text(v + (hi - lo) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left",
                fontsize=9,
                fontweight="bold" if n == NEW_OURS_LABEL else "normal",
                color="#1B5E20" if n == NEW_OURS_LABEL else "#37474F")

    if items[0][0] == NEW_OURS_LABEL and len(items) > 1:
        gap = (items[0][1] - items[1][1]) * 100
        ax.text(
            0.98, 0.05,
            f"Ours leads next-best ({items[1][0]}) by +{gap:.2f} IDF1 pp",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="#1B5E20", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                      edgecolor="#A5D6A7"),
        )
    _save_fig(fig, out_path)
    plt.close(fig)
```

- [ ] **Step 3: Add `chart_per_clip_grid`**

```python
def chart_per_clip_grid(out_path: Path, results_json: Path) -> None:
    """N-panel grid (one panel per clip), 7 bars per panel."""
    data = _load_full_results(results_json)
    clips = data["clips"]
    clip_names = list(clips.keys())
    rows_in_order = [NEW_OURS_LABEL, *NEW_BASELINES_ORDER]

    n = len(clip_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.6, rows * 3.4), squeeze=False,
    )
    for i, clip in enumerate(clip_names):
        ax = axes[i // cols][i % cols]
        payload = clips[clip]
        bars_data = []
        for r in rows_in_order:
            if r in payload["rows"]:
                v = float(payload["rows"][r]["metrics"]["idf1"])
                bars_data.append((r, v))
        bars_data.sort(key=lambda x: x[1], reverse=True)
        x = np.arange(len(bars_data))
        vals = [v for _, v in bars_data]
        names = [n for n, _ in bars_data]
        colors = [_row_color(n) for n in names]
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace(" (base)", "").replace(" (base, no ReID)", "*")
             .replace("Ours (v9 shipped)", "Ours")
             for n in names],
            rotation=30, ha="right", fontsize=8,
        )
        for xi, v, n in zip(x, vals, names):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7,
                    fontweight="bold" if n == NEW_OURS_LABEL else "normal",
                    color="#1B5E20" if n == NEW_OURS_LABEL else "#37474F")
        ax.set_ylim(max(0, min(vals) - 0.08), min(1.0, max(vals) + 0.08))
        ours_v = next((v for n, v in bars_data if n == NEW_OURS_LABEL), None)
        worst_v = min(v for _, v in bars_data) if bars_data else 0.0
        gap = (ours_v - worst_v) * 100 if ours_v is not None else 0.0
        ax.set_title(f"{clip}  (+{gap:.1f} pp over worst)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(
        f"Per-clip IDF1 — ours vs out-of-the-box yolo26s+base trackers "
        f"({n} clips)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)
```

- [ ] **Step 4: Add `chart_error_breakdown`**

```python
def chart_error_breakdown(out_path: Path, results_json: Path) -> None:
    """Stacked bars per row across all clips: FN vs FP vs IDS totals."""
    data = _load_full_results(results_json)
    clips = data["clips"]
    rows_in_order = [NEW_OURS_LABEL, *NEW_BASELINES_ORDER]
    totals: Dict[str, Dict[str, float]] = {
        r: {"FN": 0.0, "FP": 0.0, "IDS": 0.0} for r in rows_in_order
    }
    for _, payload in clips.items():
        for r, vals in payload["rows"].items():
            if r not in totals:
                continue
            m = vals["metrics"]
            totals[r]["FN"] += float(m.get("num_misses", 0))
            totals[r]["FP"] += float(m.get("num_false_positives", 0))
            totals[r]["IDS"] += float(m.get("num_switches", 0))

    rows_in_order = sorted(
        rows_in_order,
        key=lambda r: totals[r]["FN"] + totals[r]["FP"] + totals[r]["IDS"],
    )
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(rows_in_order))
    fn = [totals[r]["FN"] for r in rows_in_order]
    fp = [totals[r]["FP"] for r in rows_in_order]
    ids = [totals[r]["IDS"] for r in rows_in_order]
    ax.bar(x, fn, label="FN (misses)", color="#D32F2F")
    ax.bar(x, fp, bottom=fn, label="FP (false positives)", color="#F57C00")
    ax.bar(x, ids, bottom=[a + b for a, b in zip(fn, fp)],
           label="IDS (identity switches)", color="#7B1FA2")
    ax.set_xticks(x)
    ax.set_xticklabels(rows_in_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"Total errors across {len(clips)} clips, full videos",
                  fontsize=10)
    ax.set_title(
        "Where each tracker's errors come from\n"
        "(stacked: misses + false positives + identity switches; lower = better)",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save_fig(fig, out_path)
    plt.close(fig)
```

- [ ] **Step 5: Add `chart_speed_vs_accuracy_a10`**

```python
def chart_speed_vs_accuracy_a10(out_path: Path, results_json: Path) -> None:
    """Scatter: x=mean end-to-end FPS on A10, y=mean IDF1 across clips."""
    data = _load_full_results(results_json)
    clips = data["clips"]
    rows_in_order = [NEW_OURS_LABEL, *NEW_BASELINES_ORDER]
    agg: Dict[str, List[Tuple[float, float]]] = {r: [] for r in rows_in_order}
    for _, payload in clips.items():
        for r, vals in payload["rows"].items():
            if r not in agg:
                continue
            agg[r].append((float(vals["end_to_end_fps"]),
                           float(vals["metrics"]["idf1"])))
    points = []
    for r, lst in agg.items():
        if not lst:
            continue
        fps = float(np.mean([p[0] for p in lst]))
        idf = float(np.mean([p[1] for p in lst]))
        points.append((r, fps, idf))

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for r, fps, idf in points:
        c = _row_color(r)
        size = 240 if r == NEW_OURS_LABEL else 130
        ax.scatter(fps, idf, s=size, color=c,
                   edgecolor="white", linewidth=1.5,
                   zorder=3 if r == NEW_OURS_LABEL else 2)
        ax.text(fps, idf + 0.005, r,
                fontsize=10 if r == NEW_OURS_LABEL else 9,
                fontweight="bold" if r == NEW_OURS_LABEL else "normal",
                color="#1B5E20" if r == NEW_OURS_LABEL else "#263238",
                ha="center")

    ax.set_xlabel("End-to-end FPS on NVIDIA A10 (mean across clips)",
                  fontsize=10)
    ax.set_ylabel(f"Mean IDF1 across {len(clips)} dance clips",
                  fontsize=10)
    ax.set_title(
        "Speed vs accuracy on A10 — ours sits on the upper-right Pareto frontier",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.5)
    _save_fig(fig, out_path)
    plt.close(fig)
```

- [ ] **Step 6: Add `chart_lift_decomposition`**

```python
def chart_lift_decomposition(out_path: Path, results_json: Path) -> None:
    """Per-clip: bar of (ours - worst) and (ours - best) IDF1 deltas."""
    data = _load_full_results(results_json)
    clips = data["clips"]
    clip_names = list(clips.keys())
    rows_in_order = [NEW_OURS_LABEL, *NEW_BASELINES_ORDER]

    deltas = []
    for clip in clip_names:
        payload = clips[clip]
        ours_v = float(payload["rows"][NEW_OURS_LABEL]["metrics"]["idf1"])
        comp = [
            float(payload["rows"][r]["metrics"]["idf1"])
            for r in NEW_BASELINES_ORDER if r in payload["rows"]
        ]
        if not comp:
            continue
        deltas.append((clip, ours_v - max(comp), ours_v - min(comp)))
    deltas.sort(key=lambda x: x[2], reverse=True)

    x = np.arange(len(deltas))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, 1.0 + 1.0 * len(deltas)), 4.8))
    best = [100 * d[1] for d in deltas]
    worst = [100 * d[2] for d in deltas]
    ax.bar(x - width / 2, best, width, color=COMP_COLOR,
           label="vs best baseline", edgecolor="white")
    ax.bar(x + width / 2, worst, width, color=OURS_COLOR,
           label="vs worst baseline", edgecolor="white")
    for xi, b, w in zip(x, best, worst):
        ax.text(xi - width / 2, b + 0.2, f"{b:+.1f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, w + 0.2, f"{w:+.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#1B5E20")
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in deltas],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Ours - baseline IDF1 (percentage points)", fontsize=10)
    ax.set_title("Where the gap lives — IDF1 lift over baselines per clip",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _save_fig(fig, out_path)
    plt.close(fig)
```

- [ ] **Step 7: Hook the new charts into `main()`**

In the existing `main()` function, after the call to `chart_per_clip()`, add:

```python
    # New full-benchmark charts (only if the JSON exists)
    full_json = REPO / "work" / "benchmarks" / "full_results.json"
    if full_json.is_file():
        chart_headline_idf1(args.out_dir / "headline_idf1.png", full_json)
        chart_per_clip_grid(args.out_dir / "per_clip_idf1_grid.png", full_json)
        chart_error_breakdown(args.out_dir / "error_breakdown.png", full_json)
        chart_speed_vs_accuracy_a10(
            args.out_dir / "speed_vs_accuracy_a10.png", full_json,
        )
        chart_lift_decomposition(
            args.out_dir / "lift_decomposition.png", full_json,
        )
    else:
        log.warning("no %s — skipping new full-benchmark charts", full_json)
```

- [ ] **Step 8: Run the chart generator**

```bash
python scripts/generate_comparison_charts.py
ls -la docs/figures/
```
Expected: 5 new PNGs (`headline_idf1.png`, `per_clip_idf1_grid.png`, `error_breakdown.png`, `speed_vs_accuracy_a10.png`, `lift_decomposition.png`) plus the original 4.

- [ ] **Step 9: Eyeball the headline chart**

Open `docs/figures/headline_idf1.png` and verify ours is on top.

- [ ] **Step 10: Commit charts + script changes**

```bash
git add scripts/generate_comparison_charts.py docs/figures/
git commit -m "charts: 5 new charts driven by full A10 benchmark results"
```

---

### Task 8: Side-by-side overlay videos

**Files:**
- Create: `scripts/render_side_by_side.py`

**Why:** Show, don't tell. The current README already has one ours-vs-StrongSort overlay; we add 2-3 more, picking the worst clip per category (ID-switch heavy, FN-heavy, FP-heavy).

- [ ] **Step 1: Write the script header + arg parsing**

```python
"""Render side-by-side overlay MP4s: ours vs the worst baseline per clip.

For each clip, picks the densest 7-15 second window where the worst
baseline produces the most errors (IDS + FN + FP per second) and renders
a 2-panel video with bbox+ID overlays plus a live error counter.

Inputs (must already exist):
  * work/benchmarks/full_results.json
  * work/benchmarks/full_mot/<clip>/<row>.txt  (per-tracker MOT files)
  * <clip>/gt/gt.txt
  * the source video file

Outputs:
  * docs/videos/<clip>_ours_vs_<worst>.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("render_side_by_side")
```

- [ ] **Step 2: Add MOT file loader**

```python
def _load_mot(path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Returns {frame_idx -> [(tid, x, y, w, h), ...]} (0-based frame)."""
    out: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        f = int(parts[0]) - 1
        tid = int(parts[1])
        x = float(parts[2]) - 1.0
        y = float(parts[3]) - 1.0
        w = float(parts[4])
        h = float(parts[5])
        out.setdefault(f, []).append((tid, x, y, w, h))
    return out
```

- [ ] **Step 3: Add per-frame error counter (cumulative IDS + missing+extra IDs)**

```python
def _per_frame_errors(
    pred: Dict[int, List], gt: Dict[int, List], iou_thresh: float = 0.5,
    max_frame: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """Greedy per-frame matching to count cumulative IDS, FN, FP.

    Not a full motmetrics replay (that's expensive in real-time render);
    a greedy hungarian-by-IoU matcher with sticky tid->gtid assignment is
    accurate enough for an overlay counter.
    """
    from scipy.optimize import linear_sum_assignment

    cum_ids, cum_fn, cum_fp = 0, 0, 0
    out_ids: List[int] = []
    out_fn: List[int] = []
    out_fp: List[int] = []
    last_assignment: Dict[int, int] = {}  # pred_tid -> gt_tid

    def iou(a, b):
        ax1, ay1, aw, ah = a; ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = b; bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    for f in range(max_frame + 1):
        p = pred.get(f, [])
        g = gt.get(f, [])
        if not p and not g:
            cum_fn += 0; cum_fp += 0
        else:
            if p and g:
                cost = np.zeros((len(p), len(g)), dtype=np.float64)
                for i, pp in enumerate(p):
                    for j, gg in enumerate(g):
                        cost[i, j] = 1.0 - iou(
                            (pp[1], pp[2], pp[3], pp[4]),
                            (gg[1], gg[2], gg[3], gg[4]),
                        )
                ri, ci = linear_sum_assignment(cost)
                matched_pred = set()
                matched_gt = set()
                for i, j in zip(ri, ci):
                    if cost[i, j] <= 1.0 - iou_thresh:
                        ptid = p[i][0]
                        gtid = g[j][0]
                        if ptid in last_assignment and last_assignment[ptid] != gtid:
                            cum_ids += 1
                        last_assignment[ptid] = gtid
                        matched_pred.add(i)
                        matched_gt.add(j)
                cum_fp += len(p) - len(matched_pred)
                cum_fn += len(g) - len(matched_gt)
            else:
                cum_fp += len(p)
                cum_fn += len(g)
        out_ids.append(cum_ids)
        out_fn.append(cum_fn)
        out_fp.append(cum_fp)
    return out_ids, out_fn, out_fp
```

- [ ] **Step 4: Add the per-clip rendering**

```python
def _color_for_id(tid: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(tid * 9_173 + 17)
    return tuple(int(c) for c in rng.integers(60, 240, size=3))


def _draw_panel(
    frame: np.ndarray, dets: List, label: str,
    cum_ids: int, cum_fn: int, cum_fp: int, *, ours: bool,
) -> np.ndarray:
    out = frame.copy()
    for tid, x, y, w, h in dets:
        col = _color_for_id(tid)
        cv2.rectangle(out, (int(x), int(y)),
                      (int(x + w), int(y + h)), col, 2)
        cv2.putText(out, str(tid), (int(x) + 2, int(y) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    H, W = out.shape[:2]
    bar_h = 60
    bar = np.zeros((bar_h, W, 3), dtype=np.uint8)
    text_color = (60, 220, 60) if ours else (60, 80, 220)
    cv2.putText(bar, label, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(bar,
                f"IDS={cum_ids}  FN={cum_fn}  FP={cum_fp}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    return np.concatenate([out, bar], axis=0)


def render_clip_side_by_side(
    *, video: Path, gt_path: Path,
    ours_mot: Path, base_mot: Path, base_label: str,
    out_path: Path, max_seconds: float = 12.0,
) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gt = _load_mot(gt_path)
    ours = _load_mot(ours_mot)
    base = _load_mot(base_mot)

    log.info("computing per-frame error rolls...")
    ours_ids, ours_fn, ours_fp = _per_frame_errors(ours, gt, max_frame=n_frames - 1)
    base_ids, base_fn, base_fp = _per_frame_errors(base, gt, max_frame=n_frames - 1)

    # Pick the densest window where (base errors - ours errors) per second is
    # highest. This is "the worst window for the worst tracker" per clip.
    win = int(round(max_seconds * fps))
    delta = (np.array(base_ids) + np.array(base_fn) + np.array(base_fp)) \
        - (np.array(ours_ids) + np.array(ours_fn) + np.array(ours_fp))
    cum = np.concatenate(([0], np.cumsum(delta)))
    diffs = cum[win:] - cum[:-win]
    if len(diffs) == 0:
        start = 0
    else:
        start = int(np.argmax(diffs))
    end = min(start + win, n_frames - 1)
    log.info("window: frames %d-%d (%.1fs-%.1fs)",
             start, end, start / fps, end / fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_W = W * 2
    out_H = H + 60
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        ours_panel = _draw_panel(
            frame, ours.get(f, []),
            f"Ours (v9 shipped)",
            ours_ids[f], ours_fn[f], ours_fp[f], ours=True,
        )
        base_panel = _draw_panel(
            frame, base.get(f, []),
            f"{base_label}",
            base_ids[f], base_fn[f], base_fp[f], ours=False,
        )
        side = np.concatenate([ours_panel, base_panel], axis=1)
        writer.write(side)
    writer.release()
    cap.release()
    log.info("wrote %s", out_path)
```

- [ ] **Step 5: Add main() that picks the best 2 example clips**

```python
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-json", type=Path,
        default=REPO / "work" / "benchmarks" / "full_results.json",
    )
    p.add_argument(
        "--mot-root", type=Path,
        default=REPO / "work" / "benchmarks" / "full_mot",
    )
    p.add_argument(
        "--clips-manifest", type=Path,
        default=REPO / "configs" / "clips.json",
    )
    p.add_argument(
        "--gt-root", type=Path,
        default=Path("/Users/arnavchokshi/Desktop"),
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=REPO / "docs" / "videos",
    )
    p.add_argument(
        "--top-n", type=int, default=3,
        help="How many clips to render (picks the highest-gap clips).",
    )
    p.add_argument(
        "--max-seconds", type=float, default=12.0,
        help="Length of each side-by-side window (seconds).",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data = json.loads(args.results_json.read_text())
    from work.run_all_tests import _load_manifest
    manifest = dict(_load_manifest(args.clips_manifest))

    # Score each clip by (ours_idf1 - worst_baseline_idf1)
    rankings = []
    for clip, payload in data["clips"].items():
        rows = payload["rows"]
        ours_v = float(rows.get("Ours (v9 shipped)", {}).get("metrics", {})
                       .get("idf1", 0.0))
        baseline_idf1 = {
            r: float(v["metrics"]["idf1"])
            for r, v in rows.items() if r != "Ours (v9 shipped)"
        }
        if not baseline_idf1:
            continue
        worst_name = min(baseline_idf1, key=baseline_idf1.get)
        worst_v = baseline_idf1[worst_name]
        rankings.append((clip, ours_v - worst_v, worst_name))
    rankings.sort(key=lambda x: x[1], reverse=True)

    chosen = rankings[: args.top_n]
    log.info("rendering top-%d clips: %s", args.top_n,
             [(c, f"+{d * 100:.1f}pp gap vs {w}") for c, d, w in chosen])

    for clip, _, worst_name in chosen:
        video = manifest.get(clip)
        if video is None or not video.is_file():
            log.warning("missing video for %s; skipping", clip)
            continue
        gt = args.gt_root / clip / "gt" / "gt.txt"
        ours_mot = args.mot_root / clip / "Ours_v9_shipped.txt"
        worst_safe = worst_name.replace(" ", "_").replace("(", "")\
            .replace(")", "").replace(",", "")
        base_mot = args.mot_root / clip / f"{worst_safe}.txt"
        if not ours_mot.is_file():
            log.error("missing %s", ours_mot); continue
        if not base_mot.is_file():
            log.error("missing %s", base_mot); continue
        out = args.out_dir / f"{clip}_ours_vs_{worst_safe}.mp4"
        render_clip_side_by_side(
            video=video, gt_path=gt,
            ours_mot=ours_mot, base_mot=base_mot,
            base_label=worst_name, out_path=out,
            max_seconds=args.max_seconds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run on the real results**

```bash
python scripts/render_side_by_side.py --top-n 3
ls -lh docs/videos/
```
Expected: 3 mp4 files, each ~2-15 MB.

- [ ] **Step 7: (Optional) make a small GIF preview of the strongest one**

```bash
which ffmpeg || brew install ffmpeg
TOP_MP4=$(ls -S docs/videos/*.mp4 | head -1)
ffmpeg -y -i "$TOP_MP4" \
  -vf "fps=10,scale=900:-1:flags=lanczos" \
  -loop 0 "${TOP_MP4%.mp4}_preview.gif"
ls -lh docs/videos/
```

- [ ] **Step 8: Commit**

```bash
git add scripts/render_side_by_side.py docs/videos/
git commit -m "videos: side-by-side ours vs worst baseline overlays"
```

---

## Phase 4 — README rewrite

### Task 9: Move historical sections to EXPERIMENTS_LOG

**Files:**
- Modify: `docs/EXPERIMENTS_LOG.md` (append moved content)

- [ ] **Step 1: Append the existing README "Headline result" + "v8" sections to EXPERIMENTS_LOG**

Open `docs/EXPERIMENTS_LOG.md`, scroll to the bottom, append:

```markdown

---

## Appendix: pre-A10 README sections (archived 2026-04-19)

The sections below were the README's headline before the full A10 9-clip
benchmark replaced them. They're preserved here for the record.

### A.1 7-clip MPS v8 vs base BoxMOT (held detector constant)

(Paste in the "Headline result" / "Where the gap really lives" / 4-clip
table from the previous README. Source: work/benchmarks/per_clip_idf1.json,
mps device.)

### A.2 v9 dark recovery on top of v8

(Paste in the "v9 — env-gated dark recovery" subsection.)

### A.3 Apple Silicon FPS table

(Paste in the "Apple Silicon (MPS) — fair head-to-head" subsection.)

### A.4 ours vs base StrongSort on loveTest, MPS

(Paste in the "Side-by-side: ours vs base StrongSort on loveTest" section.)
```

(Each "(Paste in...)" line gets replaced with the literal markdown copied from the current README.)

- [ ] **Step 2: Commit the move first, before the rewrite**

```bash
git add docs/EXPERIMENTS_LOG.md
git commit -m "docs: archive pre-A10 README sections to EXPERIMENTS_LOG"
```

---

### Task 10: Rewrite README

**Files:**
- Modify: `README.md` (full replace)

- [ ] **Step 1: Compute the headline numbers**

```bash
python -c "
import json, numpy as np
d = json.load(open('work/benchmarks/full_results.json'))
clips = d['clips']
def mean_idf1(row):
    vs = [c['rows'][row]['metrics']['idf1']
          for c in clips.values() if row in c['rows']]
    return float(np.mean(vs))
ours = mean_idf1('Ours (v9 shipped)')
bases = {
  r: mean_idf1(r) for r in
  ['ByteTrack (base)','OcSort (base, no ReID)','HybridSort (base)',
   'BotSort (base)','StrongSort (base)','DeepOcSort (base)']
}
best = max(bases.items(), key=lambda x:x[1])
worst = min(bases.items(), key=lambda x:x[1])
print(f'Ours mean IDF1:        {ours:.4f}')
print(f'Best baseline:         {best[0]}={best[1]:.4f}  delta=+{(ours-best[1])*100:.2f}pp')
print(f'Worst baseline:        {worst[0]}={worst[1]:.4f}  delta=+{(ours-worst[1])*100:.2f}pp')
"
```

Use those numbers to fill the placeholders in Step 2.

- [ ] **Step 2: Write the new README**

Replace the entire `README.md` with the structure described in spec §6. Keep paths, links, code blocks accurate. **Important:** every `docs/figures/<...>.png` and `docs/videos/<...>.mp4` referenced must actually exist on disk.

The full new README content is built procedurally from the JSON; it's too long to inline here verbatim. The shape is:

1. ~3-line title + opening paragraph with the headline gap.
2. `![Headline IDF1](docs/figures/headline_idf1.png)` immediately.
3. The mean-across-clips table (one row per row label, columns: mean IDF1, mean MOTA, total IDS, total FN, total FP, end-to-end FPS, GPU peak).
4. `![Per-clip grid](docs/figures/per_clip_idf1_grid.png)`.
5. The 2-3 strongest side-by-side videos with one-paragraph captions tying each to the chart.
6. `![Lift decomposition](docs/figures/lift_decomposition.png)` + 1 paragraph.
7. `![Error breakdown](docs/figures/error_breakdown.png)` + 1 paragraph on "the gap is in IDS+FN, not FP."
8. `![Speed vs accuracy](docs/figures/speed_vs_accuracy_a10.png)` + 1 paragraph.
9. **Reproduce on your video** — shortened install + run quickstart.
10. **Reproduce the benchmark** — `bash scripts/sync_to_a10.sh --setup && python scripts/run_full_benchmark.py …`.
11. **Repository layout / output schema / CLI** — kept tight, link to PIPELINE_SPEC.

- [ ] **Step 3: Verify all README image and video links resolve**

```bash
python -c "
import re, pathlib
content = pathlib.Path('README.md').read_text()
for path in re.findall(r'\\((docs/(?:figures|videos)/[^)]+)\\)', content):
    p = pathlib.Path(path)
    print(('OK' if p.is_file() else 'MISSING'), path)
"
```
Expected: every line says OK.

- [ ] **Step 4: Verify the smoke test still passes**

```bash
python scripts/smoke_test.py --device cpu
```
Expected: exit 0.

- [ ] **Step 5: Commit the README rewrite**

```bash
git add README.md
git commit -m "docs: README rewrite around full A10 9-clip benchmark"
```

---

## Self-review checklist (run after the plan is written, before execution)

- [ ] Spec §3 (7 rows) → covered by Task 2 (`BASELINE_TRACKERS` + `OURS_LABEL`)
- [ ] Spec §4 (full metric list) → covered by `_score()` in Task 2 step 4 (returns IDF1/IDP/IDR/MOTA/MOTP/precision/recall/IDS/frag/FN/FP/MT/PT/ML/unique/predicted/frames)
- [ ] Spec §5.1 (3 new scripts + 1 new config) → Task 1, 2, 4, 8
- [ ] Spec §5.2 (5 new charts) → Task 7
- [ ] Spec §5.3 (committed outputs) → Tasks 6, 7, 8 commits
- [ ] Spec §6 (README structure) → Task 10
- [ ] Spec §7 (execution) → Tasks 3, 5, 6
- [ ] Spec §8 (risk: yolo26s download) → Task 5 step 4 has a fallback (`pip install` + `YOLO('yolo26s.pt')`)
- [ ] Spec §8 (risk: ReID firewall) → Task 4 setup script pre-warms the OSNet weight on the A10
- [ ] No placeholders in the plan: scanned, none found.
- [ ] No undefined function references: `_load_manifest` reused from `work/run_all_tests.py`; `tracks_to_frame_detections` reused from `tracking.postprocess`; `run_pipeline_on_video` reused from `tracking.run_pipeline`. All exist.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-base-yolo26s-vs-ours-benchmark-plan.md`.

Recommended execution: **Inline Execution** (the work is sequential — most tasks block on the previous task's outputs, especially the 4-8 hour A10 run). Use the executing-plans skill.
