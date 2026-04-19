# BEST_ID_STRAT — production 2D person-tracking + ID-assignment

A single, end-to-end pipeline that turns a dance video into a stable
`dict[track_id -> Track]` of bounding boxes. Built around a
fine-tuned YOLO26s detector, BoxMOT's DeepOcSort with OSNet ReID, and
a small chain of post-processing stages whose every constant was
verified on a 7-clip benchmark with a strict no-regression rule.

> **Documentation map**
> - This file — overview + headline numbers vs alternatives.
> - [`docs/PIPELINE_SPEC.md`](docs/PIPELINE_SPEC.md) — exhaustive
>   reproduction spec (every model, every config value, every kwarg).
> - [`docs/EXPERIMENTS_LOG.md`](docs/EXPERIMENTS_LOG.md) — why each
>   value was chosen and what was tried and rejected.

---

## Quickstart

**Prereqs.** Python **3.11** (everything is pinned against 3.11 — newer
Python may not have wheels for the pinned `numpy 1.26.4` / `torch 2.11`).
For GPU you also need a working CUDA driver (NVIDIA) or macOS 13+
(Apple Silicon / MPS).

```bash
git clone https://github.com/arnavchokshi/swaySort.git
cd swaySort
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 1) Install torch FIRST (the wheel depends on your platform):
#    -- NVIDIA + CUDA 12.x:
pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu124
#    -- Apple Silicon (mps) or CPU-only:
pip install torch==2.11.0 torchvision==0.26.0

# 2) Install the rest of the pinned deps (exact versions used to
#    measure every IDF1 / FPS number in this README):
pip install -r requirements.txt

# 3) Verify the install end-to-end (~15s on CPU, no GPU required):
python scripts/smoke_test.py --device cpu     # or --device cuda:0 / mps

# 4) Run the production pipeline on your own video:
python -m tracking.run_pipeline \
    --video path/to/dance.mp4 \
    --out   work/dance/tracks.pkl \
    --device cuda:0                           # or mps / cpu
```

Outputs `work/dance/tracks.pkl` (final tracks) and
`work/dance/tracks.pkl.cache.pkl` (intermediate raw detections, kept on
disk so post-process re-runs are free).

> **Reproducibility note.** Every dependency in `requirements.txt` is
> pinned to the *exact* version that produced the numbers in the
> "Headline result" and "Speed" sections below. The shipped
> `weights/best.pt` (57 MB, dance-fine-tuned YOLO26s) is what every
> per-clip IDF1 in the headline table was measured against. The OSNet
> ReID checkpoint is auto-downloaded by BoxMOT on first run
> (~5 MB). **No data outside the repo is needed to reproduce** —
> bring any input video and a Torch device.

---

## What's inside

```
video.mp4
   │
   ▼  multi-scale YOLO26s @ {768, 1024}, conf 0.34, NMS-union 0.6
   ▼  DeepOcSort + OSNet x0.25 ReID  (Kalman jitter patch installed)
   ▼  prune + interp + ID merge  (pre_min_total_frames=20,
   │                              id_merge gap=48 / iou=0.10,
   │                              ReID cos≥0.7 gate)
   ▼  post-merge AND-gate         (len≥60 ∧ mean≥0.55 ∧ p90≥0.84)
   ▼  bbox continuity stitch       (gap=400, jump=2000 px, size=4×)
   ▼  CV-gated size smoother       (cv≤0.20 ⇒ const, else 21-median)
   ▼  per-track center median      (window=21)
   │
   ▼
tracks.pkl   →  dict[int, Track] : frames, bboxes, confs
```

Five sequential post-process stages on top of one detector + one
tracker + one ReID head. Full kwarg-level spec in
[`docs/PIPELINE_SPEC.md`](docs/PIPELINE_SPEC.md).

---

## Headline result

7-clip dance benchmark, mean IDF1 — same multi-scale YOLO detector
across every row, only the tracker stack differs:

![Tracker accuracy comparison](docs/figures/accuracy_overall.png)

### Where the gap really lives — per-clip head-to-head

Mean IDF1 hides the real story. Every tracker scores ~1.000 on the
easy clips (`easyTest`, `gymTest`, `BigTest`, `adiTest`); the field
only separates on the hard ones. Below is every base BoxMOT tracker
re-run on **the same machine, the same cached YOLO multi-scale
detections** as our pipeline, then scored against
`/Users/.../<clip>/gt/gt.txt` at IoU 0.5 with `py-motmetrics`:

![Per-clip IDF1 vs each base tracker](docs/figures/per_clip_competitors.png)

| Clip | dancers | This pipeline (v8) | Best competitor | Worst competitor | Our gap over worst |
|---|---:|---:|---:|---:|---:|
| `MotionTest` | 14 fast motion | **0.861** | OcSort 0.795 | HybridSort 0.733 | **+12.8 pp** |
| `loveTest` | 15 close-contact | **0.836** | BotSort 0.786 | OcSort 0.735 | **+10.2 pp** |
| `shorterTest` | 9 fast cuts | **0.923** | BotSort 0.888 | OcSort 0.841 | **+8.2 pp** |
| `mirrorTest` | 9 + reflection | **0.993** | ByteTrack 0.966 | StrongSort 0.950 | **+4.2 pp** |

The gap *grows* with clip difficulty — the harder the clip, the
bigger our lead. Every per-clip number above was produced by
`scripts/eval_per_clip.py` on this Mac (MPS); raw output (CLEAR + ID
metrics, per-tracker FN/FP/IDS) is committed at
[`work/benchmarks/per_clip_idf1.json`](work/benchmarks/per_clip_idf1.json).

> Reproduce locally:
> ```bash
> python scripts/eval_per_clip.py \
>     --clips loveTest MotionTest shorterTest mirrorTest \
>     --gt-root /path/to/your/gt-root --device mps
> ```

The full per-version (`v1` → `v8`) breakdown of what every
post-process stage buys you, plus every knob we swept and rejected,
lives in
[`docs/EXPERIMENTS_LOG.md`](docs/EXPERIMENTS_LOG.md#11-per-version-lift)
— total post-process lift is **+0.0291 mean IDF1 with zero regressions
on any individual clip**.

---

## Speed

### A100, single-GPU (production target)

| Stage | Per-frame |
|---|---:|
| YOLO multi-scale (768 + 1024) | ~28–32 ms (≈ 32–35 FPS) |
| DeepOcSort + OSNet x0.25 | ~30–50 ms |
| Post-process chain (whole 1k-frame clip) | ~0.7–0.9 s |

**End-to-end: ~1–1.5 minutes per minute of dance video on an A100.**

### Apple Silicon (MPS) — fair head-to-head against alternative trackers

We re-ran every tracker against **the same cached YOLO multi-scale
detections** on `loveTest` (820 frames @ 1080p) on an M-series Mac,
so the only variable per row is the tracker itself.
Numbers are end-to-end FPS (detection + tracker + post-process):

![Tracker speed (MPS)](docs/figures/speed_bars.png)

| Tracker | Tracker latency | End-to-end FPS | Unique IDs<br>(15 real dancers) |
|---|---:|---:|---:|
| ByteTrack (base) | 1.3 ms | 13.33 | 21 |
| OcSort (base, no ReID) | 1.4 ms | 13.32 | 19 |
| **Ours (DeepOcSort + post-process)** | **42.8 ms** | **8.58** | **20** |
| HybridSort (base) | 58.9 ms | 7.54 | 18 |
| BotSort (base) | 74.6 ms | 6.74 | 18 |
| StrongSort (base) | 119.3 ms | 5.18 | **38** |

Reading this chart: ByteTrack and OcSort are 1.5× faster end-to-end,
but **only because they skip ReID entirely** — and they pay -3.0 to
-5.6 IDF1 points for it. The three other ReID-based competitors
(BotSort, StrongSort, HybridSort) are **all slower than us *and*
strictly less accurate**. StrongSort produces 38 unique IDs across
820 frames despite there being 15 real dancers — that's 23 identity
swaps the tracker never recovered from.

Reproduce locally: see `scripts/benchmark_trackers.py`.

### Speed vs accuracy — the Pareto picture

![Speed vs accuracy](docs/figures/speed_vs_accuracy.png)

We sit on the upper-right Pareto frontier: **nothing in the field is
both faster *and* more accurate**.

---

## Side-by-side: ours vs ByteTrack on the largest-gap clip

`loveTest` is the worst-case clip (15 same-uniform dancers in
sustained close contact). On the same cached YOLO detections + same
machine, our shipped pipeline scores **IDF1 = 0.836** vs base
ByteTrack at **IDF1 = 0.744** — a **−9.2 pp** drop just from
swapping in a motion-only tracker
(see [`work/benchmarks/per_clip_idf1.json`](work/benchmarks/per_clip_idf1.json)).
Below is 6 seconds from the middle of the clip, our pipeline on the
left, base ByteTrack on the right. Each bounding box is colored by
track ID — **stable colors across frames = stable identity, color
flips = identity swaps the tracker did not recover from**.

![Ours vs ByteTrack on loveTest — 6-second preview](docs/videos/love_ours_vs_bytetrack_preview.gif)

Watch the full 27-second comparison:
[`docs/videos/love_ours_vs_bytetrack.mp4`](docs/videos/love_ours_vs_bytetrack.mp4).

What to look for (numbers from `work/benchmarks/render_run.log`):
- **Total ID space across the whole clip.** Ours emits **14 final
  IDs**; ByteTrack emits **21**. Ground truth is 15 dancers — six
  ByteTrack IDs are split-and-rebuilt copies of the same dancer, our
  ID-merge stage collapsed them.
- **Color stability on close-contact frames.** Around the middle
  cluster (frames where 4+ dancers occlude each other), ByteTrack's
  IDs flicker as boxes are reassigned; ours is held stable by the
  OSNet-cosine-gated ID merge in stage 3 of the post-process chain.
- **Mean active dancers per frame is the same** (~14.3 for both),
  so ByteTrack isn't dropping people — it's *renaming* them, which
  is exactly what IDF1 punishes.

This isn't ByteTrack failing in a pathological way — it's the
expected behavior of a motion-only tracker on a same-uniform
crowded scene, and exactly the failure mode our DeepOcSort + ReID +
post-process chain was built to fix.

---

## Per-clip lift from the v1 → v8 post-process chain

Same detector, same tracker — only the post-process chain changes.
The v1 → v8 lift recovered the most ground on the hard clips
(MotionTest +12.2 pp, loveTest +4.4 pp) without regressing any of
the easy ones:

![Per-clip v1 -> v8 progression](docs/figures/per_clip_v1_to_v8.png)

---

## Repository layout

```
README.md                       this file
requirements.txt                python deps (torch installed separately)
pyproject.toml                  project + pytest config
.gitignore

configs/
  best_pipeline.json            post-process JSON config
  clips.example.json            template manifest for batch scripts

docs/
  PIPELINE_SPEC.md              exhaustive reproduction spec
  EXPERIMENTS_LOG.md            decisions + things tried & rejected
  figures/                      README comparison charts (PNG + Mermaid)
  videos/                       README side-by-side video assets

prune_tracks.py                 FrameDetections cache dataclass
tracking/
  run_pipeline.py               entry point: video -> tracks.pkl
  multi_scale_detector.py       multi-scale YOLO ensemble
  deepocsort_runner.py          DeepOcSort + Kalman jitter patch
  postprocess.py                stage-3 prune/interp/merge logic
  best_pipeline.py              stages 3-7 driver + helpers
  bbox_stitch.py                long-gap bbox continuity stitch

scripts/
  smoke_test.py                 fresh-clone install verifier
  benchmark_trackers.py         fair head-to-head tracker speed bench
  eval_per_clip.py              per-clip IDF1 eval vs base BoxMOT trackers
  generate_comparison_charts.py regenerates the README PNGs + Mermaid

work/
  benchmarks/
    tracker_speeds.json         scripts/benchmark_trackers.py output
    per_clip_idf1.json          scripts/eval_per_clip.py output
  results/                      per-clip tracks.pkl + overlay videos
  run_all_tests.py              batch driver (uses configs/clips.json)
  render_overlays.py            batch overlay renderer (same manifest)

weights/
  best.pt                       dance-fine-tuned YOLO26s (load-bearing)
```

---

## Output schema

`joblib.load("tracks.pkl")` → `dict[int, tracking.postprocess.Track]`:

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `track_id` | `int` | scalar | unique track id |
| `frames` | `np.int64` | `(T,)` | frame indices (0-based) |
| `bboxes` | `np.float32` | `(T, 4)` | xyxy |
| `confs` | `np.float32` | `(T,)` | per-frame detection confidence |

---

## CLI

```
python -m tracking.run_pipeline \
    --video VIDEO          input video file or directory of frames
    --out OUT              output tracks.pkl path
    --device DEVICE        cuda:0 | mps | cpu (default cuda:0)
    --weights WEIGHTS      YOLO weights (default weights/best.pt)
    --cfg CFG              post-process config JSON
                           (default configs/best_pipeline.json)
    --reid-weights NAME    ReID checkpoint
                           (default osnet_x0_25_msmt17.pt)
    --max-frames N         optional cap on input frames (testing)
    --cache PATH           explicit cache path (default <out>.cache.pkl)
    --force                re-run YOLO + DeepOcSort even if cache exists
    --log-level LEVEL      python logging level (default INFO)
```

---

## Programmatic use

End-to-end on a new video:

```python
from pathlib import Path
from tracking.run_pipeline import run_pipeline_on_video

tracks = run_pipeline_on_video(
    video=Path("dance.mp4"),
    out=Path("work/dance/tracks.pkl"),
    device="cuda:0",
)
```

Re-run only the post-process on an existing cache (sub-second):

```python
from pathlib import Path
from tracking.best_pipeline import build_tracks

tracks = build_tracks(
    cache_path=Path("work/dance/tracks.pkl.cache.pkl"),
    cfg_path=Path("configs/best_pipeline.json"),
    save_to=Path("work/dance/tracks_v2.pkl"),
)
```
