# Pipeline reproduction spec

This is the single source of truth for **what** the production pipeline
does, with every model, every config value, and every technique stated
explicitly. Reading this file is sufficient to rebuild the system from
scratch.

For **why** any value was chosen (sweep tables, rejected alternatives),
see [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md).

---

## 1. Environment

| Component | Pin | Notes |
|---|---|---|
| Python | `>=3.11` | uses PEP 604 union types |
| OS | macOS arm64 (mps) **or** Linux x86_64 (CUDA 12) | tested on A100 + Apple M-series |
| Torch | install separately, **before** `requirements.txt` | platform-specific wheel |
| Ultralytics YOLO | `>=8.4.37, <9` | YOLO26 family supported here |
| BoxMOT | `>=10.0.52` | DeepOcSort + OSNet ReID |
| lap | `>=0.5.12` | required by BoxMOT for assignment |
| numpy | `>=1.26, <3` | |
| scipy | `>=1.11` | `ndimage.median_filter`, `signal.medfilt`, `interpolate.interp1d` |
| opencv-python-headless | `>=4.9, <4.11` | video frame I/O |
| imageio + imageio-ffmpeg | `>=2.34`, `>=0.5` | |
| ffmpeg-python | `>=0.2` | |
| joblib | `>=1.3` | cache + tracks pickle format |

CUDA install (Linux):
```bash
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Apple Silicon (MPS):
```bash
pip install torch torchvision   # default index
pip install -r requirements.txt
```

---

## 2. Model weights

### 2.1 Detector — YOLO26s, dance-fine-tuned

| Field | Value |
|---|---|
| Architecture | YOLO26s (Ultralytics) |
| Backbone | stock YOLO26s |
| Fine-tune dataset | dance footage, 1-class (`person`) |
| File | `weights/best.pt` |
| Approximate size | 60 MB |

The fine-tuned checkpoint is **load-bearing**. Stock YOLO26{n,s,m,l,x}
and YOLO11{s,m,l} all underperform by 4–8 % IDF1 on the hardest clip
(see `EXPERIMENTS_LOG.md §3.6`).

### 2.2 ReID — OSNet x0.25

| Field | Value |
|---|---|
| Architecture | OSNet (omni-scale ReID) |
| Width multiplier | `x0.25` |
| Pretraining dataset | MSMT17 |
| File name | `osnet_x0_25_msmt17.pt` |
| Source | auto-downloaded by BoxMOT to `~/.cache/boxmot/` on first run |

Heavier OSNet x1.0 / OSNet AIN x1.0 yield zero gain at 4× the
embedding cost (see `EXPERIMENTS_LOG.md §3.2`).

---

## 3. Pipeline stages

The end-to-end driver is `tracking.run_pipeline.run_pipeline_on_video`.
It executes **two phases**:

```
                                          stages 1-2 (cached)
                ┌─────────────────────────────────────────────────────┐
   video.mp4 ──▶│ multi-scale YOLO ──▶ DeepOcSort ──▶ FrameDetections │
                │                                       cache         │
                └────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────────────┐
                │ build_tracks(cache) ──▶ tracks.pkl             │
                │     stage 3: postprocess_tracks                │
                │     stage 4: filter_tracks_post_merge          │
                │     stage 5: bbox_continuity_stitch            │
                │     stage 6: size_smooth_cv_gated              │
                │     stage 7: smooth_centers_median             │
                └────────────────────────────────────────────────┘
```

The cache (`<out>.cache.pkl`, joblib-pickled
`list[prune_tracks.FrameDetections]`) is preserved on disk so
post-process tweaks don't have to re-run YOLO + DeepOcSort.

### 3.1 Stage 1 — multi-scale YOLO detection ensemble

Implementation: `tracking.multi_scale_detector.make_multi_scale_detector`.

Per-frame algorithm:

1. Run YOLO `predict` at `imgsz=768`.
2. Run YOLO `predict` at `imgsz=1024`.
3. Concatenate per-scale boxes.
4. Run a single torchvision NMS at `iou_threshold=ensemble_iou` to
   union duplicates across scales (score reduction = max).
5. Return `[x1, y1, x2, y2, conf, cls]`.

Exact configuration:

| Knob | Value |
|---|---|
| `weights` | `weights/best.pt` |
| `imgsz_list` | `[768, 1024]` |
| `conf` | **`0.34`** (`tracking.best_pipeline.DET_CONF`) |
| `iou` (per-scale NMS) | `0.70` |
| `ensemble_iou` (cross-scale NMS) | `0.60` |
| `classes` | `[0]` (person) |
| `tta_flip` | `False` |
| `device` | `cuda:0` / `mps` / `cpu` (caller's choice) |

### 3.2 Stage 2 — DeepOcSort + OSNet ReID

Implementation: `tracking.deepocsort_runner.make_tracker`.

The tracker is BoxMOT's `DeepOcSort` (or `DeepOCSort` /
`DeepOCSORT` depending on the BoxMOT version — resolved at runtime by
`_resolve_deepocsort_class`). Construction is version-tolerant: we
introspect `__init__` and pass only the kwargs it accepts.

Exact construction kwargs:

| BoxMOT param | Value | Notes |
|---|---|---|
| `model_weights` *or* `reid_weights` | `Path("osnet_x0_25_msmt17.pt")` | BoxMOT auto-downloads if file is absent |
| `device` | `torch.device(...)` | matches caller `--device` |
| `fp16` *or* `half` | `False` | FP16 was 0.9–0.96× as fast on A100 (negative speedup) |
| `det_thresh` | BoxMOT default (typically `0.3`) | only set if the param exists in this BoxMOT version |

DeepOcSort's own (BoxMOT-internal) hyperparameters are left at their
defaults — none are overridden:

| Param | Default we keep |
|---|---|
| `max_age` | `30` (do **not** raise — `100`/`150` cost -0.0228 IDF1 on `loveTest`) |
| `min_hits` | `3` |
| `iou_threshold` | `0.3` |
| `delta_t` | `3` |
| `inertia` | `0.2` |
| `w_association_emb` | `0.5` |
| `alpha_fixed_emb` | `0.95` |

**Required patch** (installed once via
`tracking.deepocsort_runner.install_kalman_jitter_patch`):

The BoxMOT Kalman filter occasionally produces a covariance matrix
that is barely positive-semi-definite, causing `scipy.linalg.cho_factor`
to raise `LinAlgError` and silently drop a frame's track update. We
wrap `BaseKalmanFilter.update` (or `.update_state`) and on
`LinAlgError` retry up to 6 times, adding `eps * I` to `P` with
`eps = 1e-6 * 10**attempt`. Idempotent (a flag prevents re-patching).

Per-frame loop:

```python
tracker = make_tracker(...)
for idx, frame_bgr in iter_video_frames(video):
    dets = detect(frame_bgr)           # (N, 6)
    tracks_per_frame = tracker.update(dets, frame_bgr)
    # tracks_per_frame: [x1, y1, x2, y2, id, conf, cls, det_index]
    cache.append(FrameDetections(xyxys, confs, tids))
```

### 3.3 Stage 3 — `postprocess_tracks`

Implementation: `tracking.postprocess.postprocess_tracks`. Operates on
`list[RawTrack]` produced from the cache via
`frame_detections_to_raw_tracks`.

Exact kwargs (all values fixed; none come from the JSON config when a
constant is named):

| Kwarg | Value | Source |
|---|---|---|
| `min_box_w` | `10` | constant |
| `min_box_h` | `10` | constant |
| `min_total_frames` | **`20`** | `best_pipeline.PRE_MIN_TOTAL_FRAMES` |
| `min_conf` | `0.38` | `cfg["pp_min_conf"]` |
| `max_gap_interp` | `12` | `cfg["pp_max_gap_interp"]` |
| `id_merge_max_gap` | **`48`** | `best_pipeline.ID_MERGE_MAX_GAP` |
| `id_merge_iou_thresh` | **`0.10`** | `best_pipeline.ID_MERGE_IOU_THRESH` |
| `id_merge_osnet_cos_thresh` | `0.7` | `cfg["pp_id_merge_osnet_cos_thresh"]` (gate of last resort — do not loosen) |
| `medfilt_window` | `11` | `cfg["pp_medfilt_window"]` |
| `gaussian_sigma` | `3.0` | `cfg["pp_gaussian_sigma"]` |
| `num_max_people` | `25` | `cfg["pp_num_max_people"]` |
| `overlap_merge_iou_thresh` | `0.7` | `cfg["pp_overlap_merge_iou_thresh"]` |
| `overlap_merge_min_frames` | `5` | `cfg["pp_overlap_merge_min_frames"]` |
| `edge_trim_conf_thresh` | `0.0` (off) | `cfg["pp_edge_trim_conf_thresh"]` |
| `edge_trim_max_frames` | `0` (off) | `cfg["pp_edge_trim_max_frames"]` |
| `pose_extractor` | `None` | pose-cos disabled |
| `pose_cos_thresh` | `0.0` | pose-cos disabled |
| `pose_max_gap` | `120` | `cfg["pp_pose_max_gap"]` |
| `pose_min_iou_for_pair` | `0.0` | `cfg["pp_pose_min_iou_for_pair"]` |
| `pose_max_center_dist` | `150.0` | `cfg["pp_pose_max_center_dist"]` |
| `frame_loader` | `None` | |

What this stage does, in order:

1. Drops boxes with `w < 10` or `h < 10`.
2. Drops tracks with fewer than 20 total frames.
3. Drops detections with `conf < 0.38`.
4. Linearly interpolates bboxes across gaps `<= 12` frames.
5. Median-filters bbox centroids with window 11.
6. Gaussian-smooths bbox centroids with sigma 3.0.
7. **ID merge**: for each pair of tracks `(A, B)` where the head of `B`
   begins within 48 frames after the tail of `A`, with IoU between the
   two endpoint boxes `>= 0.10`, AND OSNet ReID cosine similarity
   `>= 0.7`, merge `B` into `A`.
8. **Overlap merge**: for any two co-existing tracks with sustained
   IoU `>= 0.7` for `>= 5` frames, merge into one.
9. **Pose merge**: disabled (`pose_cos_thresh = 0`).
10. Cap simultaneous track count at 25 per frame.

### 3.4 Stage 4 — `filter_tracks_post_merge`

Implementation: `tracking.best_pipeline.filter_tracks_post_merge`.

For each track from stage 3, keep iff **all three** hold:

| Predicate | Threshold |
|---|---|
| `len(track.frames) >= POST_MIN_LEN` | `60` |
| `mean(track.confs) >= POST_MIN_CONF` | `0.55` |
| `np.percentile(track.confs, 90) >= POST_MIN_P90_CONF` | `0.84` |

The `AND` is critical — `OR` semantics re-introduce the mirror-reflection
phantom (`mean_conf=0.49`, `len=114`).

### 3.5 Stage 5 — `bbox_continuity_stitch`

Implementation: `tracking.bbox_stitch.bbox_continuity_stitch`.

Loose bbox-only stitch for long off-frame walkouts. Algorithm:

1. For every ordered pair `(A, B)` of tracks where `B` starts after
   `A` ends:
2. Compute frame gap `g = first_frame(B) - last_frame(A)`. Reject if
   `g > max_gap_frames`.
3. Compute median velocity `v` over the last `velocity_window` frames
   of `A`. Compute extrapolated position
   `p_pred = last_pos(A) + v * g`, with `||v * g||` capped at
   `velocity_extrapolate_cap_px`.
4. Reject if `||p_pred - first_pos(B)|| > max_position_jump_px`.
5. Reject if the area ratio `area(B[0]) / area(A[-1])` (or its inverse)
   exceeds `max_size_ratio`.
6. Otherwise merge `B` into `A`.

Exact kwargs:

| Kwarg | Value |
|---|---|
| `max_gap_frames` | `400` (≈ 13 s @ 30 FPS) |
| `max_position_jump_px` | `2000.0` |
| `max_size_ratio` | `4.0` |
| `velocity_window` | `5` (default) |
| `velocity_extrapolate_cap_px` | `200.0` (default) |

### 3.6 Stage 6 — `size_smooth_cv_gated`

Implementation: `tracking.best_pipeline.size_smooth_cv_gated`.

Per track:

1. Compute `cv_w + cv_h = std(w)/mean(w) + std(h)/mean(h)`.
2. **If `<= cv_thresh`**: replace per-frame `(w, h)` with the per-track
   median `(w, h)`. Centers preserved.
3. **Else**: median-filter `(w, h)` with `mode="nearest"` and a window
   of `fallback_window` (clamped to track length, made odd).
4. Recombine smoothed `(w, h)` with the per-frame `(cx, cy)` to get
   new bboxes.

Exact kwargs:

| Kwarg | Value |
|---|---|
| `cv_thresh` | `0.20` |
| `fallback_window` | `21` |

### 3.7 Stage 7 — `smooth_centers_median`

Implementation: `tracking.best_pipeline.smooth_centers_median`.

Per track:

1. Decompose bbox to `(cx, cy, w, h)`.
2. Median-filter `cx` and `cy` independently with `mode="nearest"`
   and a window of `window` (clamped to track length, made odd).
3. Recombine smoothed `(cx, cy)` with the *unchanged* `(w, h)` (which
   stage 6 already cleaned) to get the final bbox.

Exact kwargs:

| Kwarg | Value |
|---|---|
| `window` | `21` |

---

## 4. Data structures

### 4.1 Cache: `list[FrameDetections]`

Defined in `prune_tracks.FrameDetections` (a frozen dataclass):

```python
@dataclass(frozen=True)
class FrameDetections:
    xyxys: np.ndarray  # (N, 4)  float32
    confs: np.ndarray  # (N,)    float32
    tids:  np.ndarray  # (N,)    float32  (DeepOcSort track IDs)
```

One entry per processed frame. Empty frames are
`FrameDetections(empty (0,4), empty (0,), empty (0,))`. Persisted via
`joblib.dump(frames, "cache.pkl")`.

### 4.2 Output: `dict[int, Track]`

Defined in `tracking.postprocess.Track`:

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `track_id` | `int` | scalar | globally unique track id |
| `frames` | `np.int64` | `(T,)` | video-frame indices (0-based, may have gaps) |
| `bboxes` | `np.float32` | `(T, 4)` | xyxy bounding boxes |
| `confs` | `np.float32` | `(T,)` | per-frame detection confidence |
| `masks` | optional | `(T, H, W)` or `None` | unused in this pipeline |
| `detected` | optional | `(T,)` `bool` or `None` | unused in this pipeline |

Persisted via `joblib.dump(tracks, "tracks.pkl")`.

### 4.3 Configuration: `configs/best_pipeline.json`

The complete file:

```json
{
  "best_pipeline_cfg": {
    "pp_min_total_frames": 60,
    "pp_min_conf": 0.38,
    "pp_id_merge_max_gap": 8,
    "pp_id_merge_iou_thresh": 0.5,
    "pp_id_merge_osnet_cos_thresh": 0.7,
    "pp_gaussian_sigma": 3.0,
    "pp_max_gap_interp": 12,
    "pp_medfilt_window": 11,
    "pp_num_max_people": 25,
    "pp_overlap_merge_iou_thresh": 0.7,
    "pp_overlap_merge_min_frames": 5,
    "pp_edge_trim_conf_thresh": 0.0,
    "pp_edge_trim_max_frames": 0,
    "pp_pose_cos_thresh": 0.0,
    "pp_pose_max_gap": 120,
    "pp_pose_min_iou_for_pair": 0.0,
    "pp_pose_max_center_dist": 150.0
  }
}
```

`pp_min_total_frames`, `pp_id_merge_max_gap`, and `pp_id_merge_iou_thresh`
are kept here as the historical Optuna values for reference. The
production code overrides them at the call-site with the constants in
`tracking.best_pipeline` (`PRE_MIN_TOTAL_FRAMES = 20`,
`ID_MERGE_MAX_GAP = 48`, `ID_MERGE_IOU_THRESH = 0.10`).

---

## 5. Reproduction

### 5.1 Install

```bash
git clone <this repo>
cd BEST_ID_STRAT
python -m venv .venv && source .venv/bin/activate

pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124   # or default for mps/cpu
pip install -r requirements.txt
```

Place the dance-fine-tuned YOLO checkpoint at `weights/best.pt`. (The
ReID checkpoint downloads itself the first time DeepOcSort runs.)

### 5.2 Run

```bash
python -m tracking.run_pipeline \
    --video path/to/dance.mp4 \
    --out   work/dance/tracks.pkl \
    --device cuda:0
```

The driver writes:

- `work/dance/tracks.pkl`  — final `dict[int, Track]`.
- `work/dance/tracks.pkl.cache.pkl` — intermediate
  `list[FrameDetections]` (re-runs of the post-process are free past
  this; pass `--force` to invalidate).

### 5.3 CLI flags

```
--video VIDEO            Input video file or directory of frames.
--out   OUT              Output tracks.pkl (joblib).
--weights WEIGHTS        YOLO weights (default weights/best.pt).
--cfg CFG                Post-process config JSON
                         (default configs/best_pipeline.json).
--reid-weights NAME_OR_PATH
                         ReID checkpoint (default osnet_x0_25_msmt17.pt
                         — BoxMOT auto-downloads).
--device DEVICE          cuda:0 / mps / cpu (default cuda:0).
--max-frames N           Optional cap on input frames (testing).
--cache PATH             Explicit cache path (default <out>.cache.pkl).
--force                  Re-run YOLO + DeepOcSort even if cache exists.
--log-level LEVEL        Python logging level (default INFO).
```

### 5.4 Programmatic

```python
from pathlib import Path
from tracking.run_pipeline import run_pipeline_on_video

tracks = run_pipeline_on_video(
    video=Path("dance.mp4"),
    out=Path("work/dance/tracks.pkl"),
    device="cuda:0",
)
# tracks: dict[int -> tracking.postprocess.Track]
```

To re-run only the post-process on an existing cache:

```python
from pathlib import Path
from tracking.best_pipeline import build_tracks

tracks = build_tracks(
    cache_path=Path("work/dance/tracks.pkl.cache.pkl"),
    cfg_path=Path("configs/best_pipeline.json"),
    save_to=Path("work/dance/tracks_v2.pkl"),
)
```

---

## 6. Repository layout

```
.
├── README.md                       overview + headline numbers
├── requirements.txt                python deps (torch separate)
├── pyproject.toml
├── .gitignore
├── configs/
│   └── best_pipeline.json          post-process JSON config (§4.3)
├── docs/
│   ├── PIPELINE_SPEC.md            this file
│   └── EXPERIMENTS_LOG.md          why every value was chosen
├── prune_tracks.py                 FrameDetections dataclass (§4.1)
├── tracking/
│   ├── __init__.py
│   ├── run_pipeline.py             entry point: video -> tracks.pkl
│   ├── multi_scale_detector.py     stage 1 (§3.1)
│   ├── deepocsort_runner.py        stage 2 (§3.2) + Kalman patch
│   ├── postprocess.py              stage 3 logic (§3.3)
│   ├── best_pipeline.py            stages 3-7 driver + helpers (§3.3-§3.7)
│   └── bbox_stitch.py              stage 5 (§3.5)
└── weights/
    ├── best.pt                     dance-fine-tuned YOLO26s (§2.1)
    └── README.txt
```

---

## 7. Performance envelope (A100, single-GPU)

| Stage | Wall time per frame |
|---|---|
| YOLO multi-scale detection (768 + 1024) | ~28–32 ms |
| DeepOcSort + OSNet x0.25 forward | ~30–50 ms |
| Stages 3–7 post-process (whole clip, ~1k frames) | ~0.7–0.9 s total |

For a 30-second 30-FPS clip on an A100: **~1–1.5 minutes** end-to-end.
On Apple M-series MPS: ~2–4× slower depending on the model.

---

## 8. Known constraints

- **Do not raise DeepOcSort `max_age`** above 30. Extending it keeps
  phantom tracks alive across reappearances and costs IDF1 on close-
  contact clips.
- **Do not loosen `pp_id_merge_osnet_cos_thresh`** below 0.7. This is
  the gate of last resort that prevents `id_merge_iou_thresh = 0.10`
  from producing wrong merges.
- **Do not add `imgsz=1280` to the detector ensemble.** YOLO splits
  large foreground dancers into multiple boxes at 1280 px and defeats
  the cross-scale NMS, costing −0.144 IDF1 on `BigTest`.
- **Do not enable pose-cos identity merge** (`pose_cos_thresh > 0`).
  Helps `BigTest` ~+0.003 but costs `loveTest` ~−0.020.
- **Do not enable detector horizontal-flip TTA** (`tta_flip=True`).
  Cost is 2× detector forwards; tested deltas are in the noise band.
- **Do not switch off `bbox_continuity_stitch`'s velocity cap**
  (`velocity_extrapolate_cap_px = 200`). Without it, a few high-velocity
  end-of-clip tracks extrapolate off-frame and break the position-jump
  gate.
