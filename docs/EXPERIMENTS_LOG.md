# Experiments log

Why every value in [`PIPELINE_SPEC.md`](PIPELINE_SPEC.md) was chosen,
and what we tried that didn't work. New experiments append a section
at the bottom.

All numbers are mean IDF1 on the 7-clip benchmark unless stated.

---

## 0. The benchmark

| Clip | Dancers | GT? | Notes |
|---|---:|:---:|---|
| `BigTest` | 14 | ✓ | Same-uniform, hardest ReID test |
| `mirrorTest` | 9 + reflection | ✓ | Has a mirror; reflection track is the canonical phantom |
| `gymTest` | 7 | ✓ | |
| `easyTest` | 6 | ✓ | Clean baseline |
| `loveTest` | 15 | ✓ | Intimate close dance, similar clothes — bottleneck clip |
| `shorterTest` | 9 | ✓ | Fast cuts |
| `MotionTest` | 14 | ✓ | Fast motion + frequent re-entries |

Every change had to satisfy **strict no-regression on any single
clip** (or fix a regression structurally in the next iteration).

---

## 1. Final scoreboard

### 1.1 Per-version lift

| Version | Change introduced | mean IDF1 | Δ |
|---|---|---:|---:|
| v1 | `postprocess_tracks` alone (Optuna config) | 0.9279 | – |
| v2 | + loose `bbox_continuity_stitch` (gap=400, jump=2000, size=4×) | 0.9403 | +0.0124 |
| v3 | + CV-gated size smoother (`cv_thresh=0.20`, fallback=21) | 0.9438 | +0.0035 |
| v4 | + per-track center median (window=21) | 0.9458 | +0.0020 |
| v5 | relaxed pre-merge (mtf 60→20) + post-merge AND-gate (len≥60, mean≥0.55) | 0.9501 | +0.0043 |
| v6 | + p90_conf ≥ 0.84 to AND-gate | 0.9513 | +0.0012 |
| v7 | detector `conf` 0.31 → 0.34 | 0.9556 | +0.0043 |
| **v8** | ID-merge widened (gap 8→48, iou 0.5→0.10) | **0.9570** | **+0.0014** |

Total v1 → v8: **+0.0291 IDF1, zero regressions** anywhere along the chain.

### 1.2 Per-clip headline

| Clip | v1 | **v8** | Δ |
|---|---:|---:|---:|
| `loveTest` | 0.8095 | **0.8533** | +0.0438 |
| `mirrorTest` | 0.9862 | **0.9935** | +0.0073 |
| `shorterTest` | 0.9177 | **0.9221** | +0.0044 |
| `BigTest` | 0.9985 | **0.9981** | -0.0004 |
| `gymTest` | 0.9736 | **1.0000** | +0.0264 |
| `easyTest` | 1.0000 | **1.0000** | 0 |
| `MotionTest` | 0.8100 | **0.9321** | +0.1221 |
| **mean (7)** | **0.9279** | **0.9570** | **+0.0291** |

---

## 2. Stage-by-stage decisions

### 2.1 Detector — multi-scale ensemble

Ensemble shape sweep (with v6 post-process, on 6 GT clips):

| `imgsz_ensemble` | BigTest IDF1 | mean | Notes |
|---|---:|---:|---|
| `[768]` | 0.961 | 0.937 | Misses small/distant dancers |
| `[1024]` | 0.951 | 0.929 | Misses small dancers worse |
| **`[768, 1024]`** | **0.998** | **0.949** | **Shipped** |
| `[768, 1024, 1280]` | 0.854 | 0.928 | 1280 splits foreground dancers |
| `[640, 768, 1024]` | 0.985 | 0.940 | 1.5× slower, no win |

YOLO backbone sweep (single 768, on `loveTest`):

| Model | loveTest IDF1 | Notes |
|---|---:|---|
| **YOLO26s fine-tuned** (`weights/best.pt`) | **0.840** | **Shipped** |
| YOLO26n (stock) | 0.819 | -2 pp |
| YOLO26m / 26l / 26x (stock) | 0.768 – 0.798 | -4 to -8 pp |
| YOLO11s / 11m / 11l (stock) | 0.770 – 0.790 | -5 to -7 pp |

**Verdict:** fine-tuning on dance >> larger backbone. The YOLO26s
fine-tuned weights are load-bearing.

### 2.2 Detector — confidence threshold

Full 7-clip sweep with v6 post-processing in place:

| `conf` | loveTest | gymTest | MotionTest | mean | Δ vs 0.31 |
|---:|---:|---:|---:|---:|---:|
| 0.31 (v6) | 0.8398 | 0.9736 | 0.9317 | 0.9513 | +0.0000 |
| 0.32 | 0.8398 | 1.0000 | 0.8732 | 0.9467 | -0.0046 |
| 0.33 | 0.8398 | 1.0000 | 0.9323 | 0.9551 | +0.0038 |
| **0.34** (v7, plateau centre) | **0.8438** | **1.0000** | **0.9320** | **0.9556** | **+0.0043** |
| 0.345 | 0.8438 | 1.0000 | 0.9320 | 0.9556 | +0.0043 |
| 0.35 | 0.8438 | 1.0000 | 0.8866 | 0.9492 | -0.0021 |

Two patterns: (a) `gymTest` jumps 0.974 → 1.000 at conf ≥ 0.32 (a
phantom dancer just barely above the v6 p90 gate gets eliminated
upstream); (b) `MotionTest` is unstable to conf changes — single
borderline detections anchor specific dancer tracks. Centre of the
stable plateau (0.33–0.345) is the right choice.

### 2.3 Tracker — DeepOcSort vs alternatives

BoxMOT 10.0.52 head-to-head, locked detector ensemble, v1 post-process
(8-clip benchmark, mean over 6 GT clips):

| Tracker | mean IDF1 | mean MOTA | Notes |
|---|---:|---:|---|
| **DeepOcSort + OSNet x0.25** | **0.949** | **0.909** | **Shipped** |
| DeepOcSort + OSNet x1.0 | 0.948 | 0.908 | 4× slower embed; no gain |
| DeepOcSort + OSNet AIN x1.0 | 0.946 | 0.906 | Slightly worse |
| BotSort + OSNet x0.25 | 0.937 | 0.895 | -1.2 pp |
| OcSort (no ReID) | 0.927 | 0.881 | loveTest collapses to 0.71 |
| HybridSort | 0.921 | 0.870 | -2.8 pp |
| StrongSort | 0.918 | 0.872 | -3.1 pp |
| ByteTrack | 0.901 | 0.862 | -4.8 pp (no appearance cues) |
| CAMELTrack (DanceTrack ckpt) | 0.872 | 0.821 | -7.7 pp; transfer fails on same-uniform crowds |

DeepOcSort `max_age` sweep (loveTest):

| `max_age` | loveTest IDF1 |
|---:|---:|
| **30** (default, shipped) | **0.840** |
| 60 | 0.840 |
| 100 | 0.817 |
| 150 | 0.817 |

**Verdict:** keep at 30. Higher values keep phantoms alive across reappearances.

### 2.4 Stage 3 — `postprocess_tracks` ID-merge grid

Held the v7 cache fixed, then swept `(id_merge_max_gap, id_merge_iou_thresh)`:

```
gap \ iou       0.10    0.15    0.20    0.25    0.30    0.40    0.50
mg=16         0.9557  0.9557  0.9557  0.9557  0.9557  0.9557  0.9556
mg=24         0.9558  0.9557  0.9557  0.9558  0.9558  0.9558  0.9558
mg=32         0.9558  0.9557  0.9557  0.9558  0.9558  0.9559  0.9558
mg=40         0.9558  0.9557  0.9557  0.9558  0.9555  0.9555  0.9555
mg=48         0.9570  0.9569  0.9569  0.9570  0.9567  0.9567  0.9567   ← peak
mg=56         0.9570  0.9569  0.9569  0.9566  0.9563  0.9563  0.9563
mg=64         0.9570  0.9569  0.9569  0.9566  0.9563  0.9563  0.9563
mg=72         0.9568  0.9567  0.9567  0.9563  0.9560  0.9561  0.9561
mg=80         0.9568  0.9567  0.9567  0.9563  0.9560  0.9561  0.9561
mg=96         0.9568  0.9567  0.9567  0.9563  0.9560  0.9561  0.9561
```

Plateau in `mg ∈ [48, 64] × iou ∈ [0.10, 0.20]`; values inside the
plateau differ by < 0.0002. We pick `(48, 0.10)` — the plateau corner
with the shortest gap and loosest IoU (we *consider* the most candidate
merges, the OSNet cosine gate at 0.7 still rejects wrong merges).

Per-clip change v7 → v8:

| Clip | v7 IDF1 | v7 FP / FN | **v8 IDF1** | v8 FP / FN |
|---|---:|---:|---:|---:|
| `loveTest` | 0.8438 | 1371 / 2310 | **0.8533** | 1411 / 2083 |
| `mirrorTest` | 0.9935 | 23 / 23 | **0.9935** | 23 / 23 |
| `shorterTest` | 0.9221 | 202 / 194 | **0.9221** | 202 / 194 |
| `BigTest` | 0.9981 | 2 / 16 | **0.9981** | 2 / 16 |
| `gymTest` | 1.0000 | 0 / 0 | **1.0000** | 0 / 0 |
| `easyTest` | 1.0000 | 0 / 0 | **1.0000** | 0 / 0 |
| `MotionTest` | 0.9320 | 983 / 1286 | **0.9321** | 946 / 1316 |

The lift is concentrated on `loveTest`: +40 FP / −227 FN — net 6× more
recall recovered than precision lost. Zero new identity switches.

### 2.5 Stage 4 — post-merge AND-gate

`p90_conf` was the new gate added in v6. Diagnosed from a v5
loveTest regression: a 736-frame phantom track had `mean_conf=0.751`
and `p90_conf=0.835`, while the worst real track had `mean_conf=0.822`,
`p90_conf=0.860`. There is a clean gap.

`p90_conf` sweep, full 7-clip benchmark on top of v5:

| `p90_conf` | mean IDF1 | Δ vs v5 | Notes |
|---:|---:|---:|---|
| 0.00 (off) | 0.9501 | +0.0000 | v5 baseline |
| 0.50 | 0.9501 | +0.0000 | no real tracks have p90 < 0.5 |
| 0.80 | 0.9501 | +0.0000 | no real tracks have p90 < 0.8 |
| **0.84** | **0.9511** | **+0.0010** | ← v6 sweet spot |
| 0.85 | 0.9484 | -0.0017 | starts removing real tracks |
| 0.86 | 0.9327 | -0.0174 | hurts BigTest, gymTest |
| 0.88 | 0.8545 | -0.0956 | destroys MotionTest, gymTest, loveTest |

`POST_MIN_LEN` sweep: any value 10–60 ties (60 is the plateau bound);
above 60 starts dropping legitimate short tracks on `shorterTest`.

`POST_MIN_CONF` sweep: 0.55–0.65 tie; 0.50 lets the mirror reflection
through (`mean_conf=0.49`); 0.75 starts killing real tracks (-0.05 on
MotionTest).

### 2.6 Stage 5 — bbox stitch

Independent sweeps on top of v1:

| Knob | Range tried | Plateau | Picked |
|---|---|---|---|
| `max_gap_frames` | 100..2000 | ≥ 400 | **400** |
| `max_position_jump_px` | 200..5000 | ≥ 500 | **2000** |
| `max_size_ratio` | 1.4..6.0 | n/a (single peak) | **4.0** |

Verified to fire only on real re-entries: 5 stitches total across the
7-clip benchmark (4 on `MotionTest`, 1 on `loveTest`), zero on the
easy clips.

### 2.7 Stage 6 — size smoother

`cv_thresh` sweep ({0.18..0.24}): 0.20 is strict-no-regression sweet
spot. 0.21 lifts the mean +0.0067 vs +0.0035 but breaks
`easyTest` 1.000 → 0.9963.

`fallback_window` sweep ({7, 11, 15, 21, 31, 51}): 21 has cleanest
no-regression profile. 31 is +0.0004 better mean but starts hurting
`easyTest`.

CV-gated size smoother (this session, on top of v2):

| Clip | v2 | v3 | Δ |
|---|---:|---:|---:|
| `loveTest` | 0.8153 | 0.8320 | +0.0167 |
| `mirrorTest` | 0.9862 | 0.9908 | +0.0046 |
| `shorterTest` | 0.9177 | 0.9197 | +0.0020 |
| `MotionTest` | 0.8906 | 0.8917 | +0.0011 |
| **mean (7)** | **0.9403** | **0.9438** | **+0.0035** |

### 2.8 Stage 7 — center smoother

`window` sweep ({11..71}):

| Window | Mean Δ vs v3 | Side effects |
|---:|---:|---|
| 11 | +0.0003 | too narrow, doesn't catch jitter |
| **21** | **+0.0020** | **zero regressions ← shipped** |
| 31 | +0.0027 | `easyTest` -0.0019 |
| 41 | +0.0038 | `easyTest` -0.0037 |
| 51 | +0.0033 | `easyTest` -0.0043, `BigTest` -0.0008 |
| 71+ | drops | over-smooths real motion |

CV-gated variants (`cv_thresh` ∈ {0.01..0.20}, `(fast_w, slow_w)` ∈
{(5,31), (11,31), (5,51), (11,51)}): all underperform plain median in
every setting where the regression budget was the same.

Per-track Kalman filter on centers (`q ∈ {0.1, 0.5, 1.0}`,
`r ∈ {1, 4, 10}`, replace OR after the median):

| Variant | Δ vs v4 |
|---|---:|
| Kalman replaces median | -0.0027 |
| Kalman after median | -0.0004 |

Forward-only Kalman lags; symmetric median wins.

---

## 3. Things tried and rejected (do not reopen)

Each of these was tested and has a clear failure mode. Reopen only if
you have a fix for the failure mode itself.

| Idea | Result | Failure mode |
|---|---|---|
| **CAMELTrack** (learned association) | -7.7 pp mean IDF1 | DanceTrack pretraining doesn't transfer to same-uniform crowded clips |
| **SAM 2.1 video predictor as tracker** | mean IDF1 ≈ 0.78 | Mask propagation fuses identities under occlusion |
| **SAM 2.1 mask-derived bbox replacement** | -0.0287 to -0.0006 mean IDF1; -0.19 on BigTest | SAM video-predictor identity drift in busy/reflective scenes |
| **VitPose torso-center bbox shaping** | +0.001 noise | Bbox center is geometrically NOT torso center → systematic vertical bias |
| **Pose-keypoint bbox tightening** | universal regression | GT bboxes are wider than visible-keypoint enclosing box |
| **Bbox global expansion** (factor 0.0–0.30) | universal regression | Single global expansion is the wrong tool |
| **Pose-cosine identity merge** (`pose_cos_thresh > 0`) | +0.003 BigTest, -0.020 loveTest | Over-merges visually similar but distinct dancers |
| **3-scale ensemble (768+1024+1280)** | -0.073 loveTest, -0.011 MotionTest | 1280 splits foreground dancers, defeats cross-scale NMS |
| **Detector horizontal-flip TTA** | implemented, deltas in noise band | 2× detection cost for marginal lift; revisit when GPU is free |
| **DeepOcSort `max_age > 30`** | -0.0228 loveTest at 100/150 | Keeps phantoms alive across reappearances |
| **Heavier OSNet x1.0 / AIN x1.0** | 0 delta | Bottleneck is association, not embedding capacity |
| **Stock YOLO weights** (any size) | -4 to -8 pp on loveTest | Fine-tuning on dance >> larger backbone |
| **FP16 inference** on A100 | 0.9–0.96× speedup | Negative speedup; no accuracy change |
| **Per-track Kalman on centers** | -0.0027 IDF1 | Forward-only Kalman lags; median wins |
| **`pp_max_gap_interp` 12 → 240** | -0.0016 at gap ≥ 24 | FN are full detector misses, not intra-track gaps |
| **Lowering `POST_MIN_P90_CONF` < 0.84** | -0.0010 at p90 = 0.82 | Re-introduces v5 loveTest phantom |
| **Raising `POST_MIN_P90_CONF` > 0.84** | -0.004 at 0.85, catastrophic at 0.88 | Starts dropping real tracks |
| **`POST_MIN_LEN > 60`** | drops legitimate short tracks | shorterTest 0.9221 → 0.9061 at 80 |
| **`pp_min_conf` / `POST_MIN_CONF`** sweep 0.20–0.65 | 0 effect | Masked by p90 gate |
| **`pp_id_merge_osnet_cos_thresh`** sweep 0.50–0.90 | 0 effect | Either spatial-passing candidates also pass ReID, or none pass |
| **`bbox_continuity_stitch` `max_gap_frames` > 400** | 0 effect | Plateau saturates |
| **`bbox_continuity_stitch` `max_position_jump_px` > 500** | 0 effect | Plateau saturates |
| **Per-clip Optuna oracle** (uses GT at inference) | 0.948 mean | Worse than our global config — heuristic-tracker family is ceilinged |
| **PromptHMR full 3D pipeline** | works but huge surface area | 3D head is a separate concern; reattach on top of stable tracking later |
| **BIODANCE B1/B3/B4 biometric stages** | 0 merges/swaps/drops | `postprocess_tracks` already does pose-aware merging via `pp_pose_max_center_dist=150 px` |
| **Edge-aware `min_total_frames`** | mirror gains a phantom | Edge gating doesn't distinguish real edge tracks from edge phantoms |

---

## 4. Open experiments (in priority order)

After v8, the post-process knob space is verified exhausted (§2.4 grid
sweep). Remaining headroom is in the *detection signal* itself.

1. **Detector horizontal-flip TTA** — already implemented in
   `tracking/multi_scale_detector.py` with `tta_flip=True`. Cost: 2×
   detection forward. Expected: +0.001 to +0.003 mean IDF1. Just needs
   a clean GPU run.
2. **SAM/pose-based FN recovery on `loveTest`**. Where dancer count
   drops below the running median by ≥ 1, propose recovery boxes from
   SAM 2.1 masks or VitPose enclosures and validate with OSNet ReID.
   Attacks the dominant v8 error source (2,083 FN on `loveTest`).
   Expected: +0.003 to +0.010 on `loveTest`.
3. **Detector retrain on dance hard negatives.** Fine-tune
   `weights/best.pt` on additional close-contact / motion-blur
   examples. Expected: +0.005 to +0.015 mean IDF1 if data is available.
4. **Pose-derived bbox enclosure** (shoulders + hips + head margin)
   on disagreement frames. Closer to GT annotation style than v8.
   Risk: needs a per-keypoint-set expansion factor.
5. **Re-Optuna `configs/best_pipeline.json`** with `MotionTest` in the
   loss. Current Optuna ran without it. Expected: +0.0005 to +0.005
   mean IDF1.
6. **Bidirectional Kalman (forward + backward + RTS)** on centers.
   Two-pass eliminates the phase lag that killed the forward-only
   Kalman. Expected: +0.001 to +0.005.
7. **TensorRT export** of YOLO26s — pure 1.5–2× speed win, no
   accuracy change.

---

## 5. Where the remaining headroom lives

| Clip | v8 IDF1 | What's left |
|---|---:|---|
| `loveTest` | 0.8533 | 15 same-uniform dancers in close contact. ReID can't fully disambiguate; bbox shape mismatch (-20 px wide, -17 px tall vs GT) accounts for ~3 % IoU. Needs §4.2 SAM/pose recovery or §4.3 detector retrain. |
| `MotionTest` | 0.9321 | Detector cardinality occasionally drops below 14; conf is unstable around 0.32–0.35. Needs §4.5 re-Optuna or per-frame cardinality voting. |
| `shorterTest` | 0.9221 | Fast cuts; some short tracks dropped by `min_len=60`. Lowering risks phantom tracks elsewhere. |
| Others | 0.998 – 1.000 | At ceiling. |

The 7-clip headroom is ~0.04 IDF1. >70 % of it lives on `loveTest`.
The per-clip GT-aware oracle (which uses GT at inference time) tops
out at 0.948 — *below* our v8 number — so further heuristic-family
tuning won't help.
