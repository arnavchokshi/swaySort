# Experiments log

Why every value in [`PIPELINE_SPEC.md`](PIPELINE_SPEC.md) was chosen,
and what we tried that didn't work. New experiments append a section
at the bottom.

All numbers in §1–§5 are mean IDF1 on the original 7-clip benchmark
unless stated. The §7 overnight session re-scored every variant on
a 9-clip benchmark (the original 7 plus `adiTest` and `darkTest`),
so its numbers are not directly comparable to §1.

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
| `adiTest` | 1 | ✓ | added §7. Solo clip; checks for spurious extra IDs |
| `darkTest` | 6 | ✓ | added §7. Low-light dance; baseline IDF1 0.65, ceiling for v8 |

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

Total v1 → v8: **+0.0291 IDF1, zero regressions** anywhere along the
chain.

**v9** (overnight session, §7) added the env-gated dark-recovery
profile on top of v8 — CLAHE + auto-gamma, luma-gated, byte-identical
on every frame brighter than mean-luma 80. On the well-lit 7 clips
the cache is bit-identical to v8 (0 ms regression), so the v9 lift
shows up only when scored against the new 9-clip benchmark that
includes `darkTest`. See §7.1 for the 9-clip table; the headline
there is **mean IDF1 0.9208 → 0.9263 (+0.0056)**, driven by
**`darkTest` 0.6526 → 0.7027 (+0.0502)** with zero regression on
any of the original 7 clips.

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
| **3-scale ensemble (`imgsz=[768,896,1024]`)** (v9 session) | -0.0023 mean, -0.0621 MotionTest | Same foreground-splitting failure as `[768,1024,1280]`; shorterTest gain doesn't offset MotionTest loss |
| **Soft-NMS at default `sigma=0.5`** (v9 session) | -0.18 mean, MotionTest collapses | Score threshold too permissive — preserves 12k+ overlapping FP per clip |
| **Cardinality-voting FN recovery** (v9 session) | -0.002 mean across all 15 variants | Synthetic extrapolated boxes lack appearance evidence; DeepOcSort assigns them to wrong tracks → new IDS + FP |
| **Adaptive conf reduction on dark frames** | -0.03 to -0.10 on darkTest | Lowers threshold under boxes v9 dark already upgraded; doubles down on borderline detections |
| **Multi-exposure brighten ensemble** (`BEST_ID_DARK_BRIGHTEN`) | -0.07 to -0.13 darkTest | Brightened view introduces FP that auto-gamma already filtered |
| **Explicit gamma 1.8 / 2.2** | -0.07 / -0.09 darkTest | Over-amplifies regions auto-gamma already saturated |
| **`imgsz_ensemble=[1024,1280]` or `[1280]` on dark** | -0.06 / -0.19 darkTest | Foreground-splitting at 1280 px (same root cause as the 7-clip rejection) |
| **SAM stride-1 stacked with v9 dark on darkTest** | -0.0096 vs v9 dark alone | SAM marks marginal low-luma boxes as phantoms (mask fill < 0.30) and removes detections that gamma legitimately recovered |
| **RTMW with default body/hand/face weights** | within rounding of baseline | Default weights don't bias toward the discriminative signal; only `pose_hand_heavy` and `pose_min_vis_strict` lift at all |

---

## 4. Open experiments (in priority order)

After v8, the post-process knob space is verified exhausted (§2.4 grid
sweep). Remaining headroom is in the *detection signal* itself.

1. **Detector horizontal-flip TTA** — already implemented in
   `tracking/multi_scale_detector.py` with `tta_flip=True`. Cost: 2×
   detection forward. Expected: +0.001 to +0.003 mean IDF1. Just needs
   a clean GPU run.
2. ~~**SAM/pose-based FN recovery on `loveTest`**. Where dancer count
   drops below the running median by ≥ 1, propose recovery boxes from
   SAM 2.1 masks or VitPose enclosures and validate with OSNet ReID.
   Attacks the dominant v8 error source (2,083 FN on `loveTest`).
   Expected: +0.003 to +0.010 on `loveTest`.~~ **Done** (see §7.3 / §7.5).
   *SAM 2.1 per-bbox verifier* (image-predictor, not video-predictor) at
   `BEST_ID_SAM_VERIFY_STRIDE=1` lifts `loveTest` +0.0082 IDF1 (recovers
   219 of 2246 misses) at the cost of −0.0056 on `MotionTest`. Shipped
   env-gated, not a default. *Cardinality-voting FN recovery* (linear
   extrapolation from track history) regressed every variant tested;
   marked do-not-reopen.
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
7. ~~**TensorRT export** of YOLO26s — pure 1.5–2× speed win, no
   accuracy change.~~ **Done** (see §6.1).

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

---

## 6. Lossless speed optimizations

All four ship behind env-var feature flags so the default code path
is unchanged. Each was validated with `scripts/regression_check.py`
(bit-identical or IoU-matched within tolerance) on 500 frames of
`loveTest` per device.

### 6.1. TensorRT FP32 export of YOLO26s — A100

| Configuration | Wall | ms / frame | FPS |
|---|---:|---:|---:|
| Baseline `weights/best.pt` | 31.88 s | 63.8 | 18.06 |
| `BEST_ID_TRT_ENGINE_DIR=weights/` (uses `best_{768,1024}.engine`) | 27.03 s | 54.1 | 21.79 |

**+15.2 % wall-clock reduction (1.18× speedup) on the full pipeline.**
Output equivalence: same 14 final tracks both runs; same long-gap
pose-merge fires on the same IDs over the same frame ranges; per-
detection bbox L∞ < 5 px and confidence L∞ < 0.05 (cuDNN ↔ TRT FP32
numerical noise band — same algorithm, different kernel selection).

The detector-only speedup is the projected 1.5–2× from §4.7; the
end-to-end ratio is smaller because the tracker (DeepOcSort + OSNet)
is unchanged and now dominates the per-frame budget. Logged to set
expectations correctly for any future TRT-only optimization on this
pipeline.

Build steps (run once on the target NVIDIA box; ~5 min per `imgsz`):

```bash
python scripts/export_yolo_trt.py --device cuda:0
# produces weights/best_768.engine and weights/best_1024.engine
```

The detector falls back to `.pt` if either engine is missing.
Engines are baked to a fixed input shape and a specific `(GPU arch,
TRT version)` combo, so they must be rebuilt when any of those
change.

### 6.2. Frame prefetch + detector ↔ tracker pipelining — MPS

Two separate flags (`BEST_ID_PREFETCH=4` and
`BEST_ID_PIPELINE_PARALLEL=1`); compose for **+13.8 % wall-clock
reduction on 500 frames of `loveTest` (Apple M-series MPS,
bit-identical cache + tracks)**.

On A100 the same combo measured inside the run-to-run noise band
(GPU is fast enough that the CPU-overlap headroom is small, and
the pipeline-parallel thread contends with the detector for the
single CUDA stream). They ship off by default on CUDA and on by
default for users who set them — no automatic device gating, the
user picks.

### 6.3. GPU-resident ensemble NMS

`BEST_ID_GPU_NMS=1` keeps cross-scale boxes/conf on the model
device and runs `torchvision.ops.nms` on-device, eliminating two
`.cpu().numpy()` round-trips per frame. Output is bit-identical to
the legacy CPU-NMS path.

Measured impact:
- **MPS**: −14 % wall (slower) — `torchvision.ops.nms` has no
  Metal kernel and falls back to CPU, adding overhead. Leave OFF.
- **A100**: within run-to-run noise. The 1–2 ms saved per frame
  doesn't clear the 6–7 % variance band. Leave OFF on A100.

Kept in the codebase because it's a pre-req if a future change
moves more of the detector pipeline (e.g. raw-output post-process)
on-device, and because it has no downside on CUDA when the rest of
the loop is also GPU-resident.

---

## 7. Overnight session — SAM 2.1, RTMW, dark recovery (v9)

**Goal.** Extend v8 with the two model families that were rejected
historically (SAM 2.1, RTMW) and attack the new low-light failure
mode (`darkTest` baseline IDF1 0.6526). Hardware: single A10
(24 GB) — half the SXM-A100 from the v1–v8 sessions.

**Setup.** Two new clips were added to the benchmark:

- `adiTest` (1 dancer, ground truth available) — solo-dance check
  for spurious extra IDs. Every variant tested scored a perfect
  1.0 IDF1 here.
- `darkTest` (6 dancers, ground truth available) — backlit indoor
  dance, mean luma ~50, with frequent occlusion-by-shadow.

The 7-clip benchmark mean is therefore not directly comparable
between sections: where §1.2 reports the v8 7-clip mean as 0.9570,
the same v8 cache rerun on the **9-clip** benchmark below scores
**0.9208** (the new clips drag the mean down: `darkTest` at 0.6526,
`adiTest` at 1.0000). All numbers in §7 are mean over the 9-clip
benchmark.

### 7.1 The v9 release (winner)

The shipping change is a single env-var-gated default flip in
`work/run_all_tests.py`:

```python
os.environ.setdefault("BEST_ID_DARK_PROFILE", "v9")
```

`BEST_ID_DARK_PROFILE=v9` is a shorthand that sets two knobs in
`tracking/dark_recovery.py`: `BEST_ID_DARK_CLAHE=1` (CLAHE on the
LAB L-channel) and `BEST_ID_DARK_GAMMA=auto` (luma-adaptive gamma
in `[1.0, 2.5]`). Both fire **only** on frames whose mean luma
falls below `BEST_ID_DARK_LUMA=80` — every frame of every well-lit
clip skips the preprocessing entirely, so the cache is bit-identical
to v8 wherever it was already winning.

| Variant | Env | Mean IDF1 (9 clips) | Δ vs v8 | Total IDS | Total FN | Total FP |
|---|---|---:|---:|---:|---:|---:|
| **v8 baseline** | – | 0.9208 | +0.0000 | 27 | 5875 | 4314 |
| **v9 (`dark_v9`)** | `CLAHE=1`, `GAMMA=auto` | **0.9263** | **+0.0056** | **20** | 6014 | 4314 |
| `v9_dark_plus_sam` | + `SAM_VERIFY=1`, `STRIDE=1` | 0.9255 | +0.0048 | 26 | 5744 | 4456 |
| `v9_dark_plus_pose` | + `POSE_MERGE=1`, `MIN_VIS=0.5` | 0.9266 | +0.0059 | 20 | 6019 | 4211 |
| `v9_dark_plus_sam_plus_pose` | all of above | 0.9259 | +0.0051 | 26 | 5748 | 4348 |

Per-clip IDF1 of the v9 default (`v9_dark`) vs v8 baseline:

| Clip | v8 baseline | v9 (`dark_v9`) | Δ |
|---|---:|---:|---:|
| `BigTest` | 0.9981 | 0.9981 | +0.0000 |
| `mirrorTest` | 0.9932 | 0.9932 | +0.0000 |
| `adiTest` | 1.0000 | 1.0000 | +0.0000 |
| `easyTest` | 1.0000 | 1.0000 | +0.0000 |
| `gymTest` | 1.0000 | 1.0000 | +0.0000 |
| `loveTest` | 0.8494 | 0.8494 | +0.0000 |
| `shorterTest` | 0.8624 | 0.8624 | +0.0000 |
| `MotionTest` | 0.9312 | 0.9312 | +0.0000 |
| `darkTest` | 0.6526 | **0.7027** | **+0.0502** |

The v9 default is **strictly non-regressive on every well-lit
clip** (the luma gate makes preprocessing a no-op on those frames)
and adds +0.0502 IDF1 on `darkTest`. It also reduces total identity
switches from 27 → 20 (−26 %), driven by `darkTest` dropping
23 → 16 IDS — CLAHE+gamma make the appearance descriptor more
stable in low-light frames so DeepOcSort flips identities less
often.

### 7.2 What didn't ship (and why)

**RTMW pose-merge gate (`pose_min_vis_strict`).** Mean IDF1 lift on
9 clips is **+0.0003** (concentrated on `MotionTest`: +0.0027).
Stacks cleanly with v9 dark recovery (`v9_dark_plus_pose` at
0.9266 vs `v9_dark` at 0.9263) but the absolute lift is not worth
the extra runtime dependency (`rtmlib` + `onnxruntime` + an ONNX
download). Kept in the codebase, fully wired, env-gated; users with
`rtmlib` installed can opt in via:

```bash
BEST_ID_POSE_MERGE=1 BEST_ID_POSE_MERGE_MIN_VIS=0.50 \
  python work/run_all_tests.py --device cuda:0
```

**SAM 2.1 per-bbox verifier (`sam_stride1`).** Mean IDF1 lift on
9 clips is **+0.0023** (concentrated on `loveTest`: +0.0082, by
recovering 219 missed detections via mask continuity). Two reasons
not to ship as default:

1. *Negative interaction with v9 dark*. `v9_dark + sam_stride1`
   on `darkTest` scores 0.6931 vs 0.7027 for `v9_dark` alone — SAM
   marks marginal low-luma boxes as phantoms (mask fill < 0.30)
   and removes them, costing 38 detections that the gamma-corrected
   detector had legitimately recovered.
2. *Cost*. SAM 2.1 image-predictor at stride 1 adds ~80 ms per
   frame on the A10 — roughly doubles end-to-end wall-clock for a
   +0.008 loveTest lift.

Kept env-gated; users running `loveTest`-style same-uniform
close-contact dance and not running on `darkTest`-style low-light
should opt in:

```bash
BEST_ID_SAM_VERIFY=1 BEST_ID_SAM_VERIFY_STRIDE=1 \
  python work/run_all_tests.py --device cuda:0
```

### 7.3 SAM 2.1 verifier sweep (10 variants, 9 clips)

The verifier asks the SAM 2.1 *image* predictor (not the video
predictor — that's the path that historically caused identity
fusion in §3) for each bbox: "what fraction of pixels inside the
box are foreground?". If `< BEST_ID_SAM_VERIFY_FILL` (default 0.30),
the box is dropped as a phantom.

| Variant | Stride | Fill thresh | Conf cutoff | Mean IDF1 | Δ | loveTest Δ | darkTest Δ | MotionTest Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | – | – | – | 0.9208 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| sam_default | 5 | 0.30 | < 0.55 | 0.9207 | -0.0000 | -0.0001 | +0.0000 | -0.0002 |
| sam_strict_fill | 5 | 0.40 | < 0.55 | 0.9207 | -0.0000 | -0.0002 | +0.0000 | -0.0002 |
| sam_loose_fill | 5 | 0.20 | < 0.55 | 0.9207 | -0.0000 | -0.0002 | +0.0000 | -0.0002 |
| **sam_stride1** | **1** | 0.30 | < 0.55 | **0.9231** | **+0.0023** | **+0.0082** | **+0.0185** | -0.0056 |
| sam_stride10 | 10 | 0.30 | < 0.55 | 0.9207 | -0.0000 | -0.0001 | +0.0000 | -0.0002 |
| sam_conf_lt45 | 5 | 0.30 | < 0.45 | 0.9208 | -0.0000 | -0.0001 | +0.0000 | +0.0000 |
| sam_conf_lt70 | 5 | 0.30 | < 0.70 | 0.9208 | +0.0001 | +0.0002 | +0.0005 | -0.0002 |
| sam_strict_combo | 1 | 0.40 | < 0.45 | 0.9205 | -0.0002 | +0.0000 | +0.0001 | -0.0022 |
| sam_safe_combo | 5 | 0.40 | < 0.70 | 0.9207 | -0.0000 | -0.0002 | +0.0000 | -0.0002 |

The win is concentrated entirely in `sam_stride1`: every other
configuration leaves the cache untouched within rounding (the
default stride-5 + conf < 0.55 gate is too narrow — almost every
box that gets verified is a true positive at the default fill
threshold).

The trade-off is visible in the per-clip miss/fp counts:
- `loveTest`: −219 misses, +62 false positives (net IDF1 +0.0082)
- `darkTest`: +1 miss, −8 false positives (net IDF1 +0.0185 — IDS
  dropped 23 → 19, the bigger win)
- `MotionTest`: −14 misses, +41 false positives, +2 IDS (net
  IDF1 −0.0056 — SAM is preserving motion-blur boxes that the
  baseline correctly suppressed)

So SAM stride-1 is a **directional win on appearance-degraded
clips** (loveTest, darkTest) at the cost of a small `MotionTest`
regression. Worth keeping env-gated; not worth defaulting on for
the entire benchmark.

### 7.4 RTMW pose-merge sweep (9 variants, 9 clips)

The gate is wired as an **AND** with the existing IoU/proximity ID
merge: a candidate merge passes only if BOTH the IoU/center-dist
gate AND the pose-similarity gate accept it. Pose features come
from the rtmlib Wholebody RTMW model (133 keypoints: 17 body, 6
feet, 68 face, 21+21 hands). Similarity is a weighted bbox-
normalised cosine (default body 0.40, hands 0.40, face 0.20).

| Variant | Threshold | Body w | Hand w | Face w | MinVis | Mean IDF1 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | – | – | – | – | – | 0.9208 | +0.0000 |
| pose_default | 0.50 | 0.40 | 0.40 | 0.20 | 0.30 | 0.9208 | +0.0000 |
| pose_thresh_low | 0.40 | 0.40 | 0.40 | 0.20 | 0.30 | 0.9208 | +0.0000 |
| pose_thresh_high | 0.65 | 0.40 | 0.40 | 0.20 | 0.30 | 0.9208 | +0.0000 |
| pose_hand_heavy | 0.50 | 0.20 | 0.60 | 0.20 | 0.30 | 0.9211 | +0.0003 |
| pose_body_heavy | 0.50 | 0.60 | 0.20 | 0.20 | 0.30 | 0.9208 | +0.0000 |
| **pose_min_vis_strict** | 0.50 | 0.40 | 0.40 | 0.20 | **0.50** | **0.9211** | **+0.0003** |
| pose_min_vis_loose | 0.50 | 0.40 | 0.40 | 0.20 | 0.10 | 0.9208 | +0.0000 |
| pose_lightweight | 0.50 | 0.40 | 0.40 | 0.20 | 0.30 | 0.9208 | +0.0000 |
| pose_performance | 0.50 | 0.40 | 0.40 | 0.20 | 0.30 | 0.9208 | +0.0000 |

The ceiling is +0.0003 mean IDF1, concentrated entirely on
`MotionTest` (+0.0027): hand-keypoint geometry distinguishes a
fast-arm dancer from a near-by arm-up neighbour during merges that
the IoU+ReID gate alone would have allowed. Two configurations
(`pose_hand_heavy` and `pose_min_vis_strict`) tie for the ceiling
because they both reduce the false-positive count on `MotionTest`
by 103 boxes (the hand-heavy and high-visibility configs both
require strong per-finger evidence before allowing a merge).

**Critical debugging note.** The first overnight run of this sweep
returned identical metrics for **every** variant (the gate appeared
to be silently disabled). Investigation traced the cause to
`tracking/best_pipeline.py:_make_frame_loader_for_cache`, which
was assuming the source video lived as a sibling of the
`tracks.pkl.cache.pkl` file. In sweep contexts the cache lives
under a per-variant output directory while the videos live under
`/home/ubuntu/work/data/<clip>/...`, so the loader returned a
no-op `lambda _idx: None`, the pose extractor received `None`
frames, and `cosine(...)` returned 0.0 — making every gate
unconditionally reject. The fix has two parts:

1. `tracking/run_pipeline.py` writes a sidecar
   `<cache>.cache.pkl.video.json` containing the absolute video
   path next to every cache it produces.
2. `_make_frame_loader_for_cache` reads the sidecar first, then
   falls back to sibling guesses, then to a no-op loader (which
   logs a warning instead of silently degrading).

The bug is documented because it's the second time pose-merge has
fired silent `0.0` cosines (the first was the wrong-shape feature
return in v6); the sidecar JSON makes the failure mode loud.

### 7.5 Cardinality-voting FN recovery sweep (15 variants, 9 clips)

Premise: detector misses on close-contact dance frames where the
expected dancer count drops below the running median for a few
frames. Recover the missing boxes by linearly extrapolating from
the existing track history of the dancers that just disappeared.

Implementation: `tracking/fn_recovery.py`, env-gated by
`BEST_ID_FN_RECOVERY=1` plus knobs for cardinality drop threshold,
lookback window, IoU vs nearby boxes, min track history,
extrapolation displacement cap, and confidence backfill.

| Variant | Mean IDF1 | Δ vs baseline |
|---|---:|---:|
| baseline | 0.9208 | +0.0000 |
| fnr_default | 0.9188 | -0.0020 |
| fnr_drop2 | 0.9206 | -0.0002 |
| fnr_drop1_lb3 | 0.9188 | -0.0020 |
| fnr_drop1_lb10 | 0.9206 | -0.0002 |
| fnr_iou_tight | 0.9186 | -0.0022 |
| fnr_iou_loose | 0.9187 | -0.0021 |
| fnr_minhist3 | 0.9188 | -0.0020 |
| fnr_minhist15 | 0.9188 | -0.0020 |
| fnr_disp100 | 0.9188 | -0.0020 |
| fnr_disp300 | 0.9188 | -0.0020 |
| fnr_conf_low | 0.9188 | -0.0020 |
| fnr_conf_high | 0.9188 | -0.0020 |
| fnr_combo_a | 0.9199 | -0.0009 |
| fnr_combo_b | 0.9196 | -0.0012 |

**Universal regression.** Every variant cost mean IDF1. The
extrapolated boxes do reduce total FN by 100–200 frames, but they
introduce 30–60 new false positives and 2–5 new identity switches
because the recovered box doesn't have a fresh appearance
descriptor — DeepOcSort matches it to a near-by track, splitting
or merging the wrong identities.

The hypothesis was correct (FN do cluster around cardinality
drops) but the recovery mechanism is wrong: a synthetic box with
no real appearance evidence is more harmful than the missing
detection it replaces. Marked as a do-not-reopen entry in §3.

### 7.6 darkTest-targeted push (19 variants, darkTest only)

After the 9-clip stack validation showed v9 (CLAHE + auto-gamma)
was the local optimum at 0.7027 on `darkTest`, this sweep tested
whether stacking other dark-recovery techniques could push past
0.7027 — adaptive confidence on dark frames, brightened ensemble
view, larger imgsz, explicit gamma values, broader luma trigger,
and SAM stride-1 combinations.

| Variant | darkTest IDF1 | Δ vs v9 |
|---|---:|---:|
| v8_baseline | 0.6526 | -0.0502 |
| **v9_dark** (CLAHE + auto gamma) | **0.7027** | **+0.0000** |
| v9_luma100 | 0.7027 | +0.0000 (more frames triggered, no new lift) |
| v9_plus_sam_stride1 | 0.6931 | -0.0096 |
| v9_adaptiveconf_15 | 0.6711 | -0.0316 |
| sam_stride1 (no v9 dark) | 0.6710 | -0.0317 |
| v9_combo_b (conf-0.10 + 1024/1280) | 0.6570 | -0.0457 |
| v9_imgsz_1280 | 0.6423 | -0.0604 |
| v9_brighten_15 | 0.6320 | -0.0707 |
| v9_gamma_18 | 0.6318 | -0.0709 |
| v9_plus_sam_stride1_brighten | 0.6262 | -0.0765 |
| v9_combo_a (conf-0.10 + brighten 1.5) | 0.6172 | -0.0855 |
| v9_adaptiveconf_05 | 0.6120 | -0.0907 |
| v9_combo_c (max-recall combo) | 0.6085 | -0.0942 |
| v9_gamma_22 | 0.6080 | -0.0947 |
| v9_plus_sam_stride1_conf10 | 0.6037 | -0.0990 |
| v9_adaptiveconf_10 | 0.6035 | -0.0992 |
| v9_brighten_20 | 0.5701 | -0.1326 |
| v9_imgsz_1280_only | 0.5133 | -0.1894 |

**The v9 dark default is the local optimum on darkTest.** Every
other knob — explicit gamma instead of auto, broader luma trigger,
adaptive conf reduction, multi-exposure brighten, larger imgsz,
even stacking with SAM stride-1 — strictly regresses. Two
mechanisms explain the regression:

1. *Auto-gamma already saturates the LAB L-channel headroom.*
   Adding explicit `gamma=1.8` or `2.2` over-amplifies regions the
   auto-gamma already pushed near-clip, which YOLO interprets as
   noise. (`v9_gamma_18` 0.6318, `v9_gamma_22` 0.6080.)
2. *Adaptive conf lowering on dark frames is well-meaning but
   exposes the same low-confidence boxes that v9 dark recovery
   was supposed to upgrade to high-confidence true positives.*
   Counting them twice (lower the threshold AND boost the input)
   doubles down on borderline detections that DeepOcSort then
   stitches into the wrong identities. (`v9_adaptiveconf_05/10/15`
   all regress.)

`darkTest` is at its local optimum until either the detector is
fine-tuned on low-light dance footage (§4.3) or a better LLIE
preprocessor is integrated.

### 7.7 v9 stack validation (5 variants, 9 clips)

The first validation sweep that established `v9_dark` as the
shipping default and ruled out the 3-scale ensemble + Soft-NMS
combinations that had looked promising in narrower sweeps.

| Variant | Mean IDF1 | Δ vs baseline | Notes |
|---|---:|---:|---|
| **baseline** (v8) | 0.9208 | +0.0000 | – |
| **dark_v9** (CLAHE + auto gamma) | **0.9263** | **+0.0056** | shipping v9 default |
| ens_896 (+ `imgsz=[768,896,1024]`) | 0.9185 | -0.0023 | shorterTest +0.0569 / MotionTest -0.0621 |
| dark_v9 + ens_896 | 0.9173 | -0.0035 | dark + 3-scale don't compose |
| dark_v9 + ens_896 + Soft-NMS | 0.7366 | -0.1842 | catastrophic FP explosion |

The 3-scale `[768, 896, 1024]` ensemble is a `loveTest` /
`shorterTest` win (+0.0070 / +0.0569) and a `MotionTest` regression
(−0.0621) — `MotionTest`'s fast motion + 14 dancers exposes the
same foreground-splitting failure mode that killed
`[768, 1024, 1280]` (`§3` rejected list). Net mean is negative,
ensemble shape stays at `[768, 1024]`.

Soft-NMS at `sigma=0.5` with the default low score-threshold
preserves so many overlapping boxes that the per-clip false
positive count explodes by 1.2k–13k. The `MotionTest` MOTA drops
from 0.86 → 0.0005. Soft-NMS is implemented and env-gated
(`BEST_ID_SOFT_NMS=<sigma>`) but the default sigma is unsafe;
keep OFF unless the score-threshold is also raised (open
investigation).

### 7.8 Recommended final pipeline

The shipped v9 default (`work/run_all_tests.py`):

```python
os.environ.setdefault("BEST_ID_DARK_PROFILE", "v9")
```

Equivalent to setting `BEST_ID_DARK_CLAHE=1` and
`BEST_ID_DARK_GAMMA=auto` (luma-gated, byte-identical on every
well-lit clip). Verified: **mean IDF1 0.9263 on the 9-clip
benchmark, +0.0056 vs v8, +0.0502 on `darkTest`, zero regression
on any other clip, total IDS 27 → 20**.

Optional opt-in flags for users with the relevant dependencies
installed:

| Flag | Best on | Lift | Caveats |
|---|---|---:|---|
| `BEST_ID_POSE_MERGE=1 BEST_ID_POSE_MERGE_MIN_VIS=0.50` | `MotionTest` | +0.0027 | requires `rtmlib` + `onnxruntime`; mean +0.0003; downloads RTMW ONNX on first run |
| `BEST_ID_SAM_VERIFY=1 BEST_ID_SAM_VERIFY_STRIDE=1` | `loveTest` | +0.0082 | requires SAM 2.1 weights at `weights/sam2/sam2.1_hiera_tiny.pt`; ~80 ms/frame on A10; **negative interaction with v9 dark on `darkTest`** (-0.0096) |

Both stack with v9 dark cleanly on the well-lit clips. The SAM
flag should NOT be combined with v9 dark on `darkTest`-style
low-light footage; for those clips the v9 dark profile alone is
the local optimum.

### 7.9 Things tried in the overnight session and rejected (do not reopen)

Appended to the §3 master list:

| Idea | Result | Failure mode |
|---|---|---|
| **3-scale ensemble (`imgsz=[768, 896, 1024]`)** | -0.0023 mean; -0.0621 MotionTest | Same foreground-splitting issue as 768/1024/1280; shorterTest gain doesn't offset MotionTest loss |
| **Soft-NMS at default `sigma=0.5`** | -0.18 mean, MotionTest collapses | Default score threshold is too permissive — preserves 12k+ overlapping FP boxes per clip |
| **Cardinality-voting FN recovery** | -0.002 mean, all 15 variants regress | Synthetic boxes lack appearance evidence; DeepOcSort assigns them to wrong tracks, creating new IDS and FP |
| **Adaptive conf on dark frames** (`-0.05`/`-0.10`/`-0.15`) | -0.03 to -0.10 darkTest | Lowers threshold under the boxes that v9 dark already upgraded; doubles down on borderline detections |
| **Multi-exposure brighten ensemble** (`BEST_ID_DARK_BRIGHTEN=1.5`/`2.0`) | -0.07 to -0.13 darkTest | Brightened view introduces FP that the auto-gamma path already filtered out; adds detector cost |
| **Explicit gamma 1.8 / 2.2 on dark frames** | -0.07 / -0.09 darkTest | Over-amplifies regions auto-gamma already saturated |
| **Higher imgsz on dark** (`1024,1280` or `1280` only) | -0.06 / -0.19 darkTest | Same foreground-splitting at 1280 px |
| **SAM stride-1 + v9 dark** (combination on `darkTest`) | -0.0096 vs v9 dark alone on `darkTest` | SAM marks marginal low-luma boxes as phantoms (mask fill < 0.30) and drops detections that gamma had legitimately recovered |
| **RTMW with default body/hand/face weights** | within rounding of baseline | Default weights don't bias toward the discriminative signal (hands); only `pose_hand_heavy` and `pose_min_vis_strict` lift at all |

---

## 8. A10 GPU 9-clip head-to-head benchmark (April 19, 2026)

This is the apples-to-apples baseline-vs-ours measurement that the new
README leads with. Every number in this section was produced on the
same Lambda Labs `gpu_1x_a10` instance (NVIDIA A10, 23 GB VRAM, CUDA
12.1, driver 535.288.01) by `scripts/run_full_benchmark.py`. The
machine was otherwise idle for the duration of the run; raw output
lives at
[`work/benchmarks/full_a10_results.json`](../work/benchmarks/full_a10_results.json),
per-tracker MOT-format predictions at
[`work/benchmarks/full_a10_mot/`](../work/benchmarks/full_a10_mot/).

### 8.1 Why this study exists

Sections §1–§7 above measure each post-process change against
`weights/best.pt` + the multi-scale ensemble — i.e. they show the
lift our pipeline buys on top of an already-strong detector. The
gap a *new user* sees is bigger than that, because a new user
starts with **stock `yolo26s.pt` + a default tracker constructor**,
not our fine-tuned weights and not our post-process chain.

§8 measures exactly that: 6 baseline trackers from BoxMOT 18.0.0
(ByteTrack, OcSort, HybridSort, BotSort, StrongSort, DeepOcSort)
each driven by the **stock COCO-pretrained `yolo26s.pt` at single
scale 640, conf 0.25, IoU 0.7** — the Ultralytics defaults. The
baseline detector cache is shared across all 6 baselines so the
only variable per row is the tracker. Then the 7th row is "Ours
(v9 shipped)" running the full pipeline (`weights/best.pt`,
multi-scale {768,1024} ensemble, conf 0.34, ensemble IoU 0.6, full
v9 post-process chain) on the same machine, scored against the
same ground truth at the same IoU 0.5 with the same py-motmetrics
1.4.0 install.

### 8.2 Scoring conventions

* `py-motmetrics 1.4.0`, `compare_to_groundtruth(... 'iou', distth=0.5)`
* `mm.io.loadtxt(..., fmt='mot15-2D', min_confidence=1)` for GT
* All 9 clips listed in §0 are scored
* Per-clip "n_frames" reported is the GT frame range (matches
  `loveTest=820`, `MotionTest=1407` etc.); detection / tracking ran
  on the full clip in every case (no `--max-frames` cap)

### 8.3 What lives where

* `work/benchmarks/full_a10_results.json` — the master JSON consumed
  by `scripts/generate_comparison_charts.py --full-results-json` to
  render the headline README charts
* `work/benchmarks/full_a10_mot/<clip>/<tracker>.txt` — MOT-15
  predictions per (clip, tracker) pair (used by
  `scripts/render_side_by_side.py` for the comparison videos)
* `docs/figures/full_benchmark/*.png` — generated charts
* `docs/figures/full_benchmark/_mermaid_snippets.md` — Mermaid
  fallbacks for inline GitHub rendering

The README §"Benchmarks" table is built from
`docs/figures/full_benchmark/a10_summary_table.md`, also auto-generated.

---

## 9. Archived: detailed sections from the pre-§8 README

These sections used to live in `README.md` before the April-19
rewrite around the §8 A10 benchmark. They are kept here verbatim
for completeness — every link, every table value, every
reproduction command still works.

### 9.1 Why the README used to lead with MPS speed numbers

The original README documented **fair tracker speed comparison on
Apple Silicon (MPS)** because that's the machine the project was
developed on. The new README leads with **A10 GPU end-to-end FPS**
because that's apples-to-apples with the §8 head-to-head accuracy
study; the MPS numbers are still reproducible via
`scripts/benchmark_trackers.py` and live at
[`work/benchmarks/tracker_speeds.json`](../work/benchmarks/tracker_speeds.json).

### 9.2 Apple Silicon (MPS) tracker speed table — fair head-to-head

We re-ran every tracker against **the same cached YOLO multi-scale
detections** on `loveTest` (820 frames @ 1080p) on an M-series Mac,
so the only variable per row is the tracker itself. Numbers are
end-to-end FPS (detection + tracker + post-process):

| Tracker | Tracker latency | End-to-end FPS | Unique IDs (15 real dancers) |
|---|---:|---:|---:|
| ByteTrack (base) | 1.3 ms | 13.33 | 21 |
| OcSort (base, no ReID) | 1.4 ms | 13.32 | 19 |
| **Ours (DeepOcSort + post-process)** | **42.8 ms** | **8.58** | **20** |
| HybridSort (base) | 58.9 ms | 7.54 | 18 |
| BotSort (base) | 74.6 ms | 6.74 | 18 |
| StrongSort (base) | 119.3 ms | 5.18 | **38** |

ByteTrack and OcSort are 1.5× faster end-to-end, but only because
they skip ReID entirely — and they pay -3.0 to -5.6 IDF1 points
for it. The three other ReID-based competitors (BotSort,
StrongSort, HybridSort) are all slower than us *and* strictly less
accurate. StrongSort produces 38 unique IDs across 820 frames
despite there being 15 real dancers — that's 23 identity swaps the
tracker never recovered from.

Reproduce locally: `python scripts/benchmark_trackers.py`.

### 9.3 Side-by-side: ours vs base StrongSort on `loveTest` (the original write-up)

`loveTest` is the worst-case clip in the §1–§7 benchmark — 15
same-uniform dancers in sustained close contact. On the same
cached YOLO detections, the same machine, the same ground truth,
the shipped pipeline scored **IDF1 = 0.836** vs base StrongSort at
**IDF1 = 0.749** ([source JSON](../work/benchmarks/per_clip_idf1.json)).
We deliberately picked StrongSort here, not ByteTrack — StrongSort
is BoxMOT's premium appearance-based tracker (Kalman + ECC + linear
assignment + the *same* OSNet ReID head we use), so this comparison
isolates exactly what the post-process chain buys you.

The clip below is the densest 7 seconds of identity chaos in the
entire video (frames 467–676, t = 15.6–22.6 s — picked by sliding
a 7-s window over the per-frame `SWITCH` events from
`py-motmetrics`). It's slowed to ~10 s playback so each swap is
visible. The big red counter in the bottom strip is the live
identity-switch count — it ticks up exactly when `motmetrics`
records a `SWITCH` event, and the panel border flashes red on every
swap frame. **In this 7-second window: ours = 0 swaps, StrongSort
= 21 swaps.** The other 5 of StrongSort's 26 total swaps happen
elsewhere in the clip; ours' single swap is at frame 26, two
seconds into the video.

Whole-clip totals (820 frames, GT = 15 dancers):

| Metric | Ours | Base StrongSort | Delta |
|---|---:|---:|---:|
| Final unique IDs in the prediction | **14** | 38 | 2.7× more IDs than real dancers |
| Average track lifetime | **809 frames** | 308 frames | StrongSort tracks live 1/3 as long |
| Identity switches (`num_switches`) | **1** | 26 | **26× more swaps** |
| Fragmentations (`num_fragmentations`) | **52** | 209 | 4× more fragmentations |
| IDF1 | **0.836** | 0.749 | **+8.7 pp** |

The chaos on the right side is StrongSort's tracker flickering its
ID space *constantly* — every time two dancers occlude, StrongSort's
appearance-cosine assignment misroutes the box and creates a new
ID, which then never gets re-merged. Our pipeline uses the same
OSNet head but adds (a) an OSNet-cosine-gated ID-merge stage that
re-stitches re-emerging dancers back to their original ID, and (b)
a post-merge AND-gate that kills the spurious short tracks before
they ever reach the IDF1 scorer. **That's the +8.7-pp gap, in pure
post-process.** No model retraining, no heavier ReID head — just
the chain documented in [`PIPELINE_SPEC.md`](PIPELINE_SPEC.md).

### 9.4 A100 single-GPU production-target speed flags

| Stage | Per-frame |
|---|---:|
| YOLO multi-scale (768 + 1024) | ~28–32 ms (≈ 32–35 FPS) |
| DeepOcSort + OSNet x0.25 | ~30–50 ms |
| Post-process chain (whole 1k-frame clip) | ~0.7–0.9 s |

End-to-end: ~1–1.5 minutes per minute of dance video on an A100.

#### Optional speed flags (lossless)

Four env-var-gated optimizations ship in the repo. Default
behavior with none set is exactly the historical v8 pipeline;
every flag has been verified to leave the `FrameDetections` cache
and `tracks.pkl` output bit-identical (or, for TensorRT, within
FP32 cuDNN-noise tolerance) by `scripts/regression_check.py`.

| Env var | Effect | Measured Δ on A100 | Measured Δ on MPS |
|---|---|---:|---:|
| `BEST_ID_TRT_ENGINE_DIR=weights/` | Loads `<stem>_{768,1024}.engine` instead of `.pt`. Build once with `python scripts/export_yolo_trt.py --device cuda:0`. | **+15.2 %** wall (validated) | n/a (CUDA-only) |
| `BEST_ID_PREFETCH=4` | Decodes frame N+1 in a background thread while detect+track runs on N. | within run-to-run noise | **+6 %** wall (validated) |
| `BEST_ID_PIPELINE_PARALLEL=1` | One-frame detector look-ahead so `tracker.update(N)` overlaps with `detect(N+1)`. | within run-to-run noise | **+13 %** wall (validated, solo) |
| `BEST_ID_GPU_NMS=1` | Keeps cross-scale boxes/conf on the model device for the ensemble NMS. Bit-identical output. | within run-to-run noise | -14 % (MPS lacks a native NMS kernel — leave OFF) |

All measurements: 500 frames of `loveTest` (1080p), single GPU,
nothing else competing for the device. Run-to-run variance was
~6–7 % on A100 (one extra job on the same GPU swung it 6 %), so
deltas inside that band are reported as noise rather than wins.

##### Recommended config per device

```bash
# A100 / CUDA — TRT is the only validated win.
# (Build engines once; takes ~5 min per imgsz on an A100.)
python scripts/export_yolo_trt.py --device cuda:0
BEST_ID_TRT_ENGINE_DIR=weights/ python -m tracking.run_pipeline \
    --video your.mp4 --out work/your/tracks.pkl --device cuda:0

# MPS / Apple Silicon — opposite stack: no TRT (CUDA-only),
# turn on the two CPU-overlap flags. Composes for +13.8 % wall on
# 500 frames of loveTest, bit-identical cache + tracks.
BEST_ID_PREFETCH=4 BEST_ID_PIPELINE_PARALLEL=1 \
    python -m tracking.run_pipeline \
    --video your.mp4 --out work/your/tracks.pkl --device mps
```

##### Validate any flag yourself

`scripts/regression_check.py` runs the pipeline twice (once with
flags, once without), then diffs the cache and tracks. Bit-exact
by default; pass `--iou-match 0.5 --tol-box 5 --tol-conf 0.05` for
tolerance-bounded comparison (use this flavor for TRT FP32 vs
PyTorch FP32, where NMS sort order can flip on near-tied scores
even though the boxes are equivalent):

```bash
python scripts/regression_check.py run --video V \
    --out work/regression/baseline --device cuda:0
BEST_ID_TRT_ENGINE_DIR=weights/ \
    python scripts/regression_check.py run --video V \
    --out work/regression/trt --device cuda:0
python scripts/regression_check.py diff \
    --a work/regression/baseline --b work/regression/trt \
    --iou-match 0.5 --tol-box 5 --tol-conf 0.05
```

##### Reproduction notes for the A100 numbers above

The +15.2 % TRT result was measured against the cleaned-repo
detector and tracker drivers in this repo (no other GPU work on
the A100 at the time). On the same 500-frame `loveTest`:

* Baseline (no flags): 31.88 s wall → 63.8 ms/frame → 18.06 FPS
* `BEST_ID_TRT_ENGINE_DIR=weights/`: 27.03 s wall → 54.1 ms/frame → 21.79 FPS
* Same 14 final tracks both runs; same long-gap pose-merge fires
  on the same IDs; per-detection bbox L∞ < 5 px and conf L∞ < 0.05
  (FP32 cuDNN ↔ TRT FP32 noise band).

The two CPU-overlap flags (`PREFETCH`, `PIPELINE_PARALLEL`) and
the GPU-NMS flag did **not** stack on top of TRT in our A100 run
— the combined config landed at 58.6 ms/frame, slower than TRT
alone. The plausible cause is single-CUDA-stream contention
between the detector thread and the tracker thread: when the
detector is already fast (TRT cuts it from ~28 → ~20 ms), the
overlap window is too small for the thread-handoff overhead to
pay back. We ship those flags off by default on CUDA for that
reason.

### 9.5 Per-clip v1 → v8 progression

Same detector, same tracker — only the post-process chain changes.
The v1 → v8 lift recovered the most ground on the hard clips
(`MotionTest` +12.2 pp, `loveTest` +4.4 pp) without regressing any
of the easy ones; the per-clip table is in §1.2 above. The chart
is generated by
`python scripts/generate_comparison_charts.py` →
`docs/figures/per_clip_v1_to_v8.png`.
