I have enough context. Now let me compose the deep research report.

---

# Deep Research Report: Non-SAM / Non-RTMW IDF1 Improvements for `loveTest` and `darkTest`

**Pipeline anchor.** Production v8 (mean IDF1 0.957 across 7 clips). Five-stage post-process on top of multi-scale YOLO26s + DeepOcSort/OSNet x0.25. Driver in `tracking/run_pipeline.py:249-286`; constants in `tracking/best_pipeline.py:60-120`. All proposals below explicitly avoid the 26 entries in `EXPERIMENTS_LOG.md §3` "Things tried and rejected" unless they specifically address that idea's documented failure mode.

**Operating envelope.** A10 24 GB. darkTest = 24 fps × 23 s × 720p ≈ 552 frames, ~14 dancers. loveTest dominant residual: **2,083 FN vs 1,411 FP at IDF1 0.853** (`EXPERIMENTS_LOG.md §2.4`).

---

## Section 1 — Low-light handling (darkTest)

darkTest is a domain-shift problem, not a tracker problem. Three places enhancement can plug in:

- **(P1) Pre-detector** — only path that recovers detector confidence. Risk: changes pixel statistics, can cost mAP on well-lit clips.
- **(R1) Per-frame ReID crop only** — only path that improves OSNet embeddings. Risk: zero (well-lit crops untouched if gated).
- **(B1) Both** — full effect, double the cost.

Anything we add must be **gated by frame mean luma** (e.g. `Y < 70/255`) so the well-lit clips run the original code path bit-for-bit and the no-regression rule is structurally preserved.

### 1.1 Adaptive CLAHE (per-channel, LAB L-channel) — strategy A

| Field | Value |
|---|---|
| **Description** | Convert BGR → LAB; apply OpenCV `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` to L; convert back. |
| **Where to apply** | **B1**. Pre-detector: helps YOLO26s on dark dancers. Per-ReID-crop: helps OSNet (empirical: OSNet was trained on MSMT17, mostly well-lit). |
| **Detector impact** | +5-10 % recall on near-threshold detections (dark frames). Side effect: possible halos around dancers in already-bright regions, which would defeat purpose. |
| **ReID impact** | Embeddings stabilize across in-clip lighting variance (the 14 dancers may not all be equally dark). |
| **Latency** | ~0.5 ms / frame at 720p on CPU; <0.1 ms on GPU via OpenCV CUDA. **~280 ms total for darkTest.** |
| **Risk** | LOW with luma gate. CLAHE on a bright frame slightly flattens contrast but doesn't shift colors; if gated on `Y < 70`, **zero effect on `easyTest` etc.** |

### 1.2 Gamma correction with adaptive γ — strategy B

| Field | Value |
|---|---|
| **Description** | `out = ((in/255) ** (1/γ)) * 255` with γ chosen per-frame: γ = 1 + clamp((90 − Y_mean)/30, 0, 1.5). So γ ∈ [1.0, 2.5], no-op on bright frames. |
| **Where to apply** | **P1** primarily; ReID gets no benefit from monotonic gamma. |
| **Latency** | LUT-based: ~0.1 ms / frame on CPU. Negligible. |
| **Risk** | LOW (luma-gated, monotonic). |
| **Expected** | Smaller than CLAHE; gamma boosts dark pixels but doesn't add local contrast — useful when the issue is global underexposure. |

### 1.3 Multi-exposure ensemble — strategy C ★

| Field | Value |
|---|---|
| **Description** | Mirror the existing multi-scale pattern: detect at native frame **and** at γ=2.0 brightened version. Concatenate boxes, NMS-union at `ensemble_iou=0.6` (same as cross-scale fusion). |
| **Where to apply** | **P1**. Slot directly into `tracking/multi_scale_detector.py:165-241` `_detect_legacy` after the per-scale loop, mirroring the `tta_flip` block at lines 190-213. |
| **Detector impact** | This is exactly the documented mechanism behind the 768+1024 win — same backbone, different input statistics, NMS-union catches the disjoint set of recoveries. The brightened view recovers dark dancers; the native view preserves precision on already-visible ones. |
| **Latency** | 2× detector forward when triggered. Native detector latency = 28-32 ms / frame. With ~50 % darkTest frames triggered: +14-16 ms / frame avg. **Total darkTest: ~552 × 16 = +9 s wall.** |
| **Risk** | Same risk profile as the rejected `tta_flip` (which was 2× detector cost for noise-band lift). Difference: multi-exposure is *gated* on darkness so the 2× cost only fires when there's a recovery to be made. |
| **Why this is the strongest classical option** | The pipeline's whole detector design philosophy is "let NMS-union pick the right detection from a disjoint set of failure modes." Multi-exposure is a literal copy of that pattern in the brightness axis. |

### 1.4 Zero-DCE++ (learned, zero-reference) — strategy D

| Field | Value |
|---|---|
| **Description** | Tiny LLIE network (10 K params, 0.115 G FLOPs). Outputs an iterative tone curve. **Zero-reference**: trained on unpaired well-lit data only, no need for paired darkTest GT. |
| **Where to apply** | **B1**. Pre-detector for FN recovery; on ReID crops for embedding stability. |
| **Detector impact** | Published Zero-DCE++ → +mAP downstream on EXDark dataset; expect comparable lift on darkTest. |
| **ReID impact** | OSNet embeddings on uniformly-relit crops should be more stable than on raw dark crops (see CENet, 2024). |
| **Latency** | ~1 ms / 720p frame on A10 (1000 FPS at 1200×900 on a 2080 Ti per the Zero-DCE++ paper; A10 is comparable). **Total: ~0.6 s for darkTest.** |
| **Risk** | LOW with luma-gating. Zero-DCE++ is well-known not to introduce halos or color shifts because it's monotonic per channel. Add a `weights/zero_dce_pp.pt` (~50 KB) to the repo; load it lazily inside `make_multi_scale_detector` when env var `BEST_ID_LLIE=zerodce` is set. |
| **No-regression** | Run `scripts/regression_check.py` on `loveTest` with `BEST_ID_LLIE=zerodce` enabled but luma-gated. Bright frames bypass the network entirely. |

### 1.5 Retinexformer (heavier, transformer-based) — strategy E

| Field | Value |
|---|---|
| **Description** | ICCV 2023 / NTIRE 2024 runner-up. One-stage Retinex transformer; published has a downstream "low-light object detection" application section that demonstrates +mAP on EXDark / DARK FACE. |
| **Where to apply** | **P1**. ReID crops are small (~100×200 px) so the heavier IGT inside Retinexformer is overkill there — Zero-DCE++ is the right choice for crops. |
| **Latency** | ~25 ms / 720p frame on A10 for the small ECCV-2024 variant. **Total: ~14 s** if every darkTest frame is enhanced; ~7 s if luma-gated to the dark half. |
| **Risk** | MEDIUM. Retinexformer is a real generative network; it can hallucinate color/shape on out-of-distribution dark frames. Validate on a held-out frame from each well-lit clip with the env-var off (default) and on (gated). |
| **Versus Zero-DCE++** | Retinexformer is +3 dB PSNR over Zero-DCE++ on LOL benchmarks. The downstream gain on detection mAP is **smaller than the PSNR gap suggests** (0.5-1.5 mAP, in published tables). For our use case, **prefer Zero-DCE++ for Tier 1 trial** because of the 25× lower latency; only escalate to Retinexformer if Zero-DCE++ doesn't move darkTest IDF1. |

### 1.6 Adaptive detector confidence — strategy F ★

| Field | Value |
|---|---|
| **Description** | Per-frame: if `Y_mean(frame) < 80`, set `conf = 0.34 - 0.06 = 0.28` for that single frame. Recovers low-confidence detections that the global 0.34 plateau threshold blocks. |
| **Where to apply** | Inside the per-frame detector loop in `tracking/multi_scale_detector.py:171-188` (the `model.predict(...)` call). YOLO26s' `conf` kwarg can be set per-call. |
| **Detector impact** | +recall on dark frames at cost of FP. Critically: the FP rate **on a dark frame** is still bounded by the multi-scale NMS-union and the downstream AND-gate (`tracking/best_pipeline.py:90-92`, `len≥60 ∧ mean≥0.55 ∧ p90≥0.84`). FPs on individual dark frames are absorbed by the p90 gate as long as they don't form 60-frame phantom tracks. |
| **Latency** | ZERO. Just a different scalar. |
| **Risk** | LOW. The conf-sweep table at `EXPERIMENTS_LOG.md §2.2` shows MotionTest is unstable to conf changes — but that's *globally* changing conf. *Per-frame, gated on darkness*, only frames that have no representation in well-lit clips are affected. **Strict no-regression guaranteed by the gate.** |
| **Why I rate this near top** | Free + leverages the existing AND-gate as the FP backstop. Should be the *first* darkTest-specific knob tried. |

### 1.7 Detector fine-tune on a darkTest-like subset — strategy G

| Field | Value |
|---|---|
| **Description** | Brief fine-tune on a few hundred darkTest-style frames with pseudo-GT from the brightest pass. Steps: run multi-exposure ensemble (1.3), accept the union as pseudo-GT for training, fine-tune `weights/best.pt` for 5-10 epochs at low LR (1e-5) on this subset only. |
| **Where to apply** | Replace the model loaded in `tracking/multi_scale_detector.py:155` with the dark-aware checkpoint. Or ship as `weights/best_dark.pt` and gate via env var `BEST_ID_DETECTOR=dark`. |
| **Data needed** | 200-500 frames, 5-15 dancers each, weak labels from the multi-exposure pass at conf ≥ 0.5. |
| **Latency** | ZERO at inference. |
| **Risk** | MEDIUM. Fine-tuning on weak labels can drift YOLO toward the brightened domain and hurt well-lit clips. **Mitigation:** mix 50/50 with a frozen subset of original training data; validate strict-no-regression on 7-clip benchmark before shipping. The published precedent (`EXPERIMENTS_LOG.md §2.1`) is "fine-tuning on dance >> larger backbone" so this is the highest-ceiling option overall, but slowest to set up. |

### 1.8 White-balance normalization — strategy H

Per-frame Gray-World or learned-white-point WB before detection. Fixes color casts (yellow stage lights) without changing brightness. ~0.2 ms / frame. **Low expected lift** unless darkTest has a strong color cast — please inspect a sample frame to decide.

### 1.9 Per-clip ReID-cosine threshold tuning (degenerate-embedding regime) — strategy I

In low-light the OSNet embedding manifold collapses (all-dark crops are similar). The `pp_id_merge_osnet_cos_thresh` global sweep was rejected (zero effect, `EXPERIMENTS_LOG.md §3`) because *spatial-passing candidates also pass ReID in the well-lit case*. **In the dark-embedding case, this assumption breaks** — too many spatially-plausible candidates pass the cosine gate (false merges) or none do (missed merges).

Implementation: in `tracking/best_pipeline.py:283`, replace the constant with a per-clip value driven by an embedding-degeneracy heuristic (e.g. mean pairwise cosine of all detection embeddings on first 30 frames; if > 0.85 the embedding is degenerate, raise threshold from 0.7 → 0.85 to compensate).

### 1.10 Ranked recommendation for darkTest

| Rank | Strategy | Cost | Expected lift | Why first |
|---|---|---|---|---|
| 1 | **F** Adaptive conf (luma-gated, 0.34→0.28) | 0 ms | +0.005 to +0.020 darkTest IDF1 | Zero risk, zero latency, leverages existing AND-gate |
| 2 | **C** Multi-exposure ensemble (gated) | +9-16 s wall | +0.005 to +0.025 darkTest IDF1 | Mirrors the proven multi-scale design pattern |
| 3 | **A** CLAHE on LAB-L (B1, gated) | +0.5 s wall | +0.003 to +0.015 darkTest IDF1 | Classical, bounded risk, helps both detector AND ReID |
| 4 | **D** Zero-DCE++ (B1, gated) | +0.6 s wall | +0.005 to +0.020 darkTest IDF1 | Strongest learned LLIE per ms; tiny weight; clean integration |
| 5 | **G** Detector fine-tune | days of setup | +0.010 to +0.030 darkTest IDF1 | Highest ceiling; needs labels; slowest path |
| 6 | **E** Retinexformer (P1, gated) | +7-14 s wall | marginal over D | Only if D fails |

Tier 1 (try first, cheap & non-overlapping): **F + C** — combinable, both gate on luma, total +9-16 s wall budget, no GPU memory pressure on A10.

---

## Section 2 — FN recovery for `loveTest` (close-contact occlusion)

The dominant v8 loveTest error is **2,083 FN vs 1,411 FP**: detector misses dancers when partial occlusion drops single-detection conf below 0.34 *or* when two dancers fuse into one detection and the fused box doesn't survive cross-scale NMS. Eight strategies, ranked by expected impact and integration ease.

### 2.1 Per-frame cardinality voting + propagated synthetic detections ★★★

**Core idea.** Estimate active dancer count per second; if a frame's detection count drops by ≥ 2 from the running median, kick in a "Kalman propagation" recovery pass.

**Mechanism.** For each confirmed track lost in frame N:
1. Predict bbox at N from Kalman state (DeepOcSort already maintains this internally; OC-SORT's "observation-centric re-update" — Deep OC-SORT paper, ICIP 2023 — does exactly this concept).
2. Accept the predicted box as a synthetic detection iff:
   - No other detection in frame N has IoU > 0.30 with the prediction (no double-count).
   - The Kalman state's covariance trace is below a threshold (track is well-tracked, not already drifting).
   - The track has been alive ≥ k = 10 frames.
3. Tag the synthetic detection with conf = 0.35 (just above gate) and let it propagate normally.

**Where to integrate.** Add a post-tracker hook after `tracker.update(...)` returns at `tracking/run_pipeline.py:159` (`_safe_tracker_update`), before `_record_tracker_output` at line 177. Operate on `tracks_per_frame` to inject synthetic rows for missing well-tracked dancers.

**Expected lift.** This directly attacks the FN dominance. On loveTest where median active count is ~14 and the FN budget is 2,083 / 820 frames ≈ 2.5 missing/frame, recovering even 60 % of those is ~1,250 fewer FN. **Expected loveTest IDF1: 0.853 → 0.875-0.895.**

**Risk.** MEDIUM. Synthetic detections can persist a wrong identity through occlusion (chaining a Kalman ghost). **Mitigation:** require an OSNet ReID match on the *next real* detection of that track before committing the synthetic frames to the cache. Implementation: write the cache normally, then have a "synthetic frame validator" pass that drops synthetic frames if the next real detection's ReID cos to the synthetic-frame-extrapolated pre-occlusion crop is < 0.7.

**Why this is the #1 priority.** It's the only proposal here that operates on the *gap between tracker.update and cache*, where the Kalman predictions already exist for free. Incremental code, no new model, no GPU cost.

### 2.2 imgsz=896 added to the multi-scale ensemble ★★

**Core idea.** The 1280 attempt failed because YOLO splits foreground dancers (`EXPERIMENTS_LOG.md §3`, -0.073 loveTest). The shipped (768, 1024) misses the medium scale where loveTest's close-contact pairs sit. **896 fills exactly that gap** without crossing into the splitting regime.

**Where to integrate.** `tracking/run_pipeline.py:119` change `DETECTOR_IMGSZ_ENSEMBLE = (768, 1024)` to `(768, 896, 1024)`. The detector at `tracking/multi_scale_detector.py:142` already accepts arbitrary lists.

**Expected lift.** +0.002 to +0.008 mean IDF1, with most concentrated on loveTest. The 1280 failure mode (splitting) was specific to 1280; 896 is between the working 768 and 1024 so should not introduce that.

**Latency.** +50 % detector forward → +12-18 ms / frame on A10. Total ~+15 s wall on loveTest 820 frames.

**Kill criterion.** If loveTest IDF1 lift < 0.003 after a single full sweep with 896 added, abandon — the headroom in the multi-scale axis is exhausted.

### 2.3 Soft-NMS for cross-scale ensemble fusion ★

**Core idea.** Currently the cross-scale NMS at `tracking/multi_scale_detector.py:233` (`torchvision.ops.nms`) hard-rejects boxes with IoU > 0.6 against a higher-scoring box. In a close-contact scene, two real dancers can have IoU = 0.7 (overlapping torsos), and one is silently dropped. Soft-NMS replaces the binary keep/drop with a continuous score multiplier `s_i' = s_i × exp(-IoU²/σ)` so a high-IoU box stays alive at reduced confidence and can survive the AND-gate downstream.

**Where to integrate.** Replace `nms` call at `tracking/multi_scale_detector.py:233`, also at `:301` for the GPU-NMS path. Use the standard Soft-NMS formula with σ=0.5 (Bodla et al., ICCV 2017; the canonical reference is the Soft-NMS paper).

**Expected lift.** +0.001 to +0.005 loveTest. Smaller than 2.1/2.2 because the AND-gate is conservative; many soft-NMS-saved boxes won't form long enough tracks.

**Risk.** LOW. Mathematically a strict superset of hard NMS (σ→0 → hard NMS).

**Caveat.** May **increase** FP slightly. Since loveTest already has 1,411 FP, any solution that increases FP needs to be balanced against FN reduction. Soft-NMS is precisely the trade-off knob: tune σ on loveTest to find the FN/FP optimum.

### 2.4 Persistence-based low-confidence promotion ★★

**Core idea.** Run the detector at `conf=0.15` (much lower than 0.34) but **only emit detections that fall inside the predicted Kalman gate of an existing confirmed track for k consecutive frames**. This is the OccluTrack 2024 mechanism (CBKF — confidence-based Kalman filter) and the OC-SORT virtual-trajectory idea, adapted to our two-stage detect→track separation.

**Implementation.**
1. Add a "shadow" detector path that re-runs YOLO with `conf=0.15` only when the main detector returns < median-1 detections.
2. For each shadow detection, check IoU with all active track Kalman predictions. If IoU > 0.4 with exactly one prediction, mark as "candidate-recovery."
3. After k = 5 consecutive frames of consistent candidate-recovery for the same track, promote into the cache stream as conf = 0.36.

**Where to integrate.** New module `tracking/persistence_recovery.py`; called from `tracking/run_pipeline.py` between detector and tracker. Has access to tracker's predicted states via DeepOcSort's internal Kalman objects (BoxMOT exposes `tracker.trackers[i].kf` post-update).

**Expected lift.** +0.003 to +0.010 loveTest. Higher upper bound than 2.3 because it explicitly chases the FN axis.

**Latency.** Conditional shadow pass: ~+15 ms / frame when triggered. ~50 % of loveTest frames trigger → +6 s wall.

### 2.5 Multi-resolution (half-res scout + full-res confirm) ★

**Core idea.** Run a fast half-res (384px) detection to enumerate all candidate locations, then re-run the existing 768/1024 ensemble cropped to ROIs from the scout. Catches small/distant dancers that even the 1024 misses, and concentrates compute on real ROIs.

**Where to integrate.** Replace the per-frame loop body of `tracking/multi_scale_detector.py:171-188`. Need to add a 384 model_by_scale entry.

**Expected lift.** +0.001 to +0.005 mean. Probably more useful for clips with size variance (BigTest) than loveTest where dancers are uniformly close.

**Latency.** Net neutral or slight win because the 384 pass is ~5 ms and the cropped 1024 is much faster than full-frame 1024.

**Risk.** MEDIUM. ROI cropping can chop a dancer at the boundary if the scout box is tight. Mitigate by 30 % padding around scout boxes before re-detecting.

### 2.6 Test-time augmentation that wasn't horizontal flip ★

The horizontal-flip TTA is in noise band (`EXPERIMENTS_LOG.md §3`, "implemented, deltas in noise band"). Try alternates that are NOT horizontal symmetric:

- **Vertical translation TTA**: shift frame ±20 px vertically, detect, un-shift. Helps with dancers near image edges on jump moves.
- **Small rotation TTA**: rotate ±5°, detect, un-rotate boxes via affine inverse. Helps with dancers in tilted poses (lifts, dips).
- **Brightness multiplication TTA**: ×0.85 and ×1.15 versions (cheap, similar to multi-exposure 1.3 but tighter range).

**Where to integrate.** Same as `tta_flip` block at `tracking/multi_scale_detector.py:190-213`, parameterized by transform type.

**Expected lift.** +0.001 to +0.004 loveTest. Each TTA is 1× detector, so 2 alternates = 4× cost. **Recommend brightness ×0.85/×1.15 only**, since it's the most likely mechanism to recover the partial-occlusion FN (occluded torsos look darker because of the occluder's shadow).

### 2.7 Two-stage detection: candidate region classifier ★

**Core idea.** A second-stage binary classifier ("is this a person?") on candidate regions YOLO emits at low conf (0.15-0.30). Use a tiny ResNet18 trained on positive samples from the high-conf detections of the same clip plus negatives from background regions.

**Cost.** Single fp16 forward of a tiny CNN (~1 ms / candidate). At ~50 candidates / frame in low-conf regime, ~50 ms / frame additional. **Probably too expensive for the marginal lift expected (+0.001 to +0.005).**

**Verdict:** lower priority than 2.1/2.2/2.4.

### 2.8 Detector confidence calibration (temperature scaling) — per-clip

**Core idea.** YOLO confidences are not calibrated. Apply temperature scaling: `p_calibrated = sigmoid(logit(p) / T)` with T fitted on a held-out validation subset to minimize ECE. Recent work (Trakulwaranont et al. 2025, "Determining the Optimal T-Value...for YOLO-World") cuts ECE from 6.78 % → 2.31 % with T scaling alone.

**Per-clip variant.** Fit T per-clip using detector's own self-consistency: take the highest-scoring 20 % of detections as pseudo-positives, lowest 20 % as pseudo-negatives, fit T with ML estimation on this self-supervised split. **No GT needed at inference**, satisfying the per-clip-without-GT constraint the user noted in §4.

**Where to integrate.** Wrap the detector callable returned from `make_multi_scale_detector` to apply T to the conf column. T computed on first 100 frames of the clip and frozen.

**Expected lift.** Indirect — improves the AND-gate's ability to discriminate phantom from real (current p90 ≥ 0.84 gate at `tracking/best_pipeline.py:92` is a calibration gate). Probably +0.001 to +0.005 across hard clips.

**Risk.** MEDIUM. Recalibrated p90 means the 0.84 threshold needs to be re-swept per clip. Better integrated *with* a per-clip threshold tuner (proposal 4.3).

### 2.9 Custom loveTest fine-tune on hard 30-second segments ★

**Core idea.** Identify the FN-densest 30-second window in loveTest (e.g. via the per-frame `SWITCH` events the README already does), generate pseudo-GT from a stronger ensemble (multi-exposure + 896 + Soft-NMS), and fine-tune `weights/best.pt` for 5 epochs on that window's 720 frames + 50 % well-lit baseline data.

**Cost.** 30 min training on A10. Zero inference cost.

**Expected lift.** +0.005 to +0.020 loveTest. Same magnitude expected as the original "fine-tuning on dance >> larger backbone" win at `EXPERIMENTS_LOG.md §2.1`.

**Risk.** MEDIUM. Overfitting to loveTest's specific dancers/lighting can hurt other clips. **Mitigation:** validate strict-no-regression on the 7-clip benchmark before promoting `weights/best.pt`.

### 2.10 Ranked recommendation for FN recovery

| Rank | Strategy | Cost | Expected loveTest lift | Why this order |
|---|---|---|---|---|
| 1 | **2.1** Cardinality-voting Kalman synthetic detections | code only | +0.020 to +0.040 | Direct attack on FN; uses existing Kalman state |
| 2 | **2.2** imgsz=896 added | +15 s wall | +0.003 to +0.010 | Cheap, mirrors proven (768,1024) win |
| 3 | **2.4** Persistence-based low-conf promotion | +6 s wall | +0.003 to +0.010 | Robust to FP via k=5 persistence |
| 4 | **2.9** Custom loveTest fine-tune | 30 min train | +0.005 to +0.020 | Highest ceiling but needs validation |
| 5 | **2.3** Soft-NMS cross-scale | code only | +0.001 to +0.005 | Cheap, but may add FP |

Tier 1 (try first, free): **2.1 + 2.2** — independent, additive.

---

## Section 3 — ID-merge improvements (no SAM/RTMW)

The post-process knob space is documented exhausted (`EXPERIMENTS_LOG.md §2.4` 2-D grid sweep). Headroom requires *new signals*, not retuning existing ones.

### 3.1 Per-track motion fingerprint (Fourier + DTW) ★

**Core idea.** Each dancer has a unique rhythmic signature — phase, amplitude, dominant frequency of (cx, cy) trajectory over a 60-frame window. When OSNet appearance is degenerate (same uniform), motion fingerprint is the orthogonal signal.

**Algorithm.**
1. For each track, compute FFT of (cx[t], cy[t]) over a sliding 60-frame window. Take magnitude spectrum, top-3 peaks → 6-D motion fingerprint.
2. In the ID-merge step at `tracking/postprocess.py:331-456` `_id_merge`, for candidates passing IoU + ReID, additionally require motion fingerprint distance < threshold OR boost score by motion match.
3. Alternative (simpler): Dynamic Time Warping distance between A's tail trajectory and B's head trajectory, projected forward across the gap.

**Where to integrate.** New helper `_motion_fingerprint(track)` in `tracking/postprocess.py`. Add as a third gate after the OSNet cosine check at `tracking/postprocess.py:406-411`.

**Expected lift.** +0.001 to +0.008 on loveTest. Stronger when ReID is degenerate (also helps darkTest).

**Reference.** Gait-recognition Fourier descriptor literature (Boulgouris & Chi 2007; Cunado et al. 2003) — well-established that 6-10 Hz periodic components are ID-discriminative.

**Risk.** LOW. Used only as additive evidence, never as sole rejection. The OSNet cos≥0.7 gate remains the gate of last resort.

### 3.2 Bidirectional Kalman + RTS smoother on bbox centers ★ (already in §4 of EXPERIMENTS_LOG)

**Core idea.** Forward-only Kalman lagged the median (`EXPERIMENTS_LOG.md §2.8`, -0.0027 IDF1). The Rauch-Tung-Striebel smoother does forward filter + backward smoother in a batch pass; eliminates the phase lag. Result is closer to centered median but with explicit motion-model dynamics, so it interpolates across detection gaps better.

**Algorithm.** Constant-velocity model on (cx, cy, vx, vy). Forward pass produces filtered estimate at each frame; backward pass re-smooths using future evidence. Standard textbook result; numerically stable in float64.

**Where to integrate.** Replace or complement `smooth_centers_median` at `tracking/best_pipeline.py:205-247`. Could be a new stage 7b, sequential after the median.

**Expected lift.** +0.001 to +0.005 mean (per `EXPERIMENTS_LOG.md §4.6` open experiment).

**Risk.** LOW. Reversible (controlled by env var or config flag). 21-frame median already gives most of the benefit; RTS adds principled gap-handling.

**Reference.** Rauch, Tung & Striebel (1965); Invariant RTS Smoother (arXiv:2403.00075, 2024).

### 3.3 Per-track HSV color histogram (interior-masked) ★★

**Core idea.** Even with same uniforms, dancers have **different skin tones, hair colors, accessories**. An HSV histogram of the bbox interior (cropped to the inner 60% to exclude background) is a cheap appearance descriptor *orthogonal to OSNet's learned embedding* — it captures colors OSNet may have learned to ignore (since OSNet was trained for generalization, not a specific scene).

**Algorithm.**
1. Per detection: crop bbox, take interior 60% rectangle (avoid background), compute 8×8×8 HSV histogram, L1-normalize.
2. Per track: store running mean histogram across all detections.
3. In `_id_merge` at `tracking/postprocess.py:331-456`, add chi-squared distance between A's tail histogram and B's head histogram as an additive gate.

**Where to integrate.** New per-detection histogram stored in `RawTrack` at `tracking/postprocess.py:39-47`. Compute in the detector→tracker bridge in `_record_tracker_output` at `tracking/run_pipeline.py:126-147`. Need access to the source frame, which requires plumbing `frame_bgr` through the cache. **Or** compute histograms lazily during `build_tracks` if the frame source is reachable.

**Expected lift.** +0.003 to +0.012 loveTest. Higher because dance lighting on a same-uniform group accentuates per-dancer skin/hair variation more than uniform variation.

**Risk.** LOW. Additive evidence only.

**Latency.** ~0.05 ms per detection on CPU.

### 3.4 Cross-track exclusion in ID-merge ★

**Core idea.** Currently `_id_merge` greedily picks the best (A, B) pair (`tracking/postprocess.py:413-425`) but does not check whether some other living track C exists in the same window competing for B's identity. If C also has high cos to B but isn't picked, C might be the true continuation of B.

**Algorithm.** When considering merge A→B, compute also `cos(C, B)` for every other track C alive in the gap window. Reject the A→B merge iff some C exists with `cos(C, B) > cos(A, B)` AND `C` is geometrically plausible (predicted center within R px of B's first center).

**Where to integrate.** In `_id_merge` at `tracking/postprocess.py:381-415`, after scoring candidate B for A, do a competitive check across all other A' before accepting.

**Expected lift.** +0.001 to +0.005. Specifically helps loveTest where 15 dancers create dense merge graphs.

**Risk.** LOW. Strictly conservative — only blocks merges that are ambiguous.

### 3.5 Trajectory-continuity score in id_merge candidate ranking ★

**Core idea.** Currently the merge score at `tracking/postprocess.py:413` is `iou + 0.5 * cos_sim`. Add a third term: extrapolated-velocity match. If A's tail velocity vector is (vx, vy) and B's head velocity is (vx', vy'), include `cos(velocity_A, velocity_B)` as an additional ranker.

**Where to integrate.** Modify the score computation at `tracking/postprocess.py:413`. Use velocities computed in the bbox-stitch helper at `tracking/bbox_stitch.py:125-135`.

**Expected lift.** +0.001 to +0.003. Probably most useful when IoU is borderline (0.10-0.20 range, exactly the v8 plateau).

**Risk.** LOW. Re-ranking only.

### 3.6 Per-clip OSNet cos threshold tuning (clip-adaptive)

The global `pp_id_merge_osnet_cos_thresh` sweep is 0-effect (`EXPERIMENTS_LOG.md §3`) — but **per-clip** is structurally different. The mechanism the user describes (darkTest's degraded embeddings) is real.

**Heuristic.** Compute the all-pairs cosine distribution of detections in the first N frames. Set the threshold at percentile 20 of that distribution. In a degenerate-embedding regime (everyone looks similar), the percentile-20 cosine is high, automatically raising the gate to compensate.

**Where to integrate.** In `tracking/best_pipeline.py:283`, replace the constant with a function call that reads from a small per-clip JSON (`work/<clip>/cos_thresh.json`) populated by a new `scripts/calibrate_cos.py`.

**Expected lift.** +0.001 to +0.010 on **darkTest specifically** (where the mechanism applies). Near-zero on well-lit clips because the embedding manifold is healthy.

**Risk.** LOW with the percentile-based heuristic — well-lit clips will produce a percentile-20 close to 0.7, so behavior is unchanged.

### 3.7 Re-Optuna with darkTest + MotionTest in the loss

Already noted in `EXPERIMENTS_LOG.md §4.5`. Worth listing for completeness.

**Risk note.** Per-clip Optuna oracle (uses GT at inference) **underperforms our global config** at 0.948 (`EXPERIMENTS_LOG.md §3`). This implies the heuristic-tracker family is ceilinged. Re-Optuna should be considered a small-tweak experiment, not a strategic move. Expected: +0.0005 to +0.005 mean.

### 3.8 Ranked recommendation for ID-merge

| Rank | Strategy | Code surface | Expected lift |
|---|---|---|---|
| 1 | **3.3** HSV histogram per-track | medium (new field in RawTrack + frame access) | +0.003 to +0.012 loveTest |
| 2 | **3.1** Motion fingerprint (FFT/DTW) | medium | +0.001 to +0.008 |
| 3 | **3.2** RTS bidirectional Kalman | small (replaces smoother) | +0.001 to +0.005 |
| 4 | **3.4** Cross-track exclusion | small | +0.001 to +0.005 |
| 5 | **3.6** Per-clip cos threshold | small | +0.001 to +0.010 darkTest |
| 6 | **3.5** Trajectory-continuity score | trivial (one line) | +0.001 to +0.003 |

---

## Section 4 — Pipeline-level ideas

### 4.1 Adaptive multi-scale (per-frame imgsz selection) ★

**Idea.** Skip 1024 on frames where the previous frame's median bbox area was > 30 K px² (large dancers, no need for the small-scale pass). Skip 768 when prior median area < 10 K px² (small dancers, no need for the large-scale pass).

**Where.** State machine inside `tracking/multi_scale_detector.py` `_detect_legacy`/`_detect_gpu_nms`. Track previous frame's median area as closure state.

**Expected.** Speedup ~30 % when applicable; **no accuracy impact intended** — adapt only when one scale is provably redundant for that frame's regime.

**Risk.** LOW if conservative thresholds. Add a hysteresis (3-frame moving average) so single-frame outliers don't flip scales.

### 4.2 Two-pass tracking (re-detect at expected locations)

**Idea.** First pass produces stable tracks. Second pass re-runs detection at conf=0.20 only inside ROIs centered on the first-pass track centroids ± 100 px, then unions any new detections back through DeepOcSort-replay.

**Where.** New `tracking/two_pass_runner.py`; called from `run_pipeline_on_video` at `tracking/run_pipeline.py:289-354` as an optional second iteration after the first cache write.

**Expected.** Marginal improvement on FN-dominated clips. Cost is +1× detector total.

**Risk.** MEDIUM. Two-pass introduces order-dependence that can be hard to reason about.

### 4.3 GT-aware threshold finder per clip (without GT at inference) ★

**Idea.** The user's exact specification: cluster detector cardinalities per frame; the most common count = expected dancer count; tune `conf` so detector cardinality matches mode count.

**Algorithm.**
1. Run detector at conf=0.15 on first 30 % of frames, count detections per frame.
2. Build histogram, find mode (most common active count).
3. Sweep `conf` in {0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36}.
4. For each, compute new median cardinality. Pick `conf` whose median is closest to (mode + 0).
5. Use chosen conf for the full clip.

**Where.** New `scripts/calibrate_conf.py` writing `work/<clip>/conf.json`. `tracking/run_pipeline.py:121` reads from JSON if present.

**Expected.** +0.003 to +0.010 mean. Captures most of the per-clip Optuna headroom without using GT.

**Risk.** MEDIUM. Mode estimation can be wrong if the clip has long FP-heavy or FN-heavy stretches. Mitigate with median-of-windows instead of pure mode.

### 4.4 Track-level confidence rebalancing per clip

The AND-gate at `tracking/best_pipeline.py:90-92` (`len≥60 ∧ mean≥0.55 ∧ p90≥0.84`) is global. The 0.84 p90 threshold was tuned for the 7-clip mix — clip-specific calibration could push it further.

**Per-clip heuristic.** Cluster all post-stage-1 tracks by `(mean_conf, p90_conf, len)`. Find the bimodal split between phantom and real cluster. Set thresholds at the cluster-separating midline.

**Risk.** HIGH. The 0.84 → 0.85 sweep already shows -0.0017 IDF1 (`EXPERIMENTS_LOG.md §2.5`); narrow margin. Implement only after Section 5's Tier 1 has shipped.

### 4.5 Newer YOLO variants comparison

YOLO11/12/13 are released; the EXPERIMENTS_LOG only tested stock YOLO11 at -5 to -7 pp loveTest (`§2.1`) — but those were stock weights, not fine-tuned. **The fine-tuning gap (`fine-tuning on dance >> larger backbone`) is documented as much larger than the architecture gap.**

**Expected.** Re-fine-tuning YOLO12/13 on the dance dataset is the path. Bare-bones YOLO13 stock will likely underperform our fine-tuned YOLO26s. Recommendation: **defer until Section 2.9 fine-tune is done; then compare fine-tuned-YOLO12 vs fine-tuned-YOLO26s on the same dataset.**

**YOLO13 specifically** introduced HyperACE for high-order correlations — could plausibly help in close-contact crowds. Worth a post-fine-tune ablation.

### 4.6 Detector ensemble of two backbones ★

**Idea.** YOLO26s + RTMDet-tiny. Different label-assignment schemes, different anchor designs → different FP modes. NMS-union over both reduces architecture-shared FNs.

**Where.** Extend `tracking/multi_scale_detector.py` to accept a second `weights` path with a different YOLO loader. Or wrap RTMDet inference behind a thin adapter that emits `[x1,y1,x2,y2,conf,cls]`.

**Cost.** ~+25-40 ms / frame on A10 (RTMDet-tiny is ~5-10 ms; YOLO26s 768/1024 is the bulk).

**Expected.** +0.002 to +0.008. Most gain concentrated where YOLO26s has consistent failure modes (close-contact loveTest).

**Risk.** MEDIUM. RTMDet outputs need label-space alignment (both must emit class-0 person). NMS at `ensemble_iou=0.6` should fuse cleanly.

### 4.7 Adaptive frame-rate handling

**Idea.** Run detector + tracker at full 24 fps; for visualization/output, sub-sample. Or vice-versa: at fast cuts run all frames, at slow scenes run every other and interpolate.

**Verdict.** Doesn't help accuracy on loveTest (already a steady-fast clip). Only a speed lever. **Not recommended for IDF1 work.**

---

## Section 5 — Top 5 recommended experiments

Picked for: (a) highest expected IDF1 lift per implementation hour, (b) low-risk integration that respects the no-regression rule, (c) coverage of both loveTest and darkTest, (d) zero overlap with parallel SAM/RTMW research streams.

---

### Experiment 5.1 — Cardinality-voting Kalman synthetic detections (Section 2.1)

**Pseudocode integration.**

```python
# new module: tracking/synthetic_recovery.py
def maybe_inject_synthetic(
    tracker, dets, frame_idx, history,
    *, lookback=30, min_track_age=10, conf_label=0.36,
):
    """If detection cardinality dropped, inject Kalman-predicted boxes
    for well-tracked tracks that are missing this frame.
    """
    median_card = np.median([h.n for h in history[-lookback:]])
    if len(dets) >= median_card - 1:
        return dets
    synthetic = []
    for trk in tracker.trackers:
        if trk.time_since_update == 0 or trk.age < min_track_age:
            continue
        if trk.kf.P.trace() > COV_TRACE_LIMIT:
            continue
        pred_xyxy = trk.kf.predict_box()
        if max(_iou(pred_xyxy, d[:4]) for d in dets) > 0.30:
            continue
        synthetic.append([*pred_xyxy, conf_label, 0])
    if synthetic:
        dets = np.vstack([dets, np.asarray(synthetic, dtype=np.float32)])
    return dets

# In tracking/run_pipeline.py around line 175 (_safe_detect call):
dets = _safe_detect(detect, frame_bgr)
dets = maybe_inject_synthetic(tracker, dets, idx, frame_history)  # NEW
tracks_per_frame = _safe_tracker_update(tracker, dets, frame_bgr, idx)
```

**Exact metric.** loveTest IDF1, FN count, FP count, num_switches, num_fragmentations from py-motmetrics. Compare per-frame cardinality histogram before/after.

**Expected lift.** loveTest **+0.020 to +0.040**, darkTest **+0.005 to +0.015** (also benefits because darkTest's FNs share the occlusion-Kalman-ghost mechanism). Mean across 7 clips: **+0.005 to +0.012**.

**Kill criterion.** If after 1 day of integration + tuning, loveTest IDF1 stays below **0.860**, abandon this approach. Mean must stay ≥ 0.957 (no regression).

---

### Experiment 5.2 — Multi-exposure detector ensemble + adaptive conf (Sections 1.3 + 1.6)

**Pseudocode integration.**

```python
# tracking/multi_scale_detector.py — extend make_multi_scale_detector
def _detect_legacy(frame_bgr):
    luma = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())
    is_dark = luma < 80
    eff_conf = 0.28 if is_dark else conf  # adaptive conf knob (Section 1.6)
    extra_views = []
    if is_dark and brighten_on_dark:  # Section 1.3
        bright = np.clip(frame_bgr.astype(np.float32) * 1.8, 0, 255)
        extra_views.append(bright.astype(np.uint8))
    for view in [frame_bgr] + extra_views:
        for imgsz in sorted_imgsz:
            results = models_by_scale[imgsz].predict(
                view, imgsz=int(imgsz), conf=eff_conf, iou=iou,
                device=device, verbose=False, classes=classes,
            )
            # ... collect into all_xyxy, all_conf, all_cls (existing code)
    # ... existing NMS-union at ensemble_iou=0.6
```

Gate behind env var `BEST_ID_DARK_RECOVERY=1` — default off for the well-lit benchmark, default on for darkTest.

**Exact metric.** darkTest IDF1, FN, FP, num_switches. Also evaluate on the 7-clip benchmark with flag OFF to verify zero regression.

**Expected lift.** darkTest **+0.010 to +0.030** combined. Mean (when applied selectively to darkTest): **+0.001 to +0.004**.

**Kill criterion.** If combined treatment doesn't lift darkTest IDF1 by ≥ +0.010 versus baseline, escalate to learned LLIE (Zero-DCE++, Section 1.4).

---

### Experiment 5.3 — HSV histogram per-track (Section 3.3)

**Pseudocode integration.**

```python
# Extend tracking/postprocess.py RawTrack at line 39-47 with hsv_hist field.
@dataclass
class RawTrack:
    track_id: int
    frames: np.ndarray
    bboxes: np.ndarray
    confs: np.ndarray
    masks: Optional[np.ndarray] = None
    embeds: Optional[np.ndarray] = None
    hsv_hists: Optional[np.ndarray] = None  # (n, 512) flattened 8x8x8

# Compute in frame_detections_to_raw_tracks (need access to frames).
# Store the running mean per track.
def _track_hist(track) -> np.ndarray:
    return track.hsv_hists.mean(axis=0)

# In _id_merge at tracking/postprocess.py line 405-415, after OSNet cos check:
if ti.hsv_hists is not None and tj.hsv_hists is not None:
    chi2 = float(_chi_square(_track_hist(ti), _track_hist(tj)))
    if chi2 > HSV_CHI2_GATE:  # e.g. 0.5
        continue
    score = iou + 0.5 * cos_sim + 0.3 * (1 - chi2)
```

**Frame access.** Cache currently does not store frames; either (a) re-decode at build_tracks time for histogram computation (slow), or (b) compute and store in cache during the detect+track pass. Option (b) requires extending `FrameDetections` at `prune_tracks.py` with a `hsv_hists` array — invasive but right.

**Exact metric.** loveTest IDF1, num_switches, num_fragmentations. Also darkTest IDF1 (HSV is more invariant to brightness than OSNet).

**Expected lift.** loveTest **+0.005 to +0.015**, darkTest **+0.003 to +0.010**, mean **+0.002 to +0.005**.

**Kill criterion.** If loveTest IDF1 lift < +0.003 with histogram added, abandon — appearance variance among the 15 dancers is too low for histogram to help.

---

### Experiment 5.4 — Soft-NMS for cross-scale ensemble (Section 2.3)

**Pseudocode integration.**

```python
# tracking/multi_scale_detector.py replace nms call at line 233:
def _soft_nms(boxes, scores, iou_thresh, sigma=0.5, score_thresh=0.001):
    """Linear soft-NMS variant (Bodla et al., ICCV 2017)."""
    indices = np.arange(len(boxes))
    keep = []
    while len(indices):
        i = np.argmax(scores)
        keep.append(indices[i])
        if len(indices) == 1:
            break
        # ... compute IoU of box i vs all remaining
        # ... s_j' = s_j * exp(-iou^2 / sigma) for high-IoU
        # ... drop boxes with s' < score_thresh
    return np.array(keep)

# Replace at line 233:
keep = _soft_nms(xyxy, conf_arr, ensemble_iou, sigma=0.5)
```

**Exact metric.** loveTest IDF1, FN, FP. Critical: track FP carefully because Soft-NMS preserves overlapping boxes; tune sigma to balance.

**Expected lift.** loveTest **+0.002 to +0.008**, mean **+0.001 to +0.003**.

**Kill criterion.** If FP increase exceeds FN decrease (so net IDF1 ≤ baseline), abandon. Try sigma ∈ {0.3, 0.5, 0.7}.

---

### Experiment 5.5 — Bidirectional Kalman + RTS smoother (Section 3.2)

**Pseudocode integration.**

```python
# Replace tracking/best_pipeline.py:205-247 smooth_centers_median, OR
# add as stage 7b after the median.

def smooth_centers_rts(tracks, *, q=1.0, r=4.0):
    """Forward-backward Rauch-Tung-Striebel smoother on bbox centers."""
    out = {}
    for tid, tr in tracks.items():
        bb = np.asarray(tr.bboxes, dtype=np.float64)
        if len(bb) < 5:
            out[tid] = tr; continue
        cx = (bb[:, 0] + bb[:, 2]) / 2
        cy = (bb[:, 1] + bb[:, 3]) / 2
        w = bb[:, 2] - bb[:, 0]
        h = bb[:, 3] - bb[:, 1]
        cx_s, cy_s = _rts_smooth_2d(cx, cy, q=q, r=r)
        new_bb = np.stack(
            [cx_s - w / 2, cy_s - h / 2, cx_s + w / 2, cy_s + h / 2], axis=1,
        )
        out[tid] = type(tr)(
            track_id=tr.track_id, frames=tr.frames,
            bboxes=new_bb, confs=tr.confs,
            masks=getattr(tr, "masks", None),
            detected=getattr(tr, "detected", None),
        )
    return out

# In build_tracks at tracking/best_pipeline.py:308:
final = smooth_centers_rts(stage4)  # replaces median  — OR sequential after
```

**Exact metric.** Mean IDF1 across 7 clips, per-clip table. Strict no-regression.

**Expected lift.** Mean **+0.001 to +0.005**. Per the open-experiments note in `EXPERIMENTS_LOG.md §4.6`.

**Kill criterion.** If any single clip regresses by > 0.002 vs v8, abandon and stick with the median.

---

### Summary table: top 5 prioritized

| # | Experiment | Target | Expected loveTest | Expected darkTest | Expected mean | Risk |
|---|---|---|---|---:|---:|---|
| 1 | Cardinality-voting Kalman synthetic | FN | +0.020 to +0.040 | +0.005 to +0.015 | +0.005 to +0.012 | MEDIUM |
| 2 | Multi-exposure + adaptive conf (gated) | darkTest | +0.000 | +0.010 to +0.030 | +0.001 to +0.004 | LOW |
| 3 | HSV histogram per-track | ID-merge | +0.005 to +0.015 | +0.003 to +0.010 | +0.002 to +0.005 | LOW |
| 4 | Soft-NMS cross-scale ensemble | FN | +0.002 to +0.008 | +0.002 to +0.005 | +0.001 to +0.003 | LOW |
| 5 | RTS bidirectional Kalman | smoothing | +0.001 to +0.003 | +0.001 to +0.003 | +0.001 to +0.005 | LOW |

If implemented in series (each gated, no-regression validated): **expected mean IDF1 0.957 → 0.965-0.972**, with darkTest IDF1 from baseline (TBD) → competitive level, and loveTest from 0.853 → 0.880-0.910.

---

## Section 6 — References (2024-2026)

**Multi-object tracking and occlusion**
- Cao et al. **Observation-Centric SORT (OC-SORT)**, CVPR 2023 — virtual-trajectory observation-centric re-update for occlusion. ArXiv: [2203.14360](https://arxiv.org/abs/2203.14360). Direct precedent for the cardinality-voting Kalman synthetic mechanism (Section 2.1 / Experiment 5.1).
- Maggiolino et al. **Deep OC-SORT**, ICIP 2023 — adaptive appearance-association on top of OC-SORT. ArXiv: [2302.11813](https://arxiv.org/abs/2302.11813). Best-published DanceTrack HOTA = 61.3 — sets a ceiling expectation for our same-uniform regime.
- **OA-SORT (Occlusion-Aware SORT)**, March 2026 — plug-and-play OAM/OAO/BAM modules. 63.1 % HOTA, 64.2 % IDF1 on DanceTrack. ArXiv: [2603.06034](https://arxiv.org/abs/2603.06034v2).
- **OccluTrack**, 2024 — abnormal-motion suppression in Kalman, pose-guided ReID for partial occlusion. ArXiv: [2309.10360](https://arxiv.org/abs/2309.10360).
- **PD-SORT**, January 2025 — pseudo-depth + Depth Volume IoU for heavy occlusions. ArXiv: [2501.11288](https://arxiv.org/abs/2501.11288).

**Crowded human detection**
- **Density-based Object Detection in Crowded Scenes**, April 2025 — DG-NMS (density-guided NMS). ArXiv: [2504.09819](https://arxiv.org/abs/2504.09819). Directly applicable refinement of the Soft-NMS proposal (Experiment 5.4).
- Bodla et al. **Soft-NMS — Improving Object Detection with One Line of Code**, ICCV 2017 (canonical Soft-NMS reference).
- **Do We Still Need Non-Maximum Suppression?**, WACV 2024 — accurate confidence estimates question the necessity of NMS at all.

**Low-light enhancement (LLIE)**
- Cai et al. **Retinexformer**, ICCV 2023 + NTIRE 2024 runner-up + ECCV 2024 enhanced version. Demonstrates downstream low-light object-detection benefit. ArXiv: [2303.06705](https://arxiv.org/abs/2303.06705). [GitHub](https://github.com/caiyuanhao1998/Retinexformer).
- Li et al. **Zero-DCE++**, TPAMI 2021 — 10K-param LLIE, ~1000 FPS on 2080Ti at 1200×900. [Project](https://li-chongyi.github.io/Proj_Zero-DCE++.html). Strongest learned LLIE per ms; Tier 1 for darkTest (Section 1.4).
- **IGDNet** — Zero-Shot Robust Underexposed Image Enhancement via Illumination-Guided and Denoising, July 2025. ArXiv: [2507.02445](https://arxiv.org/abs/2507.02445). 20.41 dB PSNR / 0.860 SSIM, beats 14 unsupervised baselines. Worth evaluating against Zero-DCE++ on darkTest.

**Person ReID under low-light**
- **NightReID** Benchmark + EDA framework (Image Enhancement and Denoising + Data Distribution Alignment), 2025. [OpenReview](https://openreview.net/forum?id=CFwtGTJPE4). 1,500 nighttime IDs. Validates the "enhance crops then ReID" approach used in Strategy A/D (Section 1).
- **Illumination Distillation Framework (IDF)** — master nighttime branch + illumination-enhancement branch fused for nighttime ReID. ArXiv: [2308.16486](https://arxiv.org/pdf/2308.16486v1.pdf).
- **Collaborative Enhancement Network (CENet)**, 2024 — parallel relighting + ReID with feature interaction. ArXiv: [2312.16246](https://arxiv.org/pdf/2312.16246).

**Detector calibration**
- Trakulwaranont et al. **Determining the Optimal T-Value for Temperature Scaling Calibration on YOLO-World**, MDPI Applied Sciences 2025 — ECE 6.78 % → 2.31 %. [Paper](https://www.mdpi.com/2076-3417/15/22/12062). Supports Section 2.8.

**YOLO variants**
- Ultralytics. **Ultralytics YOLO Evolution: YOLO26, YOLO11, YOLOv8, YOLOv5**. ArXiv: [2510.09653](https://arxiv.org/html/2510.09653v2).
- **YOLOv12: Attention-centric Real-Time Object Detectors**, ArXiv: [2502.12524](https://arxiv.org/pdf/2502.12524). +2.1 % mAP over YOLOv10-N.
- **YOLOv13: HyperACE for high-order correlations**, 2025. ArXiv: [2506.17733](https://arxiv.org/pdf/2506.17733). +3.0 % mAP over YOLO11-N. Directly relevant to Section 4.5 if a fine-tuning campaign is launched.
- Lyu et al. **RTMDet: An Empirical Study of Designing Real-Time Object Detectors**, 2022. [Paper](https://arxiv.gg/paper/2212.07784). 52.8 % AP, 300+ FPS on 3090. Diverse-architecture ensemble candidate (Section 4.6).

**Smoothing / tracking**
- **The Invariant Rauch-Tung-Striebel Smoother**, 2024. ArXiv: [2403.00075](https://arxiv.org/abs/2403.00075). Modern bidirectional smoother on Lie-group states; canonical RTS reference for Section 3.2 / Experiment 5.5.

**Gait / motion fingerprinting**
- Boulgouris & Chi 2007, **Gait recognition based on Fourier descriptors**, IEEE — phase-weighted Fourier captures ID. Foundation for Section 3.1.
- Cunado et al. 2003, **Automatic extraction and description of human gait models for recognition purposes**, CVIU — anatomical-landmark Fourier signatures.

---

## Appendix: Concrete code-citation index

| Proposal | File | Lines | Action |
|---|---|---|---|
| 1.3 / 5.2 Multi-exposure | `tracking/multi_scale_detector.py` | 165-241, 243-313 | Add brightened view loop mirroring the `tta_flip` block (190-213) |
| 1.6 Adaptive conf | `tracking/multi_scale_detector.py` | 173-176 | Compute per-frame `eff_conf` from luma before `model.predict` |
| 2.1 / 5.1 Synthetic injection | `tracking/run_pipeline.py` | 175-177 | New helper between detect and track |
| 2.2 imgsz=896 | `tracking/run_pipeline.py` | 119 | Change tuple to `(768, 896, 1024)` |
| 2.3 / 5.4 Soft-NMS | `tracking/multi_scale_detector.py` | 233, 301 | Replace `torchvision.ops.nms` with `_soft_nms` |
| 2.4 Persistence promotion | `tracking/run_pipeline.py` | 165-184 | New shadow-detector path |
| 2.9 Custom fine-tune | `weights/best.pt` | n/a | Replace with `weights/best_v9.pt` after training |
| 3.1 Motion fingerprint | `tracking/postprocess.py` | 405-415 | Add fingerprint check after OSNet cos |
| 3.2 / 5.5 RTS smoother | `tracking/best_pipeline.py` | 205-247, 308 | New `smooth_centers_rts` replaces or follows `smooth_centers_median` |
| 3.3 / 5.3 HSV histogram | `tracking/postprocess.py` `RawTrack` | 39-47 | Add `hsv_hists` field; cache extension at `prune_tracks.py` `FrameDetections` |
| 3.4 Cross-track exclusion | `tracking/postprocess.py` | 381-415 | Competitive-merge gate after candidate scoring |
| 3.5 Trajectory continuity | `tracking/postprocess.py` | 413 | Add velocity-cos to merge score |
| 3.6 Per-clip cos | `tracking/best_pipeline.py` | 283 | Read from `work/<clip>/cos_thresh.json` |
| 4.1 Adaptive imgsz | `tracking/multi_scale_detector.py` | 142, 171-188 | State-machine wrapper around `sorted_imgsz` |
| 4.3 Threshold finder | new `scripts/calibrate_conf.py` | n/a | Writes `work/<clip>/conf.json`; read at `tracking/run_pipeline.py:121` |
| 4.6 Detector ensemble | `tracking/multi_scale_detector.py` | 95-163 | Accept second backbone path |

All proposals can be gated behind environment variables (the established pattern at `tracking/multi_scale_detector.py:39-49` `_resolve_gpu_nms_flag`) so the default code path remains the v8 production pipeline byte-for-byte.

 I've already reviewed the experiments log thoroughly, so I can move forward with writing the comprehensive research report while being careful to exclude the rejected approaches—CAMELTrack, SAM 2.1 video predictor as a tracker, and SAM 2.1 m—from any new proposals.

Now I'm going through the extensive list of techniques and hyperparameter adjustments I've tested, noting which ones showed promise, which hit plateaus, and which ones I should avoid re-proposing based on their performance history.