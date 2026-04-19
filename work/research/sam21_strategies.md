I have everything I need. Let me write the report now. (No code changes — pure research, as requested.)

---

# SAM 2.1 integration strategies for offline dance-video tracking

**Target pipeline:** v8 production pipeline (mean IDF1 = 0.957 across 7 clips).
**Bottleneck clip:** `loveTest` (IDF1 = 0.853, **2,083 FN vs 1,411 FP** — detector recall is the dominant error source on close-contact frames).
**New clip:** `darkTest` (low-light, ~14 dancers, 720p, 24 fps, 23 s).
**Hardware target:** NVIDIA A10 (24 GB VRAM, ~31 TFLOPS FP16).
**Past SAM failures to avoid:** (a) SAM 2.1 video predictor *fuses* identities under occlusion, (b) SAM mask-derived bbox replacement causes identity drift in busy/reflective scenes (notably −0.19 IDF1 on `BigTest`).

The two failure modes share a common root: **SAM 2's memory-bank propagation lets one object's appearance contaminate another's mask once they overlap.** Every strategy below either (a) avoids the video predictor entirely (uses the *image* predictor only), or (b) uses video propagation only over very short windows with explicit exclusion prompts, or (c) keeps SAM downstream of the existing tracker as a verifier/gate, never as the primary source of identity.

---

## Section 1 — Twelve novel SAM 2.1 integration strategies

### Strategy 1 — Exclusion-prompted SAM 2.1 image-predictor for cardinality-gated FN recovery   `[priority: HIGH]`

**Concrete description.** This is the single most direct attack on the loveTest bottleneck.
1. After `_detect_and_track_serial` finishes (`tracking/run_pipeline.py:166-184`) but before `build_tracks`, walk the cache and compute the per-frame detected dancer count `N(t)`. Compute the running median `Ñ` over a 30-frame window.
2. For every frame `t` where `N(t) ≤ Ñ - 1` (a "missing-dancer frame"), do the following:
   - For each track that was *active in [t-5, t-1]* but *absent at t*, extrapolate its bbox forward to frame `t` (reuse `tracking.postprocess._track_extrapolate_box` at line 163 of `tracking/postprocess.py`).
   - Build a single SAM 2.1 image-predictor call with:
     - `box = predicted_bbox` for the missing track (positive),
     - `point_coords = [center of every other tracked dancer's bbox at t]` with `point_labels = [0, 0, 0, ...]` (all negatives — "everyone else is *not* the target"),
     - `multimask_output=False`.
   - Take the returned mask, derive its tight bbox, gate by: (a) mask area ≥ 0.4 × predicted bbox area (real person, not a sliver), (b) mask area ≤ 1.5 × predicted bbox area (not the whole crowd), (c) **OSNet cosine ≥ 0.75 against the missing track's most recent embedding** (the existing 0.7 threshold from `pp_id_merge_osnet_cos_thresh`, hardened by 0.05 because we're synthesising the box).
   - If all three pass, append a synthetic detection `(bbox_from_mask, conf=0.50, tid=missing_track_id)` to the cache for that frame.
3. Then run `build_tracks` as normal.

**Why this avoids past failure modes.** No video predictor at any point — every SAM call is the *image* predictor with no memory bank, so cross-object identity fusion is structurally impossible. The negative point prompts act as explicit "exclude these other dancers" signals. The OSNet ReID gate at the end is the same gate of last resort the v8 pipeline already trusts.

**Expected impact.**
- `loveTest`: directly attacks the 2,083 FN. Even if SAM only reliably recovers dancers in 25 % of "missing-dancer frames", that's ~150-300 FN recovered. Estimated: **+0.008 to +0.018 IDF1** (taking loveTest from 0.853 to ~0.86-0.87). Could be higher if recovery rate is closer to 40 %.
- `MotionTest` (currently 0.932, 946 FP / 1316 FN): similar mechanism applies; cardinality drops in high-motion frames are the failure pattern. Estimated **+0.003 to +0.008**.
- `darkTest` (no baseline yet): SAM 2.1 was trained on SA-V (heterogeneous lighting) so it should detect dancers the YOLO detector misses in low light. This may be the *single biggest* win on darkTest.

**Implementation cost.** SAM 2.1 image-predictor on A10:
- Image encoder forward (one per frame): ~50 ms with `sam2.1_hiera_base_plus`, ~25 ms with `sam2.1_hiera_tiny` (scaled from RTX 4090 / A100 numbers in the SAM 2 paper).
- Mask decoder forward (per prompt set, after encoder is cached): ~3-5 ms.

For a 700-frame `loveTest` clip, "missing-dancer frames" are likely ~15 % = ~100 frames. Total SAM cost: 100 × 50 ms ≈ **5 sec on A10 with `base_plus`**, or 2.5 sec with `tiny`. Memory: +4 GB on A10 (well within budget — see §2).

**Estimated IDF1 lift on loveTest.** **+0.008 to +0.018**, depending on SAM recovery rate. Mid-estimate: +0.012 (loveTest → 0.865).

**Risk / failure modes.**
- *Synthetic detections may pollute the OSNet gallery.* Mitigation: mark synthetic detections in the cache (e.g., `conf = 0.50` exactly, never produced by YOLO since its threshold is 0.34 with much higher typical values for true positives), and exclude them from the ReID-embedding gallery used by DeepOcSort's next frame. This requires a small change to the cache schema (a `synthetic` flag on `FrameDetections`), not a code change to DeepOcSort itself.
- *SAM may segment the wrong person* when the target is fully occluded. Mitigation: the OSNet gate ≥ 0.75 catches this in expectation; also add a maximum displacement gate (recovered bbox center must be within 200 px of the extrapolated center).

**Rank: HIGH.** Largest expected lift, smallest blast radius, no change to the tracker or DeepOcSort kalman state.

---

### Strategy 2 — SAM 2.1 per-bbox VERIFIER (mask coverage + connected-components)   `[priority: MEDIUM-HIGH]`

**Concrete description.**  After stage 4 (`filter_tracks_post_merge` in `tracking/best_pipeline.py:123-146`), for each *surviving* track, sample K=8 evenly-spaced frames. At each sampled frame, run SAM 2.1 image predictor with `box = track.bboxes[t]`. Compute two diagnostics:
1. `coverage = mask.sum() / bbox_area` — should be in `[0.35, 0.85]` for a real person (lower = bbox too big / phantom; higher = bbox too tight / bad).
2. `n_components` — number of 4-connected components in the mask. A real person mask is one large blob (with possibly small detached limb fragments). Phantoms (mirror reflections at oblique angles, posters of dancers, two-people-fused-into-one detections) often have either very low coverage or fragmented masks.

Add a fourth predicate to `filter_tracks_post_merge` (after the `p90 ≥ 0.84` gate at line 144): keep iff *at least 6 of 8 sampled frames* satisfy `coverage ∈ [0.35, 0.85]` AND `largest_component_area / total_mask_area ≥ 0.7`.

**Why this avoids past failure modes.** Image-predictor only; no propagation; verdicts are per-frame and per-track-independent so there's no opportunity for identity fusion.

**Expected impact.** Targets the same class of phantoms the current `p90 ≥ 0.84` gate catches (mirror reflections, posters), but uses a structurally different signal (geometry vs confidence). Likely catches the 0.85 / 0.86 boundary phantoms that the p90 gate has to leave behind because raising it to 0.85 already kills real tracks (per §2.5 of the experiments log).

- `mirrorTest` (currently 0.9935): may catch the residual 0.0065 IDF1 gap. **+0.001 to +0.005.**
- `loveTest` (currently 0.8533): the FP component is 1,411 and shrinking it affects IDF1's precision side. Could shave 100-200 FP. **+0.002 to +0.005.**
- Other clips: at ceiling already, neutral or +0.001.

**Implementation cost.** ~14 surviving tracks × 8 sampled frames = ~110 SAM image-predictor calls per clip. With `tiny`: ~3 sec. Per-frame encoder cost is amortised across the 8 samples since they're at different frames (no caching benefit), but you can batch them.

**Risk / failure modes.** A real dancer in a dramatic pose (e.g., arabesque with extended limbs going outside the bbox) may have low coverage and trigger a false-positive rejection. Mitigation: use 6-of-8 majority voting rather than all-frames, and validate the threshold on `easyTest` / `gymTest` first — if either drops, abort.

**Rank: MEDIUM-HIGH.** Cleanly attacks the FP side that the current p90 gate is saturated on.

---

### Strategy 3 — Mask-IoU-gated occlusion detection for ReID gallery hygiene   `[priority: HIGH]`

**Concrete description.** This is the most novel idea — and it directly follows the SAM2MOT (April 2025) "Cross-Object Interaction" paper's central insight, adapted to a *post-process* setting. Once per N=3 frames during the `_detect_and_track_serial` loop (`tracking/run_pipeline.py:172`), do:
1. Run SAM 2.1 image predictor *batched* over all current bboxes in a single encoder forward (the image encoder is the expensive part — adding more box prompts to the same image is ~3 ms each).
2. Compute pairwise mask-IoU between every pair of currently active tracks.
3. If `mask_iou(A, B) > 0.4`, mark **both A and B's current frame** as `occluded`.
4. Append the `occluded` flag to the FrameDetections cache (new boolean column).

Then in `tracking.postprocess._id_merge` (line 331), when computing the OSNet cosine similarity at line 405-411, instead of using the *last* embedding of A and *first* embedding of B, use the last *non-occluded-flagged* embedding of A and the first *non-occluded-flagged* embedding of B. This means we're never comparing appearances of two dancers when one was visually fused with someone else.

**Why this avoids past failure modes.** No mask propagation. The image predictor's output for each box is independent, so the "two masks merge into one" failure of the video predictor cannot occur. SAM2MOT v2 paper (arXiv:2504.04519) reports +4.5 IDF1 on DanceTrack from exactly this kind of cross-object occlusion logic — and DanceTrack is a similar same-uniform-crowd setting to loveTest.

**Expected impact.**
- `loveTest`: the OSNet cosine gate at 0.7 is currently the gate of last resort for ID merges. Cleaner embeddings ⇒ fewer wrong rejections of correct merges. Likely recovers some of the 2,083 FN that get fragmented into multiple short tracks dropped by the `min_len ≥ 60` post-filter. **+0.005 to +0.012.**
- `BigTest` (currently 0.9981): essentially at ceiling, neutral.
- `MotionTest`: occluded-frame detection is less of an issue here (errors are detector cardinality, not appearance contamination). Neutral.

**Implementation cost.** SAM image predictor encoder once every 3 frames = ~230 forwards on a 700-frame clip. With `base_plus` on A10: ~12 sec. With `tiny`: ~6 sec. Mask decoder for ~14 boxes per frame = 14 × 4 ms × 230 = 13 sec extra. Total: **~25 sec extra on A10 with `base_plus`** (current baseline is ~30 sec, so this ~doubles wall time).

**Risk / failure modes.**
- The 0.4 mask-IoU threshold needs validation (SAM2MOT used 0.8). The dance scenario likely needs lower because dancers in close contact partially overlap without being occluded. Sweep on a small held-out clip first.
- "Occluded" frames of a track may already be sparse (only contributes to the gallery on that frame), so the win from purging them depends on how often DeepOcSort actually looks at occluded frames when computing its cosine similarity.

**Rank: HIGH.** Strong literature precedent (SAM2MOT +4.5 IDF1 on DanceTrack). The right lever for the ReID-contamination root cause.

---

### Strategy 4 — Mask IoU as additional signal in `_id_merge`   `[priority: MEDIUM]`

**Concrete description.** In `tracking.postprocess._id_merge` (line 331), the current spatial gate is `iou >= 0.10` between the extrapolated tail of A and the head of B (line 401-403). This bbox IoU is a poor signal in the dance setting because two dancers in close contact often have very high bbox IoU but very different mask shapes. Augment the gate as follows:
- Keep the existing extrapolated-bbox-IoU gate (line 401-403) as a *cheap* pre-filter (≥ 0.10).
- For every pair that passes, run SAM 2.1 image predictor on the tail-frame and head-frame of each track. Compute mask IoU between (a) A's mask at A's last frame, projected forward to B's first frame using A's tail velocity, and (b) B's mask at B's first frame.
- New gate: `bbox_iou ≥ 0.10 AND mask_iou ≥ 0.30 AND osnet_cos ≥ 0.7`.

**Why this avoids past failure modes.** Per-call image predictor; no propagation across the gap; no possibility of the two tracks' masks fusing because they're computed independently.

**Expected impact.**
- The gate sits inside `_id_merge`. Today the OSNet cos ≥ 0.7 is the only fine-grained discriminator. Adding mask IoU adds an *orthogonal* signal: appearance (OSNet) is brittle on same-uniform clips; pose silhouette (mask shape) is much less so.
- `loveTest`: marginal lift on FN (more correct merges admitted), maybe +0.002 to +0.005.
- `BigTest`: neutral (already 0.998).
- Risk of regression on `mirrorTest` (the canonical phantom is a mirror reflection — its mask would have similar shape to the real dancer, so mask IoU would not help reject it; but the OSNet gate already rejects it).

**Implementation cost.** Number of merge candidates per clip is small (~50-150 pairs from the post-process log). 2 SAM forwards per candidate = 100-300 SAM calls = **~5-15 sec on A10 with `base_plus`**.

**Risk / failure modes.** Mask IoU and bbox IoU are positively correlated, so the additional information may be small. Sweep on `loveTest` only first.

**Rank: MEDIUM.** Modest lift, modest cost.

---

### Strategy 5 — Mask-area-trajectory phantom filter   `[priority: MEDIUM-HIGH]`

**Concrete description.** Real dancers have stable mask-area trajectories with smooth depth-driven variation. Phantoms (mirror reflections, posters, partial-body misfires) typically have one of:
- Bimodal area (e.g., a poster suddenly being occluded for 5 frames, then visible again at very different size).
- Non-physical area trajectories (mirror reflections at oblique angles produce abrupt area jumps when the dancer crosses behind a mirror frame edge).
- Saturated low CV (a poster has near-zero CV(area), much lower than even the v8 size-smoother gate of 0.20).

Add a per-track diagnostic stage between `filter_tracks_post_merge` and `bbox_continuity_stitch` (i.e., insert at line 302 of `tracking/best_pipeline.py`):
1. For each surviving track, sample K=15 frames evenly across its lifetime.
2. Run SAM 2.1 image predictor at each sample (using the bbox as prompt).
3. Compute mask-area trajectory `A_t`. Drop the track iff:
   - `CV(A_t) ≥ 0.6` (extreme variation, suggests track is jumping between different objects), OR
   - the mask-area trajectory has a discontinuity > 3× the median delta (a bimodal area distribution), OR
   - more than 30 % of sampled frames have `coverage = mask_area / bbox_area < 0.2` (mostly background — the bbox doesn't actually contain a person).

**Why this avoids past failure modes.** Image-predictor only. The decision is per-track (independent of other tracks) and per-sampled-frame (independent across the track), so identity fusion is structurally impossible.

**Expected impact.**
- This is the *structural* answer to phantom tracks that the v8 p90 gate just barely catches. Per the experiments log §2.5, the v5-introduced phantom had `mean_conf=0.751`, `p90_conf=0.835`, while the worst real track was at `p90=0.860`. A 0.025 margin. Fragile. The mask-area-trajectory check is a much harder-to-fake signal.
- `mirrorTest`: likely catches the canonical mirror reflection robustly. **+0.002 to +0.005** (taking it from 0.9935 closer to 1.0).
- `loveTest`: catches some FP-side phantoms. **+0.002 to +0.005.**
- Could allow loosening the `p90 ≥ 0.84` gate to `0.82`, recovering some real short tracks on `shorterTest`. Indirect lift: **+0.001 to +0.003 on shorterTest.**

**Implementation cost.** ~14 surviving tracks × 15 sampled frames = ~210 SAM forwards. With `tiny`: **~5 sec** per clip.

**Risk / failure modes.** A dancer transitioning from a stand to a leap can have a 2× area change in a single frame. Use median-filter-then-detect-jump rather than raw deltas, and require *two* consecutive jumps for rejection.

**Rank: MEDIUM-HIGH.** Cheap, structural, addresses a known fragile gate.

---

### Strategy 6 — Mask-aware ReID embeddings (SAM-masked OSNet crops)   `[priority: MEDIUM]`

**Concrete description.** When `_id_merge` (`tracking/postprocess.py:405-411`) computes OSNet cosine between two tracks' embeddings, the embeddings come from DeepOcSort's internal gallery, which used the *full* bbox crop (background, occluders, and all). On loveTest's same-uniform close-contact frames, the background is *another dancer in the same uniform*, which is exactly what corrupts the embedding.

Fix: at every frame where DeepOcSort is about to produce a ReID embedding, first run SAM 2.1 image predictor with the bbox as prompt, get the mask, and zero-out (or set to 128 grey, which OSNet was trained on) the background pixels. Re-extract the OSNet embedding from this masked crop.

This requires touching the BoxMOT internals or inserting a wrapper. The *minimally invasive* version: don't touch DeepOcSort at all; instead, separately compute and store SAM-masked OSNet embeddings *only for the tail-frame of A and head-frame of B* in `_id_merge`. Use those for the cosine gate at line 405-411 of `tracking/postprocess.py`.

**Why this avoids past failure modes.** Image-predictor only. The mask is used to filter pixels going into OSNet — SAM's role is purely to define "what's background" for the appearance-matching step.

**Expected impact.** Literature (AOANet 2025 on Occluded-Duke, Mask-Aware Hierarchical Aggregation Transformer 2024) shows mask-aware ReID gives 5-15 percentage-point lifts on occluded-person ReID benchmarks. In the `_id_merge` gate, this likely raises the cosine similarity of true matches and lowers it for false ones, which means the existing 0.7 threshold becomes a tighter gate.

- `loveTest`: probably the biggest beneficiary because the close-contact frames contaminate embeddings the most. **+0.003 to +0.010 IDF1.**
- `BigTest`: at ceiling (0.998), neutral.
- `MotionTest`: less applicable (errors are detection cardinality), **+0.001.**

**Implementation cost.** Only ~50-150 merge candidates per clip × 2 SAM image-predictor calls each = ~100-300 SAM forwards. With `tiny`: **~3-7 sec per clip.** Plus 2 OSNet forwards per pair, which is microseconds.

**Risk / failure modes.**
- OSNet x0.25 was pretrained on MSMT17 *with backgrounds intact*. Feeding it a masked-out crop is an out-of-distribution input. May actually *degrade* embedding quality. Mitigation: replace background with a per-image grey rather than pure black, so the network sees a constant-luminance background rather than zero.
- For dancers in extreme poses where the SAM mask is fragmented, the masked crop may be too sparse. Mitigation: dilate the mask by 5 px before applying.

**Rank: MEDIUM.** Strong literature support but OOD risk for OSNet is real; needs validation.

---

### Strategy 7 — SAM-derived per-region cardinality voting   `[priority: LOW-MEDIUM]`

**Concrete description.** SAM 2.1 has a built-in "automatic mask generator" mode that produces all salient masks in an image given a uniform grid of point prompts (typical: 32×32 grid). For each frame in the cache:
1. Run automatic-mask-generator at coarse grid (16×16 = 256 points).
2. Filter masks to those with: area in `[2000, 30000]` px² (person-sized at 1080p), aspect ratio in `[1.5, 4.5]` (vertical/standing), and centered above the floor of the frame.
3. Count = SAM-vote estimate of dancer count `N_SAM(t)`.
4. Compare to detector count `N_YOLO(t)`. When `N_SAM(t) ≥ N_YOLO(t) + 1`, mark this frame as a "potential FN frame" and feed it into Strategy 1's recovery loop.

This replaces the running-median trigger in Strategy 1 with a SAM-vote trigger.

**Why this avoids past failure modes.** Image predictor only.

**Expected impact.** A more principled trigger than the running median — SAM-vote is right for the actual frame, not a temporal smoothing. But: largely subsumed by Strategy 1; the running median is a pretty good trigger already.

**Implementation cost.** Automatic-mask-generator is the expensive mode of SAM. ~250-500 ms per frame for `base_plus`. Running on all 700 frames of loveTest is **~3-5 minutes**, which violates the <10-min wall budget and is probably overkill.

**Risk / failure modes.** Person-shaped mask filtering is fragile (a dancer mid-leap doesn't satisfy aspect ratio in [1.5, 4.5]).

**Rank: LOW-MEDIUM.** Useful as an alternative trigger for Strategy 1 *if* the running-median trigger turns out to miss systematically. Don't run as a standalone experiment.

---

### Strategy 8 — SAM 2.1 reverse-direction re-segmentation for missed identity recovery   `[priority: MEDIUM]`

**Concrete description.** Current pipeline runs forward only. When the v8 final tracks hit GT eval and a track was *split* into two (one ends at frame `t`, another starts at frame `t+5`), we currently rely on the post-process to merge them. But if the merge is rejected (insufficient ReID cosine, e.g.), both halves contribute to FN.

Reverse pass after `build_tracks` returns:
1. For each pair `(A, B)` where A ends at `t1` and B starts at `t2` with `t2 > t1` and OSNet cos was *just below* 0.7 (e.g., in `[0.55, 0.7]` — borderline):
2. Run SAM 2.1 image predictor *backwards in time* from frame `t2-1` to frame `t1+1`, with B's first bbox as the seed and using each previous frame's predicted mask to re-prompt the previous frame's image. (This is essentially SAM 2.1's video-predictor mode but constrained to a *single* track's recovery and with a tight gap, so identity fusion is bounded.)
3. If the reverse-propagated mask at frame `t1+1` overlaps A's last bbox by mask-IoU ≥ 0.4, accept the merge.

**Why this avoids past failure modes.** Video propagation is restricted to a *single object* over a *short gap* (≤ 48 frames per the existing `id_merge_max_gap = 48`). Identity fusion requires two tracked objects whose masks merge — here we're propagating *one* object backwards.

**Expected impact.**
- Small. Targets borderline merge rejections. Likely 5-15 saves per clip.
- `loveTest`: **+0.002 to +0.005.**
- Other clips: **+0.001.**

**Implementation cost.** Borderline merges are rare (~10-30 per clip). Each invocation runs SAM video predictor over up to 48 frames at ~30 ms/frame on A10 with `base_plus` = ~1.4 sec per merge candidate. Total: **~15-45 sec per clip.**

**Risk / failure modes.** The reverse propagation can still drift in busy scenes — exactly the failure mode that killed mask-bbox-replacement in past experiments. Mitigation: hard-cap the propagation length (≤ 16 frames) and reject if the propagated mask area changes by more than 2× across the gap.

**Rank: MEDIUM.** Modest expected lift, not the cheapest strategy, but it's a clean way to use the video predictor in a *constrained* way that avoids its known failure mode.

---

### Strategy 9 — SAM-shape-based phantom rejection for posters / mirror reflections   `[priority: LOW-MEDIUM]`

**Concrete description.** Mirror reflections and posters of dancers have characteristically different mask geometries from real 3D-projected dancers:
- Reflections typically have *vertically flipped* depth cues (light direction reversed).
- Posters are *flat* — no depth-driven mask-area variation across frames (very low CV).
- Both have characteristic *aspect-ratio stability* very different from a moving dancer.

For each surviving track, sample 10 frames, compute SAM masks, compute three shape statistics across the track:
1. `CV(area)` — should be > 0.10 for real moving dancers.
2. `CV(aspect_ratio)` — should be > 0.05 (real dancers change pose).
3. Mask boundary "jaggedness" (perimeter² / area) — flat objects have characteristic values.

Use a small logistic regression (3 features → real/phantom) trained on a held-out clip with hand-labeled phantoms.

**Why this avoids past failure modes.** Image-predictor only; per-track.

**Expected impact.** Largely subsumed by Strategy 5 (mask-area trajectory). Worth running as an *ablation* on Strategy 5 to see if the extra features help, but probably not as a standalone.

**Rank: LOW-MEDIUM.** Subsumed by Strategy 5.

---

### Strategy 10 — SAM as tie-breaker for borderline `_id_merge` decisions only   `[priority: HIGH for cost/benefit]`

**Concrete description.** This is the *minimum-effort* SAM integration. Most merge candidates are handled robustly by the existing IoU + OSNet gate. Only ~5-10 % per clip are borderline (OSNet cos in `[0.6, 0.75]`, IoU in `[0.10, 0.20]`).

For only those borderline pairs, run SAM 2.1 image predictor on the tail of A and head of B, compute mask IoU at the projected bbox. If mask IoU ≥ 0.35 → accept. If < 0.35 → reject. This is essentially Strategy 4 but invoked only on borderline pairs.

**Why this avoids past failure modes.** Image predictor only; per-pair; never invoked when OSNet/IoU already give a clear answer.

**Expected impact.**
- Few invocations, but each one is decisive. Probably saves 5-15 correct merges per clip.
- `loveTest`: **+0.002 to +0.005.**

**Implementation cost.** ~5-15 SAM image-predictor calls per clip × 2 forwards = 10-30 SAM forwards = **<1 sec per clip on A10 with `tiny`.** Effectively free.

**Risk / failure modes.** Mask-IoU threshold needs validation. Test on `loveTest` only first.

**Rank: HIGH for cost/benefit.** Tiny implementation effort, tiny cost, modest but real lift, zero blast radius.

---

### Strategy 11 — SAM-validated long-gap stitch for `bbox_continuity_stitch`   `[priority: LOW-MEDIUM]`

**Concrete description.** `bbox_continuity_stitch` (`tracking/bbox_stitch.py:53-224`) currently rejects stitches when `pos_diff > max_position_jump_px = 2000`. For borderline rejections (pos_diff in `[1500, 2000]`), invoke SAM 2.1 at a single intermediate frame (mid-gap) with the predicted bbox center as a *positive point prompt* (not box, since we're uncertain about the bbox). If SAM returns a person-shaped mask centered within 100 px of the predicted center, accept the stitch.

**Why this avoids past failure modes.** Image predictor only; one frame.

**Expected impact.** `bbox_continuity_stitch` fires only 5 times across the entire 7-clip benchmark (per experiments log §2.6). Borderline rejections are rare. Likely 0-2 additional stitches per clip. **+0.0005 to +0.002 mean IDF1.**

**Rank: LOW-MEDIUM.** Useful as a defense-in-depth measure but not a top-3 candidate.

---

### Strategy 12 — `darkTest` low-light supplementary detection   `[priority: HIGH for darkTest specifically]`

**Concrete description.** The new `darkTest` clip is low-light. YOLO26s is dance-fine-tuned but the training set's lighting distribution may not cover this regime. SAM 2.1 was trained on SA-V which is heterogeneous in lighting. Run SAM 2.1 in *automatic-mask-generator* mode at low spatial resolution (downsample frame 2× before feeding to SAM, since dark scenes don't have fine features anyway) on every 5th frame of `darkTest`. Filter to person-shaped masks. Use as supplementary detections alongside YOLO output, gated by ReID cosine ≥ 0.6 against existing tracks (or accepted as new tracks if mask area/shape is consistent).

**Why this avoids past failure modes.** Image-predictor only.

**Expected impact.**
- darkTest has no baseline; this could be the difference between IDF1 ~ 0.7 (if YOLO26s drops 30 % of dancers) and IDF1 ~ 0.85.
- Other clips: don't run this; dispatch by clip metadata or by a low-light detector (mean luminance < threshold).

**Implementation cost.** 552 frames at 24 fps × 23 s, every 5th frame = 110 frames × 250 ms (auto-mask at 540p) = **~28 sec per clip on A10 with `base_plus`.**

**Risk / failure modes.** Auto-mask generator produces *lots* of false positives in dark scenes too; the person-shape filter must be tight.

**Rank: HIGH for darkTest specifically; not for other clips.**

---

### Summary table

| # | Strategy | Phase | SAM mode | Priority | loveTest IDF1 lift estimate | Cost (A10) |
|---|---|---|---|---|---:|---:|
| 1 | Exclusion-prompted FN recovery | post-stage 2 | image (box+neg-points) | **HIGH** | **+0.008-0.018** | ~5 sec |
| 2 | Per-bbox verifier (coverage+CC) | stage 4 | image (box) | MEDIUM-HIGH | +0.002-0.005 | ~3 sec |
| 3 | Mask-IoU occlusion gate | inside detect/track + stage 3 | image (box, batched) | **HIGH** | +0.005-0.012 | ~25 sec |
| 4 | Mask-IoU in `_id_merge` | stage 3 | image (box) | MEDIUM | +0.002-0.005 | ~10 sec |
| 5 | Mask-area-trajectory phantom | between 4 and 5 | image (box) | MEDIUM-HIGH | +0.002-0.005 | ~5 sec |
| 6 | Mask-aware OSNet at merge | stage 3 | image (box) | MEDIUM | +0.003-0.010 | ~7 sec |
| 7 | Per-region cardinality voting | post-stage 2 | image (auto-mask) | LOW-MEDIUM | (subsumed by #1) | ~5 min (too slow) |
| 8 | Reverse-direction re-segmentation | post-`build_tracks` | video, single-obj | MEDIUM | +0.002-0.005 | ~30 sec |
| 9 | Phantom shape stats | stage 4 | image (box) | LOW-MEDIUM | (subsumed by #5) | ~3 sec |
| 10 | Borderline-only SAM tie-breaker | stage 3 | image (box) | **HIGH** (cost/benefit) | +0.002-0.005 | <1 sec |
| 11 | SAM-validated long-gap stitch | stage 5 | image (point) | LOW-MEDIUM | +0.0005-0.002 | <1 sec |
| 12 | darkTest low-light supplement | stage 1 | image (auto-mask) | HIGH for darkTest | (darkTest only) | ~28 sec |

---

## Section 2 — Hardware / cost analysis on A10 GPU

### A10 specs and budget

| Resource | A10 | Used by current pipeline | Available for SAM |
|---|---|---|---|
| VRAM | 24 GB | YOLO26s ~3 GB (with TRT engine), OSNet x0.25 ~0.5 GB, OS+CUDA overhead ~2 GB | ~18 GB free |
| FP16 TFLOPS (dense) | 31 | shared | shared (SAM 2.1 inference is FP16 on A10) |
| Wall-time budget | <10 min for ~30 s clip | currently ~30-60 s | ~9 min headroom |

### SAM 2.1 model size and per-image latency on A10

A10 is roughly 0.4× A100 dense FP16 throughput. The SAM 2 paper (arXiv:2408.00714) reported A100 numbers; scaling to A10:

| Variant | Params | VRAM (load) | Image-encoder latency (A100) | Image-encoder latency (A10, scaled) | Image-predictor end-to-end latency (A10) |
|---|---|---|---|---|---|
| `sam2.1_hiera_tiny` | 39 M | ~2 GB | ~11 ms (91 FPS) | ~25 ms (40 FPS) | ~28 ms |
| `sam2.1_hiera_small` | 46 M | ~2.5 GB | ~12 ms (83 FPS) | ~28 ms (35 FPS) | ~32 ms |
| `sam2.1_hiera_base_plus` | 81 M | ~4 GB | ~22 ms (44 FPS) | ~50 ms (20 FPS) | ~55 ms |
| `sam2.1_hiera_large` | 224 M | ~8 GB | ~35 ms (28 FPS) | ~80 ms (12 FPS) | ~85 ms |

Add ~3-5 ms per additional prompt set (mask-decoder forward), assuming the image encoder cache is reused.

For batched per-frame multi-object inference (Strategy 3), the encoder runs once per frame and the decoder runs N times (N = current dancer count). For 14 dancers per frame, encoder + 14 × 4 ms decoder ≈ 50 + 56 = 106 ms with `base_plus`, or 25 + 56 = 81 ms with `tiny`.

### Throughput envelope

For a 30 s 30 fps clip (900 frames):

| Strategy invocation pattern | Frames touched | Per-frame cost (`tiny`) | Per-frame cost (`base_plus`) | Total wall time (`tiny` / `base_plus`) |
|---|---|---|---|---|
| Every frame, one box | 900 | ~28 ms | ~55 ms | 25 s / 50 s |
| Every frame, batched (N=14 boxes) | 900 | ~80 ms | ~110 ms | 72 s / 100 s |
| Every 3rd frame, batched | 300 | ~80 ms | ~110 ms | 24 s / 33 s |
| Cardinality-gated only (~15% of frames) | ~135 | ~28 ms | ~55 ms | 4 s / 8 s |
| Per-merge-candidate only (~50 pairs × 2) | 100 | ~28 ms | ~55 ms | 3 s / 6 s |

**Recommendation:** start with `sam2.1_hiera_base_plus` for all experiments. It's the sweet spot — the SAM 2 paper shows accuracy plateaus between `base_plus` and `large`, and `base_plus` is 1.5× faster on A10. Drop to `tiny` only if Strategy 3's full per-frame per-object inference is needed and 25 sec/clip is too slow.

### Memory headroom

YOLO26s (3 GB) + OSNet (0.5 GB) + SAM 2.1 base_plus (4 GB) + activations (~3 GB) + CUDA runtime (~2 GB) ≈ **12.5 GB** total. Comfortable on a 24 GB A10. Even SAM 2.1 large fits with room to spare.

If running Strategy 3 alongside the existing pipeline (i.e., SAM model resident the whole time), the steady-state VRAM of base_plus + YOLO + OSNet is ~10 GB, leaving ~14 GB for activations.

---

## Section 3 — Top 3 recommended experiments

### Experiment 1 — Exclusion-prompted SAM 2.1 FN recovery (Strategy 1)

**Why this one:** Largest expected lift on the dominant error source (loveTest's 2,083 FN), structurally avoids identity fusion (no video predictor), small implementation footprint (touches only `tracking.run_pipeline._detect_and_track_serial` via a post-loop pass and the cache schema).

**Pseudocode:**

```python
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.postprocess import _track_extrapolate_box, Track

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")

def recover_fn_with_sam(frames_cache, video_path, osnet_extractor, *,
                        running_median_window=30,
                        cardinality_drop_thresh=1,
                        mask_area_min_ratio=0.4,
                        mask_area_max_ratio=1.5,
                        osnet_cos_thresh=0.75,
                        recovered_conf=0.50):
    """Augment frames_cache (list[FrameDetections]) in-place with
    synthetic FN-recovery detections, gated by SAM 2.1 + OSNet."""

    counts = np.array([len(fd.tids) for fd in frames_cache])
    running_median = np.array([
        np.median(counts[max(0, i-running_median_window//2):
                         min(len(counts), i+running_median_window//2)])
        for i in range(len(counts))
    ])

    all_video_frames = list(iter_video_frames(video_path))

    raw_tracks = build_track_index(frames_cache)

    for t, fd in enumerate(frames_cache):
        if counts[t] > running_median[t] - cardinality_drop_thresh:
            continue

        active_recently = [
            tr for tr in raw_tracks.values()
            if t-5 <= tr.frames[-1] < t and t not in tr.frames
        ]
        if not active_recently:
            continue

        present_centers = [
            ((fd.xyxys[k,0]+fd.xyxys[k,2])/2, (fd.xyxys[k,1]+fd.xyxys[k,3])/2)
            for k in range(len(fd.tids))
        ]

        frame_bgr = all_video_frames[t][1]
        predictor.set_image(frame_bgr)

        new_dets = []
        for tr in active_recently:
            fake = Track(track_id=tr.track_id, frames=tr.frames,
                         bboxes=tr.bboxes, confs=tr.confs, masks=None,
                         detected=np.ones(len(tr.frames), dtype=bool))
            pred_box = _track_extrapolate_box(fake, t)

            point_coords = np.array(present_centers, dtype=np.float32) \
                if present_centers else None
            point_labels = np.zeros(len(present_centers), dtype=np.int32) \
                if present_centers else None

            masks, scores, _ = predictor.predict(
                box=pred_box[None, :],
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = masks[0]

            mask_area = mask.sum()
            bbox_area = (pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1])
            if not (mask_area_min_ratio*bbox_area <= mask_area
                    <= mask_area_max_ratio*bbox_area):
                continue

            ys, xs = np.where(mask)
            sam_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()],
                               dtype=np.float32)

            crop = frame_bgr[int(sam_box[1]):int(sam_box[3]),
                             int(sam_box[0]):int(sam_box[2])]
            embed_new = osnet_extractor(crop)
            embed_old = tr.embeds[-1]
            cos = float(np.dot(embed_new/np.linalg.norm(embed_new),
                               embed_old/np.linalg.norm(embed_old)))
            if cos < osnet_cos_thresh:
                continue

            new_dets.append((sam_box, recovered_conf, tr.track_id))

        if new_dets:
            xyxys = np.concatenate([fd.xyxys, np.stack([d[0] for d in new_dets])])
            confs = np.concatenate([fd.confs, np.array([d[1] for d in new_dets],
                                                      dtype=np.float32)])
            tids = np.concatenate([fd.tids, np.array([d[2] for d in new_dets],
                                                    dtype=np.float32)])
            frames_cache[t] = FrameDetections(xyxys, confs, tids)

    return frames_cache
```

This slots into `tracking/run_pipeline.py` after line 344 (between cache write and `build_tracks` call).

**Exact metric:** mean IDF1 across the 7-clip benchmark, plus per-clip `(IDF1, FP, FN, IDS)` from `scripts/eval_per_clip.py`.

**Expected lifts:**
| Clip | Current | After Exp 1 | Δ |
|---|---:|---:|---:|
| `loveTest` | 0.8533 | **0.860 - 0.870** | +0.007 to +0.017 |
| `MotionTest` | 0.9321 | **0.935 - 0.940** | +0.003 to +0.008 |
| `darkTest` | (no baseline) | (significant lift expected) | — |
| Mean (7 clips) | 0.9570 | **0.960 - 0.962** | +0.003 to +0.005 |

**Kill criterion:** abandon if (a) `loveTest` lift is < +0.003, OR (b) any clip regresses by > -0.002 (the v8 strict-no-regression rule), OR (c) wall time per clip exceeds 90 s (3× current baseline).

---

### Experiment 2 — Mask-IoU occlusion-aware ReID gallery hygiene (Strategy 3)

**Why this one:** Strong literature precedent (SAM2MOT +4.5 IDF1 on DanceTrack from the analogous Cross-Object Interaction module, arXiv:2504.04519). Attacks the *root cause* of ReID gate fragility on close-contact clips, which the v8 OSNet cosine ≥ 0.7 gate can't structurally fix.

**Pseudocode:**

```python
def annotate_occlusion_in_cache(frames_cache, video_path, *,
                                stride=3, mask_iou_thresh=0.4):
    """Add per-frame `occluded_tids: set[int]` annotations to the cache."""
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2.1-hiera-base-plus"
    )
    all_frames = list(iter_video_frames(video_path))

    occluded_tids_per_frame = [set() for _ in frames_cache]

    for t in range(0, len(frames_cache), stride):
        fd = frames_cache[t]
        if len(fd.tids) < 2:
            continue

        predictor.set_image(all_frames[t][1])
        boxes = fd.xyxys.astype(np.float32)
        masks, _, _ = predictor.predict(box=boxes, multimask_output=False)

        masks = masks[:, 0] if masks.ndim == 4 else masks

        for i in range(len(fd.tids)):
            for j in range(i+1, len(fd.tids)):
                inter = (masks[i] & masks[j]).sum()
                union = (masks[i] | masks[j]).sum()
                if union == 0:
                    continue
                iou = inter / union
                if iou > mask_iou_thresh:
                    occluded_tids_per_frame[t].add(int(fd.tids[i]))
                    occluded_tids_per_frame[t].add(int(fd.tids[j]))

    return occluded_tids_per_frame

def hygiene_aware_id_merge_cos(ti, tj, occluded_per_frame):
    """Replacement for the OSNet cos calculation at
    tracking/postprocess.py:405-411. Uses last non-occluded
    embedding of A and first non-occluded embedding of B."""
    if ti.embeds is None or tj.embeds is None:
        return 1.0

    a_idx = next(
        (k for k in range(len(ti.frames)-1, -1, -1)
         if int(ti.track_id) not in occluded_per_frame[ti.frames[k]]),
        len(ti.frames)-1
    )
    b_idx = next(
        (k for k in range(len(tj.frames))
         if int(tj.track_id) not in occluded_per_frame[tj.frames[k]]),
        0
    )

    a = ti.embeds[a_idx] / (np.linalg.norm(ti.embeds[a_idx]) + 1e-9)
    b = tj.embeds[b_idx] / (np.linalg.norm(tj.embeds[b_idx]) + 1e-9)
    return float(np.dot(a, b))
```

This requires:
1. Cache schema extension: `FrameDetections` (`prune_tracks.py:14-18`) gains an optional `occluded_tids: np.ndarray` field. Backwards-compatible default.
2. The `_id_merge` function (`tracking/postprocess.py:331`) takes an extra `occluded_per_frame` arg, threaded from `build_tracks` (`tracking/best_pipeline.py:275-296`).

**Exact metric:** mean IDF1 + per-clip IDF1, *plus* report the count of (a) merge candidates whose cos changed by ≥ 0.05 between baseline and SAM-hygiene OSNet, and (b) merge decisions flipped by the new cos.

**Expected lifts:**
| Clip | Current | After Exp 2 | Δ |
|---|---:|---:|---:|
| `loveTest` | 0.8533 | **0.860 - 0.866** | +0.007 to +0.013 |
| `MotionTest` | 0.9321 | 0.933 - 0.935 | +0.001 to +0.003 |
| `BigTest` | 0.9981 | 0.998 - 0.999 | 0 to +0.001 |
| Mean (7 clips) | 0.9570 | **0.959 - 0.961** | +0.002 to +0.004 |

**Kill criterion:** abandon if (a) `loveTest` lift < +0.005, OR (b) any clip regresses > -0.002, OR (c) the count of "flipped merge decisions" is < 5 across the 7-clip benchmark (= occlusion gating isn't actually changing anything in practice; the existing OSNet cos is already robust enough).

---

### Experiment 3 — Mask-area-trajectory phantom filter (Strategy 5)

**Why this one:** Cheap, structurally orthogonal to the existing AND-gate, attacks the residual fragility of the `p90 ≥ 0.84` confidence gate (the gap between phantom 0.835 and worst-real 0.860 is only 0.025, per experiments log §2.5). If it works, it both raises the IDF1 and *enables loosening the p90 gate* to recover real short tracks on shorterTest.

**Pseudocode:**

```python
def filter_tracks_by_mask_trajectory(tracks, video_path, *,
                                     n_samples=15,
                                     cv_area_thresh=0.6,
                                     bg_coverage_thresh=0.2,
                                     min_largest_component_ratio=0.7):
    """Drop tracks whose SAM mask trajectory looks non-physical.
    Slot in between filter_tracks_post_merge and bbox_continuity_stitch
    in tracking/best_pipeline.py:299-302."""
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2.1-hiera-base-plus"
    )

    out = {}
    for tid, tr in tracks.items():
        n = len(tr.frames)
        sample_idxs = np.linspace(0, n-1, num=min(n_samples, n)).astype(int)

        areas = []
        coverages = []
        largest_comp_ratios = []
        for k in sample_idxs:
            t = int(tr.frames[k])
            box = tr.bboxes[k]

            frame_bgr = read_frame(video_path, t)
            predictor.set_image(frame_bgr)
            masks, _, _ = predictor.predict(
                box=box[None, :], multimask_output=False,
            )
            mask = masks[0]

            area = mask.sum()
            bbox_area = (box[2]-box[0]) * (box[3]-box[1])

            from scipy.ndimage import label
            labeled, n_comp = label(mask)
            if n_comp == 0:
                continue
            largest = max((labeled == c).sum() for c in range(1, n_comp+1))

            areas.append(area)
            coverages.append(area / max(bbox_area, 1.0))
            largest_comp_ratios.append(largest / max(area, 1.0))

        if len(areas) < 5:
            out[tid] = tr
            continue

        areas = np.array(areas)
        cv_area = float(np.std(areas) / max(np.mean(areas), 1.0))
        bg_coverage_count = sum(1 for c in coverages if c < bg_coverage_thresh)
        bad_largest = sum(1 for r in largest_comp_ratios
                          if r < min_largest_component_ratio)

        if (cv_area > cv_area_thresh
                or bg_coverage_count > 0.4 * len(coverages)
                or bad_largest > 0.4 * len(largest_comp_ratios)):
            log.info("phantom-filter: dropping tid=%d cv_area=%.2f "
                     "bg_count=%d bad_largest=%d",
                     tid, cv_area, bg_coverage_count, bad_largest)
            continue

        out[tid] = tr

    return out
```

Insert at `tracking/best_pipeline.py:301` (between `filter_tracks_post_merge` and `bbox_continuity_stitch`).

**Exact metric:** per-clip `(IDF1, FP, FN, n_tracks_kept, n_tracks_dropped)`.

**Expected lifts:**
| Clip | Current | After Exp 3 | Δ |
|---|---:|---:|---:|
| `loveTest` | 0.8533 | 0.855 - 0.858 | +0.002 to +0.005 |
| `mirrorTest` | 0.9935 | **0.996 - 1.000** | +0.003 to +0.007 |
| `shorterTest` (with relaxed p90 to 0.82) | 0.9221 | 0.924 - 0.928 | +0.002 to +0.006 |
| Mean (7 clips) | 0.9570 | **0.959 - 0.961** | +0.002 to +0.004 |

**Kill criterion:** abandon if (a) more than 1 real (high-IDF1-contributor) track is dropped on any clip in the 7-clip benchmark (this is the strict no-regression rule applied at the track level), OR (b) `mirrorTest` doesn't lift by ≥ +0.001, OR (c) cv_area/coverage thresholds need >5 sweep iterations to find a no-regression setting (it's not a robust signal).

---

### Stretch goal — Strategy 10 as a "free" addendum

Once Experiment 1 has been validated, Strategy 10 (SAM as borderline-only `_id_merge` tie-breaker) costs <1 sec per clip and adds independent signal — recommend running as a follow-up experiment with no kill criterion (it's effectively free).

---

## Section 4 — References

### Primary SAM 2.1 references

- **SAM 2 paper.** Ravi, Gabeur, Hu et al., "SAM 2: Segment Anything in Images and Videos," arXiv:2408.00714, July 2024. The canonical paper, including A100 latency tables for Hiera-tiny / -small / -base+ / -large.
- **SAM 2.1 release.** [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2), September 2024 update. Improved checkpoints, same architecture. Model checkpoints:
  - [`facebook/sam2.1-hiera-tiny`](https://huggingface.co/facebook/sam2.1-hiera-tiny) (39 M params, 2 GB VRAM)
  - [`facebook/sam2.1-hiera-small`](https://huggingface.co/facebook/sam2.1-hiera-small) (46 M params, 2.5 GB VRAM)
  - [`facebook/sam2.1-hiera-base-plus`](https://huggingface.co/facebook/sam2.1-hiera-base-plus) (81 M params, 4 GB VRAM) **← recommended**
  - [`facebook/sam2.1-hiera-large`](https://huggingface.co/facebook/sam2.1-hiera-large) (224 M params, 8 GB VRAM)
- **`SAM2ImagePredictor` API.** [`sam2/sam2_image_predictor.py`](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py). Supports `box`, `point_coords` + `point_labels` (positive=1, negative=0), `mask_input`, and `multimask_output`. Multiple boxes batched per call return `(N, 1, H, W)` masks.

### SAM-for-MOT papers (most relevant first)

- **SAM2MOT.** Jiang, Wang, Zhao et al., "SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation," arXiv:2504.04519v2, April 2025. ETH/Huawei. **Directly relevant** — Cross-Object Interaction module uses mIoU > 0.8 on segmentation masks to detect occlusion, then logits-variance to identify the occluded object and purge corrupted memory frames. Reports +4.5 IDF1 on DanceTrack vs the previous best (a same-uniform-crowd benchmark closely analogous to `loveTest`). Code: [`TripleJoy/SAM2MOT`](https://github.com/TripleJoy/SAM2MOT). This is the strongest evidence that SAM-mask-based occlusion handling materially improves MOT in your setting.
- **MASA.** Li, Tu, Li et al., "Matching Anything by Segmenting Anything," CVPR 2024, arXiv:2406.04221. ETH/INSAIT. Trains a tracking adapter on SAM-derived dense proposals + contrastive learning, no labeled video needed. Project page: [matchinganything.github.io](https://matchinganything.github.io/). Less directly relevant (focused on open-vocabulary), but the "SAM-as-proposal-generator" framing supports Strategies 1 and 7.
- **SAMURAI.** Yang, Loy et al., "SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory," 2024. Adds Kalman-based motion modeling and memory cleaning to SAM 2. Code: [`yangchris11/samurai`](https://github.com/yangchris11/samurai). Relevant for understanding the SAM-2-as-tracker failure mode the user already hit; the SAMURAI memory-cleaning idea is what motivates the *occluded-frame purging* in Strategy 3.
- **SambaMOTR (ICLR 2025).** Synchronized set-of-sequences modeling with `MaskObs` for occlusion-uncertain frames. SOTA on DanceTrack, BFT, SportsMOT. Project: [sambamotr.github.io](http://sambamotr.github.io/).
- **EfficientTAM (ICCV 2025).** Xiong et al., "Efficient Track Anything," arXiv:2411.18933. 1.6× speedup over SAM 2 Hiera-B+ on A100. EfficientTAM-Mobile is 4.6× faster. **Useful as a swap-in for `sam2.1_hiera_base_plus` if Strategy 3's per-frame cost is too high.**
- **Efficient-SAM2 (ICLR 2026).** 1.68× speedup on SAM2.1-L with 1.0% accuracy drop via Sparse Window Routing + Sparse Memory Retrieval, post-training. arXiv:2602.08224.

### Mask-aware ReID (relevant for Strategy 6)

- **AOANet.** "Adaptive Occlusion-Aware Network for Occluded Person Re-Identification," IEEE TCSVT 2025, vol 35 p 5067. 70.6 % mAP on Occluded-Duke with adaptive occlusion-weighted features.
- **Mask-Aware Hierarchical Aggregation Transformer.** IEEE 2024-2025 (DOI: 10.1109/[…]/10844683). Transformer-based mask-aware ReID.
- **RGANet.** "Region Generation and Assessment Network," arXiv:2309.03558. CLIP-based body-region prototypes for occlusion-robust ReID.
- **Background Suppression Framework.** Zhao et al., arXiv:2509.03032, 2025. Adversarial background suppression for ReID. Relevant for the "mask out other dancers in same uniform" use case.

### Code repositories worth pulling

- [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2) — official reference implementation, includes both image and video predictors, `SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")`.
- [`TripleJoy/SAM2MOT`](https://github.com/TripleJoy/SAM2MOT) — full pipeline including the Cross-Object Interaction module (Strategy 3 inspiration).
- [`yangchris11/samurai`](https://github.com/yangchris11/samurai) — motion-aware memory selection (relevant for any future video-predictor experiment).
- [`yformer/EfficientTAM`](https://github.com/yformer/EfficientTAM) — drop-in faster SAM2 alternative if image-encoder cost is a concern.

### Implementation guidance specific to this codebase

The minimum viable integration touches:
- `tracking/run_pipeline.py:289-354` (`run_pipeline_on_video`) — add a `--sam-fn-recovery` flag and call the recovery function between cache write (line 346) and `build_tracks` (line 350).
- `prune_tracks.py:14-18` (`FrameDetections`) — extend with optional `occluded_tids` and `synthetic` fields (Strategy 1, 3) — backwards-compatible defaults.
- `tracking/postprocess.py:331` (`_id_merge`) — accept an optional `occluded_per_frame` arg and use it in the cosine calculation at lines 405-411 (Strategy 3, 6, 10).
- `tracking/best_pipeline.py:299-302` — insert the `filter_tracks_by_mask_trajectory` stage between `filter_tracks_post_merge` and `bbox_continuity_stitch` (Strategy 5).

Every strategy is **additive and feature-flagged**, so it can be A/B tested against the v8 baseline using `scripts/regression_check.py` (which the repo already uses for lossless-speed-flag validation).

---

## Closing recommendation

Run experiments in this order, each gated by its kill criterion before proceeding:

1. **Experiment 1 (Strategy 1) first.** Highest expected lift on the dominant error source. If it lifts loveTest by ≥ +0.005 with no regression, ship it.
2. **Experiment 3 (Strategy 5) second.** Cheapest, most orthogonal to existing gates. If `mirrorTest` lifts and no real track is dropped, ship it. Then explore loosening `POST_MIN_P90_CONF` from 0.84 to 0.82 (which the experiments log shows currently regresses by -0.0010 — but with the SAM phantom filter as a safety net, that regression should disappear).
3. **Experiment 2 (Strategy 3) third.** Highest cost, highest theoretical ceiling. Run only after Experiments 1 and 3 have established a new baseline. If it lifts loveTest by ≥ +0.005 *on top of* the new baseline, ship it.
4. **Strategy 10** as a free addendum once Experiment 1 has been validated.

Total expected mean-IDF1 lift if all three top experiments succeed:  **0.957 → 0.965-0.972** (+0.008 to +0.015), with `loveTest` plausibly reaching **0.870-0.885** — the first crack of the close-contact bottleneck since the v1 → v8 chain saturated.

 I'm pulling the official SAM 2 paper specs as my baseline reference, then scaling down to A10 hardware using the typical 0.4-0.5x performance ratio for FP16 operations between the two GPUs.

The paper reports around 44 FPS on A100 for the larger Hiera-B+ model (roughly 22ms per image), which translates to about 50ms on A10, while the smaller Hiera-tiny hits 91 FPS on A100 (11ms) and would run at around 25ms on A10.

I have the numbers I need now, so I'm ready to write up the final report with specific file references and actionable details.