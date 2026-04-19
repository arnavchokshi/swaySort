# Base YOLO26s vs Ours — full-video, A10-GPU benchmark

**Status:** drafted, awaiting approval
**Date:** 2026-04-19
**Hardware:** Lambda Labs A10 (`ubuntu@141.148.49.145`, key `~/.ssh/pose-tracking.pem`)

## 1. Goal in one sentence

Re-benchmark this repo's *shipped* pipeline against the truly out-of-the-box
"`pip install ultralytics boxmot`" experience for **6 trackers × 9 clips ×
full video length**, measure every metric the user could want, and rewrite
the README around the result.

## 2. Why we're re-doing the existing benchmark

The current `work/benchmarks/per_clip_idf1.json` already does ours-vs-base
trackers, but it's not the comparison the user wants:

- **Existing benchmark** holds the YOLO front-end *constant* across all rows
  (our fine-tuned `weights/best.pt` @ multi-scale {768, 1024} ensemble).
  This isolates the *tracker* contribution, hiding the value of our custom
  weights.
- **What the user wants**: a head-to-head where the only thing we share with
  baselines is the YOLO architecture (YOLO26s). Baselines get
  Ultralytics' COCO-trained `yolo26s.pt` @ default settings; ours gets
  `weights/best.pt` + multi-scale + post-process chain. This shows the
  *combined* value of (custom weights + custom pipeline) vs the swap-in
  baseline a new user would actually get.
- The existing benchmark also only covers 4 clips (loveTest, MotionTest,
  shorterTest, mirrorTest). New benchmark covers all 9 (adds easyTest,
  gymTest, BigTest, adiTest, darkTest), full video length each.

## 3. The seven rows per clip

| Row label | Detector | Tracker | Post-process |
|---|---|---|---|
| **Ours (v9 shipped)** | `weights/best.pt` multi-scale {768, 1024}, conf=0.34, ensemble NMS @ 0.6, dark recovery v9 | DeepOcSort + OSNet x0.25 + Kalman jitter patch | Full chain (prune+interp+merge, AND-gate, bbox stitch, CV-gated size smoother, center median) |
| ByteTrack (base) | stock `yolo26s.pt` single-scale 640, conf=0.25, IoU=0.7, classes=[0] | `boxmot.ByteTrack()` | none |
| OcSort (base) | same | `boxmot.OcSort()` | none |
| HybridSort (base) | same | `boxmot.HybridSort(reid_weights=osnet_x0_25, with_reid=True)` | none |
| BotSort (base) | same | `boxmot.BotSort(reid_weights=osnet_x0_25)` | none |
| StrongSort (base) | same | `boxmot.StrongSort(reid_weights=osnet_x0_25)` | none |
| **DeepOcSort (base)** | same | `boxmot.DeepOcSort(reid_weights=osnet_x0_25)` (no Kalman patch, no postprocess) | none |

Six baseline rows + one "ours" row = 7 rows × 9 clips = **63 full-video runs**.

## 4. Metrics captured per (clip, row)

From `py-motmetrics` at IoU 0.5 on `<clip>/gt/gt.txt`:

- **Identity**: `idf1`, `idp`, `idr`, `num_unique_objects`,
  `num_predicted_unique_objects`
- **CLEAR**: `mota`, `motp`, `precision`, `recall`
- **Errors**: `num_switches` (IDS), `num_fragmentations`,
  `num_misses` (FN), `num_false_positives` (FP)
- **Track quality**: `mostly_tracked`, `partially_tracked`, `mostly_lost`
- **Frames**: `num_frames`

From wall-clock timing on the A10 (single-GPU, nothing else competing):

- `det_ms_per_frame` (mean), `tracker_ms_per_frame` (mean / median / p95),
  `postprocess_ms_total`, `wall_seconds`,
  `end_to_end_fps` (frames / wall),
  `gpu_peak_mb` (via `torch.cuda.max_memory_allocated`).

## 5. Architecture

### 5.1 New scripts (committed to repo)

- `scripts/run_full_benchmark.py` — driver. For each clip:
  1. Build the **base detector cache** (stock yolo26s.pt @ 640, single scale)
     once per clip, save next to the video as `<stem>.base_det_cache.pkl`.
  2. Replay through every base tracker, score, time, dump per-clip artifacts
     to `work/benchmarks/full/<clip>/<row>.{txt,json}`.
  3. Run the full shipped pipeline (`run_pipeline_on_video`) once per clip,
     score and time it.
  4. Roll up everything into `work/benchmarks/full_results.json`.
- `scripts/sync_to_a10.sh` — one-command rsync of repo + weights + the 9
  clip directories (videos + `gt/`) to the A10. Also sets up the venv on
  the remote and downloads `yolo26s.pt`.
- `scripts/render_all_overlays.py` — produces a side-by-side overlay video
  for each clip showing **Ours vs Worst-base** so the README can embed
  short looping clips that "show, not tell."

### 5.2 New chart generation

`scripts/generate_comparison_charts.py` is extended (NOT replaced) to also
produce, from `work/benchmarks/full_results.json`:

- `docs/figures/headline_idf1.png` — single horizontal bar per row, mean
  IDF1 across 9 clips, ours highlighted, with the headline gap annotated.
- `docs/figures/per_clip_idf1_grid.png` — 9-panel grid (one panel per
  clip), 7 bars per panel.
- `docs/figures/error_breakdown.png` — stacked bar per row: (FP, FN, IDS)
  totals across all 9 clips. Shows *what kind* of errors each tracker
  makes, not just the IDF1 number.
- `docs/figures/speed_vs_accuracy.png` — refresh of existing chart with
  A10 numbers and all 7 rows.
- `docs/figures/lift_decomposition.png` — for each clip, three deltas:
  (ours - best base), (ours - mean base), (ours - worst base). Tells the
  reader where the gap is biggest.

### 5.3 Outputs committed

- `work/benchmarks/full_results.json` — the single source of truth
  (every row × every clip × every metric × every timing).
- `work/benchmarks/full_summary.md` — auto-generated markdown table the
  README pulls from.
- `docs/figures/*.png` — the new charts.
- `docs/videos/*.mp4` — short side-by-side overlays per clip
  (≤ 15 sec, the densest GT-tracked window per clip).

## 6. README rewrite outline

Per user request, the README needs to read like a *breakthrough headline*,
not a doc index. Proposed structure (target ≤ ~250 lines):

1. **Top fold** (no scroll needed):
   - One-paragraph pitch ("X dance dataset, Y-clip benchmark, +Z IDF1
     pp over the best out-of-the-box YOLO26s+BoxMOT pipeline").
   - The headline IDF1 bar chart (`headline_idf1.png`).
   - The two strongest side-by-side overlay GIFs (loveTest + MotionTest).
2. **Per-clip results** — the full 9-clip table + grid PNG.
3. **Where the gap lives** — error-breakdown chart + 1-2 paragraphs
   explaining "the gap is in IDS and FN, not FP."
4. **Speed** — A10 FPS table + speed-vs-accuracy scatter, no MPS
   detour in the headline.
5. **Reproduce on your own video** — the `pip install`, `python -m
   tracking.run_pipeline …` recipe (current Quickstart, slimmed).
6. **Reproduce the benchmark** — one paragraph + one command pointing at
   `scripts/run_full_benchmark.py`.
7. **Repository layout / output schema / programmatic use / CLI** — kept
   short, links to `docs/PIPELINE_SPEC.md` for details.

The current "Headline result" / "Side-by-side ours vs StrongSort" /
"v1→v8 lift" sections move to `docs/EXPERIMENTS_LOG.md` as historical
notes. Their data is preserved.

## 7. Execution plan (high level)

1. Spec + plan committed.
2. Write `scripts/run_full_benchmark.py` + `scripts/sync_to_a10.sh` locally.
   Smoke-test the driver on one clip (capped at 50 frames) on the user's
   Mac to catch any bugs offline.
3. `sync_to_a10.sh` pushes everything (~5-15 GB of video) to A10.
4. Set up the A10 env: torch+cu124, ultralytics 8.4.37, boxmot 18.0.0,
   `yolo26s.pt` download, `osnet_x0_25_msmt17.pt` ReID weight (auto on
   first BoxMOT run).
5. Run `scripts/run_full_benchmark.py` on the A10 — full 9 clips × 7 rows
   end-to-end. Estimated wall-time: 4-8 hours (will measure & update).
6. Pull results back, regenerate charts locally, render overlay videos
   locally (mp4 encoding is faster on Mac than installing ffmpeg-with-h264
   on the A10).
7. Rewrite README as outlined in §6, update `EXPERIMENTS_LOG.md` with
   the moved sections, commit everything.
8. Sanity check: `scripts/smoke_test.py` still passes; the README links
   resolve; the `docs/figures/*.png` actually exist (they currently do
   NOT in the repo, but are referenced — this fixes that pre-existing
   bug as a side effect).

## 8. Risks + mitigations

- **A10 disk fill**: 9 clips can be 5+ GB. We have 1.3 TB free — fine.
- **A10 already running another job**: `nvidia-smi` shows no processes,
  but if one starts mid-bench we re-run the affected clip.
- **ReID weight download fails behind firewall**: pre-download
  `osnet_x0_25_msmt17.pt` locally and rsync to remote.
- **`yolo26s.pt` not at expected URL**: Ultralytics' platform.ultralytics.com
  may require account login. Fallback: use `YOLO("yolo26s.pt")` which
  triggers Ultralytics' auto-download from their CDN; if that fails,
  we manually download and place it.
- **Ground-truth FPS mismatch**: each clip's `gt.txt` is 1-indexed;
  `eval_per_clip.py` already handles this. We reuse that scoring code.
- **boxmot 18.0.0 quirks per tracker**: existing `_build_tracker` already
  knows the constructor signatures; we reuse it verbatim and only swap
  the detector cache.

## 9. Out of scope

- Sweeping hyperparameters of any baseline tracker (we keep them at
  out-of-box defaults — that's the whole point).
- Comparing against non-BoxMOT trackers (CAMELTrack, BoT-SORT++, etc).
  The user listed exactly 5 trackers + ours; we honor the list.
- Re-training anything. Custom weights are loaded from
  `weights/best.pt`, untouched.
- Mobile / Jetson / on-device speed numbers. A10 only.
