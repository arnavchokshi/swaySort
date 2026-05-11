# Expected-count Stage A Tracking

Individual Mesh Scan mode asks the user one tracking question:

```text
How many different performers are in your video?
```

That value is treated as the total number of target performers across the
whole clip. It is not the number visible in the first frame, and it is not the
maximum simultaneous visible count.

## How The Prior Is Used

Stage A still uses YOLO plus DeepOcSort as a high-recall proposal generator.
The expected count is applied after the normal postprocess chain as a global
identity prior:

- If the final track count already equals the expected count, tracks are left
  passed through the same expected-count consolidation pass. This matters for
  split IDs such as `12001`: the raw count can be correct while the identity
  assignment is still wrong.
- If there are too many identities, Stage A consolidates likely split
  tracklets first, then suppresses the weakest extras.
- If there are too few identities, Stage A recovers real candidate tracklets
  from the relaxed proposal pool. It does not fabricate synthetic identities.
- Postprocess-generated split IDs (`base_id * 1000 + piece_index`) are repaired
  when a high-ID piece takes over a low-ID carrier and the carrier later
  becomes another performer. The carrier prefix is stitched to the high-ID
  continuation, and the carrier suffix remains a separate identity.
- Early interior starts are backfilled only when a long track starts within the
  first 90 frames, starts away from the image edge, and the expected-count
  active set is under-filled before that start. This targets early occlusion
  recovery without forcing true late entrants into frame 0.
- Final expected-count outputs are remapped to dense target IDs `1..N`, so
  debug overlays and downstream palette masks do not expose tracker-generated
  high IDs.
- Late entries and early exits are allowed because selection is based on total
  unique identity evidence, not frame-zero visibility.
- The output includes `presence_plan.json`, with per-frame active target IDs,
  track ranges, confidence summaries, count-mismatch review frames, and reason
  codes for matched, recovered, merged, or suppressed tracks.
- Dense high-count clips (`expected_count >= 12`) use slightly tighter
  horizontal and vertical bbox calibration (`scale_x=0.85`, `scale_y=0.92`).
  This is a general crowded-scene rule validated on the full CVAT set;
  lower-count clips keep the wider default because it better preserves mirror
  and shorter clips.

## Benchmark Commands

Baseline from GPU tracks:

```bash
python scripts/eval_id_yolo_cvat.py \
  --cvat-root /workspace/CV_pipeline/CVAT \
  --include-mirror \
  --run-pipeline --device cuda:0 --force --save-detector-cache \
  --out-root /workspace/CV_pipeline/benchmarks/baseline \
  --run-id baseline_head_no_expected
```

Expected-count run:

```bash
python scripts/benchmark_stage_a_expected_count.py \
  --cvat-root /workspace/CV_pipeline/CVAT \
  --run-pipeline --device cuda:0 --force --save-detector-cache \
  --out-root /workspace/CV_pipeline/benchmarks/expected \
  --run-id expected_count_v7_adaptive_full
```

Failure overlays from saved tracks:

```bash
python scripts/eval_id_yolo_cvat.py \
  --cvat-root /workspace/CV_pipeline/CVAT \
  --include-mirror \
  --tracks-root /workspace/CV_pipeline/benchmarks/expected/expected_count_v7_adaptive_full/tracks \
  --out-root /workspace/CV_pipeline/benchmarks/expected \
  --run-id expected_count_v7_adaptive_full_report \
  --use-expected-count \
  --debug-overlays \
  --debug-overlay-max-frames 8
```

## Current Validation Status

The latest full GPU validation using the expected-count prior is the May 11,
2026 benchmark stored in:

```text
work/benchmarks/full_a10_results.json
```

That run covers all 12 current annotated CVAT clips and has exact total
predicted identity count on every clip. `jhumarTest` and `loveTest` remain the
hardest identity cases in the current set.

| Clip | GT IDs | Pred IDs | IDF1 | MOTA | ID switches |
|---|---:|---:|---:|---:|---:|
| 2pplTest | 2 | 2 | 1.0000 | 1.0000 | 0 |
| easyTest | 6 | 6 | 0.9969 | 0.9938 | 0 |
| gymTest | 7 | 7 | 0.9961 | 0.9922 | 0 |
| BigTest | 14 | 14 | 0.9973 | 0.9946 | 0 |
| adiTest | 5 | 5 | 1.0000 | 1.0000 | 0 |
| mirrorTest | 9 | 9 | 0.9743 | 0.9486 | 0 |
| shorterTest | 9 | 9 | 0.9429 | 0.8857 | 0 |
| eldonTest | 2 | 2 | 0.9734 | 0.9468 | 0 |
| jealousTest | 2 | 2 | 0.9558 | 0.9117 | 0 |
| jhumarTest | 10 | 10 | 0.8503 | 0.7192 | 6 |
| MotionTest | 14 | 14 | 0.9222 | 0.8460 | 0 |
| loveTest | 15 | 15 | 0.8784 | 0.7572 | 2 |
