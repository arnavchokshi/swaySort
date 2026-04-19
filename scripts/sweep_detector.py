"""Tier-2 (detector) sweep harness for the BEST_ID pipeline.

Re-runs the full pipeline (detect+track+postprocess) per variant. Each
variant supplies a set of environment variables that toggle dark-recovery
features in ``tracking.dark_recovery`` (gamma, CLAHE, multi-exposure,
adaptive conf, Soft-NMS) plus optional ``BEST_ID_*`` overrides.

Variants live in a JSON file::

    {
      "variants": [
        {
          "name": "baseline",
          "env": {}
        },
        {
          "name": "gamma_auto",
          "env": {"BEST_ID_DARK_GAMMA": "auto"}
        },
        {
          "name": "clahe_only",
          "env": {"BEST_ID_DARK_CLAHE": "1"}
        }
      ]
    }

For each variant, the runner:
  1. Sets the env vars in the current process.
  2. Runs the full pipeline on each requested clip.
  3. Scores the resulting tracks vs. ground truth.
  4. Writes per-clip + summary results to
     ``<out_dir>/<variant>/scores.json`` and a single aggregated
     ``<out_dir>/sweep_results.json``.

Per-variant outputs are reused if they already exist so a partially
completed sweep can be resumed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("sweep_detector")


# Env vars we forward to subordinate workers (and reset between variants).
# Anything BEST_ID_* that the pipeline reads should be listed here so the
# variant gets the right value AND so previous variants' values don't leak.
ENV_KEYS = (
    # Detector + dark recovery
    "BEST_ID_GPU_NMS",
    "BEST_ID_DARK_PROFILE",
    "BEST_ID_DARK_GAMMA",
    "BEST_ID_DARK_CLAHE",
    "BEST_ID_DARK_LUMA",
    "BEST_ID_ADAPTIVE_CONF",
    "BEST_ID_DARK_BRIGHTEN",
    "BEST_ID_SOFT_NMS",
    "BEST_ID_IMGSZ_ENSEMBLE",
    "BEST_ID_ENSEMBLE_IOU",
    "BEST_ID_TTA_FLIP",
    # Cardinality-voting FN recovery
    "BEST_ID_FN_RECOVERY",
    "BEST_ID_FN_RECOVERY_DROP",
    "BEST_ID_FN_RECOVERY_WINDOW",
    "BEST_ID_FN_RECOVERY_LOOKBACK",
    "BEST_ID_FN_RECOVERY_IOU",
    "BEST_ID_FN_RECOVERY_CONF",
    "BEST_ID_FN_RECOVERY_MIN_HIST",
    "BEST_ID_FN_RECOVERY_MAX_DISP",
    # SAM 2.1 per-bbox verifier
    "BEST_ID_SAM_VERIFY",
    "BEST_ID_SAM_VERIFY_WEIGHTS",
    "BEST_ID_SAM_VERIFY_CFG",
    "BEST_ID_SAM_VERIFY_FILL",
    "BEST_ID_SAM_VERIFY_CONF_MAX",
    "BEST_ID_SAM_VERIFY_AREA_MAX",
    "BEST_ID_SAM_VERIFY_STRIDE",
    "BEST_ID_SAM_VERIFY_DEVICE",
    # RTMW pose-aware ID merge gate
    "BEST_ID_POSE_MERGE",
    "BEST_ID_POSE_MERGE_THRESH",
    "BEST_ID_POSE_MERGE_BODY_W",
    "BEST_ID_POSE_MERGE_HAND_W",
    "BEST_ID_POSE_MERGE_FACE_W",
    "BEST_ID_POSE_MERGE_MIN_VIS",
    "BEST_ID_POSE_MERGE_MODE",
)


def _set_env(env: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Set each key in env, return a snapshot of the previous values."""
    prev: Dict[str, Optional[str]] = {}
    for k in ENV_KEYS:
        prev[k] = os.environ.get(k)
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = str(v)
    return prev


def _restore_env(prev: Dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _load_variants(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    variants = data.get("variants", [])
    out: List[Dict[str, Any]] = []
    for v in variants:
        if "name" not in v:
            raise ValueError(f"variant without name: {v}")
        env = v.get("env", {})
        if not isinstance(env, dict):
            raise ValueError(f"variant env must be dict: {v}")
        out.append({"name": v["name"], "env": env, "notes": v.get("notes", "")})
    return out


def _load_clips(manifest: Path) -> List[Tuple[str, Path]]:
    data = json.loads(manifest.read_text())
    out: List[Tuple[str, Path]] = []
    for c in data.get("clips", []):
        out.append((c["name"], Path(os.path.expanduser(c["video"]))))
    return out


def _gt_path(clip_name: str, gt_root: Path) -> Path:
    """Map clip name to GT path. Convention used by score_runs.py:
       <gt_root>/<clip>/gt/gt.txt.
    """
    return gt_root / clip_name / "gt" / "gt.txt"


def _run_pipeline_for_clip(
    *, video: Path, out_pkl: Path, device: str,
    max_frames: Optional[int],
) -> Tuple[float, int]:
    """Run the full pipeline (forces fresh cache, since env may differ)."""
    # Fresh import every call avoids cached modules holding stale env state
    # for the dark-recovery helpers; cheap because Python caches the .pyc.
    rp = importlib.import_module("tracking.run_pipeline")
    importlib.reload(rp)
    t0 = time.perf_counter()
    tracks = rp.run_pipeline_on_video(
        video=video, out=out_pkl, device=device,
        max_frames=max_frames, force=True,
    )
    return time.perf_counter() - t0, len(tracks)


def _score_clip(
    *, tracks_pkl: Path, gt_path: Path,
) -> Dict[str, float]:
    """Use the existing scoring helpers in scripts.score_runs."""
    sr = importlib.import_module("scripts.score_runs")
    rows = sr._tracks_to_mot_rows(tracks_pkl)
    if not gt_path.is_file():
        return {"error": f"missing gt: {gt_path}"}
    return sr._score_mot_rows_vs_gt(rows, gt_path)


def _summary(per_clip: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean over clips that produced numeric metrics."""
    out: Dict[str, float] = {}
    keys = ("idf1", "mota", "num_switches", "num_misses",
            "num_false_positives", "num_objects")
    for k in keys:
        vals: List[float] = []
        for v in per_clip.values():
            x = v.get(k)
            if isinstance(x, (int, float)):
                vals.append(float(x))
        if vals:
            if k.startswith("num_"):
                out[k + "_total"] = float(sum(vals))
            else:
                out["mean_" + k] = float(sum(vals) / len(vals))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", required=True, type=Path,
                    help="path to variant JSON")
    ap.add_argument("--clips-manifest", default=REPO / "configs" / "clips.json",
                    type=Path, help="path to clips manifest")
    ap.add_argument("--gt-root", required=True, type=Path,
                    help="dir containing <clip>/gt.pkl")
    ap.add_argument("--out", required=True, type=Path,
                    help="output dir")
    ap.add_argument("--device", default=os.environ.get("PIPE_DEVICE", "cuda:0"))
    ap.add_argument("--clips", nargs="*", default=None,
                    help="subset of clip names (default: all in manifest)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--variant-filter", nargs="*", default=None,
                    help="only run variants whose name is in this list")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip variants whose summary already exists")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.out / "sweep_results.json"

    all_variants = _load_variants(args.variants)
    if args.variant_filter:
        wanted = set(args.variant_filter)
        all_variants = [v for v in all_variants if v["name"] in wanted]
    log.info("loaded %d variant(s) from %s", len(all_variants), args.variants)

    clips = _load_clips(args.clips_manifest)
    if args.clips:
        wanted_clips = set(args.clips)
        clips = [(n, v) for (n, v) in clips if n in wanted_clips]
    log.info("running %d clip(s): %s", len(clips), [n for n, _ in clips])

    aggregate: List[Dict[str, Any]] = []
    if aggregate_path.is_file():
        try:
            aggregate = json.loads(aggregate_path.read_text())
        except (OSError, ValueError):
            aggregate = []

    for vi, variant in enumerate(all_variants, start=1):
        v_name = variant["name"]
        v_env = variant["env"]
        v_dir = args.out / v_name
        v_dir.mkdir(parents=True, exist_ok=True)
        summary_path = v_dir / "summary.json"

        if args.skip_existing and summary_path.is_file():
            log.info("[%d/%d] SKIP %s (summary exists)",
                     vi, len(all_variants), v_name)
            continue

        log.info("[%d/%d] === variant %s env=%s ===",
                 vi, len(all_variants), v_name, v_env)
        prev = _set_env(v_env)
        per_clip: Dict[str, Dict[str, float]] = {}
        timings: Dict[str, float] = {}
        try:
            for ci, (clip_name, video) in enumerate(clips, start=1):
                if not video.is_file():
                    log.warning("clip %s missing video: %s", clip_name, video)
                    per_clip[clip_name] = {"error": f"missing video: {video}"}
                    continue
                clip_dir = v_dir / clip_name
                clip_dir.mkdir(parents=True, exist_ok=True)
                tracks_pkl = clip_dir / "tracks.pkl"
                cache_pkl = clip_dir / "tracks.pkl.cache.pkl"
                # Force-rerun: env may have changed.
                if cache_pkl.exists():
                    cache_pkl.unlink()
                if tracks_pkl.exists():
                    tracks_pkl.unlink()
                log.info("  [%d/%d] %s start", ci, len(clips), clip_name)
                t_pipe, n_tracks = _run_pipeline_for_clip(
                    video=video, out_pkl=tracks_pkl,
                    device=args.device, max_frames=args.max_frames,
                )
                timings[clip_name] = round(t_pipe, 3)
                gt_path = _gt_path(clip_name, args.gt_root)
                metrics = _score_clip(tracks_pkl=tracks_pkl, gt_path=gt_path)
                metrics["pipeline_seconds"] = round(t_pipe, 3)
                metrics["n_tracks"] = int(n_tracks)
                per_clip[clip_name] = metrics
                log.info("  [%d/%d] %s done idf1=%.4f mota=%.4f "
                         "ids=%d miss=%d fp=%d t=%.1fs",
                         ci, len(clips), clip_name,
                         float(metrics.get("idf1") or 0.0),
                         float(metrics.get("mota") or 0.0),
                         int(metrics.get("num_switches") or 0),
                         int(metrics.get("num_misses") or 0),
                         int(metrics.get("num_false_positives") or 0),
                         t_pipe)
        finally:
            _restore_env(prev)

        summary = _summary(per_clip)
        summary["variant"] = v_name
        summary["env"] = v_env
        summary["per_clip"] = per_clip
        summary["timings"] = timings
        summary_path.write_text(json.dumps(summary, indent=2))

        # Append to aggregate (replace existing entry with same name).
        aggregate = [
            r for r in aggregate if r.get("variant") != v_name
        ]
        aggregate.append({
            "variant": v_name,
            "env": v_env,
            "mean_idf1": summary.get("mean_idf1"),
            "mean_mota": summary.get("mean_mota"),
            "num_switches_total": summary.get("num_switches_total"),
            "num_misses_total": summary.get("num_misses_total"),
            "num_false_positives_total": summary.get(
                "num_false_positives_total"),
            "per_clip": {
                cn: {
                    k: per_clip[cn].get(k)
                    for k in ("idf1", "mota", "num_switches",
                              "num_misses", "num_false_positives",
                              "pipeline_seconds")
                }
                for cn in per_clip
            },
        })
        aggregate_path.write_text(json.dumps(aggregate, indent=2))
        log.info("[%d/%d] %s mean_idf1=%.4f mean_mota=%.4f IDS=%d FN=%d FP=%d",
                 vi, len(all_variants), v_name,
                 float(summary.get("mean_idf1") or 0.0),
                 float(summary.get("mean_mota") or 0.0),
                 int(summary.get("num_switches_total") or 0),
                 int(summary.get("num_misses_total") or 0),
                 int(summary.get("num_false_positives_total") or 0))

    log.info("DONE. wrote %s", aggregate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
