"""Tier-1 sweep: replay cached FrameDetections through many post-process variants.

This is the CHEAP sweep tier. Caches (work/results/<clip>/tracks.pkl.cache.pkl)
are produced once by ``work/run_all_tests.py``; this script then rebuilds
final tracks for each variant and scores them against GT. A single variant
across all 8 scored clips runs in a few seconds.

Usage::

    python scripts/sweep_postprocess.py \\
        --variants configs/sweeps/postprocess_v1.json \\
        --cache-root work/results \\
        --gt-root /home/ubuntu/work/data \\
        --out work/sweeps/postprocess_v1.json \\
        --base-cfg configs/best_pipeline.json

Variant JSON schema::

    {
      "label_prefix": "pp_v1",
      "clips": ["BigTest", "MotionTest", ...],
      "variants": [
        {
          "id": "baseline",
          "overrides": {}            # nothing changes -> sanity check
        },
        {
          "id": "lower_p90_0.80",
          "overrides": {
            "POST_MIN_P90_CONF": 0.80
          }
        },
        {
          "id": "open_idmerge",
          "overrides": {
            "ID_MERGE_MAX_GAP": 96,
            "ID_MERGE_IOU_THRESH": 0.05,
            "cfg.pp_id_merge_osnet_cos_thresh": 0.6
          }
        }
      ]
    }

Override key namespace:
- ``DET_CONF`` (re-runs detector + tracker -- only honored when caches
  do not already exist for this conf)
- ``PRE_MIN_TOTAL_FRAMES``, ``ID_MERGE_MAX_GAP``, ``ID_MERGE_IOU_THRESH``
  ``POST_MIN_LEN``, ``POST_MIN_CONF``, ``POST_MIN_P90_CONF``
- ``BBOX_STITCH_KWARGS.<key>``, ``SIZE_SMOOTHER_KWARGS.<key>``,
  ``CENTER_SMOOTHER_KWARGS.<key>``
- ``cfg.<key>`` for any key in ``best_pipeline_cfg``
- ``stages.disable_stage_2`` etc. for ablations of individual stages

Output JSON has one entry per variant with per-clip metrics + summary.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tracking import best_pipeline as bp  # noqa: E402
from tracking.bbox_stitch import bbox_continuity_stitch  # noqa: E402
from tracking.postprocess import (  # noqa: E402
    Track,
    frame_detections_to_raw_tracks,
    postprocess_tracks,
    tracks_to_frame_detections,
)

log = logging.getLogger("sweep_postprocess")


# ---------------------------------------------------------------------------
# scoring helpers (mirror of scripts/score_runs.py)
# ---------------------------------------------------------------------------


def _tracks_dict_to_mot_rows(raw: Dict[int, Track]) -> List[str]:
    n_frames = 0
    for t in raw.values():
        if hasattr(t, "frames") and len(t.frames):
            n_frames = max(n_frames, int(t.frames.max()) + 1)
    fds = tracks_to_frame_detections(raw, n_frames)
    rows: List[str] = []
    for i, fd in enumerate(fds):
        if len(fd.tids) == 0:
            continue
        frame_id = i + 1
        for j in range(len(fd.tids)):
            x1, y1, x2, y2 = (float(v) for v in fd.xyxys[j])
            conf = float(fd.confs[j])
            tid = int(fd.tids[j])
            w = x2 - x1
            h = y2 - y1
            rows.append(
                f"{frame_id},{tid},{x1 + 1.0:.6f},{y1 + 1.0:.6f},"
                f"{w:.6f},{h:.6f},{conf:.6f},1,-1"
            )
    return rows


def _score_mot_rows_vs_gt(
    pred_rows: List[str], gt_path: Path,
) -> Dict[str, float]:
    import motmetrics as mm

    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    text = "\n".join(pred_rows) + ("\n" if pred_rows else "")
    ts = mm.io.loadtxt(io.StringIO(text), fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
    mh = mm.metrics.create()
    names = [
        "idf1", "mota",
        "num_switches", "num_fragmentations",
        "num_misses", "num_false_positives",
        "mostly_tracked", "mostly_lost",
        "num_unique_objects",
    ]
    summary = mh.compute(acc, metrics=names, name="seq")
    out: Dict[str, float] = {}
    for n in names:
        try:
            out[n] = float(summary[n].iloc[0])
        except Exception:
            out[n] = float("nan")
    return out


# ---------------------------------------------------------------------------
# variant builder
# ---------------------------------------------------------------------------


def _resolve_overrides(
    overrides: Dict[str, Any], base_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split overrides into ``(constants, cfg)`` namespaces.

    ``constants`` may set: PRE_MIN_TOTAL_FRAMES, ID_MERGE_MAX_GAP,
    ID_MERGE_IOU_THRESH, POST_MIN_LEN, POST_MIN_CONF, POST_MIN_P90_CONF,
    plus nested overrides for BBOX_STITCH_KWARGS, SIZE_SMOOTHER_KWARGS,
    CENTER_SMOOTHER_KWARGS, plus stage-disable flags.
    """
    constants: Dict[str, Any] = {
        "PRE_MIN_TOTAL_FRAMES": bp.PRE_MIN_TOTAL_FRAMES,
        "ID_MERGE_MAX_GAP": bp.ID_MERGE_MAX_GAP,
        "ID_MERGE_IOU_THRESH": bp.ID_MERGE_IOU_THRESH,
        "POST_MIN_LEN": bp.POST_MIN_LEN,
        "POST_MIN_CONF": bp.POST_MIN_CONF,
        "POST_MIN_P90_CONF": bp.POST_MIN_P90_CONF,
        "BBOX_STITCH_KWARGS": copy.deepcopy(bp.BBOX_STITCH_KWARGS),
        "SIZE_SMOOTHER_KWARGS": copy.deepcopy(bp.SIZE_SMOOTHER_KWARGS),
        "CENTER_SMOOTHER_KWARGS": copy.deepcopy(bp.CENTER_SMOOTHER_KWARGS),
        "disable_stage_2": False,
        "disable_stage_3": False,
        "disable_stage_4": False,
        "disable_stage_5": False,
    }
    cfg = copy.deepcopy(base_cfg)

    for key, val in overrides.items():
        if key.startswith("cfg."):
            cfg[key[4:]] = val
        elif key.startswith("BBOX_STITCH_KWARGS."):
            constants["BBOX_STITCH_KWARGS"][key[len("BBOX_STITCH_KWARGS."):]] = val
        elif key.startswith("SIZE_SMOOTHER_KWARGS."):
            constants["SIZE_SMOOTHER_KWARGS"][key[len("SIZE_SMOOTHER_KWARGS."):]] = val
        elif key.startswith("CENTER_SMOOTHER_KWARGS."):
            constants["CENTER_SMOOTHER_KWARGS"][key[len("CENTER_SMOOTHER_KWARGS."):]] = val
        elif key in constants:
            constants[key] = val
        else:
            raise KeyError(f"unknown override key: {key}")

    return constants, cfg


def _build_tracks_from_cache(
    *,
    cache_path: Path,
    constants: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[int, Track]:
    """Inline build_tracks with overridable constants / cfg / stage gates."""
    fd = joblib.load(str(cache_path))
    raw = frame_detections_to_raw_tracks(fd)

    stage1 = postprocess_tracks(
        raw,
        min_box_w=10, min_box_h=10,
        min_total_frames=constants["PRE_MIN_TOTAL_FRAMES"],
        min_conf=cfg["pp_min_conf"],
        max_gap_interp=cfg["pp_max_gap_interp"],
        id_merge_max_gap=constants["ID_MERGE_MAX_GAP"],
        id_merge_iou_thresh=constants["ID_MERGE_IOU_THRESH"],
        id_merge_osnet_cos_thresh=cfg["pp_id_merge_osnet_cos_thresh"],
        medfilt_window=cfg["pp_medfilt_window"],
        gaussian_sigma=cfg["pp_gaussian_sigma"],
        num_max_people=cfg["pp_num_max_people"],
        overlap_merge_iou_thresh=cfg["pp_overlap_merge_iou_thresh"],
        overlap_merge_min_frames=cfg["pp_overlap_merge_min_frames"],
        edge_trim_conf_thresh=cfg["pp_edge_trim_conf_thresh"],
        edge_trim_max_frames=cfg["pp_edge_trim_max_frames"],
        pose_extractor=None, pose_cos_thresh=0.0,
        pose_max_gap=cfg["pp_pose_max_gap"],
        pose_min_iou_for_pair=cfg["pp_pose_min_iou_for_pair"],
        pose_max_center_dist=cfg["pp_pose_max_center_dist"],
        frame_loader=None,
    )

    if constants["disable_stage_2"]:
        stage2 = stage1
    else:
        stage2 = bp.filter_tracks_post_merge(
            stage1,
            min_len=constants["POST_MIN_LEN"],
            min_conf=constants["POST_MIN_CONF"],
            min_p90_conf=constants["POST_MIN_P90_CONF"],
        )

    if constants["disable_stage_3"]:
        stage3 = stage2
    else:
        stage3, _ = bbox_continuity_stitch(
            stage2, **constants["BBOX_STITCH_KWARGS"],
        )

    if constants["disable_stage_4"]:
        stage4 = stage3
    else:
        stage4 = bp.size_smooth_cv_gated(
            stage3, **constants["SIZE_SMOOTHER_KWARGS"],
        )

    if constants["disable_stage_5"]:
        final = stage4
    else:
        final = bp.smooth_centers_median(
            stage4, **constants["CENTER_SMOOTHER_KWARGS"],
        )

    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", type=Path, required=True,
                   help="Variant JSON file.")
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Directory with <clip>/tracks.pkl.cache.pkl per clip.")
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-cfg", type=Path,
                   default=REPO / "configs" / "best_pipeline.json")
    p.add_argument("--clips-override", nargs="*", default=None,
                   help="Override the variant JSON's clips field.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_cfg = json.loads(args.base_cfg.read_text())["best_pipeline_cfg"]
    spec = json.loads(args.variants.read_text())

    label_prefix = spec.get("label_prefix", "sweep")
    clips = args.clips_override or spec["clips"]
    variants: List[Dict[str, Any]] = spec["variants"]
    log.info(
        "sweep '%s': %d variants x %d clips",
        label_prefix, len(variants), len(clips),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    output: Dict[str, Any] = {
        "label_prefix": label_prefix,
        "scoring": "py-motmetrics, IoU 0.5, mot15-2D",
        "base_cfg": base_cfg,
        "clips": clips,
        "results": [],
    }

    # ALWAYS WRITE INCREMENTALLY so a crash mid-sweep doesn't lose work.
    def _flush() -> None:
        args.out.write_text(json.dumps(output, indent=2))

    _flush()

    for v_idx, variant in enumerate(variants):
        v_id = variant["id"]
        overrides = variant.get("overrides", {})
        notes = variant.get("notes", "")
        try:
            constants, cfg = _resolve_overrides(overrides, base_cfg)
        except Exception as exc:
            log.exception("variant %s: bad overrides: %s", v_id, exc)
            output["results"].append({
                "variant_id": v_id, "error": str(exc),
                "overrides": overrides,
            })
            _flush()
            continue

        per_clip: Dict[str, Dict[str, float]] = {}
        t0 = time.time()
        for clip in clips:
            cache = args.cache_root / clip / "tracks.pkl.cache.pkl"
            gt_path = args.gt_root / clip / "gt" / "gt.txt"
            if not cache.is_file():
                log.warning("[%s/%s] cache missing: %s", v_id, clip, cache)
                continue
            if not gt_path.is_file():
                log.warning("[%s/%s] gt missing: %s", v_id, clip, gt_path)
                continue
            try:
                tracks = _build_tracks_from_cache(
                    cache_path=cache, constants=constants, cfg=cfg,
                )
                rows = _tracks_dict_to_mot_rows(tracks)
                metrics = _score_mot_rows_vs_gt(rows, gt_path)
            except Exception as exc:
                log.exception("[%s/%s] failed: %s", v_id, clip, exc)
                continue
            per_clip[clip] = metrics
        dt = time.time() - t0

        if per_clip:
            idf1s = np.asarray([v["idf1"] for v in per_clip.values()])
            idss = np.asarray([v["num_switches"] for v in per_clip.values()])
            fns = np.asarray([v["num_misses"] for v in per_clip.values()])
            fps = np.asarray([v["num_false_positives"] for v in per_clip.values()])
            summary = {
                "n_clips": int(len(per_clip)),
                "mean_idf1": float(idf1s.mean()),
                "median_idf1": float(np.median(idf1s)),
                "min_idf1": float(idf1s.min()),
                "total_switches": int(idss.sum()),
                "total_misses": int(fns.sum()),
                "total_false_positives": int(fps.sum()),
                "wall_seconds": round(dt, 2),
            }
        else:
            summary = {"n_clips": 0, "wall_seconds": round(dt, 2)}

        output["results"].append({
            "variant_id": v_id,
            "notes": notes,
            "overrides": overrides,
            "constants": {
                k: v for k, v in constants.items()
                if not isinstance(v, dict) and not k.startswith("disable_")
            },
            "stages_disabled": [
                k for k, v in constants.items() if k.startswith("disable_") and v
            ],
            "summary": summary,
            "per_clip": per_clip,
        })
        _flush()

        if per_clip:
            log.info(
                "[%d/%d] %s  mean_idf1=%.4f min=%.4f IDS=%d FN=%d FP=%d  (%.1fs)",
                v_idx + 1, len(variants), v_id,
                summary["mean_idf1"], summary["min_idf1"],
                summary["total_switches"], summary["total_misses"],
                summary["total_false_positives"],
                summary["wall_seconds"],
            )
        else:
            log.warning(
                "[%d/%d] %s  produced no scored clips (%.1fs)",
                v_idx + 1, len(variants), v_id, summary["wall_seconds"],
            )

    log.info("done. wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
