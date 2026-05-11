"""Evaluate the Stage-1 ID/YOLO pipeline against local CVAT MOT exports.

This is the research harness for ID accuracy work.  It deliberately keeps
scoring separate from model changes so detector, tracker, stitching, and
pruning candidates can be compared under the same MOT metrics.

Examples:

    # Score existing per-clip tracks under runs/id_pipeline/<clip>/tracks.pkl.
    python scripts/eval_id_yolo_cvat.py \
        --clips adiTest \
        --tracks-root runs/id_pipeline

    # Run the current pipeline, then score every non-leaked CVAT clip.
    python scripts/eval_id_yolo_cvat.py \
        --run-pipeline --device cuda:0 --force
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


DEFAULT_CVAT_ROOT = Path("/Users/arnavchokshi/Desktop/CV_pipeline/CVAT")
DEFAULT_OUT_ROOT = REPO / "runs" / "id_yolo_eval"
LEAKED_CLIPS = {"mirror" + "Test"}
MOT_METRICS = [
    "num_frames",
    "idf1",
    "idp",
    "idr",
    "mota",
    "motp",
    "precision",
    "recall",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "num_fragmentations",
    "mostly_tracked",
    "mostly_lost",
    "num_unique_objects",
]


@dataclass(frozen=True)
class ClipSpec:
    name: str
    root: Path
    gt_path: Path
    video_path: Optional[Path]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ("torch", "ultralytics", "boxmot", "motmetrics", "cv2", "joblib", "numpy", "scipy"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "ok"))
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"ERR:{type(exc).__name__}:{exc}"
    return versions


def _gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
            info["torch_cuda_device_count"] = int(torch.cuda.device_count())
        info["torch_mps_available"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
    except Exception as exc:  # noqa: BLE001
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if smi:
            info["nvidia_smi"] = smi.splitlines()
    except Exception:
        pass
    return info


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _find_video(clip_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for p in clip_dir.iterdir():
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in {".mov", ".mp4", ".m4v", ".avi", ".mkv"}:
            continue
        name = p.name.lower()
        if any(tok in name for tok in ("overlay", "vis", "mask")):
            continue
        if name.endswith(".bak"):
            continue
        candidates.append(p)
    if not candidates:
        return None
    ext_rank = {".mov": 0, ".mp4": 1, ".m4v": 2, ".avi": 3, ".mkv": 4}
    candidates.sort(key=lambda p: (ext_rank.get(p.suffix.lower(), 99), p.name))
    return candidates[0]


def discover_clips(cvat_root: Path) -> List[ClipSpec]:
    clips: List[ClipSpec] = []
    for clip_dir in sorted(Path(cvat_root).iterdir(), key=lambda p: p.name):
        if not clip_dir.is_dir():
            continue
        gt_path = clip_dir / "gt" / "gt.txt"
        if not gt_path.is_file():
            continue
        clips.append(
            ClipSpec(
                name=clip_dir.name,
                root=clip_dir,
                gt_path=gt_path,
                video_path=_find_video(clip_dir),
            )
        )
    return clips


def _load_manifest(path: Path) -> List[ClipSpec]:
    payload = json.loads(Path(path).read_text())
    clips: List[ClipSpec] = []
    for item in payload.get("clips", []):
        name = str(item["name"])
        video = Path(str(item["video"])).expanduser()
        root = Path(str(item.get("root", video.parent))).expanduser()
        gt = Path(str(item.get("gt", root / "gt" / "gt.txt"))).expanduser()
        clips.append(ClipSpec(name=name, root=root, gt_path=gt, video_path=video))
    return clips


def _parse_mot_gt(gt_path: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """Read MOTChallenge-style GT rows as frame -> [(gt_id, xywh)]."""
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    with Path(gt_path).open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                frame = int(float(row[0]))
                tid = int(float(row[1]))
                x, y, w, h = (float(row[i]) for i in range(2, 6))
                mark = int(float(row[6])) if len(row) > 6 and row[6] != "" else 1
            except ValueError:
                continue
            if mark == 0:
                continue
            if w <= 0 or h <= 0:
                continue
            by_frame.setdefault(frame, []).append(
                (tid, np.asarray([x, y, w, h], dtype=np.float64))
            )
    return by_frame


def _track_frames_and_boxes(track: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray(getattr(track, "frames"), dtype=np.int64)
    bboxes = np.asarray(getattr(track, "bboxes"), dtype=np.float64)
    confs_attr = getattr(track, "confs", None)
    if confs_attr is None:
        confs = np.ones((len(frames),), dtype=np.float64)
    else:
        confs = np.asarray(confs_attr, dtype=np.float64)
    if len(confs) != len(frames):
        confs = np.ones((len(frames),), dtype=np.float64)
    return frames, bboxes, confs


def _load_predictions(
    tracks_path: Path,
    *,
    gt_max_frame: int,
) -> Dict[int, List[Tuple[int, np.ndarray, float]]]:
    """Load repo tracks and convert 0-index xyxy to 1-index MOT xywh."""
    tracks = joblib.load(str(tracks_path))
    if not isinstance(tracks, dict):
        raise ValueError(f"{tracks_path} did not contain a dict of tracks")
    by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}
    for tid_raw, track in tracks.items():
        tid = int(tid_raw)
        frames, bboxes, confs = _track_frames_and_boxes(track)
        for i, frame0 in enumerate(frames):
            frame = int(frame0) + 1
            if frame < 1 or frame > gt_max_frame:
                continue
            if i >= len(bboxes):
                continue
            x1, y1, x2, y2 = (float(v) for v in bboxes[i])
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            conf = float(confs[i]) if i < len(confs) else 1.0
            by_frame.setdefault(frame, []).append(
                (tid, np.asarray([x1, y1, w, h], dtype=np.float64), conf)
            )
    return by_frame


def _frame_count(by_frame: Dict[int, Sequence[Any]]) -> int:
    return max(by_frame.keys(), default=0)


def _bbox_iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = (float(v) for v in a[:4])
    bx1, by1, bw, bh = (float(v) for v in b[:4])
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
    return float(inter / union) if union > 0 else 0.0


def _identity_ranges_from_gt(
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for frame, items in by_frame.items():
        for tid, _box in items:
            row = out.setdefault(int(tid), {"frames": []})
            row["frames"].append(int(frame))
    for tid, row in out.items():
        frames = sorted(set(int(f) for f in row["frames"]))
        row["entry_frame"] = int(frames[0]) if frames else None
        row["exit_frame"] = int(frames[-1]) if frames else None
        row["n_frames"] = int(len(frames))
        row["active_ranges"] = _ranges(frames)
    return out


def _identity_ranges_from_pred(
    by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]],
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for frame, items in by_frame.items():
        for tid, _box, _conf in items:
            row = out.setdefault(int(tid), {"frames": []})
            row["frames"].append(int(frame))
    for tid, row in out.items():
        frames = sorted(set(int(f) for f in row["frames"]))
        row["entry_frame"] = int(frames[0]) if frames else None
        row["exit_frame"] = int(frames[-1]) if frames else None
        row["n_frames"] = int(len(frames))
        row["active_ranges"] = _ranges(frames)
    return out


def _pair_match_counts(
    *,
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]],
    iou_thresh: float,
) -> Dict[Tuple[int, int], int]:
    pair_counts: Dict[Tuple[int, int], int] = {}
    max_frame = _frame_count(gt_by_frame)
    for frame in range(1, max_frame + 1):
        gt_items = gt_by_frame.get(frame, [])
        pred_items = pred_by_frame.get(frame, [])
        if not gt_items or not pred_items:
            continue
        iou = np.zeros((len(gt_items), len(pred_items)), dtype=np.float64)
        for gi, (_gt_id, gt_box) in enumerate(gt_items):
            for pi, (_pred_id, pred_box, _conf) in enumerate(pred_items):
                iou[gi, pi] = _bbox_iou_xywh(gt_box, pred_box)
        cost = 1.0 - iou
        row_ind, col_ind = linear_sum_assignment(cost)
        for gi, pi in zip(row_ind, col_ind):
            if iou[gi, pi] < iou_thresh:
                continue
            gt_id = int(gt_items[int(gi)][0])
            pred_id = int(pred_items[int(pi)][0])
            pair_counts[(gt_id, pred_id)] = pair_counts.get((gt_id, pred_id), 0) + 1
    return pair_counts


def _identity_assignment(pair_counts: Dict[Tuple[int, int], int]) -> Dict[int, int]:
    gt_ids = sorted({int(k[0]) for k in pair_counts})
    pred_ids = sorted({int(k[1]) for k in pair_counts})
    if not gt_ids or not pred_ids:
        return {}
    gi = {tid: idx for idx, tid in enumerate(gt_ids)}
    pi = {tid: idx for idx, tid in enumerate(pred_ids)}
    weights = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for (gt_id, pred_id), count in pair_counts.items():
        weights[gi[int(gt_id)], pi[int(pred_id)]] = float(count)
    row_ind, col_ind = linear_sum_assignment(-weights)
    out: Dict[int, int] = {}
    for row, col in zip(row_ind, col_ind):
        if weights[int(row), int(col)] <= 0:
            continue
        out[int(gt_ids[int(row)])] = int(pred_ids[int(col)])
    return out


def _entry_exit_diagnostics(
    *,
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]],
    iou_thresh: float,
    late_entry_tolerance: int,
) -> Dict[str, Any]:
    gt_ranges = _identity_ranges_from_gt(gt_by_frame)
    pred_ranges = _identity_ranges_from_pred(pred_by_frame)
    pair_counts = _pair_match_counts(
        gt_by_frame=gt_by_frame,
        pred_by_frame=pred_by_frame,
        iou_thresh=iou_thresh,
    )
    assignment = _identity_assignment(pair_counts)
    pred_ids_by_gt: Dict[int, set[int]] = {}
    for (gt_id, pred_id), count in pair_counts.items():
        if count > 0:
            pred_ids_by_gt.setdefault(int(gt_id), set()).add(int(pred_id))

    rows: List[Dict[str, Any]] = []
    missed_late_entry_count = 0
    for gt_id in sorted(gt_ranges):
        gt_row = gt_ranges[gt_id]
        pred_id = assignment.get(int(gt_id))
        pred_row = pred_ranges.get(int(pred_id), {}) if pred_id is not None else {}
        gt_entry = gt_row.get("entry_frame")
        gt_exit = gt_row.get("exit_frame")
        pred_entry = pred_row.get("entry_frame")
        pred_exit = pred_row.get("exit_frame")
        entry_error = (
            int(pred_entry) - int(gt_entry)
            if pred_entry is not None and gt_entry is not None
            else None
        )
        exit_error = (
            int(pred_exit) - int(gt_exit)
            if pred_exit is not None and gt_exit is not None
            else None
        )
        if (
            gt_entry is not None
            and int(gt_entry) > 1
            and (pred_id is None or entry_error is None or abs(int(entry_error)) > late_entry_tolerance)
        ):
            missed_late_entry_count += 1
        rows.append(
            {
                "gt_id": int(gt_id),
                "assigned_pred_id": int(pred_id) if pred_id is not None else None,
                "matched_frames": int(pair_counts.get((int(gt_id), int(pred_id)), 0))
                if pred_id is not None
                else 0,
                "gt_entry_frame": gt_entry,
                "gt_exit_frame": gt_exit,
                "pred_entry_frame": pred_entry,
                "pred_exit_frame": pred_exit,
                "entry_frame_error": entry_error,
                "exit_frame_error": exit_error,
                "all_matched_pred_ids": sorted(pred_ids_by_gt.get(int(gt_id), set())),
            }
        )

    duplicate_track_count = int(
        sum(max(0, len(preds) - 1) for preds in pred_ids_by_gt.values())
    )
    return {
        "gt_identity_ranges": gt_ranges,
        "pred_identity_ranges": pred_ranges,
        "identity_assignment": {str(k): v for k, v in assignment.items()},
        "entry_exit_errors": rows,
        "duplicate_track_count": duplicate_track_count,
        "missed_late_entry_count": int(missed_late_entry_count),
    }


def _ranges(values: Iterable[int]) -> List[List[int]]:
    vals = sorted(set(int(v) for v in values))
    if not vals:
        return []
    ranges: List[List[int]] = []
    start = prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append([start, prev])
        start = prev = value
    ranges.append([start, prev])
    return ranges


def _score_clip(
    *,
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]],
    iou_thresh: float,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    max_frame = _frame_count(gt_by_frame)
    acc = mm.MOTAccumulator(auto_id=True)
    count_mismatch_frames: List[int] = []
    gt_counts: List[int] = []
    pred_counts: List[int] = []

    for frame in range(1, max_frame + 1):
        gt_items = gt_by_frame.get(frame, [])
        pred_items = pred_by_frame.get(frame, [])
        gt_ids = [tid for tid, _ in gt_items]
        pred_ids = [tid for tid, _, _ in pred_items]
        gt_boxes = np.stack([box for _, box in gt_items], axis=0) if gt_items else np.empty((0, 4))
        pred_boxes = np.stack([box for _, box, _ in pred_items], axis=0) if pred_items else np.empty((0, 4))
        distances = mm.distances.iou_matrix(
            gt_boxes,
            pred_boxes,
            max_iou=max(0.0, 1.0 - float(iou_thresh)),
        )
        acc.update(gt_ids, pred_ids, distances)
        gt_counts.append(len(gt_items))
        pred_counts.append(len(pred_items))
        if len(gt_items) != len(pred_items):
            count_mismatch_frames.append(frame)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=MOT_METRICS, name="clip")
    row = summary.loc["clip"].to_dict()
    row = {k: _json_default(v) for k, v in row.items()}

    gt_ids_all = {tid for items in gt_by_frame.values() for tid, _ in items}
    pred_ids_all = {tid for items in pred_by_frame.values() for tid, _, _ in items}
    count_exact = 0.0
    if max_frame > 0:
        count_exact = 1.0 - (len(count_mismatch_frames) / float(max_frame))
    diagnostics = {
        "gt_frame_count": max_frame,
        "gt_ids": len(gt_ids_all),
        "pred_ids": len(pred_ids_all),
        "gt_rows": int(sum(gt_counts)),
        "pred_rows": int(sum(pred_counts)),
        "count_exact": count_exact,
        "count_mismatch_frames": count_mismatch_frames,
        "count_mismatch_ranges": _ranges(count_mismatch_frames),
        "gt_count_min": int(min(gt_counts)) if gt_counts else 0,
        "gt_count_median": float(np.median(gt_counts)) if gt_counts else 0.0,
        "gt_count_max": int(max(gt_counts)) if gt_counts else 0,
        "pred_count_min": int(min(pred_counts)) if pred_counts else 0,
        "pred_count_median": float(np.median(pred_counts)) if pred_counts else 0.0,
        "pred_count_max": int(max(pred_counts)) if pred_counts else 0,
    }
    return row, acc, diagnostics


def _event_ranges_by_id(acc: Any) -> Dict[str, Any]:
    events = acc.events.reset_index()
    out: Dict[str, Any] = {
        "miss_ranges_by_gt_id": [],
        "fp_ranges_by_pred_id": [],
        "switch_windows": [],
    }

    miss = events[events["Type"] == "MISS"]
    if not miss.empty:
        for oid, group in miss.groupby("OId"):
            frames = [int(v) + 1 for v in group["FrameId"].tolist()]
            out["miss_ranges_by_gt_id"].append({
                "gt_id": int(oid),
                "count": len(frames),
                "ranges": _ranges(frames),
            })
        out["miss_ranges_by_gt_id"].sort(key=lambda d: (-int(d["count"]), int(d["gt_id"])))

    fp = events[events["Type"] == "FP"]
    if not fp.empty:
        for hid, group in fp.groupby("HId"):
            frames = [int(v) + 1 for v in group["FrameId"].tolist()]
            out["fp_ranges_by_pred_id"].append({
                "pred_id": int(hid),
                "count": len(frames),
                "ranges": _ranges(frames),
            })
        out["fp_ranges_by_pred_id"].sort(key=lambda d: (-int(d["count"]), int(d["pred_id"])))

    switches = events[events["Type"] == "SWITCH"]
    if not switches.empty:
        for _, row in switches.iterrows():
            out["switch_windows"].append({
                "frame": int(row["FrameId"]) + 1,
                "gt_id": int(row["OId"]),
                "pred_id": int(row["HId"]),
            })
    return out


def _write_events(acc: Any, path: Path) -> None:
    events = acc.events.reset_index()
    if "FrameId" in events.columns and "FrameNumber" not in events.columns:
        events.insert(1, "FrameNumber", events["FrameId"].astype(int) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, index=False)


def _range_starts(rows: Sequence[Dict[str, Any]], *, limit: int) -> List[int]:
    frames: List[int] = []
    for row in rows[:limit]:
        for start, _end in row.get("ranges", [])[:1]:
            frames.append(int(start))
    return frames


def _select_debug_overlay_frames(
    diagnostics: Dict[str, Any],
    *,
    max_frames: int,
) -> List[int]:
    """Pick a compact, deterministic set of frames that explain failures."""
    selected: List[int] = []
    selected.extend(int(item["frame"]) for item in diagnostics.get("switch_windows", [])[:6])
    for start, _end in diagnostics.get("count_mismatch_ranges", [])[:6]:
        selected.append(int(start))
    selected.extend(
        _range_starts(diagnostics.get("miss_ranges_by_gt_id", []), limit=4)
    )
    selected.extend(
        _range_starts(diagnostics.get("fp_ranges_by_pred_id", []), limit=4)
    )
    for row in diagnostics.get("entry_exit_errors", []):
        entry_error = row.get("entry_frame_error")
        exit_error = row.get("exit_frame_error")
        if entry_error is not None and abs(int(entry_error)) > 5:
            frame = row.get("gt_entry_frame")
            if frame is not None:
                selected.append(int(frame))
        if exit_error is not None and abs(int(exit_error)) > 5:
            frame = row.get("gt_exit_frame")
            if frame is not None:
                selected.append(int(frame))

    unique: List[int] = []
    seen: set[int] = set()
    for frame in selected:
        if frame <= 0 or frame in seen:
            continue
        unique.append(frame)
        seen.add(frame)
        if len(unique) >= max(0, int(max_frames)):
            break
    return unique


def _draw_xywh(
    image: np.ndarray,
    box: np.ndarray,
    *,
    color: Tuple[int, int, int],
    label: str,
) -> None:
    import cv2

    x, y, w, h = (float(v) for v in box[:4])
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    h_img, w_img = image.shape[:2]
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img - 1, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img - 1, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(12, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _write_debug_overlays(
    *,
    video_path: Optional[Path],
    clip_name: str,
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]],
    pred_by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]],
    diagnostics: Dict[str, Any],
    out_dir: Path,
    max_frames: int,
) -> List[str]:
    frames = _select_debug_overlay_frames(diagnostics, max_frames=max_frames)
    if not frames or video_path is None or not Path(video_path).is_file():
        return []
    try:
        import cv2
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    written: List[str] = []
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame) - 1))
            ok, image = cap.read()
            if not ok or image is None:
                continue
            for gt_id, box in gt_by_frame.get(int(frame), []):
                _draw_xywh(
                    image,
                    box,
                    color=(0, 220, 0),
                    label=f"GT {int(gt_id)}",
                )
            for pred_id, box, conf in pred_by_frame.get(int(frame), []):
                _draw_xywh(
                    image,
                    box,
                    color=(255, 180, 0),
                    label=f"P {int(pred_id)} {float(conf):.2f}",
                )
            cv2.putText(
                image,
                f"{clip_name} frame {int(frame)} | GT green, pred blue",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            path = out_dir / f"{clip_name}_frame_{int(frame):06d}.jpg"
            if cv2.imwrite(str(path), image):
                written.append(str(path))
    finally:
        cap.release()
    return written


def _read_timing(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_tracks_path(tracks_root: Optional[Path], clip: str, filename: str) -> Optional[Path]:
    if tracks_root is None:
        return None
    root = Path(tracks_root)
    candidates = [
        root / clip / filename,
        root / clip / "tracks.pkl",
        root / clip / "id_tracks.pkl",
        root / f"{clip}.pkl",
    ]
    return next((p for p in candidates if p.is_file()), None)


def _find_cache_path(cache_root: Optional[Path], clip: str) -> Optional[Path]:
    if cache_root is None:
        return None
    root = Path(cache_root)
    candidates = [
        root / clip / "tracks.pkl.cache.pkl",
        root / clip / "cache.pkl",
        root / clip / f"{clip}.cache.pkl",
        root / f"{clip}.cache.pkl",
    ]
    return next((p for p in candidates if p.is_file()), None)


def _run_pipeline_for_clip(
    *,
    clip: ClipSpec,
    out_dir: Path,
    device: str,
    force: bool,
    max_frames: Optional[int],
    weights: Path,
    cfg: Path,
    reid_weights: Path,
    save_detector_cache: bool,
    expected_total_performers: Optional[int],
) -> Tuple[Path, Path]:
    if clip.video_path is None or not clip.video_path.is_file():
        raise FileNotFoundError(f"{clip.name}: source video not found")
    from tracking.run_pipeline import run_pipeline_on_video

    tracks_path = out_dir / clip.name / "tracks.pkl"
    timing_path = out_dir / clip.name / "timing.json"
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    run_pipeline_on_video(
        video=clip.video_path,
        out=tracks_path,
        weights=weights,
        cfg=cfg,
        reid_weights=reid_weights,
        device=device,
        max_frames=max_frames,
        force=force,
        timing_path=timing_path,
        detections_cache_path=tracks_path.parent / "detections.pkl" if save_detector_cache else None,
        expected_total_performers=expected_total_performers,
        presence_plan_path=tracks_path.parent / "presence_plan.json",
    )
    return tracks_path, timing_path


def _build_tracks_from_cache_for_clip(
    *,
    clip: ClipSpec,
    cache_root: Path,
    out_dir: Path,
    cfg: Path,
    expected_total_performers: Optional[int],
) -> Tuple[Path, Optional[Path]]:
    cache_path = _find_cache_path(cache_root, clip.name)
    if cache_path is None:
        raise FileNotFoundError(f"{clip.name}: no cache found under {cache_root}")
    from tracking.best_pipeline import build_tracks

    tracks_path = out_dir / clip.name / "tracks.pkl"
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    build_tracks(
        cache_path=cache_path,
        cfg_path=cfg,
        save_to=tracks_path,
        expected_total_performers=expected_total_performers,
        presence_plan_path=tracks_path.parent / "presence_plan.json",
    )
    return tracks_path, None


def _mean_row(rows: List[Dict[str, Any]], *, clip_names: List[str]) -> Dict[str, Any]:
    selected = [r for r in rows if r["clip"] in set(clip_names) and r.get("status") == "ok"]
    out: Dict[str, Any] = {"clip": "MEAN_RANKING", "status": "ok", "n_clips": len(selected)}
    for key in ("idf1", "idp", "idr", "mota", "precision", "recall", "count_exact"):
        vals = [float(r[key]) for r in selected if r.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    for key in ("num_switches", "num_false_positives", "num_misses", "num_fragmentations"):
        vals = [float(r[key]) for r in selected if r.get(key) is not None]
        out[key] = int(sum(vals)) if vals else 0
    for key in ("zero_switches", "id_count_match", "mota_gate", "primary_goal_pass"):
        vals = [bool(r.get(key)) for r in selected if key in r]
        out[f"{key}_rate"] = float(np.mean(vals)) if vals else None
    return out


def _add_primary_goal_flags(row: Dict[str, Any], *, mota_threshold: float) -> None:
    """Add the user-facing Stage 1 target gates to one scoreboard row."""
    switches = float(row.get("num_switches") or 0.0)
    pred_ids = int(row.get("pred_ids") or 0)
    gt_ids = int(row.get("gt_ids") or 0)
    mota = float(row.get("mota") or 0.0)
    row["zero_switches"] = switches == 0.0
    row["id_count_match"] = pred_ids == gt_ids
    row["mota_gate"] = mota >= float(mota_threshold)
    row["primary_goal_pass"] = (
        bool(row["zero_switches"])
        and bool(row["id_count_match"])
        and bool(row["mota_gate"])
    )


def _write_scoreboard(rows: List[Dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    preferred = [
        "clip",
        "status",
        "leaked",
        "primary_goal_pass",
        "zero_switches",
        "id_count_match",
        "exact_total_count_match",
        "mota_gate",
        "idf1",
        "mota",
        "idp",
        "idr",
        "precision",
        "recall",
        "count_exact",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "num_fragmentations",
        "mostly_tracked",
        "mostly_lost",
        "gt_ids",
        "pred_ids",
        "expected_total_performers",
        "per_frame_active_count_accuracy",
        "duplicate_track_count",
        "missed_late_entry_count",
        "gt_rows",
        "pred_rows",
        "gt_frame_count",
        "tracks_path",
        "video_path",
        "error",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys and key != "diagnostics":
                keys.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps(rows, indent=2, default=_json_default))


def _build_run_metadata(args: argparse.Namespace, run_dir: Path, device: str) -> Dict[str, Any]:
    return {
        "created_unix": int(time.time()),
        "repo": str(REPO),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": sys.version,
        "args": {k: _json_default(v) for k, v in vars(args).items()},
        "resolved_device": device,
        "package_versions": _package_versions(),
        "gpu": _gpu_info(),
        "best_id_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("BEST_ID_")},
        "run_dir": str(run_dir),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cvat-root", type=Path, default=DEFAULT_CVAT_ROOT)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument(
        "--exclude-clips",
        nargs="*",
        default=None,
        help="Clip names to exclude from this run, even if discovered or listed in the manifest.",
    )
    ap.add_argument("--include-mirror", action="store_true")
    ap.add_argument("--tracks-root", type=Path, default=None)
    ap.add_argument("--tracks-filename", default="tracks.pkl")
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument(
        "--build-from-cache",
        action="store_true",
        help="Build final tracks from existing FrameDetections caches before scoring.",
    )
    ap.add_argument("--run-pipeline", action="store_true")
    ap.add_argument(
        "--use-expected-count",
        action="store_true",
        help="Use the number of unique GT IDs as simulated user input for the Stage A expected-count prior.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--save-detector-cache",
        action="store_true",
        help="When running the pipeline, also save raw detector boxes as detections.pkl.",
    )
    ap.add_argument("--weights", type=Path, default=REPO / "weights" / "best.pt")
    ap.add_argument("--cfg", type=Path, default=REPO / "configs" / "best_pipeline.json")
    ap.add_argument("--reid-weights", type=Path, default=Path("osnet_x0_25_msmt17.pt"))
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument(
        "--late-entry-tolerance",
        type=int,
        default=5,
        help="Frames of entry-frame slack before a late-entering GT identity counts as missed.",
    )
    ap.add_argument(
        "--mota-threshold",
        type=float,
        default=0.90,
        help="Primary goal gate: MOTA must be at least this value, in addition to zero switches and matching ID count.",
    )
    ap.add_argument(
        "--max-frames-mode",
        choices=("gt", "none"),
        default="gt",
        help="When running the pipeline, cap processing at GT frame count.",
    )
    ap.add_argument(
        "--allow-missing-tracks",
        action="store_true",
        help="Skip clips without existing tracks when --run-pipeline is not set.",
    )
    ap.add_argument(
        "--debug-overlays",
        action="store_true",
        help="Write GT/prediction overlay JPEGs for representative failure frames.",
    )
    ap.add_argument(
        "--debug-overlay-max-frames",
        type=int,
        default=12,
        help="Maximum debug overlay frames to write per clip.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    clips = _load_manifest(args.manifest) if args.manifest else discover_clips(args.cvat_root)
    if args.clips:
        wanted = set(args.clips)
        clips = [c for c in clips if c.name in wanted]
    if args.exclude_clips:
        excluded = set(args.exclude_clips)
        clips = [c for c in clips if c.name not in excluded]
    if not args.include_mirror:
        clips = [c for c in clips if c.name not in LEAKED_CLIPS]
    if not clips:
        raise SystemExit("no clips selected")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / run_id
    tracks_out_root = run_dir / "tracks"
    events_dir = run_dir / "events"
    diagnostics_dir = run_dir / "diagnostics"
    overlays_dir = run_dir / "debug_overlays"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(args.device))
    metadata = _build_run_metadata(args, run_dir, device)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default)
    )

    rows: List[Dict[str, Any]] = []
    ranking_clip_names: List[str] = []
    for clip in clips:
        leaked = clip.name in LEAKED_CLIPS
        if not leaked:
            ranking_clip_names.append(clip.name)
        try:
            gt_by_frame = _parse_mot_gt(clip.gt_path)
            gt_frames = _frame_count(gt_by_frame)
            if gt_frames <= 0:
                raise ValueError(f"{clip.name}: no GT frames in {clip.gt_path}")
            gt_ids_all = {tid for items in gt_by_frame.values() for tid, _ in items}
            expected_total_performers = (
                len(gt_ids_all) if args.use_expected_count else None
            )

            tracks_path = _find_tracks_path(args.tracks_root, clip.name, args.tracks_filename)
            timing_path: Optional[Path] = None
            if args.run_pipeline:
                max_frames = gt_frames if args.max_frames_mode == "gt" else None
                tracks_path, timing_path = _run_pipeline_for_clip(
                    clip=clip,
                    out_dir=tracks_out_root,
                    device=device,
                    force=bool(args.force),
                    max_frames=max_frames,
                    weights=args.weights,
                    cfg=args.cfg,
                    reid_weights=args.reid_weights,
                    save_detector_cache=bool(args.save_detector_cache),
                    expected_total_performers=expected_total_performers,
                )
            elif args.build_from_cache:
                if args.cache_root is None:
                    raise ValueError("--build-from-cache requires --cache-root")
                tracks_path, timing_path = _build_tracks_from_cache_for_clip(
                    clip=clip,
                    cache_root=args.cache_root,
                    out_dir=tracks_out_root,
                    cfg=args.cfg,
                    expected_total_performers=expected_total_performers,
                )
            elif tracks_path is None:
                if args.allow_missing_tracks:
                    rows.append({
                        "clip": clip.name,
                        "status": "missing_tracks",
                        "leaked": leaked,
                        "gt_frame_count": gt_frames,
                        "gt_path": str(clip.gt_path),
                        "video_path": str(clip.video_path) if clip.video_path else "",
                    })
                    continue
                raise FileNotFoundError(
                    f"{clip.name}: no tracks found under {args.tracks_root}; "
                    "pass --run-pipeline or --allow-missing-tracks"
                )

            pred_by_frame = _load_predictions(tracks_path, gt_max_frame=gt_frames)
            metric_row, acc, diagnostics = _score_clip(
                gt_by_frame=gt_by_frame,
                pred_by_frame=pred_by_frame,
                iou_thresh=float(args.iou_thresh),
            )
            diagnostics.update(_event_ranges_by_id(acc))
            diagnostics.update(
                _entry_exit_diagnostics(
                    gt_by_frame=gt_by_frame,
                    pred_by_frame=pred_by_frame,
                    iou_thresh=float(args.iou_thresh),
                    late_entry_tolerance=int(args.late_entry_tolerance),
                )
            )
            diagnostics["expected_total_performers"] = int(len(gt_ids_all))
            diagnostics["exact_total_count_match"] = bool(
                int(diagnostics.get("pred_ids") or 0) == int(len(gt_ids_all))
            )
            diagnostics["per_frame_active_count_accuracy"] = float(
                diagnostics.get("count_exact") or 0.0
            )
            diagnostics["gt_active_id_sets"] = [
                {
                    "frame": int(frame),
                    "gt_ids": sorted(int(tid) for tid, _box in gt_by_frame.get(frame, [])),
                }
                for frame in range(1, gt_frames + 1)
            ]
            if args.debug_overlays:
                diagnostics["debug_overlays"] = _write_debug_overlays(
                    video_path=clip.video_path,
                    clip_name=clip.name,
                    gt_by_frame=gt_by_frame,
                    pred_by_frame=pred_by_frame,
                    diagnostics=diagnostics,
                    out_dir=overlays_dir / clip.name,
                    max_frames=int(args.debug_overlay_max_frames),
                )
            if args.use_expected_count:
                try:
                    from tracking.expected_count import write_presence_plan

                    tracks_payload = joblib.load(str(tracks_path))
                    write_presence_plan(
                        diagnostics_dir / f"{clip.name}.presence_plan.json",
                        tracks_payload,
                        expected_count=int(len(gt_ids_all)),
                        num_frames=gt_frames,
                        selection_report={
                            "source": "eval_id_yolo_cvat",
                            "simulated_user_expected_count": int(len(gt_ids_all)),
                        },
                    )
                except Exception as plan_exc:  # noqa: BLE001
                    diagnostics["presence_plan_write_error"] = (
                        f"{type(plan_exc).__name__}: {plan_exc}"
                    )
            _write_events(acc, events_dir / f"{clip.name}.events.csv")
            (diagnostics_dir / f"{clip.name}.json").parent.mkdir(parents=True, exist_ok=True)
            (diagnostics_dir / f"{clip.name}.json").write_text(
                json.dumps(diagnostics, indent=2, default=_json_default)
            )
            timing = _read_timing(timing_path) if timing_path is not None else {}
            scoreboard_diag_skip = {
                "count_mismatch_frames",
                "gt_active_id_sets",
                "gt_identity_ranges",
                "pred_identity_ranges",
                "identity_assignment",
                "entry_exit_errors",
            }
            row: Dict[str, Any] = {
                "clip": clip.name,
                "status": "ok",
                "leaked": leaked,
                **metric_row,
                **{k: v for k, v in diagnostics.items() if k not in scoreboard_diag_skip},
                "tracks_path": str(tracks_path),
                "gt_path": str(clip.gt_path),
                "video_path": str(clip.video_path) if clip.video_path else "",
            }
            _add_primary_goal_flags(row, mota_threshold=float(args.mota_threshold))
            timers = timing.get("timers_s") if isinstance(timing, dict) else None
            if isinstance(timers, dict):
                row["total_run_pipeline_s"] = timing.get("total_run_pipeline_s")
                row["detect_track_loop_s"] = timers.get("detect_track_loop_s")
                row["detector_calls_s"] = timers.get("detector_calls_s")
                row["tracker_update_calls_s"] = timers.get("tracker_update_calls_s")
                row["postprocess_build_tracks_s"] = timers.get("postprocess_build_tracks_s")
            rows.append(row)
            print(
                f"{clip.name}: IDF1={float(row['idf1']):.4f} "
                f"MOTA={float(row['mota']):.4f} "
                f"count_exact={float(row['count_exact']):.4f} "
                f"IDs={row['pred_ids']}/{row['gt_ids']} "
                f"SW={int(float(row['num_switches']))} "
                f"goal_pass={row['primary_goal_pass']}"
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "clip": clip.name,
                "status": "error",
                "leaked": leaked,
                "gt_path": str(clip.gt_path),
                "video_path": str(clip.video_path) if clip.video_path else "",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{clip.name}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

    rows.append(_mean_row(rows, clip_names=ranking_clip_names))
    _write_scoreboard(rows, run_dir / "scoreboard.csv", run_dir / "scoreboard.json")
    print(f"wrote {run_dir / 'scoreboard.csv'}")
    print(f"wrote {run_dir / 'scoreboard.json'}")
    return 0 if all(r.get("status") in {"ok", "missing_tracks"} for r in rows if r.get("clip") != "MEAN_RANKING") else 1


if __name__ == "__main__":
    raise SystemExit(main())
