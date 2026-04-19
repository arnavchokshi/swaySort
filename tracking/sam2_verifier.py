"""SAM 2.1 per-bbox verifier for phantom-track removal.

Implements the top-priority SAM 2.1 integration strategy from
``work/research/sam21_strategies.md`` -- the one path that AVOIDS the
two failure modes documented in past sessions (mask-propagation
identity fusion, video-predictor identity drift).

What it does
------------
For each bounding box in a tracker frame, the verifier asks the SAM 2.1
**image** predictor (NOT the video predictor) "given this box, what
fraction of pixels inside the box belong to the foreground mask?". If
the fill ratio falls below ``BEST_ID_SAM_VERIFY_FILL`` the box is
classified as a phantom (no actual person inside) and dropped from the
cache.

This is a MASKING-style verifier, not an ID propagator: it never
re-assigns identities, never modifies bbox coordinates, and never
shares state across frames. So the past failure modes -- which all
involved the video predictor's cross-frame identity propagation --
cannot occur here.

Performance gating
------------------
Verifying every box every frame is wasteful. By default the verifier
only fires on boxes that meet ALL of:

  1. ``conf < BEST_ID_SAM_VERIFY_CONF_MAX`` (default 0.55) -- high-conf
     YOLO boxes are almost never phantoms; skip them to save GPU.
  2. ``area < BEST_ID_SAM_VERIFY_AREA_MAX`` (default 100_000 px) --
     huge boxes are also almost never phantoms.
  3. The frame is among the "suspicious" frames flagged by FN-recovery
     (cardinality drop) OR every Nth frame as a baseline check
     (env-overridable via ``BEST_ID_SAM_VERIFY_STRIDE``, default 5).

Env-var gating (env-unset == byte-identical pass-through):

  * ``BEST_ID_SAM_VERIFY``           - "1"/"true" enables the pass.
  * ``BEST_ID_SAM_VERIFY_WEIGHTS``   - path to SAM 2.1 .pt checkpoint.
                                       Default looks under
                                       ``weights/sam2/sam2.1_hiera_tiny.pt``
                                       relative to repo root.
  * ``BEST_ID_SAM_VERIFY_CFG``       - SAM 2.1 model config name.
                                       Default ``sam2.1_hiera_t.yaml``.
  * ``BEST_ID_SAM_VERIFY_FILL``      - mask fill threshold below which a
                                       box is considered a phantom.
                                       Default 0.30 (30% of bbox pixels
                                       must be foreground).
  * ``BEST_ID_SAM_VERIFY_CONF_MAX``  - skip boxes with detector
                                       conf >= this. Default 0.55.
  * ``BEST_ID_SAM_VERIFY_AREA_MAX``  - skip boxes with bbox area
                                       >= this (px). Default 100_000.
  * ``BEST_ID_SAM_VERIFY_STRIDE``    - run on every Nth frame as a
                                       baseline. Default 5. Set to 1 to
                                       run every frame (expensive).
  * ``BEST_ID_SAM_VERIFY_DEVICE``    - "cuda:0" / "cpu". Default tries
                                       cuda then falls back to cpu.

The verifier mutates the FrameDetections cache in place. Returns the
number of dropped boxes (across all frames).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("env %s=%r not float; using %.4f", key, raw, default)
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("env %s=%r not int; using %d", key, raw, default)
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    return _env_bool("BEST_ID_SAM_VERIFY", False)


# --------------------------------------------------------------------------
# Lazy SAM 2.1 image predictor singleton
# --------------------------------------------------------------------------

_PREDICTOR: Optional[Any] = None
_PREDICTOR_DEVICE: Optional[str] = None


def _resolve_device() -> str:
    raw = os.environ.get("BEST_ID_SAM_VERIFY_DEVICE", "").strip()
    if raw:
        return raw
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _resolve_weights() -> Path:
    raw = os.environ.get("BEST_ID_SAM_VERIFY_WEIGHTS", "").strip()
    if raw:
        return Path(raw)
    return REPO_ROOT / "weights" / "sam2" / "sam2.1_hiera_tiny.pt"


def _resolve_cfg() -> str:
    raw = os.environ.get("BEST_ID_SAM_VERIFY_CFG", "").strip()
    if raw:
        return raw
    return "configs/sam2.1/sam2.1_hiera_t.yaml"


def _build_predictor() -> Any:
    """Construct (or return cached) SAM 2.1 image predictor.

    Imports are lazy so that the module loads even when ``sam2`` is not
    installed (the verifier just stays disabled in that case).
    """
    global _PREDICTOR, _PREDICTOR_DEVICE
    device = _resolve_device()
    if _PREDICTOR is not None and _PREDICTOR_DEVICE == device:
        return _PREDICTOR

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        log.warning(
            "sam2 not importable (%s); SAM verifier disabled this run",
            exc,
        )
        return None

    weights = _resolve_weights()
    cfg = _resolve_cfg()
    if not weights.is_file():
        log.warning(
            "SAM 2.1 weights missing at %s; SAM verifier disabled this run",
            weights,
        )
        return None

    log.info("loading SAM 2.1 image predictor: cfg=%s weights=%s device=%s",
             cfg, weights, device)
    sam_model = build_sam2(cfg, str(weights), device=device)
    _PREDICTOR = SAM2ImagePredictor(sam_model)
    _PREDICTOR_DEVICE = device
    return _PREDICTOR


# --------------------------------------------------------------------------
# Per-frame verifier
# --------------------------------------------------------------------------


def _bbox_area(b: np.ndarray) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def _candidate_indices(
    fd: Any, *, conf_max: float, area_max: float,
) -> np.ndarray:
    """Return the indices of boxes that *could* be phantoms (worth
    verifying). High-conf or huge boxes are skipped to save GPU.
    """
    if len(fd.tids) == 0:
        return np.empty((0,), dtype=np.int64)
    confs = np.asarray(fd.confs, dtype=np.float32)
    areas = np.array([_bbox_area(fd.xyxys[i]) for i in range(len(fd.tids))],
                     dtype=np.float32)
    mask = (confs < conf_max) & (areas < area_max)
    return np.nonzero(mask)[0].astype(np.int64)


def _frames_iter(video: Path):
    """Cheap frame iterator yielding (idx, bgr) -- duplicates the
    minimal subset of tracking.deepocsort_runner.iter_video_frames.
    """
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log.warning("could not open video %s for SAM verification", video)
        return
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def _suspicious_frames(cache: List, drop_thresh: int = 1,
                       window: int = 30) -> set:
    """Identify frames where the per-frame box count drops below the
    running median by `drop_thresh`. Same idea as fn_recovery.
    """
    counts = np.asarray(
        [len(getattr(fd, "tids", [])) for fd in cache], dtype=np.float64,
    )
    n = len(cache)
    half = max(1, window // 2)
    susp = set()
    for t in range(n):
        a = max(0, t - half)
        b = min(n, t + half + 1)
        med = float(np.median(counts[a:b]))
        if counts[t] <= med - drop_thresh:
            susp.add(t)
    return susp


def verify_cache(
    cache: List,
    *,
    video: Optional[Path] = None,
) -> int:
    """Mutate ``cache`` in place by dropping boxes that fail the SAM
    foreground-coverage check. Returns the number of dropped boxes.

    ``video`` is required so we can re-decode the frames the SAM
    predictor needs (FrameDetections only stores boxes, not pixels).
    When ``video`` is None we abort with a no-op (logging a warning).
    """
    if not is_enabled():
        return 0
    if not cache:
        return 0
    if video is None:
        log.warning("SAM verifier requires the source video; skipping")
        return 0

    predictor = _build_predictor()
    if predictor is None:
        return 0

    fill_thresh = _env_float("BEST_ID_SAM_VERIFY_FILL", 0.30)
    conf_max = _env_float("BEST_ID_SAM_VERIFY_CONF_MAX", 0.55)
    area_max = _env_float("BEST_ID_SAM_VERIFY_AREA_MAX", 100_000.0)
    stride = max(1, _env_int("BEST_ID_SAM_VERIFY_STRIDE", 5))

    suspicious = _suspicious_frames(cache)
    n_dropped_total = 0
    n_checked_boxes = 0
    n_checked_frames = 0

    for idx, frame_bgr in _frames_iter(Path(video)):
        if idx >= len(cache):
            break
        fd = cache[idx]
        if len(fd.tids) == 0:
            continue
        # Frame-level gate: only verify suspicious frames OR every Nth.
        if (idx not in suspicious) and (idx % stride != 0):
            continue
        cand = _candidate_indices(fd, conf_max=conf_max, area_max=area_max)
        if len(cand) == 0:
            continue
        # Single set_image() per frame; per-box predict() cheap thereafter.
        try:
            import torch
            with torch.inference_mode():
                # SAM 2 expects RGB.
                rgb = frame_bgr[:, :, ::-1].copy()
                predictor.set_image(rgb)
                drop_mask = np.zeros(len(fd.tids), dtype=bool)
                for k in cand:
                    box = np.asarray(fd.xyxys[int(k)], dtype=np.float32)
                    masks, scores, _ = predictor.predict(
                        box=box, multimask_output=False,
                    )
                    if masks is None or len(masks) == 0:
                        continue
                    mask = masks[0] > 0
                    bx1, by1, bx2, by2 = box.astype(int)
                    bx1 = max(0, bx1); by1 = max(0, by1)
                    bx2 = min(mask.shape[1], bx2); by2 = min(mask.shape[0], by2)
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    bbox_area = float((bx2 - bx1) * (by2 - by1))
                    inside = float(mask[by1:by2, bx1:bx2].sum())
                    fill = inside / max(bbox_area, 1.0)
                    n_checked_boxes += 1
                    if fill < fill_thresh:
                        drop_mask[int(k)] = True
        except Exception as exc:
            log.exception("SAM verifier failed at frame %d: %s; "
                          "leaving boxes intact", idx, exc)
            continue

        n_checked_frames += 1
        if not drop_mask.any():
            continue

        from prune_tracks import FrameDetections
        keep = ~drop_mask
        cache[idx] = FrameDetections(
            np.asarray(fd.xyxys, dtype=np.float32)[keep],
            np.asarray(fd.confs, dtype=np.float32)[keep],
            np.asarray(fd.tids, dtype=np.float32)[keep],
        )
        n_dropped_total += int(drop_mask.sum())

    log.info(
        "SAM verifier: dropped %d / %d boxes across %d frames "
        "(stride=%d, conf_max=%.2f, area_max=%.0f, fill_thresh=%.2f)",
        n_dropped_total, n_checked_boxes, n_checked_frames,
        stride, conf_max, area_max, fill_thresh,
    )
    return n_dropped_total
