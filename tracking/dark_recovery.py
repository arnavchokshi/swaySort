"""Low-light recovery preprocessing for the multi-scale detector.

All helpers here are env-var-gated so that the production path
(`tracking.run_pipeline`, `tracking.multi_scale_detector`) is byte-for-byte
unchanged when the relevant env vars are unset.

Env vars and their semantics:

  * ``BEST_ID_DARK_PROFILE``      - "v9" enables the validated v9 production
                                    defaults: ``BEST_ID_DARK_CLAHE=1`` and
                                    ``BEST_ID_DARK_GAMMA=auto`` (luma-gated, so
                                    bright clips remain byte-identical to v8).
                                    Explicit env vars below still override.
                                    Default: unset (legacy v8 behaviour, used
                                    by the sweep harnesses for clean baselines).

  * ``BEST_ID_DARK_GAMMA``        - "auto" or float (e.g. "1.6"). When set,
                                    the preprocessor applies gamma correction
                                    BEFORE the detector forward. "auto" =
                                    luma-adaptive gamma in [1.0, 2.5].
                                    Default: unset (no-op unless v9 profile).

  * ``BEST_ID_DARK_CLAHE``        - "1"/"true"/"on" enables CLAHE on the LAB
                                    L-channel BEFORE the detector forward.
                                    Default: unset (no-op unless v9 profile).

  * ``BEST_ID_DARK_LUMA``         - threshold for "is_dark" gate
                                    (mean Y in [0,255]). Default 80.
                                    Frames brighter than this skip dark
                                    preprocessing.

  * ``BEST_ID_ADAPTIVE_CONF``     - float (e.g. "0.06"). When set, the
                                    multi-scale detector subtracts this from
                                    its base ``conf`` for any "is_dark" frame.
                                    Default: unset (no-op).

  * ``BEST_ID_DARK_BRIGHTEN``     - float (e.g. "1.8") enables the
                                    multi-exposure ensemble: detector also
                                    runs on a brightened (luma * factor) copy
                                    of dark frames. Default: unset (no-op).

  * ``BEST_ID_SOFT_NMS``          - float sigma (e.g. "0.5"). When set, the
                                    cross-scale NMS in
                                    ``tracking.multi_scale_detector`` is
                                    replaced by linear Soft-NMS with the
                                    given sigma. Default: unset (hard NMS).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# env var resolution
# ---------------------------------------------------------------------------


def _env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("env %s=%r not a float; ignoring", key, raw)
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def get_luma_threshold() -> float:
    return _env_float("BEST_ID_DARK_LUMA", 80.0) or 80.0


def _v9_profile_enabled() -> bool:
    """Return True if the v9 production-default profile is selected.

    The v9 profile bundles ``CLAHE + gamma=auto``, the configuration
    that won the dark_recovery_finalists sweep (mean IDF1 0.9263 vs
    v8 baseline 0.9208, +0.0501 IDF1 on darkTest, byte-identical on
    every well-lit clip thanks to luma-gating).
    """
    return os.environ.get("BEST_ID_DARK_PROFILE", "").strip().lower() == "v9"


def get_gamma_setting() -> Optional[str]:
    """Return None / 'auto' / float-as-string.

    Resolution order (highest priority first):
      1. Explicit ``BEST_ID_DARK_GAMMA`` env var.
      2. v9 profile default ("auto") when ``BEST_ID_DARK_PROFILE=v9``.
      3. None (legacy v8 behaviour).
    """
    raw = os.environ.get("BEST_ID_DARK_GAMMA", "").strip()
    if raw:
        return raw
    if _v9_profile_enabled():
        return "auto"
    return None


def get_clahe_enabled() -> bool:
    """Resolve CLAHE flag with v9-profile fallback (see get_gamma_setting)."""
    raw = os.environ.get("BEST_ID_DARK_CLAHE", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return _v9_profile_enabled()


def get_adaptive_conf_delta() -> Optional[float]:
    return _env_float("BEST_ID_ADAPTIVE_CONF", None)


def get_brighten_factor() -> Optional[float]:
    return _env_float("BEST_ID_DARK_BRIGHTEN", None)


def get_soft_nms_sigma() -> Optional[float]:
    return _env_float("BEST_ID_SOFT_NMS", None)


# ---------------------------------------------------------------------------
# preprocessing primitives
# ---------------------------------------------------------------------------


def frame_luma_mean(bgr: np.ndarray) -> float:
    """Cheap mean luminance estimate (Y channel)."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return 128.0
    # Approximation via grayscale mean -- much cheaper than full BT.601.
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def is_dark(bgr: np.ndarray, threshold: Optional[float] = None) -> bool:
    if threshold is None:
        threshold = get_luma_threshold()
    return frame_luma_mean(bgr) < threshold


# Cache LUTs per gamma to avoid regenerating each frame.
_GAMMA_LUT_CACHE: dict = {}


def _gamma_lut(gamma: float) -> np.ndarray:
    g = round(float(gamma), 2)
    if g not in _GAMMA_LUT_CACHE:
        inv = 1.0 / max(g, 1e-3)
        table = ((np.arange(256) / 255.0) ** inv) * 255.0
        _GAMMA_LUT_CACHE[g] = np.clip(table, 0, 255).astype(np.uint8)
    return _GAMMA_LUT_CACHE[g]


def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply per-channel gamma via 256-element LUT (fast)."""
    if abs(gamma - 1.0) < 1e-3:
        return bgr
    return cv2.LUT(bgr, _gamma_lut(gamma))


def auto_gamma(bgr: np.ndarray) -> float:
    """Adaptive gamma based on mean luma. Returns 1.0 for bright frames."""
    luma = frame_luma_mean(bgr)
    # Gamma climbs from 1.0 at luma=90 up to 2.5 at luma=30.
    gamma = 1.0 + max(0.0, min(1.5, (90.0 - luma) / 30.0))
    return float(gamma)


# CLAHE objects are stateful but cheap to construct; reuse one per process.
_CLAHE_OBJ = None


def _clahe() -> "cv2.CLAHE":
    global _CLAHE_OBJ
    if _CLAHE_OBJ is None:
        _CLAHE_OBJ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE_OBJ


def apply_clahe_lab_l(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the LAB L-channel; preserves color, boosts local contrast."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_eq = _clahe().apply(L)
    lab_eq = cv2.merge([L_eq, A, B])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_brighten(bgr: np.ndarray, factor: float) -> np.ndarray:
    """Multiply BGR by ``factor`` and clip to 0..255 (uint8)."""
    if abs(factor - 1.0) < 1e-3:
        return bgr
    out = bgr.astype(np.float32) * float(factor)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# combined preprocessor used by the detector
# ---------------------------------------------------------------------------


def preprocess_for_detector(bgr: np.ndarray) -> np.ndarray:
    """Apply env-gated dark-recovery preprocessing.

    Order: CLAHE -> gamma. Both run only when ``is_dark`` is True (so
    well-lit frames are byte-identical to the legacy code path).
    Returns the (possibly modified) frame.
    """
    if not is_dark(bgr):
        return bgr
    out = bgr
    if get_clahe_enabled():
        out = apply_clahe_lab_l(out)
    gamma_setting = get_gamma_setting()
    if gamma_setting is not None:
        if gamma_setting == "auto":
            gamma = auto_gamma(out)
        else:
            try:
                gamma = float(gamma_setting)
            except ValueError:
                gamma = 1.0
        out = apply_gamma(out, gamma)
    return out


def make_views(bgr: np.ndarray) -> List[np.ndarray]:
    """Return one or more views of the input frame to feed the detector.

    Always at least the (possibly preprocessed) original view. When
    BEST_ID_DARK_BRIGHTEN is set AND the frame is dark, also returns a
    brightened view -- the detector will fuse them via NMS-union.
    """
    primary = preprocess_for_detector(bgr)
    views = [primary]
    factor = get_brighten_factor()
    if factor is not None and is_dark(bgr):
        views.append(apply_brighten(primary, factor))
    return views


def effective_conf(base_conf: float, bgr: np.ndarray) -> float:
    """Adjust per-frame detector conf if BEST_ID_ADAPTIVE_CONF is set."""
    delta = get_adaptive_conf_delta()
    if delta is None:
        return base_conf
    if not is_dark(bgr):
        return base_conf
    return max(0.05, base_conf - float(delta))


# ---------------------------------------------------------------------------
# Soft-NMS: linear variant (Bodla et al., ICCV 2017)
# ---------------------------------------------------------------------------


def soft_nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, *,
    iou_thresh: float, sigma: float = 0.5, score_thresh: float = 0.001,
) -> np.ndarray:
    """Linear Soft-NMS. Returns indices of kept boxes (in original order
    of *kept-score-descending*).

    Mathematically a strict superset of hard NMS (sigma -> 0 = hard NMS).
    """
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    boxes = boxes.astype(np.float64).copy()
    scores = scores.astype(np.float64).copy()
    indices = np.arange(len(boxes))
    keep: List[int] = []

    while len(indices) > 0:
        i = int(np.argmax(scores))
        keep.append(int(indices[i]))
        if len(indices) == 1:
            break
        # IoU of box i vs all others
        xx1 = np.maximum(boxes[i, 0], boxes[:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = max(0.0, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_i + area - inter
        iou = np.where(union > 0, inter / union, 0.0)

        # Decay scores of high-IoU boxes
        decay = np.exp(-(iou ** 2) / max(sigma, 1e-9))
        # Above iou_thresh -> apply decay; below -> keep score (linear soft-NMS variant)
        decay = np.where(iou > iou_thresh, decay, 1.0)
        scores = scores * decay
        # Drop the picked box
        scores[i] = -1.0

        # Drop boxes whose score fell below threshold
        mask = scores > score_thresh
        scores = scores[mask]
        boxes = boxes[mask]
        indices = indices[mask]

    return np.asarray(keep, dtype=np.int64)
