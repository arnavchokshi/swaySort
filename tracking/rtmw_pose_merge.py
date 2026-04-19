"""RTMW (RTMPose-Wholebody) pose-aware ID merge gate.

Implements the top RTMW integration strategy from
``work/research/rtmw_strategies.md`` -- the AND-gate approach. For each
candidate ID merge proposed by the existing IoU/proximity pass, we
extract a 133-keypoint wholebody pose at the tail of track A and the
head of track B, compute a similarity score across body+face+hands,
and only allow the merge if the similarity exceeds a threshold.

Why an AND-gate (not OR)
------------------------
The previous pose-cosine pass that ran in v6 *replaced* the IoU gate
with pose. That regressed BigTest because two dancers with similar
body shapes scored higher than the genuine self-merge (which had a
slightly mis-cropped bbox). The AND-gate avoids that failure mode by
keeping the proven IoU/proximity gate as the primary signal and using
pose only as a *secondary* veto for visually-similar dancers.

Wholebody features (vs the old 17-COCO body-only)
-------------------------------------------------
RTMW returns 133 keypoints in COCO-Wholebody layout::

    [  0..16] body   (17, same as COCO-17)
    [ 17..22] feet   (6)
    [ 23..90] face   (68)
    [ 91..111] left hand   (21)
    [112..132] right hand  (21)

Hand keypoints carry strong identity signal (relative finger lengths,
hand pose, ring/jewellery silhouette) that the body-only extractor
totally missed. Face keypoints add identity-via-feature-spacing.

The similarity measure used here is a weighted, bbox-normalised cosine
that weighs body slightly higher than hands/face because body ratios
are more stable across frames (face/hands turn). Weight defaults are
tuned per the research report; see ``_combined_similarity``.

Env-var gating (env-unset == byte-identical pass-through)
---------------------------------------------------------

  * ``BEST_ID_POSE_MERGE``           - "1"/"true" enables the gate.
  * ``BEST_ID_POSE_MERGE_THRESH``    - cosine threshold (0..1). Tracks
                                       with sim < this are NOT merged
                                       even if IoU/proximity pass says
                                       merge. Default 0.50.
  * ``BEST_ID_POSE_MERGE_BODY_W``    - weight for body cosine. Default 0.40.
  * ``BEST_ID_POSE_MERGE_HAND_W``    - weight for combined hand cosine.
                                       Default 0.40.
  * ``BEST_ID_POSE_MERGE_FACE_W``    - weight for face cosine. Default 0.20.
  * ``BEST_ID_POSE_MERGE_MIN_VIS``   - minimum keypoint visibility
                                       (RTMW score) to include in cosine.
                                       Default 0.30.
  * ``BEST_ID_POSE_MERGE_MODE``      - rtmlib mode: "lightweight" /
                                       "balanced" / "performance". Default
                                       "balanced" (RTMW-DW-X-L 256x192).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("env %s=%r not float; using %.4f", key, raw, default)
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    return _env_bool("BEST_ID_POSE_MERGE", False)


# --------------------------------------------------------------------------
# Lazy RTMW singleton
# --------------------------------------------------------------------------

_RTMW: Optional[Any] = None


def _build_rtmw() -> Any:
    """Construct (or return cached) rtmlib Wholebody pose model.

    Lazy import so the rest of the pipeline still loads when ``rtmlib``
    or ``onnxruntime`` is not installed -- the gate just stays disabled.
    """
    global _RTMW
    if _RTMW is not None:
        return _RTMW
    try:
        import torch  # noqa: F401  -- needed before rtmlib for ORT cuda libs
        from rtmlib import Wholebody
    except ImportError as exc:
        log.warning(
            "rtmlib not importable (%s); pose-merge gate disabled this run",
            exc,
        )
        return None

    mode = os.environ.get("BEST_ID_POSE_MERGE_MODE", "balanced").strip()
    if mode not in {"lightweight", "balanced", "performance"}:
        log.warning("BEST_ID_POSE_MERGE_MODE=%r invalid; using 'balanced'",
                    mode)
        mode = "balanced"
    log.info("loading RTMW (rtmlib mode=%s)", mode)
    try:
        _RTMW = Wholebody(mode=mode, backend="onnxruntime", device="cuda")
    except Exception as exc:
        log.warning("rtmlib Wholebody init failed (%s); pose-merge disabled",
                    exc)
        _RTMW = None
    return _RTMW


# --------------------------------------------------------------------------
# Feature extraction + similarity
# --------------------------------------------------------------------------


# Wholebody keypoint slice ranges (COCO-Wholebody layout used by RTMW).
_BODY_SLICE = slice(0, 17)
_FEET_SLICE = slice(17, 23)
_FACE_SLICE = slice(23, 91)
_LHAND_SLICE = slice(91, 112)
_RHAND_SLICE = slice(112, 133)


def _crop_bbox(image: np.ndarray, bbox: np.ndarray, pad: float = 0.10
               ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad+clip bbox crop, return (crop, (x0, y0)) for coord remapping."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px = int(bw * pad)
    py = int(bh * pad)
    x0 = max(0, x1 - px)
    y0 = max(0, y1 - py)
    x3 = min(W, x2 + px)
    y3 = min(H, y2 + py)
    crop = image[y0:y3, x0:x3]
    return crop, (x0, y0)


def _bbox_normalize(kpts: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Convert keypoint xy from image-coords to (cx-relative, cy-relative)
    in [-1, 1] using the bbox center & half-extents.
    """
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    hw = max(1.0, 0.5 * (bbox[2] - bbox[0]))
    hh = max(1.0, 0.5 * (bbox[3] - bbox[1]))
    out = kpts.copy()
    out[:, 0] = (out[:, 0] - cx) / hw
    out[:, 1] = (out[:, 1] - cy) / hh
    return out


class RTMWPoseExtractor:
    """Implements the duck-typed interface expected by
    :func:`tracking.postprocess._gap_id_merge` -- ``extract(frame, bbox)``
    returns a feature dict, and the class method ``cosine(a, b)`` returns
    a scalar similarity in ``[0, 1]``.

    Returned feature dict layout::

        {
          "kpts":   (133, 2)  bbox-normalised xy
          "scores": (133,)    rtmw confidence per keypoint
          "bbox":   (4,)      original xyxy
        }
    """

    def __init__(self) -> None:
        self._model = None  # built on first extract()

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _build_rtmw()
        return self._model

    def extract(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> dict:
        """Run RTMW on the bbox crop and return normalised keypoints.

        On any failure (model unavailable, RTMW exception, no
        keypoints) returns a zero-feature dict so the cosine gate
        cleanly returns 0.0 (i.e. the merge is rejected -- safer
        default than letting an unverifiable merge through).
        """
        zero = {
            "kpts": np.zeros((133, 2), dtype=np.float32),
            "scores": np.zeros((133,), dtype=np.float32),
            "bbox": np.asarray(bbox, dtype=np.float32),
        }
        model = self._ensure_model()
        if model is None:
            return zero
        crop, (x0, y0) = _crop_bbox(frame_bgr, np.asarray(bbox, dtype=np.float32))
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return zero
        try:
            kpts_all, scores_all = model(crop)
        except Exception as exc:
            log.warning("RTMW pose extraction failed: %s", exc)
            return zero
        if kpts_all is None or len(kpts_all) == 0:
            return zero
        # Pick the pose closest to the bbox center (in case YOLOX inside
        # rtmlib found multiple people in the padded crop).
        center = np.array([crop.shape[1] / 2.0, crop.shape[0] / 2.0])
        best_i = 0
        best_d = float("inf")
        for i, kpts_i in enumerate(kpts_all):
            body = kpts_i[_BODY_SLICE]
            visible = scores_all[i][_BODY_SLICE] > 0.30
            if not visible.any():
                continue
            cm = body[visible].mean(axis=0)
            d = float(np.linalg.norm(cm - center))
            if d < best_d:
                best_d = d
                best_i = i
        kpts = kpts_all[best_i].astype(np.float32)
        scores = scores_all[best_i].astype(np.float32)
        # Map back to original-image coords, then normalise by bbox.
        kpts[:, 0] += x0
        kpts[:, 1] += y0
        kpts_norm = _bbox_normalize(kpts, np.asarray(bbox, dtype=np.float32))
        return {
            "kpts": kpts_norm.astype(np.float32),
            "scores": scores,
            "bbox": np.asarray(bbox, dtype=np.float32),
        }

    @classmethod
    def cosine(cls, a: dict, b: dict) -> float:
        """Weighted body+hands+face cosine similarity in [0, 1]."""
        if (
            not isinstance(a, dict) or not isinstance(b, dict)
            or "kpts" not in a or "kpts" not in b
        ):
            return 0.0
        kpts_a = np.asarray(a["kpts"], dtype=np.float32)
        kpts_b = np.asarray(b["kpts"], dtype=np.float32)
        sc_a = np.asarray(a["scores"], dtype=np.float32)
        sc_b = np.asarray(b["scores"], dtype=np.float32)
        if kpts_a.shape != (133, 2) or kpts_b.shape != (133, 2):
            return 0.0

        min_vis = _env_float("BEST_ID_POSE_MERGE_MIN_VIS", 0.30)
        body_w = _env_float("BEST_ID_POSE_MERGE_BODY_W", 0.40)
        hand_w = _env_float("BEST_ID_POSE_MERGE_HAND_W", 0.40)
        face_w = _env_float("BEST_ID_POSE_MERGE_FACE_W", 0.20)

        return float(_combined_similarity(
            kpts_a, sc_a, kpts_b, sc_b,
            min_vis=min_vis,
            body_w=body_w, hand_w=hand_w, face_w=face_w,
        ))


def _slice_cosine(
    kpts_a: np.ndarray, sc_a: np.ndarray,
    kpts_b: np.ndarray, sc_b: np.ndarray,
    sl: slice, *, min_vis: float,
) -> Optional[float]:
    """Cosine over a keypoint slice using only points visible in BOTH."""
    a_xy = kpts_a[sl]
    b_xy = kpts_b[sl]
    a_v = sc_a[sl] >= min_vis
    b_v = sc_b[sl] >= min_vis
    visible = a_v & b_v
    n = int(visible.sum())
    if n < 3:
        return None
    av = a_xy[visible].reshape(-1)
    bv = b_xy[visible].reshape(-1)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.dot(av, bv) / (na * nb))


def _combined_similarity(
    kpts_a: np.ndarray, sc_a: np.ndarray,
    kpts_b: np.ndarray, sc_b: np.ndarray,
    *, min_vis: float, body_w: float, hand_w: float, face_w: float,
) -> float:
    """Weighted similarity over body / (left+right hand) / face slices.

    Each component returns None if not enough mutually-visible keypoints,
    in which case its weight is redistributed to the other components.
    Final score is mapped from [-1, 1] -> [0, 1] for use as a threshold.
    """
    body_cos = _slice_cosine(kpts_a, sc_a, kpts_b, sc_b, _BODY_SLICE,
                             min_vis=min_vis)
    lhand_cos = _slice_cosine(kpts_a, sc_a, kpts_b, sc_b, _LHAND_SLICE,
                              min_vis=min_vis)
    rhand_cos = _slice_cosine(kpts_a, sc_a, kpts_b, sc_b, _RHAND_SLICE,
                              min_vis=min_vis)
    face_cos = _slice_cosine(kpts_a, sc_a, kpts_b, sc_b, _FACE_SLICE,
                             min_vis=min_vis)

    # Average the two hand cosines (whichever are present).
    hand_vals = [v for v in (lhand_cos, rhand_cos) if v is not None]
    hand_cos = float(np.mean(hand_vals)) if hand_vals else None

    components = [
        (body_cos, body_w),
        (hand_cos, hand_w),
        (face_cos, face_w),
    ]
    present = [(c, w) for (c, w) in components if c is not None]
    if not present:
        return 0.0
    total_w = sum(w for _, w in present)
    if total_w <= 0.0:
        return 0.0
    weighted = sum(c * (w / total_w) for c, w in present)
    return 0.5 * (weighted + 1.0)


def make_extractor() -> Optional[RTMWPoseExtractor]:
    """Return an extractor instance iff the gate is enabled. The
    extractor itself lazy-loads the RTMW model on first ``extract()``.
    """
    if not is_enabled():
        return None
    return RTMWPoseExtractor()


def get_pose_cos_thresh() -> float:
    """Cosine threshold used by ``postprocess._gap_id_merge``."""
    return _env_float("BEST_ID_POSE_MERGE_THRESH", 0.50)
