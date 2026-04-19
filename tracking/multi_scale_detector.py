"""Multi-scale YOLO detection ensemble.

Runs the same YOLO weights at multiple ``imgsz`` values per frame and
NMS-unions the boxes. Useful when subjects span a wide scale range
within a clip (BigTest: foreground + back-row dancers): a single
``imgsz`` always misses one band. The ensemble closes the gap without
re-training.

Phase 2 of the BigTest accuracy work uses this to reduce the
12 detector-undershoot frames remaining after Phase 0 (GT re-annotation)
and Phase 1 (proximity-gated long-gap merge).

The signature ``frame_bgr -> ndarray[N, 6]`` matches the detector hook
already used by ``eval/run_boxmot_tracker.py``, so plumbing this into a
BoxMOT tracker is a one-line drop-in.

Per-frame algorithm:
  1. Detect at each imgsz, classes=[0] (person).
  2. Concatenate all per-scale boxes.
  3. Run torchvision NMS at the supplied ``ensemble_iou`` (default 0.6).
     Score = max(conf) per merged box (the standard "max" reduction).

The ensemble is order-deterministic given the weights/seed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np


log = logging.getLogger(__name__)


def _resolve_gpu_nms_flag(explicit: Optional[bool]) -> bool:
    """Decide whether to run cross-scale NMS on the model's device.

    When explicit is None we fall back to the BEST_ID_GPU_NMS env var
    (truthy values: 1/true/yes). Default is False so legacy callers see
    no behavioural change.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("BEST_ID_GPU_NMS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_trt_engines(
    weights: Path, sorted_imgsz: List[int],
) -> Optional[dict]:
    """Map each imgsz to a TensorRT engine path if BEST_ID_TRT_ENGINE_DIR
    is set and every required engine exists.

    Convention: engines are named ``<weights_stem>_<imgsz>.engine`` (the
    layout produced by ``scripts/export_yolo_trt.py``).

    Returns None when:
      * BEST_ID_TRT_ENGINE_DIR is unset, OR
      * the directory exists but a per-scale engine is missing (we log
        a loud warning and fall back to .pt for ALL scales -- the
        ensemble must use a consistent backend so cross-scale NMS sees
        the same numerical regime).
    """
    raw = os.environ.get("BEST_ID_TRT_ENGINE_DIR", "").strip()
    if not raw:
        return None

    engine_dir = Path(raw)
    if not engine_dir.is_dir():
        log.warning("BEST_ID_TRT_ENGINE_DIR=%s is not a directory; "
                    "falling back to .pt", engine_dir)
        return None

    stem = weights.stem
    out = {}
    for imgsz in sorted_imgsz:
        candidate = engine_dir / f"{stem}_{int(imgsz)}.engine"
        if not candidate.is_file():
            log.warning(
                "BEST_ID_TRT_ENGINE_DIR=%s missing engine for imgsz=%d "
                "(expected %s); falling back to .pt for ALL scales so "
                "the ensemble stays consistent",
                engine_dir, imgsz, candidate,
            )
            return None
        out[int(imgsz)] = candidate
    log.info("TensorRT engines resolved: %s", out)
    return out


def make_multi_scale_detector(
    weights: Path,
    *,
    imgsz_list: List[int],
    conf: float,
    iou: float,
    device: str,
    ensemble_iou: float = 0.6,
    classes: Optional[List[int]] = None,
    tta_flip: bool = False,
    gpu_nms: Optional[bool] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a per-frame detector that runs YOLO at every ``imgsz`` in
    ``imgsz_list`` and NMS-unions the results.

    Output layout matches the BoxMOT detector contract:
        ``[x1, y1, x2, y2, conf, cls]``

    When ``len(imgsz_list) == 1`` this is exactly equivalent to the
    single-scale detector path used by ``eval/run_boxmot_tracker.py``,
    so callers can always wire this in unconditionally.

    ``ensemble_iou`` is the NMS IoU threshold used to fuse cross-scale
    duplicates (default 0.6 -- same person at two scales typically has
    IoU >= 0.6, while two genuinely different people sit well below).

    ``tta_flip``: when True, also predict on a horizontally-flipped copy
    of every frame at every imgsz, un-flip the boxes, and union them
    into the same NMS pool. This is classical test-time augmentation:
    detectors are not perfectly invariant to horizontal mirroring (CNN
    receptive fields, learned biases for typical pose orientations,
    asymmetric occlusion patterns), so the flipped pass catches a
    different set of borderline detections. Cost: 2x detector forwards
    per scale. Boxes are NMS-fused at ``ensemble_iou`` (the same
    threshold used for cross-scale fusion), with ``score = max(conf)``.
    Determinism: the union order is fixed (original first, then flip),
    so NMS tie-breaks are reproducible bit-for-bit across runs.
    """
    if not imgsz_list:
        raise ValueError("imgsz_list must contain at least one imgsz")
    if classes is None:
        classes = [0]

    from ultralytics import YOLO
    import torch
    from torchvision.ops import nms

    sorted_imgsz = sorted({int(s) for s in imgsz_list})
    use_gpu_nms = _resolve_gpu_nms_flag(gpu_nms)

    # One model per scale. With .pt all scales share a single YOLO
    # instance (fully dynamic). With TensorRT each scale needs its own
    # engine because TRT bakes input shapes.
    trt_engines = _resolve_trt_engines(Path(weights), sorted_imgsz)
    if trt_engines is not None:
        models_by_scale: dict = {
            sz: YOLO(str(trt_engines[sz])) for sz in sorted_imgsz
        }
        backend = "tensorrt"
    else:
        shared_model = YOLO(str(weights))
        models_by_scale = {sz: shared_model for sz in sorted_imgsz}
        backend = "pytorch"

    log.info("multi-scale detector: weights=%s imgsz=%s conf=%.3f iou=%.3f "
             "ensemble_iou=%.2f device=%s classes=%s tta_flip=%s gpu_nms=%s "
             "backend=%s",
             weights, sorted_imgsz, conf, iou, ensemble_iou, device, classes,
             tta_flip, use_gpu_nms, backend)

    # Lazy import to keep the dark-recovery module optional.
    from tracking.dark_recovery import (
        effective_conf, get_soft_nms_sigma, make_views, soft_nms_numpy,
    )

    def _detect_legacy(frame_bgr: np.ndarray) -> np.ndarray:
        """CPU-side NMS path. When the BEST_ID_DARK_* / BEST_ID_SOFT_NMS
        env vars are unset this path is byte-for-byte identical to the
        legacy v8 path (verified by regression check).
        """
        all_xyxy: List[np.ndarray] = []
        all_conf: List[np.ndarray] = []
        all_cls: List[np.ndarray] = []
        H, W = frame_bgr.shape[:2]
        eff_conf = effective_conf(conf, frame_bgr)
        views = make_views(frame_bgr)
        for view in views:
            for imgsz in sorted_imgsz:
                model = models_by_scale[imgsz]
                results = model.predict(
                    view, imgsz=int(imgsz), conf=eff_conf, iou=iou,
                    device=device, verbose=False, classes=classes,
                )
                if results:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        all_xyxy.append(
                            boxes.xyxy.cpu().numpy().astype(np.float32)
                        )
                        all_conf.append(
                            boxes.conf.cpu().numpy().astype(np.float32)
                        )
                        all_cls.append(
                            boxes.cls.cpu().numpy().astype(np.float32)
                        )

                if tta_flip:
                    # ascontiguousarray() so the YOLO predict() torch
                    # conversion doesn't trip on a negative-stride view.
                    flipped = np.ascontiguousarray(view[:, ::-1, :])
                    results_f = model.predict(
                        flipped, imgsz=int(imgsz), conf=eff_conf, iou=iou,
                        device=device, verbose=False, classes=classes,
                    )
                    if results_f:
                        boxes_f = results_f[0].boxes
                        if boxes_f is not None and len(boxes_f) > 0:
                            xyxy_f = boxes_f.xyxy.cpu().numpy().astype(np.float32)
                            new_x1 = W - xyxy_f[:, 2]
                            new_x2 = W - xyxy_f[:, 0]
                            xyxy_f[:, 0] = new_x1
                            xyxy_f[:, 2] = new_x2
                            all_xyxy.append(xyxy_f)
                            all_conf.append(
                                boxes_f.conf.cpu().numpy().astype(np.float32)
                            )
                            all_cls.append(
                                boxes_f.cls.cpu().numpy().astype(np.float32)
                            )

        if not all_xyxy:
            return np.zeros((0, 6), dtype=np.float32)

        xyxy = np.concatenate(all_xyxy, axis=0)
        conf_arr = np.concatenate(all_conf, axis=0)
        cls_arr = np.concatenate(all_cls, axis=0)

        # Single source (single scale, no TTA, no extra views) -> per-scale
        # YOLO NMS already deduped.
        if len(sorted_imgsz) == 1 and not tta_flip and len(views) == 1:
            out = np.concatenate(
                [xyxy, conf_arr[:, None], cls_arr[:, None]], axis=1,
            ).astype(np.float32)
            return out

        soft_sigma = get_soft_nms_sigma()
        if soft_sigma is not None:
            keep = soft_nms_numpy(
                xyxy, conf_arr, iou_thresh=float(ensemble_iou),
                sigma=float(soft_sigma),
            )
        else:
            boxes_t = torch.from_numpy(xyxy)
            scores_t = torch.from_numpy(conf_arr)
            keep = nms(
                boxes_t, scores_t, float(ensemble_iou),
            ).cpu().numpy()
        xyxy = xyxy[keep]
        conf_arr = conf_arr[keep]
        cls_arr = cls_arr[keep]

        out = np.concatenate(
            [xyxy, conf_arr[:, None], cls_arr[:, None]], axis=1,
        ).astype(np.float32)
        return out

    def _detect_gpu_nms(frame_bgr: np.ndarray) -> np.ndarray:
        """GPU-resident path: keep boxes/conf/cls on the model device,
        concatenate + NMS there, copy to CPU once at the very end.

        Mathematically identical to the legacy path when no env-var
        modifiers are set. With BEST_ID_DARK_* / BEST_ID_SOFT_NMS we
        fall back to the legacy CPU-side path inside the IF branches
        (these are the cheap dark-recovery features; cost is dominated
        by the YOLO forward, not the NMS).
        """
        all_xyxy_t: List[Any] = []
        all_conf_t: List[Any] = []
        all_cls_t: List[Any] = []
        H, W = frame_bgr.shape[:2]
        eff_conf = effective_conf(conf, frame_bgr)
        views = make_views(frame_bgr)
        for view in views:
            for imgsz in sorted_imgsz:
                model = models_by_scale[imgsz]
                results = model.predict(
                    view, imgsz=int(imgsz), conf=eff_conf, iou=iou,
                    device=device, verbose=False, classes=classes,
                )
                if results:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        all_xyxy_t.append(boxes.xyxy)
                        all_conf_t.append(boxes.conf)
                        all_cls_t.append(boxes.cls)

                if tta_flip:
                    flipped = np.ascontiguousarray(view[:, ::-1, :])
                    results_f = model.predict(
                        flipped, imgsz=int(imgsz), conf=eff_conf, iou=iou,
                        device=device, verbose=False, classes=classes,
                    )
                    if results_f:
                        boxes_f = results_f[0].boxes
                        if boxes_f is not None and len(boxes_f) > 0:
                            xyxy_f = boxes_f.xyxy.clone()
                            new_x1 = W - xyxy_f[:, 2]
                            new_x2 = W - xyxy_f[:, 0]
                            xyxy_f[:, 0] = new_x1
                            xyxy_f[:, 2] = new_x2
                            all_xyxy_t.append(xyxy_f)
                            all_conf_t.append(boxes_f.conf)
                            all_cls_t.append(boxes_f.cls)

        if not all_xyxy_t:
            return np.zeros((0, 6), dtype=np.float32)

        xyxy_t = torch.cat(all_xyxy_t, dim=0)
        conf_t = torch.cat(all_conf_t, dim=0)
        cls_t = torch.cat(all_cls_t, dim=0)

        # Single source (single scale, no TTA, single view) -> per-scale
        # YOLO NMS already deduped.
        if len(sorted_imgsz) == 1 and not tta_flip and len(views) == 1:
            kept_xyxy = xyxy_t
            kept_conf = conf_t
            kept_cls = cls_t
        else:
            soft_sigma = get_soft_nms_sigma()
            if soft_sigma is not None:
                # Fall through to numpy soft-NMS (cheap; dominated by YOLO).
                xyxy_np = xyxy_t.detach().cpu().numpy().astype(np.float32)
                conf_np = conf_t.detach().cpu().numpy().astype(np.float32)
                cls_np = cls_t.detach().cpu().numpy().astype(np.float32)
                keep_idx = soft_nms_numpy(
                    xyxy_np, conf_np, iou_thresh=float(ensemble_iou),
                    sigma=float(soft_sigma),
                )
                out = np.concatenate(
                    [xyxy_np[keep_idx], conf_np[keep_idx, None],
                     cls_np[keep_idx, None]], axis=1,
                ).astype(np.float32)
                return out
            keep_t = nms(xyxy_t.float(), conf_t.float(),
                         float(ensemble_iou))
            kept_xyxy = xyxy_t.index_select(0, keep_t)
            kept_conf = conf_t.index_select(0, keep_t)
            kept_cls = cls_t.index_select(0, keep_t)

        # Single device->CPU sync point (vs. 3 per scale in the legacy path).
        out_t = torch.cat(
            [kept_xyxy.float(), kept_conf.float().unsqueeze(1),
             kept_cls.float().unsqueeze(1)],
            dim=1,
        )
        return out_t.detach().cpu().numpy().astype(np.float32)

    detect = _detect_gpu_nms if use_gpu_nms else _detect_legacy

    return detect
