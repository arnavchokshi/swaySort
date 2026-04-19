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
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np


log = logging.getLogger(__name__)


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

    model = YOLO(str(weights))
    sorted_imgsz = sorted({int(s) for s in imgsz_list})
    log.info("multi-scale detector: weights=%s imgsz=%s conf=%.3f iou=%.3f "
             "ensemble_iou=%.2f device=%s classes=%s tta_flip=%s",
             weights, sorted_imgsz, conf, iou, ensemble_iou, device, classes,
             tta_flip)

    def detect(frame_bgr: np.ndarray) -> np.ndarray:
        all_xyxy: List[np.ndarray] = []
        all_conf: List[np.ndarray] = []
        all_cls: List[np.ndarray] = []
        H, W = frame_bgr.shape[:2]
        for imgsz in sorted_imgsz:
            results = model.predict(
                frame_bgr, imgsz=int(imgsz), conf=conf, iou=iou,
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
                flipped = np.ascontiguousarray(frame_bgr[:, ::-1, :])
                results_f = model.predict(
                    flipped, imgsz=int(imgsz), conf=conf, iou=iou,
                    device=device, verbose=False, classes=classes,
                )
                if results_f:
                    boxes_f = results_f[0].boxes
                    if boxes_f is not None and len(boxes_f) > 0:
                        xyxy_f = boxes_f.xyxy.cpu().numpy().astype(np.float32)
                        # Un-flip: x1 <- W - x2, x2 <- W - x1.
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

        # When we have only a single source (single scale, no TTA),
        # YOLO's per-scale NMS already de-duplicated, so we can return
        # directly.
        if len(sorted_imgsz) == 1 and not tta_flip:
            out = np.concatenate(
                [xyxy, conf_arr[:, None], cls_arr[:, None]], axis=1,
            ).astype(np.float32)
            return out

        boxes_t = torch.from_numpy(xyxy)
        scores_t = torch.from_numpy(conf_arr)
        keep = nms(boxes_t, scores_t, float(ensemble_iou)).cpu().numpy()
        xyxy = xyxy[keep]
        conf_arr = conf_arr[keep]
        cls_arr = cls_arr[keep]

        out = np.concatenate(
            [xyxy, conf_arr[:, None], cls_arr[:, None]], axis=1,
        ).astype(np.float32)
        return out

    return detect
