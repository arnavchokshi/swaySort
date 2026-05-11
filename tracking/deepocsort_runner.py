"""DeepOcSort tracker runner -- the single shipped tracker.

This is the only tracker the repo ships. It wraps BoxMOT's DeepOcSort,
plus the cholesky-jitter Kalman patch we need on long dance clips, and
emits ``RawTrack[]`` so ``tracking/postprocess.py`` consumes it
unchanged.

Why DeepOcSort: see ``docs/EXPERIMENTS_LOG.md`` for the full scoreboard
across BoxMOT trackers (DeepOcSort, BotSort, ByteTrack, OcSort,
StrongSort, HybridSort) and the CAMELTrack experiment. DeepOcSort wins
the 7-clip mean IDF1 in the v8 leaderboard.

Public API:
  * ``install_kalman_jitter_patch()`` -- one-shot BoxMOT patch.
  * ``make_tracker(reid_weights, device, half=False)`` -- build a fresh
    DeepOcSort instance with the cholesky-jitter Kalman patch applied.
  * ``iter_video_frames(path)`` -- yield ``(idx, frame_bgr)`` for a
    video file or directory of frame images.
  * ``run_deepocsort(video, yolo, cfg, ...)`` -- legacy programmatic
    driver returning ``RawTrack[]``. Newer callers should prefer
    ``tracking.run_pipeline.run_pipeline_on_video`` which produces the
    cache that ``tracking.best_pipeline.build_tracks`` consumes directly.
"""

from __future__ import annotations

import logging
import os
import inspect
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from tracking.postprocess import RawTrack


log = logging.getLogger(__name__)


# Cholesky-jitter patch is the same one ``eval/run_boxmot_tracker.py``
# installs. Keeping a single canonical implementation here so both
# callers (production pipeline + eval runner) can import it.
_JITTER_PATCH_INSTALLED = False


def install_kalman_jitter_patch() -> None:
    """Patch BoxMOT's Kalman filter ``update`` to retry on cholesky failures.

    DeepOcSort occasionally produces a covariance matrix that is barely
    positive-semi-definite; ``scipy.linalg.cho_factor`` then raises and the
    whole frame's track update is silently aborted. We retry with an
    increasing identity-jitter on the covariance diagonal. Idempotent.
    """
    global _JITTER_PATCH_INSTALLED
    if _JITTER_PATCH_INSTALLED:
        return

    import scipy.linalg

    try:
        from boxmot.motion.kalman_filters.base import BaseKalmanFilter  # type: ignore
    except ImportError:
        try:
            from boxmot.motion.kalman_filters.xysr_kf import (  # type: ignore
                KalmanFilterXYSR as BaseKalmanFilter,
            )
        except ImportError:
            log.info("kalman jitter patch skipped: no recognized BaseKalmanFilter")
            _JITTER_PATCH_INSTALLED = True
            return

    method_name = (
        "update_state" if hasattr(BaseKalmanFilter, "update_state")
        else "update" if hasattr(BaseKalmanFilter, "update")
        else None
    )
    if method_name is None:
        log.info("kalman jitter patch skipped: no compatible update method")
        _JITTER_PATCH_INSTALLED = True
        return

    orig = getattr(BaseKalmanFilter, method_name)

    def _wrapped(self, z, *args, _max_tries: int = 6, _eps0: float = 1e-6, **kwargs):
        last_exc: Optional[Exception] = None
        for attempt in range(_max_tries):
            try:
                return orig(self, z, *args, **kwargs)
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as exc:
                last_exc = exc
                eps = _eps0 * (10 ** attempt)
                try:
                    dim = self.P.shape[0]
                    self.P = self.P + np.eye(dim) * eps
                except Exception:
                    break
        log.debug("kf %s: cholesky failed after %d retries (%s)",
                  method_name, _max_tries, last_exc)

    setattr(BaseKalmanFilter, method_name, _wrapped)
    _JITTER_PATCH_INSTALLED = True
    log.info("installed BoxMOT KalmanFilter cholesky-jitter patch (%s)", method_name)


def _resolve_deepocsort_class():
    """Find the DeepOcSort class across BoxMOT versions.

    BoxMOT has shuffled its public class names across versions
    (DeepOcSort -> DeepOCSort -> DeepOCSORT). Look in the legacy
    ``boxmot.trackers`` namespace first, then the package root.
    """
    aliases = ["DeepOcSort", "DeepOCSort", "DeepOCSORT"]
    namespaces = []
    try:
        from boxmot import trackers as bm_trackers
        namespaces.append(bm_trackers)
    except ImportError:
        pass
    import boxmot as bm
    namespaces.append(bm)

    for ns in namespaces:
        for name in aliases:
            cls = getattr(ns, name, None)
            if cls is not None:
                return cls
    raise ImportError(
        "BoxMOT DeepOcSort not found. Install boxmot>=10.0.52 "
        "and verify with `python -c 'import boxmot; print(boxmot.DeepOcSort)'`."
    )


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r not float; ignoring", name, raw)
        return None


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r not int; ignoring", name, raw)
        return None


def _maybe_set_kw(
    kw: Dict[str, object],
    params: Dict[str, inspect.Parameter],
    *,
    accepts_kwargs: bool,
    key: str,
    env: str,
    cast,
) -> None:
    if key not in params and not accepts_kwargs:
        return
    value = cast(env)
    if value is not None:
        kw[key] = value


def _env_bool_override(name: str) -> Optional[bool]:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    log.warning("%s=%r not bool; ignoring", name, raw)
    return None


def make_tracker(*, reid_weights: Path, device: str, half: bool = False):
    """Construct a DeepOcSort instance with version-tolerant kwargs."""
    install_kalman_jitter_patch()
    import torch

    cls = _resolve_deepocsort_class()
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(device if ":" in str(device) else "cuda")
    elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")

    kw: Dict[str, object] = {}
    if "model_weights" in params:
        kw["model_weights"] = Path(reid_weights)
    elif "reid_weights" in params:
        kw["reid_weights"] = Path(reid_weights)
    if "device" in params:
        kw["device"] = torch_device
    if "fp16" in params:
        kw["fp16"] = bool(half)
    elif "half" in params:
        kw["half"] = bool(half)
    if "det_thresh" in params and "det_thresh" not in kw:
        default = params["det_thresh"].default
        kw["det_thresh"] = default if default is not inspect.Parameter.empty else 0.3
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="det_thresh", env="BEST_ID_TRACKER_DET_THRESH", cast=_env_float,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="max_age", env="BEST_ID_TRACKER_MAX_AGE", cast=_env_int,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="min_hits", env="BEST_ID_TRACKER_MIN_HITS", cast=_env_int,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="iou_threshold", env="BEST_ID_TRACKER_IOU_THRESHOLD", cast=_env_float,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="delta_t", env="BEST_ID_DEEPOCSORT_DELTA_T", cast=_env_int,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="inertia", env="BEST_ID_DEEPOCSORT_INERTIA", cast=_env_float,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="w_association_emb", env="BEST_ID_DEEPOCSORT_W_ASSOCIATION_EMB",
        cast=_env_float,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="alpha_fixed_emb", env="BEST_ID_DEEPOCSORT_ALPHA_FIXED_EMB",
        cast=_env_float,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="aw_param", env="BEST_ID_DEEPOCSORT_AW_PARAM", cast=_env_float,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="Q_xy_scaling", env="BEST_ID_DEEPOCSORT_Q_XY_SCALING",
        cast=_env_float,
    )
    _maybe_set_kw(
        kw, params, accepts_kwargs=accepts_kwargs,
        key="Q_s_scaling", env="BEST_ID_DEEPOCSORT_Q_S_SCALING",
        cast=_env_float,
    )
    for key, env in (
        ("embedding_off", "BEST_ID_DEEPOCSORT_EMBEDDING_OFF"),
        ("cmc_off", "BEST_ID_DEEPOCSORT_CMC_OFF"),
        ("aw_off", "BEST_ID_DEEPOCSORT_AW_OFF"),
    ):
        if key in params or accepts_kwargs:
            value = _env_bool_override(env)
            if value is not None:
                kw[key] = value
    return cls(**kw)


def iter_video_frames(video: Path, *, frame_stride: int = 1):
    """Yield (idx, frame_bgr) for either a video file or a directory of frames.

    ``frame_stride`` (default ``1``) keeps every Nth source frame and
    renumbers the yielded ``idx`` to the dense kept-frame index (so the
    consumer never sees holes).  This is what lets workstream B run the
    tracker on a 60 fps source as if it were a 30 fps source — the
    tracker, the on-disk frame extractor and the body4d input all walk
    the same dense index space.  ``stride=1`` (the default) is a no-op
    that exactly matches the legacy behaviour.
    """
    stride = max(1, int(frame_stride))
    if video.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            [p for p in video.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: p.name,
        )
        if not paths:
            raise FileNotFoundError(f"No frames in {video}")
        kept = 0
        for i, p in enumerate(paths):
            if (i % stride) != 0:
                continue
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read {p}")
            yield kept, frame
            kept += 1
        return

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video}")
    try:
        i = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (i % stride) == 0:
                yield kept, frame
                kept += 1
            i += 1
    finally:
        cap.release()


def iter_video_frames_prefetched(
    video: Path, queue_size: int = 4, *, frame_stride: int = 1
):
    """Same contract as ``iter_video_frames`` but decodes ahead in a
    background thread.

    Yields exactly the same (idx, frame_bgr) sequence as the synchronous
    iterator (cv2 / image decode is deterministic), so this is a drop-in
    replacement for any consumer. The benefit is that frame N+1's decode
    overlaps with frame N's detect+track on the GPU, hiding 5--15 ms of
    cv2 decode latency per frame.

    Args:
        video: same as ``iter_video_frames``.
        queue_size: max frames buffered in the inter-thread queue. Each
            1080p frame is ~6 MB so 4--8 is a sensible cap; 0 falls back
            to the synchronous iterator.
    """
    if queue_size <= 0:
        yield from iter_video_frames(video, frame_stride=frame_stride)
        return

    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=int(queue_size))
    stop = threading.Event()
    SENTINEL = object()

    def _producer():
        try:
            for item in iter_video_frames(video, frame_stride=frame_stride):
                if stop.is_set():
                    return
                # Bounded put -- if the consumer is slow, block until
                # there's room. Periodic timeout so we still notice stop.
                while True:
                    try:
                        q.put(item, timeout=0.25)
                        break
                    except queue.Full:
                        if stop.is_set():
                            return
        except Exception as exc:  # surface decode errors on consumer side
            try:
                q.put(("__error__", exc), timeout=1.0)
            except queue.Full:
                pass
        finally:
            try:
                q.put(SENTINEL, timeout=1.0)
            except queue.Full:
                pass

    t = threading.Thread(
        target=_producer, name="iter_video_frames_prefetch", daemon=True,
    )
    t.start()
    try:
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            if isinstance(item, tuple) and item and item[0] == "__error__":
                raise item[1]
            yield item
    finally:
        stop.set()
        # Drain so producer's blocking put() unblocks; daemon thread will
        # exit when the process exits anyway.
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        t.join(timeout=1.0)


def run_deepocsort(
    video: Path,
    *,
    yolo,
    cfg: dict,
    device: str = "cuda:0",
    max_frames: Optional[int] = None,
) -> List[RawTrack]:
    """Run YOLO (multi-scale ensemble) + DeepOcSort on a video.

    Args:
        video: input video file *or* directory of ordered frame images.
        yolo:  YOLO predictor with ``.predict(source=frame_bgr) -> Results[]``
               returning Ultralytics-style boxes. Use
               ``tracking.multi_scale_detector.MultiScaleYOLO`` for the
               shipped configuration (768+1024 ensemble).
        cfg:   config dict; consumes ``cfg['tracker']['reid_weights']``
               and (optional) ``cfg['tracker']['half']``.
        device: torch device string ("cuda:0" / "mps" / "cpu").
        max_frames: optional frame cap for testing.

    Returns:
        list[RawTrack] — same shape ``tracking.postprocess.postprocess_tracks``
        consumes. ReID embeddings are not exported (DeepOcSort holds them
        internally; postprocess only uses bbox + conf + frame index).
    """
    tcfg = cfg.get("tracker", {})
    reid_weights = Path(tcfg.get("reid_weights",
                                 "data/pretrain/reid/osnet_x0_25_msmt17.pt"))
    if not reid_weights.is_file():
        raise FileNotFoundError(
            f"ReID weights missing: {reid_weights}. "
            "BoxMOT auto-downloads osnet_x0_25_msmt17.pt to its cache; "
            "either set tracker.reid_weights to a real path or trust auto-download."
        )

    tracker = make_tracker(
        reid_weights=reid_weights, device=device,
        half=bool(tcfg.get("half", False)),
    )

    rows_per_track: Dict[int, Dict[str, list]] = {}
    n_frames = 0
    for idx, frame in iter_video_frames(Path(video)):
        if max_frames is not None and idx >= max_frames:
            break
        n_frames += 1

        # yolo.predict accepts a numpy frame and returns an
        # Ultralytics-like Results list; fused multi-scale boxes already
        # NMS-unioned when imgsz_ensemble is set.
        res = yolo.predict(source=frame)
        det = res[0]
        if det.boxes is None or len(det.boxes) == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        else:
            xyxy = det.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            conf = det.boxes.conf.detach().cpu().numpy().astype(np.float32)
            cls = det.boxes.cls.detach().cpu().numpy().astype(np.float32)
            dets = np.concatenate(
                [xyxy, conf[:, None], cls[:, None]], axis=1,
            ).astype(np.float32)

        try:
            tracks = tracker.update(dets, frame)
        except Exception as exc:
            log.exception("DeepOcSort update failed at frame %d: %s", idx, exc)
            tracks = np.zeros((0, 7), dtype=np.float32)

        if tracks is None or len(tracks) == 0:
            continue
        tracks = np.asarray(tracks, dtype=np.float32)

        # Post-update layout: [x1, y1, x2, y2, id, conf, cls, det_index].
        for row in tracks:
            x1, y1, x2, y2 = row[0:4]
            tid = int(row[4])
            cf = float(row[5]) if tracks.shape[1] > 5 else 1.0
            d = rows_per_track.setdefault(
                tid, {"frames": [], "bboxes": [], "confs": []},
            )
            d["frames"].append(idx)
            d["bboxes"].append([float(x1), float(y1), float(x2), float(y2)])
            d["confs"].append(cf)

    log.info("DeepOcSort: %d frames -> %d raw tracks", n_frames, len(rows_per_track))

    out: List[RawTrack] = []
    for tid, d in rows_per_track.items():
        out.append(RawTrack(
            track_id=int(tid),
            frames=np.asarray(d["frames"], dtype=np.int64),
            bboxes=np.asarray(d["bboxes"], dtype=np.float32),
            confs=np.asarray(d["confs"], dtype=np.float32),
        ))
    return out
