from __future__ import annotations

import numpy as np

from prune_tracks import FrameDetections
from tracking.postprocess import Track
import tracking.best_pipeline as bp
import tracking.run_pipeline as rp


def test_pipeline_parallel_default_off(monkeypatch):
    monkeypatch.delenv("BEST_ID_PIPELINE_PARALLEL", raising=False)
    assert rp._resolve_pipeline_parallel() is False


def test_pipeline_parallel_opt_out(monkeypatch):
    monkeypatch.setenv("BEST_ID_PIPELINE_PARALLEL", "0")
    assert rp._resolve_pipeline_parallel() is False


def test_pipeline_parallel_explicit_on(monkeypatch):
    monkeypatch.setenv("BEST_ID_PIPELINE_PARALLEL", "yes")
    assert rp._resolve_pipeline_parallel() is True


def test_tracker_input_sanitizer_clips_and_drops_invalid_boxes():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    dets = np.asarray([
        [-10, -5, 40, 50, 0.9, 0],
        [30, 30, 30.5, 60, 0.8, 0],
        [150, 80, 250, 150, 0.7, 0],
    ], dtype=np.float32)

    cleaned, dropped = rp._sanitize_detections_for_tracker(dets, frame)

    assert dropped == 1
    assert cleaned.shape == (2, 6)
    assert cleaned[0, 0] == 0
    assert cleaned[0, 1] == 0
    assert cleaned[1, 2] == 199
    assert cleaned[1, 3] == 99


def test_group_box_suppression_default_off(monkeypatch):
    monkeypatch.delenv("BEST_ID_GROUP_SUPPRESS", raising=False)
    dets = np.asarray([
        [0, 0, 100, 100, 0.9, 0],
        [0, 0, 30, 100, 0.8, 0],
        [70, 0, 100, 100, 0.8, 0],
    ], dtype=np.float32)

    out = rp._suppress_group_boxes_for_tracker(dets)

    assert out.shape == dets.shape


def test_group_box_suppression_drops_container(monkeypatch):
    monkeypatch.setenv("BEST_ID_GROUP_SUPPRESS", "1")
    dets = np.asarray([
        [0, 0, 100, 100, 0.9, 0],
        [0, 0, 30, 100, 0.8, 0],
        [70, 0, 100, 100, 0.8, 0],
    ], dtype=np.float32)

    out = rp._suppress_group_boxes_for_tracker(dets)

    assert out.shape == (2, 6)
    assert np.allclose(out[:, :4], dets[1:, :4])


def test_detector_edge_backfill_prepends_unoccupied_detector():
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([2, 3], dtype=np.int64),
            bboxes=np.asarray([[20, 10, 40, 50], [30, 10, 50, 50]], dtype=np.float32),
            confs=np.asarray([0.9, 0.9], dtype=np.float32),
            detected=np.asarray([True, True]),
        )
    }
    detector_frames = [
        FrameDetections(
            np.asarray([[0, 10, 20, 50]], dtype=np.float32),
            np.asarray([0.8], dtype=np.float32),
            np.asarray([0], dtype=np.float32),
        ),
        FrameDetections(
            np.asarray([[10, 10, 30, 50]], dtype=np.float32),
            np.asarray([0.8], dtype=np.float32),
            np.asarray([0], dtype=np.float32),
        ),
        FrameDetections(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ),
        FrameDetections(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ),
    ]

    out, added = bp._detector_edge_backfill(
        tracks,
        detector_frames,
        max_frames=2,
        min_run=1,
        min_conf=0.34,
        max_center_dist=15.0,
        max_size_ratio=1.2,
        occupied_iou=0.35,
    )

    assert added == 2
    assert out[1].frames.tolist() == [0, 1, 2, 3]
    assert np.allclose(out[1].bboxes[:2], [[0, 10, 20, 50], [10, 10, 30, 50]])


def test_edge_extrapolate_fills_under_count_short_edge():
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([2, 3], dtype=np.int64),
            bboxes=np.asarray([[20, 10, 40, 50], [30, 10, 50, 50]], dtype=np.float32),
            confs=np.asarray([0.8, 0.9], dtype=np.float32),
            detected=np.asarray([True, True]),
        ),
        2: Track(
            track_id=2,
            frames=np.asarray([0, 1, 2, 3], dtype=np.int64),
            bboxes=np.asarray([[80, 10, 100, 50]] * 4, dtype=np.float32),
            confs=np.asarray([0.9] * 4, dtype=np.float32),
            detected=np.asarray([True] * 4),
        ),
    }

    out, added = bp._edge_extrapolate_to_modal_count(
        tracks,
        num_frames=4,
        target_count=2,
        max_frames=2,
    )

    assert added == 2
    assert out[1].frames.tolist() == [0, 1, 2, 3]
    assert np.allclose(out[1].bboxes[0], [0, 10, 20, 50])
    assert np.allclose(out[1].bboxes[1], [10, 10, 30, 50])


def test_frame_count_cap_drops_backfilled_first():
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([0], dtype=np.int64),
            bboxes=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
            confs=np.asarray([0.1], dtype=np.float32),
            detected=np.asarray([True]),
        ),
        2: Track(
            track_id=2,
            frames=np.asarray([0], dtype=np.int64),
            bboxes=np.asarray([[20, 0, 30, 10]], dtype=np.float32),
            confs=np.asarray([0.9], dtype=np.float32),
            detected=np.asarray([False]),
        ),
    }

    out, dropped = bp._cap_frames_to_modal_count(tracks, num_frames=1, target_count=1)

    assert dropped == 1
    assert out[1].frames.tolist() == [0]
    assert out[2].frames.tolist() == []


def test_stage_prior_trims_suspicious_entry_edges():
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([0, 1, 2], dtype=np.int64),
            bboxes=np.asarray(
                [[0, 10, 45, 60], [0, 10, 50, 60], [8, 10, 65, 60]],
                dtype=np.float32,
            ),
            confs=np.asarray([0.8, 0.8, 0.8], dtype=np.float32),
            detected=np.asarray([True, True, True]),
        ),
        2: Track(
            track_id=2,
            frames=np.asarray([0, 1, 2], dtype=np.int64),
            bboxes=np.asarray([[100, 10, 140, 60]] * 3, dtype=np.float32),
            confs=np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
            detected=np.asarray([True, True, True]),
        ),
    }

    out, dropped = bp._trim_stage_edge_samples(
        tracks,
        frame_w=150,
        edge_margin=8,
        entry_visible_width=50,
        synthetic_conf_thresh=0.55,
        max_trim=3,
    )

    assert dropped == 2
    assert out[1].frames.tolist() == [2]
    assert out[2].frames.tolist() == [0, 1, 2]


def test_stage_prior_extends_inside_synthetic_entry_run():
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([5, 6, 7, 8], dtype=np.int64),
            bboxes=np.asarray(
                [[60, 10, 90, 70], [60, 10, 90, 70], [65, 10, 95, 70], [70, 10, 100, 70]],
                dtype=np.float32,
            ),
            confs=np.asarray([0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            detected=np.asarray([False, False, False, True]),
        )
    }

    out, added = bp._stage_prior_extrapolate_edges(
        tracks,
        num_frames=9,
        frame_w=200,
        edge_margin=8,
        max_frames=2,
        synthetic_run_min=3,
    )

    assert added == 2
    assert out[1].frames.tolist()[:5] == [3, 4, 5, 6, 7]


def test_stage_prior_does_not_extend_edge_entry_run():
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([5, 6, 7, 8], dtype=np.int64),
            bboxes=np.asarray(
                [[0, 10, 40, 70], [0, 10, 45, 70], [2, 10, 50, 70], [10, 10, 60, 70]],
                dtype=np.float32,
            ),
            confs=np.asarray([0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            detected=np.asarray([False, False, False, True]),
        )
    }

    out, added = bp._stage_prior_extrapolate_edges(
        tracks,
        num_frames=10,
        frame_w=200,
        edge_margin=8,
        max_frames=2,
        synthetic_run_min=3,
    )

    assert added == 0
    assert out[1].frames.tolist() == [5, 6, 7, 8]


def test_box_calibration_scales_and_shifts_center():
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([0], dtype=np.int64),
            bboxes=np.asarray([[10, 20, 30, 60]], dtype=np.float32),
            confs=np.asarray([0.9], dtype=np.float32),
            detected=np.asarray([True]),
        )
    }

    out = bp.calibrate_track_boxes(
        tracks,
        scale_x=0.5,
        scale_y=0.75,
        shift_x=-0.25,
        shift_y=0.0,
    )

    assert np.allclose(out[1].bboxes[0], [10, 25, 20, 55])


def test_box_calibration_default_valid_annotation_preset(monkeypatch):
    monkeypatch.delenv("BEST_ID_BOX_CALIBRATE", raising=False)
    monkeypatch.delenv("BEST_ID_BOX_SCALE_X", raising=False)
    monkeypatch.delenv("BEST_ID_BOX_SCALE_Y", raising=False)
    monkeypatch.delenv("BEST_ID_BOX_SHIFT_X", raising=False)
    monkeypatch.delenv("BEST_ID_BOX_SHIFT_Y", raising=False)
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([0], dtype=np.int64),
            bboxes=np.asarray([[10, 20, 30, 60]], dtype=np.float32),
            confs=np.asarray([0.9], dtype=np.float32),
            detected=np.asarray([True]),
        )
    }

    out = bp._maybe_calibrate_track_boxes(tracks)

    assert np.allclose(out[1].bboxes[0], [9.0, 20.0, 27.0, 60.0])


def test_box_calibration_uses_dense_count_preset_without_env_override(monkeypatch):
    monkeypatch.delenv("BEST_ID_BOX_CALIBRATE", raising=False)
    monkeypatch.delenv("BEST_ID_BOX_SCALE_X", raising=False)
    monkeypatch.delenv("BEST_ID_BOX_SCALE_Y", raising=False)
    monkeypatch.delenv("BEST_ID_BOX_SHIFT_X", raising=False)
    monkeypatch.delenv("BEST_ID_BOX_SHIFT_Y", raising=False)
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([0], dtype=np.int64),
            bboxes=np.asarray([[10, 20, 30, 60]], dtype=np.float32),
            confs=np.asarray([0.9], dtype=np.float32),
            detected=np.asarray([True]),
        )
    }

    out = bp._maybe_calibrate_track_boxes(tracks, expected_count=12)

    assert np.allclose(out[1].bboxes[0], [11.5, 21.6, 28.5, 58.4])


def test_box_calibration_can_be_disabled(monkeypatch):
    monkeypatch.setenv("BEST_ID_BOX_CALIBRATE", "0")
    tracks = {
        1: Track(
            track_id=1,
            frames=np.asarray([0], dtype=np.int64),
            bboxes=np.asarray([[10, 20, 30, 60]], dtype=np.float32),
            confs=np.asarray([0.9], dtype=np.float32),
            detected=np.asarray([True]),
        )
    }

    out = bp._maybe_calibrate_track_boxes(tracks)

    assert out is tracks
