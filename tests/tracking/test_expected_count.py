from __future__ import annotations

import numpy as np

from tracking.expected_count import build_presence_plan, enforce_expected_count
from tracking.postprocess import Track


def _track(tid: int, frames: list[int], conf: float = 0.9, x: float = 0.0) -> Track:
    boxes = np.asarray([[x, 0.0, x + 10.0, 40.0] for _ in frames], dtype=np.float32)
    return Track(
        track_id=tid,
        frames=np.asarray(frames, dtype=np.int64),
        bboxes=boxes,
        confs=np.asarray([conf] * len(frames), dtype=np.float32),
        detected=np.asarray([True] * len(frames), dtype=bool),
    )


def _track_with_xs(tid: int, frames: list[int], xs: list[float], conf: float = 0.9) -> Track:
    boxes = np.asarray([[x, 0.0, x + 10.0, 40.0] for x in xs], dtype=np.float32)
    return Track(
        track_id=tid,
        frames=np.asarray(frames, dtype=np.int64),
        bboxes=boxes,
        confs=np.asarray([conf] * len(frames), dtype=np.float32),
        detected=np.asarray([True] * len(frames), dtype=bool),
    )


def test_expected_count_suppresses_low_quality_extra_identity():
    tracks = {
        1: _track(1, [0, 1, 2, 3], 0.95, 0),
        2: _track(2, [0, 1, 2, 3], 0.90, 30),
        3: _track(3, [1], 0.40, 200),
    }

    out, report = enforce_expected_count(
        tracks,
        expected_count=2,
        num_frames=4,
        phase="unit",
    )

    assert sorted(out) == [1, 2]
    assert report["suppressed_track_ids"] == [3]
    assert "over_expected_count_suppressed" in report["reason_codes"]


def test_expected_count_recovers_late_candidate_without_forcing_frame_zero():
    primary = {
        1: _track(1, [0, 1, 2, 3], 0.95, 0),
    }
    candidates = {
        **primary,
        2: _track(2, [2, 3], 0.80, 30),
    }

    out, report = enforce_expected_count(
        primary,
        expected_count=2,
        candidate_tracks=candidates,
        num_frames=4,
        phase="unit",
    )

    assert sorted(out) == [1, 2]
    assert report["recovered_track_ids"] == [2]
    plan = build_presence_plan(out, expected_count=2, num_frames=4, selection_report=report)
    active = plan["per_frame_predicted_active_target_ids"]
    assert active[0]["target_ids"] == [1]
    assert active[2]["target_ids"] == [1, 2]
    assert plan["tracks"][1]["late_entry"] is True


def test_expected_count_consolidates_nonoverlapping_split_before_suppression():
    tracks = {
        1: _track(1, [0, 1, 2, 3], 0.95, 0),
        2: _track(2, [0, 1], 0.90, 40),
        3: _track(3, [1, 2, 3], 0.92, 42),
    }

    out, report = enforce_expected_count(
        tracks,
        expected_count=2,
        num_frames=4,
        phase="unit",
    )

    assert sorted(out) == [1, 2]
    assert out[2].frames.tolist() == [0, 1, 2, 3]
    assert report["merged_tracklets"][0]["from_track_ids"] == [2, 3]
    assert "tracklet_consolidation" in report["reason_codes"]


def test_expected_count_does_not_invent_missing_identity():
    primary = {1: _track(1, [0, 1, 2], 0.95, 0)}

    out, report = enforce_expected_count(
        primary,
        expected_count=2,
        candidate_tracks=primary,
        num_frames=3,
        phase="unit",
    )

    assert sorted(out) == [1]
    assert "expected_total_count_unresolved" in report["reason_codes"]


def test_expected_count_repairs_generated_split_handoff_and_keeps_suffix_identity():
    carrier_frames = [0, 1, 2, *range(3, 23), *range(40, 50)]
    carrier_xs = [0.0, 0.0, 0.0, *([0.0] * 20), *([100.0] * 10)]
    tracks = {
        1: _track(1, list(range(50)), 0.95, 200),
        8: _track_with_xs(8, carrier_frames, carrier_xs, 0.90),
        12001: _track(12001, list(range(3, 50)), 0.92, 0),
    }

    out, report = enforce_expected_count(
        tracks,
        expected_count=3,
        num_frames=50,
        phase="unit",
    )

    assert sorted(out) == [1, 2, 3]
    assert "generated_split_handoff_repaired" in report["reason_codes"]
    assert report["split_handoff_repairs"][0]["generated_split_track_id"] == 12001
    assert report["split_handoff_repairs"][0]["suffix_start_frame"] == 40
    assert report["track_id_remap"] == {"1": 1, "8": 2, "12002": 3}
    assert out[2].frames.tolist() == list(range(50))
    assert out[3].frames.tolist() == list(range(40, 50))


def test_expected_count_remaps_sparse_final_ids_to_dense_targets():
    tracks = {
        5: _track(5, [0, 1, 2], 0.95, 0),
        12001: _track(12001, [0, 1, 2], 0.90, 40),
    }

    out, report = enforce_expected_count(
        tracks,
        expected_count=2,
        num_frames=3,
        phase="unit",
    )

    assert sorted(out) == [1, 2]
    assert out[1].track_id == 1
    assert out[2].track_id == 2
    assert report["track_id_remap"] == {"5": 1, "12001": 2}
    assert "dense_id_remap" in report["reason_codes"]
