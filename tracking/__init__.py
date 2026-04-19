"""2D person tracking + ID-assignment pipeline.

The single shipped pipeline is a multi-scale YOLO ensemble (768+1024) +
BoxMOT DeepOcSort + OSNet x0.25 ReID + a 5-stage post-process chain.
See ``docs/PIPELINE_SPEC.md`` for the full reproduction spec and
``docs/EXPERIMENTS_LOG.md`` for the decision history.

Public entry point::

    from pathlib import Path
    from tracking.run_pipeline import run_pipeline_on_video

    tracks = run_pipeline_on_video(
        video=Path("dance.mp4"),
        out=Path("work/dance_tracks.pkl"),
        device="cuda:0",
    )

CLI equivalent::

    python -m tracking.run_pipeline \\
        --video dance.mp4 --out work/dance_tracks.pkl

The output of ``run_pipeline_on_video`` (and the ``tracks.pkl`` it
writes) is a ``dict[int -> tracking.postprocess.Track]`` ready for
downstream visualization or 3D-pose sidecars.

``run_pipeline_on_video`` is intentionally NOT eagerly imported here
because doing so pulls torch/ultralytics into every ``import tracking``
caller; import it directly from ``tracking.run_pipeline`` instead.
"""

__all__: list[str] = []
