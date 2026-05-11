"""GT-driven Stage A benchmark for expected-performer-count tracking.

This is a thin user-facing wrapper around ``eval_id_yolo_cvat.py``.  It
defaults to the usable CVAT/MOT set, includes mirror clips, excludes
``MotionTest`` because its annotations are not valid for acceptance, and uses
the number of unique GT track IDs as simulated user input for Stage A's
``--expected-dancer-count`` prior.

Example:

    python scripts/benchmark_stage_a_expected_count.py \
        --run-pipeline --device cuda:0 --force
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_id_yolo_cvat import main as eval_main  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    defaults = [
        "--include-mirror",
        "--use-expected-count",
        "--exclude-clips",
        "MotionTest",
    ]
    return eval_main([*defaults, *args])


if __name__ == "__main__":
    raise SystemExit(main())
