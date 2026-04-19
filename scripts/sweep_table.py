"""Pretty-print sweep results to a delta table.

Reads ``<sweep_dir>/sweep_results.json`` (the aggregate file produced by
:mod:`scripts.sweep_detector`) and emits a markdown-style table that:
  - lists each variant's mean IDF1 + delta vs the variant named ``baseline``
  - per-clip IDF1 + delta vs baseline
  - per-clip miss/fp/ids deltas on focal clips (loveTest, shorterTest,
    MotionTest, darkTest)

Useful as a one-shot "did this sweep find any wins?" summary.

Usage::

    python scripts/sweep_table.py work/sweeps/<sweep_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


CLIP_ORDER = (
    "BigTest", "mirrorTest", "adiTest", "easyTest", "gymTest",
    "loveTest", "shorterTest", "MotionTest", "darkTest",
)
FOCAL_CLIPS = ("loveTest", "shorterTest", "MotionTest", "darkTest")


def _load(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "variants" in raw:
        return raw["variants"]
    return raw


def _baseline(variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    for v in variants:
        if v.get("variant") == "baseline":
            return v
    raise SystemExit("no 'baseline' variant in sweep_results.json")


def _per_clip(v: Dict[str, Any], clip: str, key: str, default=0) -> Any:
    return v.get("per_clip", {}).get(clip, {}).get(key, default)


def _print_table(variants: List[Dict[str, Any]]) -> None:
    base = _baseline(variants)
    base_mean = base.get("mean_idf1", 0.0) or 0.0

    print("\n## Mean IDF1 across 9 clips")
    print()
    print("| variant | mean IDF1 | d vs baseline | total IDS | total FN | total FP |")
    print("|---|---:|---:|---:|---:|---:|")
    for v in variants:
        mean = v.get("mean_idf1", 0.0) or 0.0
        d = mean - base_mean
        ids = int(v.get("num_switches_total") or 0)
        fn = int(v.get("num_misses_total") or 0)
        fp = int(v.get("num_false_positives_total") or 0)
        marker = " "
        if v.get("variant") == "baseline":
            marker = "_"
        print(f"| {marker}{v['variant']}{marker} | {mean:.4f} | {d:+.4f} "
              f"| {ids} | {fn} | {fp} |")

    print("\n## Per-clip IDF1 (delta vs baseline in parens)")
    print()
    print("| clip | " + " | ".join(v["variant"] for v in variants) + " |")
    print("|---|" + "---:|" * len(variants))
    for clip in CLIP_ORDER:
        cells = [clip]
        for v in variants:
            x = _per_clip(v, clip, "idf1")
            bx = _per_clip(base, clip, "idf1")
            if isinstance(x, (int, float)) and isinstance(bx, (int, float)):
                d = x - bx
                cells.append(f"{x:.4f} ({d:+.4f})")
            else:
                cells.append(str(x))
        print("| " + " | ".join(cells) + " |")

    print("\n## Focal clips: miss / fp / ids vs baseline")
    for clip in FOCAL_CLIPS:
        print(f"\n### {clip}")
        print()
        print(f"| variant | idf1 | d_idf1 | miss | d_miss | fp | d_fp | ids |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        b = base.get("per_clip", {}).get(clip, {})
        for v in variants:
            x = v.get("per_clip", {}).get(clip, {})
            if "idf1" not in x:
                continue
            print(
                f"| {v['variant']} | {x.get('idf1', 0):.4f} "
                f"| {x.get('idf1', 0) - b.get('idf1', 0):+.4f} "
                f"| {int(x.get('num_misses', 0))} "
                f"| {int(x.get('num_misses', 0)) - int(b.get('num_misses', 0)):+d} "
                f"| {int(x.get('num_false_positives', 0))} "
                f"| {int(x.get('num_false_positives', 0)) - int(b.get('num_false_positives', 0)):+d} "
                f"| {int(x.get('num_switches', 0))} |"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", type=Path,
                    help="dir containing sweep_results.json")
    args = ap.parse_args()

    path = args.sweep_dir / "sweep_results.json"
    if not path.is_file():
        path = args.sweep_dir
    if not path.is_file():
        raise SystemExit(f"missing sweep_results.json under {args.sweep_dir}")

    variants = _load(path)
    _print_table(variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
