#!/usr/bin/env python3
"""Compute PQ / SQ / RQ, precision, recall and OSR on the held-out crops.

``--pred-dir`` holds ``<stem>_instances.npy`` files from ``scripts/infer.py``.
``--gt-dir`` holds ground truth for the same stems, as either LabelMe
``<stem>.json`` annotations or ``<stem>.npy`` label images.

Example:
    python scripts/evaluate.py --pred-dir outputs/instances --gt-dir data/test --markdown
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from bootstrapped_watershed.config import load_config
from bootstrapped_watershed.eval.panoptic import (
    METRIC_NAMES,
    format_markdown,
    load_ground_truth,
    macro_average,
    panoptic_quality,
)


def find_ground_truth(gt_dir: Path, stem: str) -> Path | None:
    for suffix in (".json", ".npy"):
        candidate = gt_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--iou-threshold", type=float, default=None)
    p.add_argument("--name", default="ours", help="Row label for the Markdown table")
    p.add_argument("--markdown", action="store_true", help="Print a README-ready table")
    p.add_argument("--csv", type=Path, default=None, help="Write per-crop metrics here")
    args = p.parse_args()

    cfg = load_config(args.config)
    threshold = args.iou_threshold if args.iou_threshold is not None else cfg.eval.iou_threshold

    per_crop, rows = [], []
    for pred_path in sorted(args.pred_dir.glob("*_instances.npy")):
        stem = pred_path.name[: -len("_instances.npy")]
        gt_path = find_ground_truth(args.gt_dir, stem)
        if gt_path is None:
            print(f"  skipping {stem}: no ground truth in {args.gt_dir}", flush=True)
            continue

        pred = np.load(pred_path).astype(np.int32)
        gt_masks = load_ground_truth(gt_path, pred.shape)
        if not gt_masks:
            continue

        metrics = panoptic_quality(pred, gt_masks, threshold)
        per_crop.append(metrics)
        rows.append({"crop_id": stem, **metrics})

    if not per_crop:
        raise SystemExit("No crops scored — check --pred-dir and --gt-dir.")

    summary = macro_average(per_crop)

    print(
        f"\n  {len(per_crop)} crops, matched one-to-one at "
        f"IoU >= {threshold}, macro-averaged\n",
        flush=True,
    )
    for metric in METRIC_NAMES:
        print(f"    {metric.upper():<10} {summary[metric]:.3f}", flush=True)

    if args.markdown:
        print("\n" + format_markdown(summary, args.name), flush=True)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  per-crop metrics -> {args.csv}", flush=True)


if __name__ == "__main__":
    main()
