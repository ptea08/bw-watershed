#!/usr/bin/env python3
"""Stage 3 — tiled full-scan inference and instance extraction.

Accepts either a single image or a directory of them. Each input produces a
``<stem>_mask.png`` semantic map and a ``<stem>_instances.npy`` label image,
which is the format ``scripts/evaluate.py`` reads.

Example:
    python scripts/infer.py --checkpoint outputs/segmenter/best.pt \\
        --scan data/scans/scan_01.tif --output-dir outputs/instances
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bootstrapped_watershed.config import load_config, resolve_device
from bootstrapped_watershed.data.dataset import IMAGE_SUFFIXES
from bootstrapped_watershed.stage3_instances.inference import load_model, run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scan", type=Path, required=True, help="Full scan, crop, or directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--save-maps",
        action="store_true",
        help="Also write the continuous foreground/boundary maps (large).",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    model = load_model(cfg, args.checkpoint, device)

    if args.scan.is_dir():
        scans = sorted(p for p in args.scan.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        scans = [args.scan]

    for index, scan in enumerate(scans, 1):
        instances = run(model, scan, args.output_dir, cfg, device, args.save_maps)
        print(f"  [{index}/{len(scans)}] {scan.name}: {int(instances.max())} instances", flush=True)


if __name__ == "__main__":
    main()
