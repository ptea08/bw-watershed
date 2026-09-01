#!/usr/bin/env python3
"""Stage 2 — train the U-Net decoder on the bootstrapped pseudo-labels.

``--bootstrap-dir`` is the output directory of ``scripts/bootstrap.py``; it
must contain ``images/`` and ``masks/`` with matching stems.

Example:
    python scripts/train.py --bootstrap-dir outputs/bootstrap --output-dir outputs/segmenter
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bootstrapped_watershed.config import load_config
from bootstrapped_watershed.stage2_segmenter.train import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--bootstrap-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=None, help="Overrides the config")
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["stage2_segmenter"]["optim"]["epochs"] = args.epochs

    best = train(cfg, args.bootstrap_dir, args.output_dir, args.resume)
    print(f"  best checkpoint: {best}", flush=True)


if __name__ == "__main__":
    main()
