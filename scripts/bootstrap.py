#!/usr/bin/env python3
"""Stage 1 — train the pixel classifier and select confident pseudo-labels.

Reads ``<data-root>/annotated/{images,masks}`` and ``<data-root>/unlabeled/``,
writes the selected pseudo-labels to ``<output-dir>/{images,masks}`` in the
layout ``scripts/train.py`` expects, plus a ``ranking.csv`` recording every
candidate's confidence so the selection can be audited.

Example:
    python scripts/bootstrap.py --data-root data/ --output-dir outputs/bootstrap
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from bootstrapped_watershed.config import load_config, resolve_device, set_seed
from bootstrapped_watershed.data.dataset import IMAGE_SUFFIXES, find_pairs
from bootstrapped_watershed.stage1_bootstrap.classifier import (
    build_classifier,
    train_classifier,
)
from bootstrapped_watershed.stage1_bootstrap.features import (
    build_token_dataset,
    load_backbone,
)
from bootstrapped_watershed.stage1_bootstrap.selection import (
    predict_crop,
    select_bootstrap_crops,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="YAML config path")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--classifier",
        choices=["mlp_shallow", "mlp_deep", "random_forest"],
        default=None,
        help="Overrides the config. Use for the MLP-D and RF ablations.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = resolve_device(cfg)

    image_out = args.output_dir / "images"
    mask_out = args.output_dir / "masks"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    backbone = load_backbone(cfg, device)

    annotated = find_pairs(args.data_root / "annotated")
    if not annotated:
        raise SystemExit(
            f"no image/mask pairs in {args.data_root / 'annotated'}. See "
            f"data/README.md for the expected layout."
        )

    print(f"  extracting tokens from {len(annotated)} annotated crops ...", flush=True)
    tokens, labels = build_token_dataset(backbone, annotated, cfg, device)
    print(f"  {len(tokens)} labelled tokens", flush=True)

    classifier = build_classifier(cfg, args.classifier)
    kind = args.classifier or cfg.stage1_bootstrap.classifier.type
    print(f"  training the {kind} classifier ...", flush=True)
    classifier = train_classifier(classifier, tokens, labels, cfg, device)

    pool = sorted(
        p for p in (args.data_root / "unlabeled").iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    print(f"  scoring a pool of {len(pool)} candidate crops ...", flush=True)

    scores, predictions = {}, {}
    for index, path in enumerate(pool, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"    [{index}/{len(pool)}] {path.name}: unreadable, skipped", flush=True)
            continue
        mask, confidence = predict_crop(classifier, backbone, image, cfg, device)
        scores[path.stem] = confidence
        predictions[path.stem] = (path, mask)
        print(f"    [{index}/{len(pool)}] {path.name}  confidence {confidence:.4f}", flush=True)

    threshold = cfg.stage1_bootstrap.selection.get("confidence_threshold")
    if threshold is None:
        print(
            "  warning: no selection.confidence_threshold set, so every candidate\n"
            "  was kept. This is not a selection — stage 2 would train on the\n"
            "  classifier's mistakes as readily as its successes. Read ranking.csv,\n"
            "  find where confidence falls off, and set a threshold. See\n"
            "  docs/TUNING.md.",
            flush=True,
        )
    selected = select_bootstrap_crops(scores, threshold)

    for stem in selected:
        source, mask = predictions[stem]
        cv2.imwrite(str(image_out / f"{stem}.png"), cv2.imread(str(source)))
        cv2.imwrite(str(mask_out / f"{stem}.png"), mask)

    with open(args.output_dir / "ranking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "crop_id", "confidence", "selected"])
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        chosen = set(selected)
        for rank, (stem, score) in enumerate(ranked, 1):
            writer.writerow([rank, stem, f"{score:.6f}", stem in chosen])

    print(f"  kept {len(selected)}/{len(scores)} crops -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
