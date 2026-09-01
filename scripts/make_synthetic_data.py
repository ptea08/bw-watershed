#!/usr/bin/env python3
"""Generate synthetic ZooScan-like data so the pipeline can be run end to end.

This exists so that someone who has just cloned the repository can confirm
their installation works before arranging any real imagery — and so the
pipeline can be smoke-tested on a compute node with no outbound network, where
downloading a public dataset is not an option.

What it produces is **not plankton**. Dark ellipses on a light field, some
deliberately overlapping so that instance separation has something to do. Any
metric computed on it is meaningless; the only question it answers is whether
the code runs.

    python scripts/make_synthetic_data.py --output-dir data/synthetic

Then, with the smoke overlay:

    python scripts/bootstrap.py --config configs/smoke.yaml \
        --data-root data/synthetic --output-dir outputs/smoke/bootstrap

Deterministic given ``--seed``, so a failure is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bootstrapped_watershed.data.masks import (
    BACKGROUND,
    BOUNDARY,
    FOREGROUND,
    encode_class_map,
)

MARGIN = 64  # keep whole organisms inside the crop


def build_labels(rng, size: int, n_organisms: int, touch_probability: float) -> np.ndarray:
    """Draw elongated blobs into an int32 instance-label image.

    Each organism is drawn onto its own uint8 canvas and then stamped into the
    label image, so a later organism overwrites an earlier one exactly the way
    two touching organisms occlude each other in a real scan. Instances stay
    disjoint, which is what the evaluation assumes.

    Roughly ``touch_probability`` of them get a partner placed just far enough
    along their major axis to abut. Those pairs are the point of the whole
    exercise: they are what gives pinch-point severing and the watershed
    something to separate. Without them stage 3 would be untested.
    """
    labels = np.zeros((size, size), dtype=np.int32)
    next_id = 1

    for _ in range(n_organisms):
        cx = int(rng.integers(MARGIN, size - MARGIN))
        cy = int(rng.integers(MARGIN, size - MARGIN))
        major = int(rng.integers(26, 44))
        minor = int(rng.integers(8, 14))
        angle = float(rng.uniform(0, 180))

        next_id = _stamp(labels, next_id, (cx, cy), (major, minor), angle)

        if rng.random() < touch_probability:
            radians = np.deg2rad(angle)
            ox = int(cx + 1.7 * major * np.cos(radians))
            oy = int(cy + 1.7 * major * np.sin(radians))
            if MARGIN <= ox < size - MARGIN and MARGIN <= oy < size - MARGIN:
                next_id = _stamp(labels, next_id, (ox, oy), (major, minor), angle)

    return labels


def _stamp(labels, label_id: int, centre, axes, angle: float) -> int:
    """Draw one filled ellipse into ``labels`` and return the next free id."""
    blob = np.zeros(labels.shape, dtype=np.uint8)
    cv2.ellipse(blob, centre, axes, angle, 0, 360, 255, -1)
    labels[blob > 0] = label_id
    return label_id + 1


def render_image(labels: np.ndarray, rng) -> np.ndarray:
    """Turn a label image into a plausible grayscale-on-BGR scan crop.

    Organisms are darker than the background with per-pixel noise on both, then
    lightly blurred so edges are not perfectly crisp. A backbone fed perfectly
    uniform regions would have almost no texture to describe, which would make
    stage 1 look better than it is.
    """
    image = rng.normal(228.0, 6.0, size=labels.shape)
    organism = labels > 0
    image[organism] = rng.normal(95.0, 12.0, size=int(organism.sum()))
    image = cv2.GaussianBlur(image.astype(np.float32), (3, 3), 0)
    grey = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def render_mask(labels: np.ndarray) -> np.ndarray:
    """Build the three-class RGB annotation mask for a label image.

    Each instance contributes an eroded core as ``foreground`` and the ring the
    erosion removed as ``boundary``. That mirrors how the real annotations are
    drawn — a closed outline around each organism — and it is what lets stage 3
    split a touching pair, because the two rings meet along the join.

    Colours come from ``encode_class_map`` rather than being written here, so
    this cannot drift from the convention the decoder expects.
    """
    class_map = np.full(labels.shape, BACKGROUND, dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    for instance_id in np.unique(labels):
        if instance_id == 0:
            continue
        mask = (labels == instance_id).astype(np.uint8)
        core = cv2.erode(mask, kernel, iterations=1)
        class_map[core > 0] = FOREGROUND
        class_map[(mask > 0) & (core == 0)] = BOUNDARY

    return encode_class_map(class_map)


def write_split(out_dir: Path, stem: str, labels, rng, *, paired: bool, reference: bool):
    """Write one crop. ``paired`` adds an RGB mask, ``reference`` an .npy label map."""
    image = render_image(labels, rng)

    if paired:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "images" / f"{stem}.png"), image)
        cv2.imwrite(str(out_dir / "masks" / f"{stem}.png"), render_mask(labels))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}.png"), image)

    if reference:
        # evaluate.py resolves ground truth as <gt-dir>/<stem>.npy
        np.save(out_dir / f"{stem}.npy", labels)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--n-annotated", type=int, default=3)
    p.add_argument("--n-unlabeled", type=int, default=12)
    p.add_argument("--n-test", type=int, default=2)
    p.add_argument(
        "--organisms",
        type=int,
        default=9,
        help="Organisms per crop before touching partners are added",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.output_dir

    def crop(size):
        return build_labels(rng, size, args.organisms, touch_probability=0.5)

    for i in range(args.n_annotated):
        write_split(
            root / "annotated", f"ann_{i:03d}", crop(args.crop_size), rng,
            paired=True, reference=False,
        )

    for i in range(args.n_unlabeled):
        write_split(
            root / "unlabeled", f"pool_{i:03d}", crop(args.crop_size), rng,
            paired=False, reference=False,
        )

    # Test crops are larger, matching the paper's held-out crops being bigger
    # than the training ones — which also exercises tiling during inference.
    for i in range(args.n_test):
        write_split(
            root / "test", f"test_{i:03d}", crop(args.crop_size * 2), rng,
            paired=False, reference=True,
        )

    print(f"  wrote synthetic data to {root}", flush=True)
    print(f"    annotated/  {args.n_annotated} image+mask pairs", flush=True)
    print(f"    unlabeled/  {args.n_unlabeled} images", flush=True)
    print(f"    test/       {args.n_test} images + .npy instance references", flush=True)
    print("\n  These are ellipses, not organisms. Any metric they produce is", flush=True)
    print("  meaningless — the only question they answer is whether it runs.", flush=True)


if __name__ == "__main__":
    main()
