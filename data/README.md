# Data

**No imagery ships with this repository.** The ZooScan corpus (20 scans of Baltic
Sea plankton samples, roughly 15000 x 25000 px each) was produced by the
**Thünen Institute of Baltic Sea Fisheries** and is not redistributable. This
directory is git-ignored apart from this file; place your own data here, or
point the scripts elsewhere with `--data-root`.

## Expected layout

```
data/
├── annotated/     # your manually annotated crops (we used 3, at 512x512)
│   ├── images/        crop_007.png
│   └── masks/         crop_007.png     <- same stem, RGB mask
├── unlabeled/     # flat directory of images; the pool stage 1 ranks
├── test/          # held-out crops + instance references (ours were 1024x1024)
└── scans/         # full scans for stage 3 (ours were ~15000x25000 px)
```

`annotated/` is split into `images/` and `masks/` subdirectories, and an image is
paired with a mask only when the **stems match exactly** — `crop_007.png` with
`crop_007.png`, not `crop_007_mask.png`. A mask whose stem does not match is
silently ignored rather than reported, so if stage 1 says it found fewer
annotated crops than you have, this is the first thing to check.

`unlabeled/` is flat: images only, no masks, since producing those masks is the
whole point of stage 1. Stage 1 then writes its selected pseudo-labels in the
same `images/` + `masks/` layout, which is why `scripts/train.py` can be pointed
straight at a bootstrap output directory.

## Mask format

Annotation masks are RGB, one class per dominant channel:

| Colour | Class | Index |
|---|---|---|
| Red | Background | 0 |
| Green | Organism interior | 1 |
| Blue | Organism boundary | 2 |

A pixel is labelled only when one channel is **strictly greater** than both
others *and* exceeds `data.mask_encoding.min_channel` (default 50). Everything
else — ties, and dark or ambiguous pixels — takes `ignore_index` (255) and
contributes nothing to any loss.

The index column is load-bearing: it is the argmax order used throughout the
code, and it is pinned by `tests/test_config.py`.

## Ground truth for evaluation

`scripts/evaluate.py` reads per-crop instance references as either:

- **LabelMe JSON** — every shape is rasterised into one instance. Class labels
  are ignored, since evaluation is class-agnostic.
- **`.npy` label image** — int array, 0 = background, one positive id per
  instance.
