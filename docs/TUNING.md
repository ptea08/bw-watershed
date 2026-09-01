# Tuning for your own data

This pipeline was developed on ZooScan scans of Baltic Sea plankton. That
dataset is not public and cannot be redistributed, so this repository is not a
replication package — it is the method, packaged so you can run it on **your**
imagery.

Doing that well means changing some numbers and leaving others alone. This page
says which is which.

The short version:

| Group | What to do |
|---|---|
| [Method constants](#1-method-constants--leave-these-alone) | Leave alone. They define the published method. |
| [Scale constants](#2-scale-constants--check-these-against-your-resolution) | Check these. They are in pixels, so they depend on your imaging resolution. |
| [Balance constants](#3-balance-constants--start-here-then-adjust) | Start at the defaults, adjust if training is unhealthy. |
| [Dataset constants](#4-dataset-constants--nothing-to-set) | Nothing to set — derived from your directories. |

---

## How to override anything

Never edit `configs/default.yaml`. Write a small overlay file containing only
what you are changing, and pass it with `--config`. It is merged on top of the
defaults, key by key, at any depth:

```yaml
# my_data.yaml
data:
  class_weights:
    boundary: 4.0

stage3_instances:
  extraction:
    min_instance_area: 600
    pinch_severing:
      dt_max: 24
```

```bash
python scripts/bootstrap.py --config my_data.yaml --data-root data --output-dir outputs/bootstrap
```

Everything you do not mention keeps its published value. Keeping your overlay
small is worth the discipline: it is a precise record of how your setup differs
from the paper, and it will not go stale when this repository updates its
defaults.

> **One YAML trap.** A key with nothing but comments under it parses as `null`,
> and `null` legitimately means "set this to nothing" — so it will erase that
> entire section rather than leave it alone:
>
> ```yaml
> stage3_instances:
>   extraction:
>     # leaving these at the defaults for now
> ```
>
> That wipes every extraction setting. If you are not changing a section,
> delete the key rather than emptying it.

`configs/smoke.yaml` is a worked example: a small overlay that shortens both
training stages so you can confirm the pipeline runs end to end before
committing to a real run.

---

## 1. Method constants — leave these alone

These *are* the method. Change one and you are no longer running Bootstrapped
Watershed; you are running something related that you should evaluate yourself.
They are marked `[paper]` in `configs/default.yaml` and pinned by
`tests/test_config.py`, so if you change one the test suite will tell you.

| Setting | Value |
|---|---|
| `stage2_segmenter.loss.tversky.alpha` / `.beta` | 0.7 / 0.3 |
| `stage2_segmenter.loss.cross_entropy.ce_weight` | 0.4 |
| `stage1_bootstrap.classifier.hidden_dim` | 128 (the 384 → 128 → 3 MLP) |
| `stage1_bootstrap.backbone.*` | DINOv3 ConvNeXt-S, frozen, 384-d @ stride 16 |
| `stage2_segmenter.optim.*` | 80 epochs, AdamW, cosine annealing with warm restarts |
| `eval.iou_threshold`, `eval.matching`, `eval.aggregate` | 0.5, Hungarian, macro |

`stage1_bootstrap.selection.class_priors` also belongs here. The multipliers
(foreground 7.0, boundary 1.5) bias the softmax before the argmax so that thin
structure is not erased by background dominance. They are a property of the
model's behaviour rather than of any particular dataset, and they are carried
unchanged from the random-forest bootstrapper so the MLP-S / MLP-D / RF
comparisons stay meaningful. Leave them.

---

## 2. Scale constants — check these against your resolution

Every value below is in **pixels**, so all of them depend on your imaging
resolution rather than on anything intrinsic to the method.

A useful thing to know about their provenance: organism size in our scans varies
a great deal — these are plankton samples containing everything from eggs to
late-stage larvae — and these settings were chosen to work across that whole
range rather than tuned to one representative organism. That makes them more
robust than a precisely calibrated set would be, and it means there is no
meaningful "typical organism diameter" to scale them by. Treat them as
reasonable defaults for imagery at roughly ZooScan resolution, and revisit them
if yours is substantially different.

| Setting | Default | What it controls |
|---|---|---|
| `extraction.min_instance_area` | 150 px² | Rejection floor — see below |
| `extraction.pinch_severing.dt_max` | 12 | Widest neck considered for severing |
| `extraction.pinch_severing.radius` | 8 | Search radius for boundary evidence |
| `extraction.pinch_severing.sever_dilation` | 3 | Width of the applied cut |
| `extraction.watershed.marker_threshold` | 5 | Minimum distance-transform peak to seed an instance |
| `extraction.watershed.smoothing_kernel` | 9 | Gaussian on the distance transform |
| `extraction.watershed.peak_kernel` | 9 | Neighbourhood a local maximum must dominate |
| `extraction.watershed.background_dilation` | 5 | Grows the mask to mark the unknown band |
| `extraction.closing_kernel` | 5 | Morphological close on the severed mask |
| `extraction.boundary_dilation` | [3, 5] | Open then dilate, forming the boundary wall |
| `extraction.median_blur` | 5 | Despeckles the argmax map |

If you do change them, keep kernel sizes odd — OpenCV requires it — and keep
every value at 1 or above. Check the result by eye on a few crops before
trusting it on a full scan.

### `min_instance_area` is a debris floor

This one is worth singling out because its name invites the wrong reading. It is
**not** a statement about how large organisms are. It is a rejection floor for
specks: connected components below 150 px² are discarded as debris or
segmentation noise, and everything above it is kept regardless of size.

So set it below the smallest thing you care about, not relative to a typical
one. The failure mode is silent — set it too high and your smallest organisms
disappear with no error, showing up only as unexplained low recall. If your
imagery is higher resolution than ours, or you are interested in genuinely small
targets, lower it and look at what comes back.

### Tiling

`stage3_instances.tiling.tile_size` (512) and `.overlap` (128) are about
compute, not organisms, with one exception: **an organism larger than the
overlap can be cut across a seam and counted twice.** If your organisms are
large relative to 128 px, raise the overlap. `tile_size` must stay a multiple of
`stage2_segmenter.decoder.stride` (32).

---

## 3. Balance constants — start here, then adjust

These are honest defaults rather than derived truths. They were arrived at by
trying several values on ZooScan data, so they encode our class balance. Yours
will differ.

### Cross-entropy class weights

```yaml
data:
  class_weights:
    background: 1.0
    foreground: 1.5
    boundary: 8.0
```

Boundary is weighted heavily because boundary pixels are a small fraction of the
image but carry nearly all of the instance-separation signal — a segmenter that
ignores them cannot split touching organisms. The weights are keyed by **name**,
not position, so they stay correct if you reorder `data.class_names`.

Adjust when: your boundary class is a very different fraction of labelled pixels
than ours. The symptom of too low a boundary weight is a model with a healthy
mIoU that nonetheless merges touching organisms into blobs — which is why
training checkpoints on **boundary IoU** rather than mIoU. Watch that number; if
it sits near zero while mIoU looks fine, raise the boundary weight.

Too *high* a weight is also possible: the model then predicts boundary
everywhere, fragmenting single organisms and driving the over-segmentation ratio
up.

### Mask decoding threshold

```yaml
data:
  mask_encoding:
    min_channel: 50
```

A pixel's dominant RGB channel must exceed this for the pixel to be assigned a
class at all; below it, the pixel is ignored. This depends on how your
annotation tool writes masks. If it emits clean, saturated colours the default
is fine. If your masks are anti-aliased, compressed, or drawn at low opacity,
raise it — or check first with `data/README.md`, which specifies the expected
format.

### Class colours

```yaml
data:
  mask_encoding:
    background: red
    foreground: green
    boundary: blue
```

Remap these if your annotation tool uses a different convention. You do not need
to recolour your masks.

---

## 4. Dataset constants — nothing to set

Earlier versions of this config carried `n_annotated: 3`, `n_bootstrap: 82`,
`n_test_crops: 15` and a hardcoded 66/16 train/val split. Those were counts from
our data, and they have been removed: the code now reads however many files you
actually have. Point the scripts at your directories and the numbers follow.

The validation fraction is still configurable, as a fraction rather than a
count:

```yaml
stage2_segmenter:
  optim:
    val_split: 0.2
```

### Choosing the bootstrap confidence threshold

This one needs your judgement, and it is worth understanding rather than
guessing.

Stage 1 trains a pixel classifier on your handful of annotated crops, predicts
on a larger unlabeled pool, and keeps the crops it is most confident about as
pseudo-labels for stage 2. Selection is by **confidence threshold**: every crop
scoring above the cut is kept.

The paper reports *N*<sub>boot</sub> = 82 crops. That is a **result, not a
setting** — 82 is simply how many of our candidates cleared the threshold. Do
not try to select 82 crops from your own pool; the right number for you depends
on your pool size and how well the stage-1 classifier generalises to it.

To choose yours, run stage 1 and read `ranking.csv` from the output directory.
It records every candidate's confidence, ranked, with a `selected` column:

```bash
python scripts/bootstrap.py --data-root data --output-dir outputs/bootstrap
```

Look for where confidence falls off. A good cut sits above that knee. Two
failure modes to steer between:

- **Threshold too high** — you keep only a handful of crops and stage 2 has too
  little supervision to improve on training directly from your annotated crops.
- **Threshold too low** — you admit crops the classifier got wrong, and stage 2
  learns those errors. This is the more dangerous direction, because the
  pseudo-labels are the *only* supervision stage 2 ever sees, so a systematic
  stage-1 failure propagates silently through everything downstream.

Open a few of the selected masks before continuing. If they look wrong, no
amount of stage-2 training will fix it.

---

## Before you start

The DINOv3 checkpoints are gated on the Hugging Face Hub. Request access to
[`facebook/dinov3-convnext-small-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m)
and export a read token as `HF_TOKEN`. Without it stages 1 and 2 fail
immediately, and on a machine without outbound network access the resulting 401
is easily mistaken for a firewall problem.

See [`RUN.md`](../RUN.md) for the full run order and [`data/README.md`](../data/README.md)
for the expected directory layout and mask format.
