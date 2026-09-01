# Running the pipeline

Steps run in order; each one consumes the previous one's output directory.
Check the "verify" note at the end of a step before starting the next — a
mistake at stage 1 is much cheaper to catch there than after 80 epochs of
stage 2.

Paths below are examples. Nothing is hardcoded: every path is a command-line
argument and every constant lives in `configs/default.yaml`.

---

## 0. Environment

```bash
cd bw-watershed
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

The DINOv3 weights are gated. Request access to
[`facebook/dinov3-convnext-small-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m),
then export a token with read access:

```bash
export HF_TOKEN=hf_...
```

Stages 1–3 all fail immediately without it. `requirements.txt` pins
`transformers>=4.56`, which is the first release with DINOv3 support.

**Verify:** `python -c "import torch; print(torch.cuda.is_available())"` prints
`True`. If it prints `False` the pipeline still runs, just far slower —
`device: cuda` in the config falls back to CPU automatically.

---

## 1. Tests

CPU-only, no weights, no data, no token. Run these before anything expensive.

```bash
pytest -q
```

**Verify:** all pass. `tests/test_config.py` pins the published constants, so a
failure there means `configs/default.yaml` has drifted from the paper.

---

## 2. Smoke test — optional, but do it before arranging real data

The tests above are unit tests. They do not run the stages, so they cannot tell
you whether the four scripts hand off to each other correctly, whether your
`HF_TOKEN` works, or whether CUDA is visible from a compute node. This step
does, in a few minutes, using generated data.

```bash
python scripts/make_synthetic_data.py --output-dir data/synthetic

python scripts/bootstrap.py --config configs/smoke.yaml \
    --data-root data/synthetic --output-dir outputs/smoke/bootstrap

python scripts/train.py --config configs/smoke.yaml \
    --bootstrap-dir outputs/smoke/bootstrap --output-dir outputs/smoke/segmenter

python scripts/infer.py --config configs/smoke.yaml \
    --checkpoint outputs/smoke/segmenter/last.pt \
    --scan data/synthetic/test --output-dir outputs/smoke/instances

python scripts/evaluate.py \
    --pred-dir outputs/smoke/instances --gt-dir data/synthetic/test --markdown
```

The generated crops are **dark ellipses on a light field, not organisms**, and
`configs/smoke.yaml` cuts training to a handful of epochs. Every number this
produces is meaningless and no checkpoint from it should be carried into a real
run. The only question it answers is whether the code runs on your machine.

Two things it deliberately does not shortcut. It needs `HF_TOKEN`, because a
gated-weights failure is exactly what you want to hit here rather than at the
start of a real run — and on a node with no outbound network the resulting 401
reads like a firewall problem. And it leaves stage 3 at its published settings,
so what you see is what those extraction constants do.

**Verify:** step 4 prints a nonzero instance count and step 5 prints a metrics
table. Use `last.pt`, not `best.pt` — a 2-epoch run will probably never improve
boundary IoU, so `best.pt` may not be written at all.

If extraction returns zero instances, that is a finding rather than a crash:
start at `min_instance_area` and read [`docs/TUNING.md`](docs/TUNING.md) §2.

`data/synthetic/` is disposable — delete it when you are done.

---

## 3. Data layout

No imagery ships with this repository. Arrange your own as:

```
data/
├── annotated/
│   ├── images/     # your annotated crops, 512x512 (we used 3)
│   └── masks/      # RGB masks, matching stems
├── unlabeled/      # candidate pool, 512x512 crops
├── test/           # held-out 1024x1024 crops + ground truth
└── scans/          # full ~15000x25000 px scans
```

Masks are RGB: red = background, green = organism interior, blue = boundary.
Ground truth in `test/` is either LabelMe `<stem>.json` or a `<stem>.npy` label
image. See [`data/README.md`](data/README.md) for the full specification.

Stems must match between `images/` and `masks/`, and between predictions and
ground truth — that is how the scripts pair files.

---

## 4. Stage 1 — bootstrap the pseudo-labels

```bash
python scripts/bootstrap.py \
    --data-root data/ \
    --output-dir outputs/bootstrap
```

Trains the shallow MLP on tokens from the annotated crops, scores every crop in
`data/unlabeled/`, and keeps those scoring above
`stage1_bootstrap.selection.confidence_threshold`.

Writes `outputs/bootstrap/{images,masks}/` plus `ranking.csv`, which records
every candidate's confidence and whether it was selected.

**On the first run the threshold is unset**, so every candidate is kept and the
script warns. That is not a usable selection — it exists so that `ranking.csv`
gets written. Read it, find where confidence falls off, set a threshold in your
config and run again. [`docs/TUNING.md`](docs/TUNING.md) covers how to choose
one. The paper's `N_boot = 82` was the count that cleared *our* threshold, not
a target to aim for.

**Verify:** the selected masks look plausible — open a few. These become the
*only* supervision stage 2 ever sees, so a systematic failure here (for
instance, everything predicted background) propagates silently through the rest
of the pipeline. `ranking.csv` is also the place to check whether the confidence
spread is meaningful at all, or whether your best candidates are barely better
than your worst.

---

## 5. Stage 2 — train the segmenter

```bash
python scripts/train.py \
    --bootstrap-dir outputs/bootstrap \
    --output-dir outputs/segmenter
```

80 epochs, AdamW, cosine annealing with warm restarts. Only the U-Net decoder
updates; the backbone stays frozen and in eval mode throughout.

Checkpoints are selected on **boundary IoU**, not mIoU — background IoU is near
1 whatever the model does, so an averaged score can look healthy while boundary
prediction has collapsed, and boundary is the only class that decides whether
touching organisms can be split.

Useful flags: `--epochs N` overrides the config for a quick smoke run,
`--resume outputs/segmenter/last.pt` continues an interrupted job.

Two checkpoints are written: `best.pt` on boundary-IoU improvement, and
`last.pt` every epoch. Resume from `last.pt` — `best.pt` can lag many epochs
behind, and on an interactive session that dies mid-run that gap is lost work.

**Verify:** `outputs/segmenter/best.pt` exists and the logged boundary IoU is
climbing rather than flat at zero. Try `--epochs 2` first to confirm the whole
loop executes before committing to the full run; a run that short may never
improve boundary IoU, in which case only `last.pt` is written and the script
says so.

---

## 6. Stage 3 — inference

Two different jobs, depending on what you want.

### 6a. On the held-out crops, to reproduce the metrics

```bash
python scripts/infer.py \
    --checkpoint outputs/segmenter/best.pt \
    --scan data/test \
    --output-dir outputs/test_instances
```

Pass the **directory**, so every test crop is processed and the output stems
line up with the ground-truth files. Step 7 depends on that pairing.

### 6b. On a full scan, to apply the method

```bash
python scripts/infer.py \
    --checkpoint outputs/segmenter/best.pt \
    --scan data/scans/scan_01.tif \
    --output-dir outputs/instances
```

Tiled at 512 px with 128 px overlap and Gaussian blending, so scan size is
bounded by host memory rather than GPU memory. Add `--save-maps` to also write
the blended logits — these are large, roughly 1.5 GB per channel for a full
scan.

**Verify:** each input yields `<stem>_mask.png` and `<stem>_instances.npy`, and
the printed instance count per crop is in a believable range. Zero instances
everywhere means stage 2 failed; tens of thousands means the watershed is
fragmenting.

---

## 7. Evaluation

```bash
python scripts/evaluate.py \
    --pred-dir outputs/test_instances \
    --gt-dir data/test \
    --markdown \
    --csv outputs/metrics.csv
```

Reports PQ, SQ, RQ, precision, recall and OSR, matched one-to-one at IoU 0.5
and macro-averaged across the crops. `--markdown` prints a README-ready table
row; `--csv` writes per-crop numbers.

**Verify:** the crop count in the header matches how many test crops you have.
If it is lower, some stems failed to pair and the script says which were
skipped — a silently smaller denominator would inflate the macro average.

Read OSR alongside precision and recall. Equal predicted and reference counts
can hide simultaneous false positives and false negatives.

---

## Ablations

`--classifier` at stage 1 swaps the bootstrapper; the rest of the pipeline is
unchanged, which is what makes the arms comparable.

```bash
python scripts/bootstrap.py --data-root data/ --output-dir outputs/rf    --classifier random_forest
python scripts/bootstrap.py --data-root data/ --output-dir outputs/mlpd  --classifier mlp_deep
```

Then run steps 5–7 against each output directory and pass `--name` to
`evaluate.py` to label the table row.

---

## Running on a Slurm cluster

Request an interactive session and run the steps inside it, rather than
submitting a batch script — the stages are quick enough to babysit, and the
verify steps above only help if you are watching.

```bash
srun --pty --gres=gpu:1 --mem=64G --time=08:00:00 bash
```

Then `source .venv/bin/activate`, `export HF_TOKEN=...`, and work through the
steps above. Note that the token has to be exported *inside* the session; it
does not survive from the login node.

Host memory matters more than GPU memory for stage 3: the blending accumulator
for a full scan is sized by the scan, not the tile.
