"""Pseudo-label prediction and confidence-based selection.

Paper, Sec. 3.1: the trained MLP predicts semantic masks for a candidate pool
of unlabeled crops sampled from the training scans. For each crop, prediction
confidence is computed as class confidence, and the crops that score above a
confidence threshold are retained (82 of them cleared it in the experiments,
which is the ``N_boot = 82`` the paper reports).

Selection is the whole point of the bootstrap: a classifier fitted to three
crops is wrong often, so rather than trusting all of its output, only the
predictions it is most sure about become supervision for stage 2.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from ..data.masks import encode_class_map, upsample_from_patches
from .classifier import predict_proba
from .features import extract_patch_tokens


def apply_class_priors(probs: np.ndarray, cfg) -> np.ndarray:
    """Reweight softmax outputs towards organism classes, then renormalise.

    A ZooScan tile is overwhelmingly background, so an unbiased argmax erases
    thin structure — especially boundaries, which are exactly what stage 2
    needs to learn. The multipliers are carried over unchanged from the
    random-forest bootstrapper so that the MLP-S, MLP-D and RF arms of Table 1
    stay directly comparable.
    """
    priors = np.ones(probs.shape[-1], dtype=np.float32)
    for name, value in cfg.stage1_bootstrap.selection.class_priors.items():
        priors[list(cfg.data.class_names).index(name)] = float(value)
    weighted = probs * priors
    return weighted / weighted.sum(axis=-1, keepdims=True)


@torch.no_grad()
def predict_crop(classifier, backbone, image_bgr: np.ndarray, cfg, device):
    """Predict one crop's pseudo-label mask and its confidence score.

    Returns ``(mask_bgr, confidence)``. The mask is at full pixel resolution
    but was decided at patch resolution, so it is blocky by construction — the
    median blur knocks the worst of the stair-stepping off without inventing
    detail the classifier never saw.
    """
    tokens, grid_h, grid_w, height, width = extract_patch_tokens(
        backbone, image_bgr, cfg, device
    )
    probs = predict_proba(classifier, tokens.reshape(-1, tokens.shape[-1]), device)
    probs = apply_class_priors(probs.reshape(grid_h, grid_w, -1), cfg)

    class_map = probs.argmax(axis=2).astype(np.uint8)
    confidence = crop_confidence(probs)

    full = upsample_from_patches(class_map, height, width)
    mask_bgr = cv2.medianBlur(
        encode_class_map(full), cfg.stage1_bootstrap.selection.median_blur
    )
    return mask_bgr, confidence


def crop_confidence(probs: np.ndarray) -> float:
    """Aggregate per-position probabilities into one score for a crop.

    The reduction is the mean of the maximum class probability — a crop scores
    highly when the classifier is decisive nearly everywhere. Mean margin and
    entropy would rank the pool differently and therefore select a different
    set of crops, so this choice is part of the method, not an implementation
    detail.
    """
    return float(np.mean(probs.max(axis=-1)))


def select_bootstrap_crops(
    scores: dict[str, float], confidence_threshold: float | None = None
) -> list[str]:
    """Return the identifiers of every crop scoring above ``confidence_threshold``.

    Selection is by threshold, not by count. The paper's ``N_boot = 82`` was the
    number of crops that cleared the cut on our data — a result of this function,
    not an input to it. A fixed count does not transfer: it is arbitrary for a
    pool of a different size, and it forces exactly 82 crops through even when
    only nine of them are any good.

    ``None`` keeps everything. That is not a useful selection — it is the
    escape hatch that lets stage 1 complete so its ``ranking.csv`` can be read
    and a real threshold chosen. Callers are expected to warn.

    Results are ordered by descending confidence, ties broken on the identifier,
    so the selection is reproducible.
    """
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if confidence_threshold is None:
        return [crop_id for crop_id, _ in ranked]
    return [crop_id for crop_id, score in ranked if score > confidence_threshold]


def split_bootstrap(
    crop_ids: list[str], val_split: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split the retained crops into train and validation sets.

    The split is a fraction rather than a count, so it holds for however many
    crops cleared the threshold. With the 82 crops of the paper and
    ``val_split = 0.2`` it reproduces the quoted 66 / 16. Deterministic given
    ``seed``, so re-running selection does not quietly move crops across the
    boundary.
    """
    n_val = max(1, int(len(crop_ids) * val_split))
    order = np.random.default_rng(seed).permutation(len(crop_ids))
    val_idx = set(order[:n_val].tolist())
    train = [c for i, c in enumerate(crop_ids) if i not in val_idx]
    val = [c for i, c in enumerate(crop_ids) if i in val_idx]
    return train, val
