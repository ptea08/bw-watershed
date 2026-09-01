"""Losses for the semantic segmenter.

Paper, Sec. 3.1: the models are optimized with Tversky loss (alpha = 0.7,
beta = 0.3) together with weighted cross-entropy. The background, boundary and
foreground weights are 1.0, 8.0 and 1.5 respectively.

The two terms do different jobs. Tversky is set-based and handles the class
imbalance directly through alpha/beta; weighted cross-entropy is per-pixel and
supplies the stable gradient that Tversky lacks on small batches, where the
boundary class may barely appear. The heavy boundary weight is what makes
touching organisms separable downstream — boundary pixels are a small fraction
of the image but carry almost all of the instance-separation signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import class_weights


class TverskyLoss(nn.Module):
    """Macro-averaged multi-class Tversky loss.

    ``alpha`` weights false negatives and ``beta`` false positives, so
    ``alpha > beta`` biases the model towards not missing organisms. Classes
    absent from a batch are skipped rather than scored as perfect, which stops
    the rare boundary class from being drowned out by easy batches.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-6,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        valid = (target != self.ignore_index).unsqueeze(1).float()
        clamped = target.clone()
        clamped[target == self.ignore_index] = 0

        one_hot = F.one_hot(clamped, num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid
        probs = probs * valid

        scores = []
        for c in range(num_classes):
            p = probs[:, c].reshape(-1)
            t = one_hot[:, c].reshape(-1)
            if t.sum() < 1:
                continue
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tversky = (tp + self.smooth) / (
                tp + self.alpha * fn + self.beta * fp + self.smooth
            )
            scores.append(1 - tversky)

        if not scores:
            return logits.sum() * 0.0
        return torch.stack(scores).mean()


class CombinedLoss(nn.Module):
    """``tversky + ce_weight * weighted_cross_entropy``.

    The relative weighting is not given in the paper; ``ce_weight`` is 0.4,
    keeping cross-entropy as a stabiliser rather than the dominant term.
    """

    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg.stage2_segmenter.loss
        ignore_index = cfg.data.ignore_index

        self.tversky = TverskyLoss(
            alpha=loss_cfg.tversky.alpha,
            beta=loss_cfg.tversky.beta,
            smooth=loss_cfg.tversky.smooth,
            ignore_index=ignore_index,
        )
        self.ce_weight = loss_cfg.cross_entropy.ce_weight
        self.ignore_index = ignore_index
        self.register_buffer(
            "class_weights", torch.tensor(class_weights(cfg), dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
        return self.tversky(logits, target) + self.ce_weight * cross_entropy


def per_class_iou(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255
) -> list[float]:
    """Per-class IoU. Absent classes yield ``nan`` rather than 0.

    Reported per class because mIoU hides the only number that matters here:
    background IoU is near 1 whatever the model does, so an averaged score can
    look healthy while boundary prediction has collapsed.
    """
    valid = target != ignore_index
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c) & valid
        target_c = (target == c) & valid
        union = (pred_c | target_c).sum().item()
        ious.append((pred_c & target_c).sum().item() / union if union else float("nan"))
    return ious
