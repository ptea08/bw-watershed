"""Shallow MLP pixel classifier over frozen DINOv3 features.

Paper, Sec. 3.1: a two-layer MLP maps features to background, boundary and
foreground logits, following 384 -> 128 -> 3. Layer normalisation and ReLU are
applied after the hidden layer, giving roughly 50 000 trainable parameters.
Only the MLP parameters are updated during this stage.

The classifier is deliberately tiny. It sees the tokens of three crops and
nothing else, so anything with real capacity would memorise them; its job is
only to be right often enough that the most confident of its predictions are
usable as supervision for stage 2.

``mlp_deep`` and ``random_forest`` are the Table 1 ablation variants
(MLP-D and RF respectively).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import class_weights


class ShallowMLP(nn.Module):
    """384 -> 128 -> 3 per-token classifier. ~50k parameters."""

    def __init__(self, in_dim: int = 384, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLP(nn.Module):
    """MLP-D ablation: 384 -> 256 -> 128 -> 3, ~165k parameters.

    Dropout sits before the output layer because with only three annotated
    crops the extra capacity overfits readily — Table 1 shows it still scores
    below the shallow variant.
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dims: tuple[int, ...] = (256, 128),
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), nn.LayerNorm(width), nn.ReLU(inplace=True)]
            prev = width
        layers += [nn.Dropout(dropout), nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_classifier(cfg, classifier_type: str | None = None):
    """Dispatch on ``cfg.stage1_bootstrap.classifier.type``.

    ``random_forest`` returns a scikit-learn estimator rather than a module;
    :func:`train_classifier` and :func:`predict_proba` handle both.
    """
    stage1 = cfg.stage1_bootstrap
    kind = classifier_type or stage1.classifier.type
    in_dim = stage1.backbone.feature_dim
    num_classes = cfg.data.num_classes

    if kind == "mlp_shallow":
        return ShallowMLP(in_dim, stage1.classifier.hidden_dim, num_classes)
    if kind == "mlp_deep":
        return DeepMLP(
            in_dim,
            tuple(stage1.classifier.deep_hidden_dims),
            num_classes,
            stage1.classifier.deep_dropout,
        )
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=stage1.random_forest.n_estimators,
            max_depth=stage1.random_forest.max_depth,
            class_weight=dict(enumerate(class_weights(cfg))),
            random_state=cfg.seed,
            n_jobs=-1,
        )
    raise ValueError(f"unknown classifier type: {kind!r}")


def train_classifier(classifier, tokens: np.ndarray, labels: np.ndarray, cfg, device):
    """Fit the pixel classifier on tokens pooled from the ``N_ann`` crops.

    ``tokens`` is ``(N, feature_dim)`` and ``labels`` is ``(N,)``; both have
    already had ignored positions dropped by
    :func:`~bootstrapped_watershed.stage1_bootstrap.features.build_token_dataset`.
    """
    if not isinstance(classifier, nn.Module):
        classifier.fit(tokens, labels)
        return classifier

    optim_cfg = cfg.stage1_bootstrap.optim
    classifier = classifier.to(device)

    dataset = TensorDataset(
        torch.from_numpy(tokens.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=optim_cfg.batch_size, shuffle=True)

    weights = torch.tensor(class_weights(cfg), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay
    )
    # Plain cosine annealing here, not the warm restarts stage 2 uses: this
    # model is small enough to converge inside a single cycle.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optim_cfg.epochs, eta_min=optim_cfg.eta_min
    )

    # 200 epochs on a small MLP is quick but not instant, and printing every
    # one buries the stage-2 output that follows. Report a tenth of them.
    report_every = max(1, optim_cfg.epochs // 10)

    best_loss, best_state = float("inf"), None
    for epoch in range(optim_cfg.epochs):
        classifier.train()
        running = 0.0
        for batch_tokens, batch_labels in loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(batch_tokens), batch_labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        epoch_loss = running / max(len(loader), 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}

        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == optim_cfg.epochs:
            print(
                f"    ep {epoch + 1:3d}/{optim_cfg.epochs}  loss {epoch_loss:.4f}  "
                f"best {best_loss:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

    if best_state is not None:
        classifier.load_state_dict(best_state)
    return classifier


@torch.no_grad()
def predict_proba(classifier, tokens: np.ndarray, device) -> np.ndarray:
    """Class probabilities for ``(N, feature_dim)`` tokens, as ``(N, C)``."""
    if not isinstance(classifier, nn.Module):
        return classifier.predict_proba(tokens)

    classifier.eval()
    batch = torch.from_numpy(tokens.astype(np.float32)).to(device)
    return torch.softmax(classifier(batch).float(), dim=1).cpu().numpy()
