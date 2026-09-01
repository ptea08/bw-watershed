"""Training loop for the semantic segmenter.

Paper, Sec. 4.2: the U-Net decoder was trained for 80 epochs using AdamW and
cosine annealing of the learning rate with warm restarts, while the
DINOv3-ConvNeXt-Small backbone remained frozen. Training was performed on a
single NVIDIA Tesla T4.

The checkpoint is selected on validation *boundary* IoU rather than mIoU.
Background IoU is near one however the model behaves, so a mean would stay
comfortable while boundary prediction quietly collapsed — and a segmenter
without boundaries cannot split touching organisms, which is the entire point
of the pipeline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from ..config import class_index, resolve_device, set_seed
from ..data.dataset import SegmentationDataset
from .decoder import ConvNeXtUNet
from .losses import CombinedLoss, per_class_iou


def build_dataloaders(cfg, bootstrap_dir: Path) -> tuple[DataLoader, DataLoader]:
    """Split the bootstrapped crops into train and validation loaders.

    The same indices drive two dataset objects: the training one augments, the
    validation one does not. Sharing a single augmented dataset would leak
    random crops into validation and make the checkpoint metric noisy.
    """
    optim_cfg = cfg.stage2_segmenter.optim

    train_source = SegmentationDataset(bootstrap_dir, cfg, train=True)
    val_source = SegmentationDataset(bootstrap_dir, cfg, train=False)

    n_val = max(1, int(len(train_source) * optim_cfg.val_split))
    n_train = len(train_source) - n_val
    train_split, val_split = random_split(
        range(len(train_source)),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        Subset(train_source, list(train_split)),
        batch_size=optim_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_source, list(val_split)),
        batch_size=optim_cfg.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch=None) -> float:
    model.train()
    total, batches = 0.0, 0

    # An epoch is one line of output, so on a large pseudo-label set the gap
    # between lines is the whole epoch. Break it up a few times.
    report_every = max(1, len(loader) // 4)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device.type):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        total += loss.item()
        batches += 1

        if batches % report_every == 0 and batches != len(loader):
            prefix = "" if epoch is None else f"ep {epoch} "
            print(
                f"      {prefix}batch {batches}/{len(loader)}  "
                f"running loss {total / batches:.4f}",
                flush=True,
            )

    return total / max(batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, cfg, device) -> tuple[float, dict[str, float]]:
    model.eval()
    num_classes = cfg.data.num_classes
    total, batches = 0.0, 0
    collected: list[list[float]] = [[] for _ in range(num_classes)]

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total += criterion(logits, labels).item()
        batches += 1

        ious = per_class_iou(
            logits.argmax(dim=1), labels, num_classes, cfg.data.ignore_index
        )
        for i, iou in enumerate(ious):
            if not np.isnan(iou):
                collected[i].append(iou)

    per_class = {
        name: (float(np.mean(collected[i])) if collected[i] else 0.0)
        for i, name in enumerate(cfg.data.class_names)
    }
    return total / max(batches, 1), per_class


def train(cfg, bootstrap_dir: Path, output_dir: Path, resume: Path | None = None) -> Path:
    """Train the decoder on the bootstrapped pseudo-labels.

    Returns the path of the best checkpoint. Only decoder weights are saved —
    the frozen backbone is reproducible from its Hub identifier, and storing
    it would multiply the checkpoint size for no gain.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = resolve_device(cfg)
    optim_cfg = cfg.stage2_segmenter.optim

    train_loader, val_loader = build_dataloaders(cfg, Path(bootstrap_dir))
    print(
        f"  {len(train_loader.dataset)} train / {len(val_loader.dataset)} val crops, "
        f"batch {optim_cfg.batch_size} -> {len(train_loader)} steps per epoch",
        flush=True,
    )
    print(f"  building the decoder on {device} ...", flush=True)
    model = ConvNeXtUNet(cfg).to(device)
    criterion = CombinedLoss(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=optim_cfg.t_0, T_mult=optim_cfg.t_mult, eta_min=optim_cfg.eta_min
    )
    use_amp = bool(cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    boundary = cfg.data.class_names[class_index(cfg, "boundary")]
    start_epoch, best_metric = 0, 0.0
    if resume is not None and Path(resume).exists():
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.decode_head.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_metric = checkpoint.get("best_metric", 0.0)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history = []

    def checkpoint(epoch: int, metric: float) -> dict:
        return {
            "epoch": epoch,
            "decoder_state_dict": model.decode_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": metric,
            "checkpoint_metric": optim_cfg.checkpoint_metric,
            "backbone_name": cfg.stage2_segmenter.backbone.name,
            "decoder_channels": list(cfg.stage2_segmenter.decoder.channels),
            "class_names": list(cfg.data.class_names),
        }

    for epoch in range(start_epoch, optim_cfg.epochs):
        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch + 1
        )
        val_loss, per_class = validate(model, val_loader, criterion, cfg, device)
        scheduler.step()

        ious = " | ".join(f"{k} {v:.3f}" for k, v in per_class.items())
        print(
            f"  ep {epoch + 1:3d}/{optim_cfg.epochs}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  IoU {ious}  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}  {time.time() - started:.1f}s",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "iou": per_class,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Written every epoch so an interrupted run can `--resume`, and so a
        # short smoke run still produces something loadable even if boundary
        # IoU never improves on its initial value.
        torch.save(checkpoint(epoch, best_metric), last_path)

        if per_class[boundary] > best_metric:
            best_metric = per_class[boundary]
            torch.save(checkpoint(epoch, best_metric), best_path)
            print(
                f"    best {optim_cfg.checkpoint_metric} {best_metric:.4f} -> {best_path}",
                flush=True,
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    if not best_path.exists():
        print(
            f"  warning: {optim_cfg.checkpoint_metric} never improved; using {last_path}",
            flush=True,
        )
        return last_path
    return best_path
