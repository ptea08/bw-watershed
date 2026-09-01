"""Paired image / RGB-mask dataset for the stage-2 segmenter.

Expects a directory holding ``images/`` and ``masks/`` subdirectories whose
files share a stem. This is the layout that stage 1 writes its pseudo-labels
into, so stage 2 can be pointed straight at a bootstrap output directory.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import decode_rgb_mask

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Match ``images/<stem>.*`` to ``masks/<stem>.*`` and return sorted pairs."""
    image_dir, mask_dir = root / "images", root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"expected {image_dir} and {mask_dir} to exist")

    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    pairs = [
        (p, masks[p.stem])
        for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in IMAGE_SUFFIXES and p.stem in masks
    ]
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs found under {root}")
    return pairs


def build_augmentation(cfg, train: bool):
    """Albumentations pipeline for training or deterministic validation.

    Both branches resize and crop identically so that train and validation
    losses stay comparable; only the training branch perturbs the sample.
    """
    import albumentations as A

    crop = cfg.data.crop_size
    common = [
        A.LongestMaxSize(max_size=max(crop + 128, 768)),
        A.PadIfNeeded(min_height=crop, min_width=crop, border_mode=cv2.BORDER_REFLECT_101),
    ]
    if not train:
        return A.Compose([*common, A.CenterCrop(crop, crop)])

    aug = cfg.stage2_segmenter.augmentation
    return A.Compose(
        [
            *common,
            A.RandomCrop(crop, crop),
            A.HorizontalFlip(p=aug.hflip),
            A.VerticalFlip(p=aug.vflip),
            A.RandomRotate90(p=aug.rot90),
            A.ShiftScaleRotate(
                shift_limit=aug.shift_scale_rotate.shift_limit,
                scale_limit=aug.shift_scale_rotate.scale_limit,
                rotate_limit=aug.shift_scale_rotate.rotate_limit,
                border_mode=cv2.BORDER_REFLECT_101,
                p=aug.shift_scale_rotate.p,
            ),
            A.ElasticTransform(
                alpha=aug.elastic.alpha, sigma=aug.elastic.sigma, p=aug.elastic.p
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug.brightness_contrast.brightness_limit,
                contrast_limit=aug.brightness_contrast.contrast_limit,
                p=aug.brightness_contrast.p,
            ),
            A.GaussNoise(p=aug.gauss_noise),
        ]
    )


def normalize(image_rgb: np.ndarray, cfg) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32, ImageNet-normalised."""
    img = image_rgb.astype(np.float32) / 255.0
    mean = np.asarray(cfg.data.normalization.mean, dtype=np.float32)
    std = np.asarray(cfg.data.normalization.std, dtype=np.float32)
    return ((img - mean) / std).transpose(2, 0, 1)


class SegmentationDataset(Dataset):
    """Yields ``(image CHW float32, label HW int64)`` for the segmenter."""

    def __init__(self, root: Path, cfg, train: bool = True):
        self.pairs = find_pairs(Path(root))
        self.cfg = cfg
        self.transform = build_augmentation(cfg, train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        label = decode_rgb_mask(
            cv2.imread(str(mask_path)),
            self.cfg.data.mask_encoding.min_channel,
            self.cfg.data.ignore_index,
        )

        augmented = self.transform(image=image, mask=label)
        image, label = augmented["image"], augmented["mask"]

        return (
            torch.from_numpy(normalize(image, self.cfg)).float(),
            torch.from_numpy(np.ascontiguousarray(label)).long(),
        )
