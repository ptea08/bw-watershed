"""Frozen DINOv3-ConvNeXt-Small feature extraction for stage 1.

Paper, Sec. 3.1: "a frozen DINOv3-ConvNeXt-Small backbone provides a
384-dimensional feature vector for each spatial position", which a shallow
classifier turns into pseudo-labels.

That description picks out ``stage3`` of the ConvNeXt hierarchy: 384 channels
at stride 16. Stage 2 loads the same checkpoint but reads all four stages, so
the two stages share one gated download rather than two.

Descriptors are consumed at their native stride-16 resolution and are never
upsampled before classification — interpolating between them manufactures
feature vectors the backbone never produced. The predicted grid is expanded to
pixels afterwards instead (see
:func:`~bootstrapped_watershed.data.masks.upsample_from_patches`).

The DINOv3 checkpoints are gated on Hugging Face. Request access, then set
``HF_TOKEN`` in your environment. Do not hardcode a token.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import require_hf_token
from ..data.masks import decode_rgb_mask, downsample_to_patches


def load_backbone(cfg, device):
    """Load the frozen DINOv3-ConvNeXt backbone in eval mode, grads disabled.

    Requires ``transformers>=4.56``, which is where DINOv3 support landed.
    """
    from transformers import DINOv3ConvNextBackbone

    backbone_cfg = cfg.stage1_bootstrap.backbone
    backbone = DINOv3ConvNextBackbone.from_pretrained(
        backbone_cfg.name,
        out_features=list(backbone_cfg.out_features),
        token=require_hf_token(),
    )
    backbone = backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone


def normalize(image_bgr: np.ndarray, cfg) -> np.ndarray:
    """BGR uint8 -> ImageNet-normalised RGB float32, still in HWC order."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(cfg.data.normalization.mean, dtype=np.float32)
    std = np.asarray(cfg.data.normalization.std, dtype=np.float32)
    return (rgb - mean) / std


def pad_to_multiple(image: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    """Reflect-pad bottom and right so both dimensions divide ``multiple``.

    Returns the padded image plus the original ``(height, width)`` so the
    padding can be trimmed off the prediction afterwards.
    """
    height, width = image.shape[:2]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return image, height, width


@torch.no_grad()
def extract_patch_tokens(backbone, image_bgr: np.ndarray, cfg, device):
    """Return ``(tokens, grid_h, grid_w, height, width)`` for one crop.

    ``tokens`` has shape ``(grid_h, grid_w, feature_dim)`` at the backbone's
    native stride-16 resolution; ``height`` and ``width`` are the crop's size
    before padding.
    """
    stride = cfg.stage1_bootstrap.backbone.stride

    padded, height, width = pad_to_multiple(image_bgr, stride)
    tensor = torch.from_numpy(normalize(padded, cfg)).permute(2, 0, 1)[None].to(device)
    with torch.autocast(device.type, enabled=bool(cfg.amp) and device.type == "cuda"):
        output = backbone(pixel_values=tensor)

    # A ConvNeXt stage is already spatial: (1, C, H/16, W/16). No prefix tokens
    # to strip and no reshape guesswork — take the grid straight off the map.
    feature_map = output.feature_maps[0][0].float()
    tokens = feature_map.permute(1, 2, 0).cpu().numpy()
    grid_h, grid_w = tokens.shape[:2]
    return tokens, grid_h, grid_w, height, width


def build_token_dataset(
    backbone, pairs: list[tuple[Path, Path]], cfg, device
) -> tuple[np.ndarray, np.ndarray]:
    """Pool ``(tokens, labels)`` from the annotated ``(image, mask)`` pairs.

    Returns ``(N, feature_dim)`` float32 tokens and ``(N,)`` int64 labels with
    ignored positions already dropped, ready for
    :func:`~bootstrapped_watershed.stage1_bootstrap.classifier.train_classifier`.
    """
    stride = cfg.stage1_bootstrap.backbone.stride
    ignore_index = cfg.data.ignore_index

    all_tokens, all_labels = [], []
    for index, (image_path, mask_path) in enumerate(pairs, 1):
        # The first pass through here is where the gated Hub download happens,
        # so a run that looks hung at [1/N] is usually fetching weights.
        print(f"    [{index}/{len(pairs)}] {image_path.name}", flush=True)

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))
        if image is None:
            raise FileNotFoundError(image_path)
        if mask is None:
            raise FileNotFoundError(mask_path)

        tokens, grid_h, grid_w, _, _ = extract_patch_tokens(backbone, image, cfg, device)
        pixel_labels = decode_rgb_mask(mask, cfg.data.mask_encoding.min_channel, ignore_index)

        # Match the reflect-padding applied to the image before encoding, so
        # that grid cell (i, j) covers the same pixels in both.
        pixel_labels = cv2.resize(
            pixel_labels,
            (grid_w * stride, grid_h * stride),
            interpolation=cv2.INTER_NEAREST,
        )
        patch_labels = downsample_to_patches(
            pixel_labels, grid_h, grid_w, cfg.data.num_classes, ignore_index
        )

        flat_tokens = tokens.reshape(-1, tokens.shape[-1])
        flat_labels = patch_labels.reshape(-1)
        keep = flat_labels != ignore_index
        all_tokens.append(flat_tokens[keep])
        all_labels.append(flat_labels[keep])

    return (
        np.concatenate(all_tokens).astype(np.float32),
        np.concatenate(all_labels).astype(np.int64),
    )
