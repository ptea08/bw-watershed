"""RGB annotation masks <-> integer class maps.

Paper, Sec. 3.1: annotations encode background in red, organism interiors in
green and organism boundaries in blue. Each pixel is assigned to the class of
its dominant colour channel; ties and other ambiguous pixels receive an ignore
index and do not contribute to the loss.

Annotations are stored as RGB rather than as index maps because they are
produced by hand, where colours are what an annotator actually sees. The
dominant channel must additionally clear ``min_channel`` for the pixel to
count at all, which discards dark and washed-out pixels where no channel
really dominates.
"""

from __future__ import annotations

import cv2
import numpy as np

# Integer class indices. These are load-bearing: they are the argmax order of
# every model in the pipeline and must agree with ``data.class_names`` in the
# config. ``tests/test_config.py`` asserts that they do.
BACKGROUND = 0
FOREGROUND = 1
BOUNDARY = 2

# Class -> BGR triple, for writing masks back out in the annotators' colours.
_CLASS_TO_BGR = {
    BACKGROUND: (0, 0, 255),  # red
    FOREGROUND: (0, 255, 0),  # green
    BOUNDARY: (255, 0, 0),  # blue
}


def decode_rgb_mask(
    mask_bgr: np.ndarray, min_channel: int = 50, ignore_index: int = 255
) -> np.ndarray:
    """Convert a BGR annotation mask to an ``(H, W)`` uint8 class map.

    ``mask_bgr`` is in OpenCV channel order, i.e. straight from ``cv2.imread``.
    A pixel is labelled only if one channel is strictly greater than both
    others *and* exceeds ``min_channel``; everything else is ``ignore_index``.
    """
    b = mask_bgr[:, :, 0].astype(np.int16)
    g = mask_bgr[:, :, 1].astype(np.int16)
    r = mask_bgr[:, :, 2].astype(np.int16)

    labels = np.full(mask_bgr.shape[:2], ignore_index, dtype=np.uint8)
    labels[(r > g) & (r > b) & (r > min_channel)] = BACKGROUND
    labels[(g > r) & (g > b) & (g > min_channel)] = FOREGROUND
    labels[(b > r) & (b > g) & (b > min_channel)] = BOUNDARY
    return labels


def encode_class_map(class_map: np.ndarray) -> np.ndarray:
    """Inverse of :func:`decode_rgb_mask`. Returns a BGR uint8 image.

    Ignored pixels have no colour of their own and come back black, which
    :func:`decode_rgb_mask` reads back as ignored — the round trip is stable.
    """
    out = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls, bgr in _CLASS_TO_BGR.items():
        out[class_map == cls] = bgr
    return out


def split_channels(mask_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pull binary ``(foreground, boundary)`` masks out of a predicted RGB mask.

    Because a predicted mask is an argmax output, green and blue are disjoint
    by construction — so the foreground mask returned here is already
    ``interior - boundary``, and step 1 of the paper's extraction procedure
    needs no separate subtraction.

    Values are 0/255 uint8, which is what the OpenCV morphology and distance
    transform calls in :mod:`bootstrapped_watershed.stage3_instances` expect.
    """
    foreground = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    boundary = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    foreground[mask_bgr[:, :, 1] == 255] = 255
    boundary[mask_bgr[:, :, 0] == 255] = 255
    return foreground, boundary


def downsample_to_patches(
    class_map: np.ndarray,
    grid_h: int,
    grid_w: int,
    num_classes: int = 3,
    ignore_index: int = 255,
) -> np.ndarray:
    """Majority-vote pixel labels down onto a ``(grid_h, grid_w)`` patch grid.

    Stage 1 classifies backbone descriptors at their native stride-16
    resolution, so the supervision has to come down to meet them rather than
    the features being upsampled to meet the supervision. Each block votes;
    blocks holding no valid pixel at all stay ignored.

    Voting rather than point-sampling matters for the boundary class, which is
    frequently a single pixel wide and would often be missed by a subsample.
    """
    height, width = class_map.shape
    block_h, block_w = height // grid_h, width // grid_w

    patch_labels = np.full((grid_h, grid_w), ignore_index, dtype=np.int64)
    for i in range(grid_h):
        for j in range(grid_w):
            block = class_map[
                i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w
            ]
            valid = block[block != ignore_index].astype(np.int64).ravel()
            if valid.size:
                patch_labels[i, j] = np.bincount(valid, minlength=num_classes).argmax()
    return patch_labels


def upsample_from_patches(patch_map: np.ndarray, height: int, width: int) -> np.ndarray:
    """Expand a patch-resolution class map back to pixel resolution.

    Nearest-neighbour only. Interpolating class indices would invent in-between
    values that no classifier ever predicted; instead each 16x16 pixel block
    simply inherits its patch's class.
    """
    return cv2.resize(
        patch_map.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
