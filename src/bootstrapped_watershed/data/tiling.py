"""Tiling and Gaussian-weighted reassembly for full-scan inference.

Paper, Sec. 3.2: a full scan is roughly 15000 x 25000 px and cannot be
processed at once. Inference runs on 512 x 512 tiles overlapping by 128 px;
within each overlap, predictions near the tile centre receive greater Gaussian
weight than predictions near its edges, and the weighted predictions are
averaged into continuous foreground and boundary maps.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    """Start offsets along one axis such that every tile is full-sized.

    The final tile is shifted inward to end flush with the edge rather than
    being padded. That costs a little extra overlap and buys freedom from
    border artefacts, which matters because a scan edge is where organisms are
    most often clipped.
    """
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] + tile_size < length:
        starts.append(length - tile_size)
    return starts


def iter_tiles(
    height: int, width: int, tile_size: int = 512, overlap: int = 128
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(y0, y1, x0, x1)`` tile bounds covering the image.

    Tiles at the right and bottom edges are shifted inward so that every tile
    is exactly ``tile_size`` — this avoids padding artefacts at scan borders.
    """
    step = tile_size - overlap
    for y0 in tile_starts(height, tile_size, step):
        for x0 in tile_starts(width, tile_size, step):
            yield y0, min(y0 + tile_size, height), x0, min(x0 + tile_size, width)


def gaussian_weight_map(tile_size: int = 512, sigma: float = 0.5) -> np.ndarray:
    """Radial Gaussian window used to weight each tile during blending.

    ``sigma`` is expressed on a normalised ``[-1, 1]`` grid spanning the tile,
    so it is resolution-independent: 0.5 places the 1-sigma contour a quarter
    of a tile from the centre, meaning a pixel's contribution has decayed
    substantially by the time it reaches a seam.

    The window is never zero, so a pixel covered by only one tile is still
    weighted correctly once the accumulated weights are divided out.
    """
    axis = np.linspace(-1.0, 1.0, tile_size)
    xx, yy = np.meshgrid(axis, axis)
    return np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2)).astype(np.float32)


class TileBlender:
    """Accumulates Gaussian-weighted tile logits into one full-scan map.

    Tiles are folded in one at a time rather than collected and combined at
    the end: a 15000 x 25000 scan needs ~1.5 GB per channel for the
    accumulator alone, and holding every tile's logits alongside it would not
    fit in host memory, let alone GPU memory.
    """

    def __init__(self, height: int, width: int, num_classes: int, weight_map: np.ndarray):
        self._logits = np.zeros((num_classes, height, width), dtype=np.float32)
        self._weights = np.zeros((height, width), dtype=np.float32)
        self._weight_map = weight_map

    def add(self, logits: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
        """Fold one tile's ``(C, h, w)`` logits in at ``(y0, y1, x0, x1)``."""
        y0, y1, x0, x1 = bounds
        window = self._weight_map[: y1 - y0, : x1 - x0]
        self._logits[:, y0:y1, x0:x1] += logits * window
        self._weights[y0:y1, x0:x1] += window

    def result(self) -> np.ndarray:
        """Return the weight-normalised ``(C, H, W)`` logit map."""
        weights = np.where(self._weights == 0.0, 1.0, self._weights)
        return self._logits / weights
