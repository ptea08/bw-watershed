"""Semantic maps -> instances.

Paper, Sec. 3.2, in order:

1. Predicted boundary pixels are removed, separating organisms the network has
   already divided.
2. Pinch-point severing: pixels where the distance transform satisfies
   ``0 < D_L2(M_g) < 12 px`` and which have boundary predictions on at least
   2 of 4 sides within an 8 px radius are cut.
3. The severed mask is closed (5 x 5) and a dilated predicted boundary wall
   (3 x 3 open, then 5 x 5 dilate) is subtracted to enforce remaining
   boundaries as hard separators, giving the foreground ``F``.
4. Watershed markers are placed at local maxima of the Gaussian-smoothed
   (9 x 9) distance transform of ``F`` exceeding a 5 px threshold.
5. Watershed partitions the refined foreground into instances, and those
   below 150 px^2 are discarded as a final cleanup step.

Step 1 needs no explicit subtraction: the semantic map is an argmax output, so
the foreground and boundary classes are disjoint by construction and reading
the foreground channel already yields ``(interior u boundary) - boundary``.
:func:`~bootstrapped_watershed.data.masks.split_channels` does this, and the
distance transform below is therefore a correct ``D_L2(M_g)``.
"""

from __future__ import annotations

import cv2
import numpy as np


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def sever_pinch_points(
    mask: np.ndarray,
    boundary: np.ndarray,
    dt_min: float = 0.0,
    dt_max: float = 12.0,
    radius: int = 8,
    min_sides: int = 2,
    sever_dilation: int = 3,
) -> np.ndarray:
    """Step 2 — cut the thin necks the network left connected.

    A pixel is severed when its L2 distance transform lies in
    ``(dt_min, dt_max)`` — i.e. it is inside the mask but close to an edge, so
    the local structure is thin; the paper's bound is ``0 < D_L2(M_g) < 12`` —
    and boundary evidence appears on at least ``min_sides`` of the
    four cardinal directions within ``radius`` px. Two opposing sides pinching
    inward is the signature of two organisms meeting; a thin region with
    boundary on only one side is more likely to be a limb or an antenna, and
    is left intact.

    Two implementation details are not specified in Sec. 3.2:

    * Sides are probed over an *annulus*, offsets ``radius // 2 .. radius``,
      rather than the full disc. Boundary pixels immediately adjacent to the
      neck belong to the organism's own outline and would fire on almost every
      thin pixel; skipping the inner half looks past them.
    * The cut is dilated by ``sever_dilation`` before being applied, widening
      it from roughly one pixel to roughly three. A one-pixel cut is often
      re-bridged by the 5 x 5 closing in :func:`build_foreground`, which would
      undo the severing entirely.
    """
    severed = mask.copy()
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    candidates = (distance > dt_min) & (distance < dt_max)

    height, width = boundary.shape
    padded = np.pad(boundary, radius, mode="constant", constant_values=0)

    above = np.zeros_like(boundary, dtype=bool)
    below = np.zeros_like(boundary, dtype=bool)
    left = np.zeros_like(boundary, dtype=bool)
    right = np.zeros_like(boundary, dtype=bool)

    for offset in range(radius // 2, radius + 1):
        r = radius
        above |= padded[r - offset : r - offset + height, r : r + width] > 0
        below |= padded[r + offset : r + offset + height, r : r + width] > 0
        left |= padded[r : r + height, r - offset : r - offset + width] > 0
        right |= padded[r : r + height, r + offset : r + offset + width] > 0

    sides = above.astype(np.uint8) + below + left + right
    pinch = candidates & (sides >= min_sides)
    if not pinch.any():
        return severed

    cut = cv2.dilate(pinch.astype(np.uint8) * 255, _ellipse(sever_dilation))
    severed[cut > 0] = 0
    return severed


def build_foreground(
    severed: np.ndarray,
    boundary: np.ndarray,
    closing_kernel: int = 5,
    boundary_dilation: tuple[int, int] = (3, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """Step 3 — close the severed mask, subtract the dilated boundary wall.

    Returns ``(F, wall)``. The wall is opened first to drop isolated boundary
    speckle that would otherwise punch holes in organisms, then dilated so
    that the surviving boundaries are thick enough to actually disconnect the
    regions they separate.
    """
    open_size, dilate_size = boundary_dilation
    closed = cv2.morphologyEx(severed, cv2.MORPH_CLOSE, _ellipse(closing_kernel))
    cleaned = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, _ellipse(open_size))
    wall = cv2.dilate(cleaned, _ellipse(dilate_size))
    return cv2.bitwise_and(closed, cv2.bitwise_not(wall)), wall


def watershed_instances(
    foreground: np.ndarray,
    image_bgr: np.ndarray,
    smoothing_kernel: int = 9,
    marker_threshold: float = 5.0,
    peak_kernel: int = 9,
    background_dilation: int = 5,
    min_area: int = 150,
) -> np.ndarray:
    """Steps 4-5 — distance-transform watershed, then small-instance cleanup.

    Markers are the local maxima of the smoothed distance transform that
    exceed ``marker_threshold``: each is a point deep inside some organism, so
    one marker means one instance. Smoothing first matters — the raw distance
    transform of a ragged mask has many spurious local maxima, each of which
    would become a separate instance.

    ``cv2.watershed`` floods ``image_bgr``, so the flooding surface is the
    original image intensity rather than the binary mask. Sec. 3.2 does not
    say which is used; image intensity lets a visible seam between two
    touching organisms guide the cut even where the predicted masks merged.

    Returns an int32 label image, 0 = background.
    """
    if foreground.sum() == 0:
        return np.zeros(foreground.shape, dtype=np.int32)

    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    smoothed = cv2.GaussianBlur(distance, (smoothing_kernel, smoothing_kernel), 0)
    peaks = cv2.dilate(smoothed, np.ones((peak_kernel, peak_kernel), np.uint8))
    seeds = np.uint8((smoothed == peaks) & (smoothed > marker_threshold)) * 255

    _, markers = cv2.connectedComponents(seeds)
    markers = markers + 1

    # Everything between the mask and its dilation is "unknown": the watershed
    # is free to assign it, which is what lets basins grow out to the edges.
    grown = cv2.dilate(
        foreground, np.ones((background_dilation, background_dilation), np.uint8)
    )
    markers[cv2.subtract(grown, foreground) == 255] = 0

    cv2.watershed(image_bgr, markers)

    partitioned = foreground.copy()
    partitioned[markers == -1] = 0

    count, blobs, stats, _ = cv2.connectedComponentsWithStats(partitioned)
    labels = np.zeros_like(blobs, dtype=np.int32)
    next_id = 1
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            labels[blobs == i] = next_id
            next_id += 1
    return labels


def extract_instances(mask_bgr: np.ndarray, image_bgr: np.ndarray, cfg) -> np.ndarray:
    """Run the full extraction sequence. This is the entry point.

    ``mask_bgr`` is the predicted RGB semantic map, ``image_bgr`` the original
    image it was predicted from. Returns an int32 instance label image.
    """
    from ..data.masks import split_channels

    extraction = cfg.stage3_instances.extraction
    foreground, boundary = split_channels(mask_bgr)

    severed = sever_pinch_points(
        foreground,
        boundary,
        dt_min=extraction.pinch_severing.dt_min,
        dt_max=extraction.pinch_severing.dt_max,
        radius=extraction.pinch_severing.radius,
        min_sides=extraction.pinch_severing.min_sides,
        sever_dilation=extraction.pinch_severing.sever_dilation,
    )
    refined, _ = build_foreground(
        severed,
        boundary,
        closing_kernel=extraction.closing_kernel,
        boundary_dilation=tuple(extraction.boundary_dilation),
    )
    return watershed_instances(
        refined,
        image_bgr,
        smoothing_kernel=extraction.watershed.smoothing_kernel,
        marker_threshold=extraction.watershed.marker_threshold,
        peak_kernel=extraction.watershed.peak_kernel,
        background_dilation=extraction.watershed.background_dilation,
        min_area=extraction.min_instance_area,
    )
