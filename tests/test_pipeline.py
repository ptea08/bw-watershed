"""CPU-only smoke tests for the array half of the pipeline.

Everything here runs on synthetic images in a fraction of a second: no GPU, no
Hugging Face download, no ZooScan data. The point is not accuracy — it is that
each stage still does the structural thing the paper says it does. A dumbbell
must come apart into two instances; tiles must cover the scan; a perfect
prediction must score PQ 1.0.

Tests that need torch live in ``test_models.py``.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from bootstrapped_watershed.config import load_config
from bootstrapped_watershed.data import masks, tiling
from bootstrapped_watershed.eval import panoptic
from bootstrapped_watershed.stage3_instances import extraction


@pytest.fixture
def cfg():
    return load_config()


# --------------------------------------------------------------------------
# data/masks.py
# --------------------------------------------------------------------------

RED = (0, 0, 255)  # BGR — background
GREEN = (0, 255, 0)  # BGR — foreground
BLUE = (255, 0, 0)  # BGR — boundary


def test_decode_assigns_each_colour_to_its_class():
    mask = np.zeros((1, 6, 3), dtype=np.uint8)
    mask[0, 0] = RED
    mask[0, 1] = GREEN
    mask[0, 2] = BLUE
    mask[0, 3] = (0, 0, 0)  # black: nothing dominates
    mask[0, 4] = (100, 100, 100)  # grey: three-way tie
    mask[0, 5] = (0, 30, 0)  # green, but under min_channel

    decoded = masks.decode_rgb_mask(mask, min_channel=50, ignore_index=255)
    assert decoded.tolist() == [
        [masks.BACKGROUND, masks.FOREGROUND, masks.BOUNDARY, 255, 255, 255]
    ]


def test_encode_decode_round_trips_including_ignored_pixels():
    class_map = np.array(
        [[masks.BACKGROUND, masks.FOREGROUND], [masks.BOUNDARY, 255]], dtype=np.uint8
    )
    round_tripped = masks.decode_rgb_mask(masks.encode_class_map(class_map))
    assert np.array_equal(round_tripped, class_map)


def test_split_channels_returns_disjoint_binary_masks():
    """Step 1 of extraction is free because argmax outputs cannot overlap."""
    class_map = np.array([[masks.FOREGROUND, masks.BOUNDARY, masks.BACKGROUND]], np.uint8)
    foreground, boundary = masks.split_channels(masks.encode_class_map(class_map))

    assert foreground.tolist() == [[255, 0, 0]]
    assert boundary.tolist() == [[0, 255, 0]]
    assert not np.any(cv2.bitwise_and(foreground, boundary))


def test_downsample_votes_rather_than_samples():
    """A one-pixel boundary must not decide a patch, but a majority must."""
    class_map = np.full((32, 32), masks.BACKGROUND, dtype=np.uint8)
    class_map[0:16, 0:16] = masks.FOREGROUND
    class_map[16:32, 16:32] = 255  # wholly unlabelled block
    class_map[0, 20] = masks.BOUNDARY  # lone pixel in a background block

    patches = masks.downsample_to_patches(class_map, 2, 2)
    assert patches[0, 0] == masks.FOREGROUND
    assert patches[0, 1] == masks.BACKGROUND  # the single boundary pixel lost
    assert patches[1, 1] == 255  # no valid pixel at all


def test_upsample_expands_patches_without_inventing_classes():
    patches = np.array([[masks.BACKGROUND, masks.BOUNDARY]], dtype=np.uint8)
    full = masks.upsample_from_patches(patches, 32, 32)

    assert full.shape == (32, 32)
    assert set(np.unique(full)) <= {masks.BACKGROUND, masks.BOUNDARY}


# --------------------------------------------------------------------------
# data/tiling.py
# --------------------------------------------------------------------------


def test_tile_starts_shifts_the_last_tile_inward():
    assert tiling.tile_starts(100, 512, 384) == [0]  # smaller than one tile
    assert tiling.tile_starts(512, 512, 384) == [0]  # exactly one tile
    starts = tiling.tile_starts(1000, 512, 384)
    assert starts[0] == 0
    assert starts[-1] + 512 == 1000  # flush with the edge, never padded


def test_tiles_cover_every_pixel_and_are_always_full_size(cfg):
    height, width = 1000, 700
    tile_size = cfg.stage3_instances.tiling.tile_size
    overlap = cfg.stage3_instances.tiling.overlap

    covered = np.zeros((height, width), dtype=np.int32)
    for y0, y1, x0, x1 in tiling.iter_tiles(height, width, tile_size, overlap):
        assert (y1 - y0, x1 - x0) == (tile_size, tile_size)
        covered[y0:y1, x0:x1] += 1

    assert covered.min() >= 1


def test_gaussian_window_peaks_at_the_centre_and_never_reaches_zero():
    window = tiling.gaussian_weight_map(65, sigma=0.5)  # odd, so a pixel sits at 0

    assert window.shape == (65, 65)
    assert window.min() > 0  # a pixel seen by one tile must still be usable
    assert window[32, 32] == pytest.approx(1.0)
    assert window.max() == window[32, 32]
    assert np.allclose(window, window[::-1, ::-1])


def test_blending_a_constant_field_returns_that_constant():
    """The weight normalisation must not tint the seams."""
    height, width, tile_size = 300, 260, 128
    window = tiling.gaussian_weight_map(tile_size, sigma=0.5)
    blender = tiling.TileBlender(height, width, 3, window)

    for bounds in tiling.iter_tiles(height, width, tile_size, 32):
        y0, y1, x0, x1 = bounds
        blender.add(np.full((3, y1 - y0, x1 - x0), 7.0, dtype=np.float32), bounds)

    assert np.allclose(blender.result(), 7.0)


# --------------------------------------------------------------------------
# stage3_instances/extraction.py
# --------------------------------------------------------------------------

CENTRE_Y = 110
NECK_ROWS = slice(107, 114)
NECK_COLS = slice(100, 140)


def dumbbell() -> tuple[np.ndarray, np.ndarray]:
    """Two disks joined by a thin neck, with boundary evidence across the neck.

    Returns ``(mask_bgr, image_bgr)``. The boundary bars sit in the gap between
    the disks, 3-6 px from the neck, which is inside the severing annulus.
    """
    class_map = np.full((220, 260), masks.BACKGROUND, dtype=np.uint8)
    cv2.circle(class_map, (70, CENTRE_Y), 38, int(masks.FOREGROUND), -1)
    cv2.circle(class_map, (170, CENTRE_Y), 38, int(masks.FOREGROUND), -1)
    class_map[NECK_ROWS, NECK_COLS] = masks.FOREGROUND
    class_map[103:106, 112:129] = masks.BOUNDARY
    class_map[115:118, 112:129] = masks.BOUNDARY

    mask_bgr = masks.encode_class_map(class_map)
    image_bgr = np.full((220, 260, 3), 240, dtype=np.uint8)
    image_bgr[class_map == masks.FOREGROUND] = 40
    return mask_bgr, image_bgr


def component_count(binary: np.ndarray) -> int:
    return cv2.connectedComponents(binary)[0] - 1


def test_severing_cuts_the_neck_the_network_left_connected(cfg):
    mask_bgr, _ = dumbbell()
    foreground, boundary = masks.split_channels(mask_bgr)
    pinch = cfg.stage3_instances.extraction.pinch_severing

    assert component_count(foreground) == 1  # joined before severing

    severed = extraction.sever_pinch_points(
        foreground,
        boundary,
        dt_max=pinch.dt_max,
        radius=pinch.radius,
        min_sides=pinch.min_sides,
        sever_dilation=pinch.sever_dilation,
    )
    assert severed[CENTRE_Y, 120] == 0
    assert component_count(severed) == 2


def test_severing_leaves_thin_structure_alone_without_boundary_evidence(cfg):
    """A limb is thin too; only boundary on >= 2 sides justifies a cut."""
    mask_bgr, _ = dumbbell()
    foreground, _ = masks.split_channels(mask_bgr)
    no_boundary = np.zeros_like(foreground)

    severed = extraction.sever_pinch_points(foreground, no_boundary)
    assert np.array_equal(severed, foreground)


def test_build_foreground_subtracts_a_dilated_boundary_wall():
    mask_bgr, _ = dumbbell()
    foreground, boundary = masks.split_channels(mask_bgr)

    refined, wall = extraction.build_foreground(foreground, boundary)
    assert wall.sum() > boundary.sum()  # opened, then dilated wider than input
    assert not np.any(cv2.bitwise_and(refined, wall))


def test_extraction_splits_a_dumbbell_into_two_instances(cfg):
    mask_bgr, image_bgr = dumbbell()
    labels = extraction.extract_instances(mask_bgr, image_bgr, cfg)

    assert labels.dtype == np.int32
    assert labels.max() == 2
    assert sorted(np.unique(labels)) == [0, 1, 2]  # contiguous ids, 0 = background
    for i in (1, 2):
        assert (labels == i).sum() >= cfg.stage3_instances.extraction.min_instance_area


def test_a_lone_organism_stays_one_instance(cfg):
    class_map = np.full((160, 160), masks.BACKGROUND, dtype=np.uint8)
    cv2.circle(class_map, (80, 80), 30, int(masks.FOREGROUND), -1)
    image_bgr = np.full((160, 160, 3), 240, dtype=np.uint8)
    image_bgr[class_map == masks.FOREGROUND] = 40

    labels = extraction.extract_instances(masks.encode_class_map(class_map), image_bgr, cfg)
    assert labels.max() == 1


def test_specks_below_the_area_floor_are_discarded(cfg):
    class_map = np.full((80, 80), masks.BACKGROUND, dtype=np.uint8)
    cv2.circle(class_map, (40, 40), 5, int(masks.FOREGROUND), -1)  # ~100 px^2 < 150
    image_bgr = np.full((80, 80, 3), 240, dtype=np.uint8)

    labels = extraction.extract_instances(masks.encode_class_map(class_map), image_bgr, cfg)
    assert labels.max() == 0


def test_an_empty_prediction_yields_no_instances(cfg):
    class_map = np.full((64, 64), masks.BACKGROUND, dtype=np.uint8)
    image_bgr = np.full((64, 64, 3), 240, dtype=np.uint8)

    labels = extraction.extract_instances(masks.encode_class_map(class_map), image_bgr, cfg)
    assert not labels.any()


# --------------------------------------------------------------------------
# eval/panoptic.py
# --------------------------------------------------------------------------


def labelled(boxes: list[tuple[int, int, int, int]], shape=(64, 64)) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    for i, (y0, y1, x0, x1) in enumerate(boxes, 1):
        labels[y0:y1, x0:x1] = i
    return labels


def test_a_perfect_prediction_scores_one_everywhere():
    pred = labelled([(0, 10, 0, 10), (20, 30, 20, 30)])
    gt = panoptic.instance_masks(pred)

    metrics = panoptic.panoptic_quality(pred, gt, 0.5)
    for name in ("pq", "sq", "rq", "precision", "recall", "osr"):
        assert metrics[name] == pytest.approx(1.0)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 0, 0)


def test_a_spurious_instance_costs_precision_but_not_recall():
    """Hand-checked: tp=2, fp=1, fn=0 -> rq = 2 / (2 + 0.5) = 0.8."""
    gt = panoptic.instance_masks(labelled([(0, 10, 0, 10), (20, 30, 20, 30)]))
    pred = labelled([(0, 10, 0, 10), (20, 30, 20, 30), (40, 50, 40, 50)])

    metrics = panoptic.panoptic_quality(pred, gt, 0.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 1, 0)
    assert metrics["sq"] == pytest.approx(1.0)
    assert metrics["rq"] == pytest.approx(0.8)
    assert metrics["pq"] == pytest.approx(0.8)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["osr"] == pytest.approx(1.5)


def test_pq_is_the_product_of_sq_and_rq():
    gt = panoptic.instance_masks(labelled([(0, 10, 0, 10), (20, 30, 20, 30)]))
    pred = labelled([(0, 10, 0, 9), (40, 50, 40, 50)])  # one sloppy, one missed

    metrics = panoptic.panoptic_quality(pred, gt, 0.5)
    assert metrics["pq"] == pytest.approx(metrics["sq"] * metrics["rq"])
    assert 0.5 <= metrics["sq"] < 1.0


def test_an_overlap_below_threshold_is_not_a_match():
    gt = panoptic.instance_masks(labelled([(0, 10, 0, 10)]))
    pred = labelled([(0, 10, 0, 4)])  # IoU = 40 / 100

    assert panoptic.panoptic_quality(pred, gt, 0.5)["tp"] == 0


def test_matching_is_one_to_one_even_at_the_degenerate_threshold():
    """One organism split in half: both halves sit at IoU exactly 0.5.

    This is the only case where one-to-one matching is not forced by the
    threshold alone, and it is precisely the over-segmentation failure the
    paper cares about. Exactly one half may be credited.
    """
    gt = panoptic.instance_masks(labelled([(0, 20, 0, 20)]))
    pred = labelled([(0, 10, 0, 20), (10, 20, 0, 20)])

    matrix = panoptic.iou_matrix(panoptic.instance_masks(pred), gt)
    assert matrix.ravel().tolist() == pytest.approx([0.5, 0.5])
    assert len(panoptic.match_instances(matrix, 0.5)) == 1

    metrics = panoptic.panoptic_quality(pred, gt, 0.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 0)
    assert metrics["osr"] == pytest.approx(2.0)  # over-segmented


def test_an_empty_prediction_scores_zero_not_nan():
    gt = panoptic.instance_masks(labelled([(0, 10, 0, 10)]))
    metrics = panoptic.panoptic_quality(np.zeros((64, 64), np.int32), gt, 0.5)

    assert metrics["pq"] == 0.0
    assert metrics["osr"] == 0.0
    assert metrics["fn"] == 1


def test_macro_average_weights_every_crop_equally():
    crowded = {name: 1.0 for name in panoptic.METRIC_NAMES}
    sparse = {name: 0.0 for name in panoptic.METRIC_NAMES}

    averaged = panoptic.macro_average([crowded, sparse])
    assert averaged["pq"] == pytest.approx(0.5)
    assert set(averaged) == set(panoptic.METRIC_NAMES)


def test_ground_truth_loads_from_a_labelme_polygon(tmp_path):
    annotation = {
        "shapes": [
            {"label": "copepod", "shape_type": "polygon",
             "points": [[10, 10], [30, 10], [30, 30], [10, 30]]},
            {"label": "detritus", "shape_type": "rectangle",
             "points": [[40, 40], [50, 50]]},
        ]
    }
    path = tmp_path / "crop_01.json"
    path.write_text(json.dumps(annotation))

    loaded = panoptic.load_ground_truth(path, (64, 64))
    assert len(loaded) == 2  # labels ignored: evaluation is class-agnostic
    assert all(mask.dtype == bool and mask.any() for mask in loaded)


def test_ground_truth_loads_from_a_label_image(tmp_path):
    path = tmp_path / "crop_01.npy"
    np.save(path, labelled([(0, 10, 0, 10), (20, 30, 20, 30)]))

    assert len(panoptic.load_ground_truth(path, (64, 64))) == 2


def test_markdown_row_matches_the_readme_column_order():
    results = {name: 0.5 for name in panoptic.METRIC_NAMES}
    table = panoptic.format_markdown(results, "ours").splitlines()

    assert table[0] == "| Method | PQ | SQ | RQ | Precision | Recall | OSR |"
    assert table[2].count("0.500") == len(panoptic.METRIC_NAMES)
