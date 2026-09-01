"""Guard the published constants against silent drift.

Every value asserted here is quoted in the paper. If someone tunes a threshold
in `configs/default.yaml`, this test fails and forces the README's results
table to be revisited. Runs in milliseconds, needs no GPU and no data.
"""

import re

import pytest

from bootstrapped_watershed.config import class_index, class_weights, load_config
from bootstrapped_watershed.data import masks


@pytest.fixture
def cfg():
    return load_config()


def test_bootstrap_constants(cfg):
    assert cfg.stage1_bootstrap.classifier.hidden_dim == 128


def test_no_dataset_specific_counts_are_configured(cfg):
    """Counts from our data must not creep back into the shipped config.

    `N_ann = 3` and `N_boot = 82` describe one dataset. Anyone else's counts
    come from their own directories, and stage 1 selects by confidence
    threshold rather than by rank, so a count is not an input at all. See
    docs/TUNING.md.
    """
    assert "n_annotated" not in cfg.stage1_bootstrap
    assert "n_test_crops" not in cfg.data
    assert "test_crop_size" not in cfg.data

    selection = cfg.stage1_bootstrap.selection
    assert "n_bootstrap" not in selection
    assert "train_split" not in selection
    assert "val_split" not in selection
    assert "confidence_threshold" in selection


def test_the_shipped_threshold_is_unset(cfg):
    """No threshold transfers between datasets, so the default must be null.

    A plausible-looking number here would be worse than nothing: it would be
    silently wrong on every dataset but ours. `bootstrap.py` warns instead.
    """
    assert cfg.stage1_bootstrap.selection.confidence_threshold is None


def test_backbones_are_frozen_and_384d(cfg):
    stage1 = cfg.stage1_bootstrap.backbone
    assert stage1.frozen is True
    assert stage1.feature_dim == 384
    # Both stages read the same ConvNeXt checkpoint: stage 1 takes stage3
    # alone (384-d @ stride 16), stage 2 takes all four for the U-Net skips.
    assert "convnext" in stage1.name
    assert stage1.name == cfg.stage2_segmenter.backbone.name
    assert list(stage1.out_features) == ["stage3"]
    assert stage1.stride == 16
    assert cfg.stage2_segmenter.backbone.frozen is True


def test_class_index_order_matches_the_code(cfg):
    """`class_names` order IS the argmax order. Drift here corrupts everything."""
    assert list(cfg.data.class_names) == ["background", "foreground", "boundary"]
    assert class_index(cfg, "background") == masks.BACKGROUND
    assert class_index(cfg, "foreground") == masks.FOREGROUND
    assert class_index(cfg, "boundary") == masks.BOUNDARY


def test_loss_constants(cfg):
    tversky = cfg.stage2_segmenter.loss.tversky
    assert (tversky.alpha, tversky.beta) == (0.7, 0.3)
    assert abs(tversky.alpha + tversky.beta - 1.0) < 1e-9

    weights = cfg.data.class_weights
    assert weights["background"] == 1.0
    assert weights["boundary"] == 8.0
    assert weights["foreground"] == 1.5
    # Boundary must stay the dominant term; it carries the separation signal.
    assert weights["boundary"] > weights["foreground"] > weights["background"]


def test_class_weight_tensor_follows_class_names(cfg):
    """The ordered tensor is derived, never written down twice."""
    ordered = class_weights(cfg)
    assert ordered == [1.0, 1.5, 8.0]
    assert ordered[masks.BOUNDARY] == 8.0


def test_tiling_constants(cfg):
    tiling = cfg.stage3_instances.tiling
    assert tiling.tile_size == 512
    assert tiling.overlap == 128
    assert 0 < tiling.overlap < tiling.tile_size
    # ConvNeXt strides by 32; a tile that is not a multiple would be padded and
    # the blending window would no longer line up with the prediction.
    assert tiling.tile_size % cfg.stage2_segmenter.decoder.stride == 0


def test_extraction_constants(cfg):
    ext = cfg.stage3_instances.extraction
    assert ext.pinch_severing.dt_min == 0
    assert ext.pinch_severing.dt_max == 12
    assert ext.pinch_severing.radius == 8
    assert ext.pinch_severing.min_sides == 2
    assert ext.closing_kernel == 5
    assert ext.watershed.smoothing_kernel == 9
    assert ext.watershed.marker_threshold == 5
    assert ext.min_instance_area == 150
    # Morphological kernels must be odd to have a well-defined centre.
    assert ext.closing_kernel % 2 == 1
    assert ext.watershed.smoothing_kernel % 2 == 1
    assert all(k % 2 == 1 for k in ext.boundary_dilation)


def test_eval_constants(cfg):
    assert cfg.eval.iou_threshold == 0.5
    assert cfg.eval.aggregate == "macro"
    # Above 0.5, one-to-one IoU matching is provably unique.
    assert cfg.eval.iou_threshold >= 0.5


def _overlay(tmp_path, text):
    path = tmp_path / "overlay.yaml"
    path.write_text(text)
    return path


def test_an_overlay_changes_only_what_it_names(tmp_path, cfg):
    """The whole point of the overlay: a user config is small and stays small.

    If omitted keys did not survive, every user would have to fork all 200
    lines of default.yaml and their copy would rot at the next update.
    """
    merged = load_config(
        _overlay(
            tmp_path,
            "data:\n"
            "  class_weights:\n"
            "    boundary: 4.0\n",
        )
    )

    assert merged.data.class_weights.boundary == 4.0
    # Siblings at every level above and beside the changed key are untouched.
    assert merged.data.class_weights.background == cfg.data.class_weights.background
    assert merged.data.class_weights.foreground == cfg.data.class_weights.foreground
    assert merged.data.crop_size == cfg.data.crop_size
    assert merged.stage2_segmenter.loss.tversky.alpha == 0.7


def test_an_overlay_reaches_arbitrarily_deep(tmp_path, cfg):
    merged = load_config(
        _overlay(
            tmp_path,
            "stage3_instances:\n"
            "  extraction:\n"
            "    pinch_severing:\n"
            "      dt_max: 24\n",
        )
    )

    severing = merged.stage3_instances.extraction.pinch_severing
    assert severing.dt_max == 24
    assert severing.radius == cfg.stage3_instances.extraction.pinch_severing.radius
    assert merged.stage3_instances.extraction.min_instance_area == 150


def test_a_list_is_replaced_not_concatenated(tmp_path):
    """`boundary_dilation: [3, 5]` is one setting, not a list to append to."""
    merged = load_config(
        _overlay(
            tmp_path,
            "stage3_instances:\n"
            "  extraction:\n"
            "    boundary_dilation: [5, 7]\n",
        )
    )

    assert merged.stage3_instances.extraction.boundary_dilation == [5, 7]


def test_an_empty_overlay_is_not_an_error(tmp_path, cfg):
    assert load_config(_overlay(tmp_path, "")).data.crop_size == cfg.data.crop_size


def test_a_missing_overlay_fails_loudly(tmp_path):
    """Silently ignoring a typo'd --config would run the paper's settings.

    The user would see a clean run and never learn their tuning was discarded.
    """
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_no_paths_or_secrets_leaked_into_the_config(cfg):
    """Paths belong on the command line; tokens belong in the environment.

    The config is the one file everyone edits, so it is the most likely place
    for a notebook path or a pasted access token to survive into a public
    repository. Matched by shape rather than by name — ``requires_hf_token``
    is a legitimate flag, ``hf_xxxxxxxx...`` is a leaked credential.
    """
    flat = repr(cfg)
    assert not re.search(r"/(kaggle|content|mnt|home)/", flat)
    assert not re.search(r"[A-Z]:\\\\", flat)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", flat)
