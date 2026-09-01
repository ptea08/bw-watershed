"""CPU-only smoke tests for the torch half of the pipeline.

Nothing here downloads a checkpoint. The decoder is driven from synthetic
feature maps through ``ConvNeXtUNet(cfg, encoder_channels=...)``, which exists
for exactly this reason: the DINOv3 weights are gated, so a test that needed
them could not run in CI or on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bootstrapped_watershed.config import class_weights, load_config  # noqa: E402
from bootstrapped_watershed.data import masks  # noqa: E402
from bootstrapped_watershed.stage1_bootstrap import classifier as clf  # noqa: E402
from bootstrapped_watershed.stage1_bootstrap import selection  # noqa: E402
from bootstrapped_watershed.stage2_segmenter import losses  # noqa: E402
from bootstrapped_watershed.stage2_segmenter.decoder import (  # noqa: E402
    ConvNeXtUNet,
    UNetDecoder,
)


@pytest.fixture
def cfg():
    return load_config()


def one_hot_logits(target: torch.Tensor, num_classes: int = 3, scale: float = 20.0):
    """Logits that predict ``target`` with near-certainty."""
    valid = target.clone()
    valid[target == 255] = 0
    return torch.nn.functional.one_hot(valid, num_classes).permute(0, 3, 1, 2).float() * scale


# --------------------------------------------------------------------------
# stage2_segmenter/losses.py
# --------------------------------------------------------------------------


def test_ce_weights_reach_the_loss_in_class_index_order(cfg):
    """The heavy weight must land on boundary, not on foreground."""
    loss = losses.CombinedLoss(cfg)

    assert loss.class_weights.tolist() == class_weights(cfg)
    assert loss.class_weights[masks.BOUNDARY].item() == 8.0
    assert loss.class_weights[masks.FOREGROUND].item() == 1.5
    assert loss.class_weights[masks.BACKGROUND].item() == 1.0


def test_tversky_rewards_a_correct_prediction_and_punishes_an_inverted_one():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, :4] = masks.FOREGROUND
    tversky = losses.TverskyLoss(alpha=0.7, beta=0.3)

    assert tversky(one_hot_logits(target), target).item() < 0.01
    inverted = one_hot_logits((target + 1) % 3)
    assert tversky(inverted, target).item() > 0.9


def test_tversky_asymmetry_penalises_misses_more_than_false_alarms():
    """alpha > beta, so missing an organism costs more than inventing one.

    The target is deliberately imbalanced (20 % foreground, as a ZooScan tile
    roughly is). On a 50/50 target the two errors cancel across the macro
    average and the asymmetry is invisible.
    """
    target = torch.zeros(1, 10, 10, dtype=torch.long)
    target[0, :2] = masks.FOREGROUND
    tversky = losses.TverskyLoss(alpha=0.7, beta=0.3)

    missed = target.clone()
    missed[0, 1] = masks.BACKGROUND  # half the organism not predicted
    spurious = target.clone()
    spurious[0, 2] = masks.FOREGROUND  # background predicted as organism

    missed_loss = tversky(one_hot_logits(missed), target)
    spurious_loss = tversky(one_hot_logits(spurious), target)
    assert missed_loss > spurious_loss


def test_ignored_pixels_contribute_nothing():
    target = torch.full((1, 8, 8), 255, dtype=torch.long)
    logits = torch.randn(1, 3, 8, 8)

    assert losses.TverskyLoss()(logits, target).item() == 0.0


def test_the_combined_loss_backpropagates(cfg):
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randint(0, 3, (2, 16, 16))
    target[0, 0, 0] = cfg.data.ignore_index

    loss = losses.CombinedLoss(cfg)(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_per_class_iou_reports_nan_for_an_absent_class():
    """A class absent from both prediction and truth is unmeasured, not perfect."""
    preds = torch.zeros(1, 4, 4, dtype=torch.long)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    preds[0, 0] = masks.FOREGROUND
    target[0, 0] = masks.FOREGROUND

    ious = losses.per_class_iou(preds, target, num_classes=3)
    assert ious[masks.BACKGROUND] == pytest.approx(1.0)
    assert ious[masks.FOREGROUND] == pytest.approx(1.0)
    assert np.isnan(ious[masks.BOUNDARY])


# --------------------------------------------------------------------------
# stage1_bootstrap
# --------------------------------------------------------------------------


def test_the_shallow_classifier_is_the_published_384_128_3(cfg):
    model = clf.build_classifier(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    assert model(torch.randn(7, 384)).shape == (7, 3)
    assert 45_000 < n_params < 55_000  # paper says "roughly 50 000"


def test_the_deep_ablation_has_more_capacity_than_the_shallow_one(cfg):
    shallow = clf.build_classifier(cfg, "mlp_shallow")
    deep = clf.build_classifier(cfg, "mlp_deep")

    assert deep(torch.randn(7, 384)).shape == (7, 3)
    assert sum(p.numel() for p in deep.parameters()) > sum(
        p.numel() for p in shallow.parameters()
    )


def test_an_unknown_classifier_type_fails_loudly(cfg):
    with pytest.raises(ValueError):
        clf.build_classifier(cfg, "transformer")


def test_predict_proba_returns_a_normalised_distribution(cfg):
    model = clf.build_classifier(cfg)
    probs = clf.predict_proba(model, np.random.randn(5, 384), torch.device("cpu"))

    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_class_priors_bias_towards_organisms_and_stay_normalised(cfg):
    uniform = np.full((2, 2, 3), 1 / 3, dtype=np.float32)
    weighted = selection.apply_class_priors(uniform, cfg)

    assert np.allclose(weighted.sum(axis=-1), 1.0)
    assert weighted[..., masks.FOREGROUND].max() > 1 / 3
    assert weighted[..., masks.BACKGROUND].max() < 1 / 3
    assert weighted[0, 0].argmax() == masks.FOREGROUND


def test_crop_confidence_is_the_mean_max_probability():
    decisive = np.array([[[0.9, 0.05, 0.05]], [[0.9, 0.05, 0.05]]], dtype=np.float32)
    uncertain = np.full((2, 1, 3), 1 / 3, dtype=np.float32)

    assert selection.crop_confidence(decisive) == pytest.approx(0.9)
    assert selection.crop_confidence(uncertain) == pytest.approx(1 / 3)
    assert selection.crop_confidence(decisive) > selection.crop_confidence(uncertain)


def test_selection_keeps_everything_above_the_threshold_and_orders_by_confidence():
    scores = {"c": 0.9, "a": 0.5, "b": 0.9, "d": 0.1}

    assert selection.select_bootstrap_crops(scores, 0.7) == ["b", "c"]
    assert selection.select_bootstrap_crops(scores, 0.05) == ["b", "c", "a", "d"]
    # Ties break on the identifier, so "b" precedes "c" at equal confidence.
    assert selection.select_bootstrap_crops(scores, 0.4) == ["b", "c", "a"]


def test_the_threshold_is_exclusive_and_can_reject_everything():
    scores = {"a": 0.5, "b": 0.5}

    assert selection.select_bootstrap_crops(scores, 0.5) == []
    assert selection.select_bootstrap_crops(scores, 0.9) == []


def test_no_threshold_keeps_every_candidate():
    """`None` is the escape hatch that lets stage 1 run so ranking.csv exists."""
    scores = {"a": 0.5, "b": 0.9, "c": 0.1}

    assert selection.select_bootstrap_crops(scores, None) == ["b", "a", "c"]
    assert selection.select_bootstrap_crops(scores) == ["b", "a", "c"]


def test_the_bootstrap_split_reproduces_the_published_66_16(cfg):
    """82 is the paper's outcome, not a setting — hence a literal here.

    The property under test is that `split_bootstrap` is a *fraction*, so it
    still divides correctly for whatever number of crops clears a user's
    threshold. Feeding it the paper's 82 should reproduce the quoted 66/16.
    """
    crop_ids = [f"crop_{i:03d}" for i in range(82)]
    train, val = selection.split_bootstrap(
        crop_ids, cfg.stage2_segmenter.optim.val_split, cfg.seed
    )

    assert len(train) == 66
    assert len(val) == 16
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(crop_ids)
    assert selection.split_bootstrap(crop_ids, 0.2, cfg.seed) == (train, val)


def test_the_split_holds_for_pool_sizes_other_than_the_papers(cfg):
    for n in (5, 40, 137):
        crop_ids = [f"crop_{i:03d}" for i in range(n)]
        train, val = selection.split_bootstrap(crop_ids, 0.2, cfg.seed)

        assert len(val) == max(1, int(n * 0.2))
        assert len(train) + len(val) == n
        assert set(train).isdisjoint(val)


# --------------------------------------------------------------------------
# stage2_segmenter/decoder.py
# --------------------------------------------------------------------------


def synthetic_features(base: int = 16, channels=(8, 16, 32, 64)):
    """Four feature maps at strides 4/8/16/32, as ConvNeXt would emit them."""
    return [
        torch.randn(1, c, base // (2**i), base // (2**i)) for i, c in enumerate(channels)
    ]


def test_the_decoder_returns_logits_at_input_resolution():
    decoder = UNetDecoder([8, 16, 32, 64], decoder_channels=(16, 8, 8, 4), num_classes=3)
    s1, s2, s3, s4 = synthetic_features(base=16)

    with torch.no_grad():
        out = decoder(s1, s2, s3, s4)
    assert out.shape == (1, 3, 64, 64)  # stride-4 skip, upsampled x4


def test_the_decoder_corrects_for_a_non_divisible_input_size():
    """A 60 px input cannot have an exactly-H/4 skip; the output must still fit."""
    decoder = UNetDecoder([8, 16, 32, 64], decoder_channels=(16, 8, 8, 4), num_classes=3)
    s1, s2, s3, s4 = synthetic_features(base=16)

    with torch.no_grad():
        out = decoder(s1, s2, s3, s4, target_hw=(60, 60))
    assert out.shape == (1, 3, 60, 60)


def test_the_decoder_can_be_built_without_touching_the_gated_hub(cfg, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    model = ConvNeXtUNet(cfg, encoder_channels=[8, 16, 32, 64])

    assert model.backbone is None
    assert list(model.trainable_parameters())  # decoder params exist and are trainable
    assert all(p.requires_grad for p in model.trainable_parameters())


def test_the_decoder_predicts_one_channel_per_class(cfg):
    model = ConvNeXtUNet(cfg, encoder_channels=[8, 16, 32, 64]).eval()
    s1, s2, s3, s4 = synthetic_features(base=16)

    with torch.no_grad():
        out = model.decode_head(s1, s2, s3, s4, target_hw=(64, 64))
    assert out.shape[1] == cfg.data.num_classes == len(cfg.data.class_names)
