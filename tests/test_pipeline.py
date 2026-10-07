"""End-to-end checks on synthetic parts with a randomly initialised ResNet-18 (no download)."""

import numpy as np
import pytest
import torch

from conftest import DEFECT_BOX
from vqgate.backbone import FrozenBackbone
from vqgate.experiment import (
    calibration_split,
    coreset_size,
    extract_layers,
    extract_patchcore_features,
    fit_padim,
    fit_patchcore,
    score_padim,
    score_patchcore,
)
from vqgate.gate import conformal_threshold, gate_counts
from vqgate.inspector import Inspector
from vqgate.metrics import auroc


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    return FrozenBackbone("resnet18", pretrained=False)


@pytest.fixture(scope="module")
def padim_inspector(backbone, synthetic_parts):
    layers = extract_layers(backbone, synthetic_parts["train"], "cpu")
    return Inspector(backbone, fit_padim(layers, seed=0, device="cpu"))


@pytest.fixture(scope="module")
def patchcore_inspector(backbone, synthetic_parts):
    features = extract_patchcore_features(backbone, synthetic_parts["train"], "cpu").flatten(0, 1)
    return Inspector(backbone, fit_patchcore(features, [0.1], seed=0, device="cpu")[0])


def test_backbone_returns_maps_at_strides_4_8_16_and_is_frozen(backbone):
    first, second, third = backbone(torch.zeros(2, 3, 224, 224))
    assert first.shape == (2, 64, 56, 56)
    assert second.shape == (2, 128, 28, 28)
    assert third.shape == (2, 256, 14, 14)
    assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())


@pytest.mark.parametrize("inspector_name", ["padim_inspector", "patchcore_inspector"])
def test_planted_defects_are_ranked_above_good_parts_and_localised(
    inspector_name, synthetic_parts, request
):
    inspector = request.getfixturevalue(inspector_name)
    good_scores, _ = inspector.inspect(synthetic_parts["good"])
    defect_scores, defect_maps = inspector.inspect(synthetic_parts["defective"])
    scores = torch.cat([good_scores, defect_scores]).numpy()
    labels = np.r_[np.zeros(8), np.ones(8)]
    assert auroc(scores, labels) == 1.0
    top, bottom, left, right = DEFECT_BOX
    for anomaly_map in defect_maps:
        row, column = np.unravel_index(anomaly_map.argmax().item(), anomaly_map.shape)
        assert top - 8 <= row < bottom + 8 and left - 8 <= column < right + 8


def test_gate_set_on_held_out_good_parts_rejects_every_planted_defect(
    patchcore_inspector, synthetic_parts
):
    held_out_scores, _ = patchcore_inspector.inspect(synthetic_parts["held_out"])
    threshold = conformal_threshold(held_out_scores.numpy(), alpha=0.05)
    assert threshold == held_out_scores.max().item()  # 19 parts at 5 %: the largest score
    test_images = torch.cat([synthetic_parts["good"], synthetic_parts["defective"]])
    scores, _ = patchcore_inspector.inspect(test_images)
    counts = gate_counts(scores.numpy(), np.r_[np.zeros(8), np.ones(8)], threshold)
    assert counts.escapes == 0 and counts.false_rejects <= 2


def test_inspector_agrees_with_the_benchmark_scoring_path(
    backbone, padim_inspector, patchcore_inspector, synthetic_parts
):
    images = synthetic_parts["defective"]
    layers = extract_layers(backbone, images, "cpu")
    features = extract_patchcore_features(backbone, images, "cpu")
    padim_scores, padim_maps, _ = score_padim(padim_inspector.detector, layers, "cpu")
    patchcore_scores, _, _ = score_patchcore(patchcore_inspector.detector, features, "cpu")
    direct_scores, direct_maps = padim_inspector.inspect(images)
    np.testing.assert_allclose(direct_scores.numpy(), padim_scores, rtol=1e-4)
    np.testing.assert_allclose(direct_maps.numpy(), padim_maps, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        patchcore_inspector.inspect(images)[0].numpy(), patchcore_scores, rtol=1e-4
    )


@pytest.mark.parametrize("inspector_name", ["padim_inspector", "patchcore_inspector"])
def test_saved_detector_gives_the_same_scores_after_reload(
    inspector_name, synthetic_parts, tmp_path, request
):
    inspector = request.getfixturevalue(inspector_name)
    inspector.save(tmp_path / "detector.pt")
    torch.manual_seed(0)  # same random backbone weights as the module fixture
    reloaded = Inspector.load(tmp_path / "detector.pt", pretrained=False)
    expected, _ = inspector.inspect(synthetic_parts["defective"])
    actual, _ = reloaded.inspect(synthetic_parts["defective"])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert reloaded.detector_megabytes() == pytest.approx(inspector.detector_megabytes())


def test_memory_bank_holds_the_requested_fraction_of_patches(patchcore_inspector):
    n_patches = 24 * 28 * 28
    assert patchcore_inspector.detector.memory.shape == (coreset_size(n_patches, 0.1), 128 + 256)
    assert coreset_size(n_patches, 0.1) == 1882 and coreset_size(5, 0.01) == 1


def test_calibration_split_is_a_seeded_partition_with_a_fifth_held_out():
    fit, held_out = calibration_split(209, seed=0)
    again_fit, again_held_out = calibration_split(209, seed=0)
    assert len(held_out) == 42 and len(fit) == 167
    assert sorted(np.r_[fit, held_out].tolist()) == list(range(209))
    assert np.array_equal(fit, again_fit) and np.array_equal(held_out, again_held_out)
    assert not np.array_equal(held_out, calibration_split(209, seed=1)[1])
