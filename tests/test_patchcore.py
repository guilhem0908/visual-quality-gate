"""PatchCore scoring checked against brute-force nearest neighbours on synthetic features."""

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

from vqgate.patchcore import PatchCoreDetector, coreset_order, patch_embeddings


def features(n_images, n_patches, dims, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_images, n_patches, dims, generator=generator)


def test_patch_scores_equal_brute_force_nearest_neighbour_distance():
    memory = features(1, 500, 12, seed=0)[0]
    queries = features(3, 16, 12, seed=1)
    scores = PatchCoreDetector(memory).patch_scores(queries).numpy()
    expected = cdist(queries.reshape(-1, 12).numpy(), memory.numpy()).min(axis=1).reshape(3, 16)
    np.testing.assert_allclose(scores, expected, rtol=1e-4, atol=1e-4)


def test_patches_stored_in_the_memory_bank_score_zero():
    memory = features(1, 64, 20, seed=2)[0]
    scores = PatchCoreDetector(memory).patch_scores(memory.view(4, 16, 20))
    assert scores.max().item() < 5e-3 * memory.norm(dim=1).mean().item()


def test_scores_do_not_change_when_feature_space_is_translated():
    memory = features(1, 300, 10, seed=3)[0]
    queries = features(2, 9, 10, seed=4)
    shift = torch.full((10,), 3.0)
    plain = PatchCoreDetector(memory).patch_scores(queries)
    shifted = PatchCoreDetector(memory + shift).patch_scores(queries + shift)
    torch.testing.assert_close(shifted, plain, atol=1e-3, rtol=1e-3)


def test_image_score_is_the_largest_patch_distance_and_the_map_peaks_there():
    memory = 0.1 * features(1, 400, 8, seed=5)[0]
    queries = 0.1 * features(1, 49, 8, seed=6)
    planted = 3 * 7 + 5  # row 3, column 5 of the 7 x 7 grid
    queries[0, planted] += 4.0
    detector = PatchCoreDetector(memory)
    patch_scores = detector.patch_scores(queries)
    image_scores, maps = detector.score_maps(patch_scores, out_side=56)
    assert image_scores[0].item() == pytest.approx(patch_scores.max().item())
    assert patch_scores.argmax().item() == planted
    peak_row, peak_col = np.unravel_index(maps[0].argmax().item(), (56, 56))
    assert abs(peak_row - (3 + 0.5) * 8) <= 4 and abs(peak_col - (5 + 0.5) * 8) <= 4


def test_embeddings_average_a_3x3_neighbourhood_and_join_two_layers():
    first = torch.zeros(1, 2, 16, 16)
    second = torch.zeros(1, 3, 8, 8)
    second[0, 0, 4, 4] = 9.0
    third = torch.ones(1, 5, 8, 8)
    third[0, :, 0, 0] = 10.0
    embeddings = patch_embeddings((first, second, third))
    assert embeddings.shape == (1, 64, 8)
    grid = embeddings[0].T.reshape(8, 8, 8)
    assert grid[0, 3:6, 3:6].eq(1.0).all() and grid[0].sum().item() == pytest.approx(9.0)
    assert grid[3, 4, 4].item() == pytest.approx(1.0)
    # at a corner the 3 x 3 mean still divides by 9: the zero padding counts
    assert grid[3, 0, 0].item() == pytest.approx((10.0 + 3 * 1.0) / 9.0)


def test_coarser_layer_is_upsampled_to_the_grid_of_the_finer_one():
    second = torch.zeros(2, 4, 12, 12)
    third = torch.full((2, 6, 6, 6), 2.0)
    embeddings = patch_embeddings((torch.zeros(2, 1, 24, 24), second, third))
    assert embeddings.shape == (2, 144, 10)
    from_third = embeddings.reshape(2, 12, 12, 10)[..., 4:]
    # away from the border the pooled coarse map is constant, and so is its bilinear upsampling
    torch.testing.assert_close(from_third[:, 3:9, 3:9], torch.full((2, 6, 6, 6), 2.0))
    # the corner keeps the pooled corner value: 4 of the 9 neighbours are inside the map
    assert from_third[0, 0, 0, 0].item() == pytest.approx(2.0 * 4 / 9)


def test_coreset_order_is_reproducible_and_nested():
    bank = features(1, 2000, 64, seed=7)[0]
    first = coreset_order(bank, 200, torch.Generator().manual_seed(1))
    again = coreset_order(bank, 200, torch.Generator().manual_seed(1))
    shorter = coreset_order(bank, 20, torch.Generator().manual_seed(1))
    assert torch.equal(first, again)
    assert torch.equal(first[:20], shorter)
    assert first.unique().numel() == 200
