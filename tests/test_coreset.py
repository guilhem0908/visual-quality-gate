"""Greedy k-centre selection: approximation guarantee, nesting, coverage, projection."""

from itertools import combinations, pairwise

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from vqgate.coreset import coverage_radius, greedy_k_center, nearest_distance, random_projection


def random_points(n_points, dims, seed):
    return torch.randn(n_points, dims, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("seed", range(5))
def test_greedy_radius_is_within_twice_the_brute_force_optimum(seed):
    points = random_points(14, 2, seed)
    k = 3
    optimum = min(
        coverage_radius(points, points[list(subset)]) for subset in combinations(range(14), k)
    )
    for start in (0, 7, 13):
        greedy = coverage_radius(points, points[greedy_k_center(points, k, start)])
        assert optimum - 1e-6 <= greedy <= 2.0 * optimum + 1e-6


def test_coverage_radius_never_grows_with_more_centres():
    points = random_points(400, 8, seed=1)
    order = greedy_k_center(points, 60, start=0)
    radii = [coverage_radius(points, points[order[:k]]) for k in range(1, 61)]
    assert all(later <= earlier + 1e-6 for earlier, later in pairwise(radii))
    assert radii[-1] < 0.8 * radii[0]


def test_centres_are_at_least_one_coverage_radius_apart():
    points = random_points(500, 5, seed=2)
    centres = points[greedy_k_center(points, 40, start=3)]
    radius = coverage_radius(points, centres)
    pairwise = torch.cdist(centres, centres) + torch.eye(40) * 1e9
    assert pairwise.min().item() >= radius - 1e-4


def test_shorter_run_is_a_prefix_of_a_longer_run():
    points = random_points(300, 16, seed=3)
    long_run = greedy_k_center(points, 50, start=11)
    short_run = greedy_k_center(points, 20, start=11)
    assert torch.equal(long_run[:20], short_run)
    assert long_run[0].item() == 11


def test_one_centre_per_cluster_when_clusters_are_far_apart():
    generator = torch.Generator().manual_seed(4)
    cluster_centres = torch.tensor([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [50.0, 50.0]])
    labels = torch.arange(200) % 4
    points = cluster_centres[labels] + 0.5 * torch.randn(200, 2, generator=generator)
    picked = greedy_k_center(points, 4, start=0)
    assert sorted(labels[picked].tolist()) == [0, 1, 2, 3]


def test_no_point_is_picked_twice_even_with_duplicates():
    points = random_points(30, 3, seed=5).repeat(2, 1)
    picked = greedy_k_center(points, 60, start=0)
    assert sorted(picked.tolist()) == list(range(60))
    with pytest.raises(ValueError):
        greedy_k_center(points, 61, start=0)


def test_nearest_distance_matches_a_kd_tree_across_chunk_boundaries():
    queries = random_points(5000, 6, seed=6)
    references = random_points(700, 6, seed=7)
    expected, _ = cKDTree(references.numpy()).query(queries.numpy())
    np.testing.assert_allclose(nearest_distance(queries, references).numpy(), expected, atol=2e-4)


def test_random_projection_preserves_distances_up_to_johnson_lindenstrauss_distortion():
    features = random_points(200, 1536, seed=8)
    projected = random_projection(features, 128, torch.Generator().manual_seed(0))
    assert projected.shape == (200, 128)
    ratio = torch.pdist(projected) / torch.pdist(features)
    # a distance ratio is chi_k / sqrt(k): mean close to 1, standard deviation close to 1/sqrt(2k)
    assert ratio.mean().item() == pytest.approx(1.0, abs=0.03)
    assert ratio.std().item() == pytest.approx((2 * 128) ** -0.5, rel=0.15)
    assert ratio.min().item() > 0.7 and ratio.max().item() < 1.3
