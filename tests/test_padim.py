"""PaDiM statistics checked against NumPy/SciPy references on synthetic Gaussians."""

import numpy as np
import pytest
import torch
from scipy.spatial.distance import mahalanobis

from vqgate.padim import PatchGaussians, patch_embeddings, random_channel_index


def gaussian_embeddings(n_samples=300, n_patches=5, dims=6, seed=0):
    """Samples x = mu_p + A_p z with a different mean and mixing matrix per patch position."""
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n_patches, dims))
    mixing = rng.normal(size=(n_patches, dims, dims)) + 2.0 * np.eye(dims)
    noise = rng.normal(size=(n_samples, n_patches, dims))
    samples = means + np.einsum("pij,npj->npi", mixing, noise)
    return torch.from_numpy(samples).float()


def test_fit_recovers_numpy_mean_and_ridge_regularised_covariance():
    embeddings = gaussian_embeddings()
    model = PatchGaussians(ridge=0.01).fit(embeddings)
    for patch in range(embeddings.shape[1]):
        samples = embeddings[:, patch].double().numpy()
        expected = np.cov(samples, rowvar=False) + 0.01 * np.eye(samples.shape[1])
        whitener = model.whitener[patch].double().numpy()
        recovered = np.linalg.inv(whitener.T @ whitener)
        np.testing.assert_allclose(model.mean[patch].numpy(), samples.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(recovered, expected, rtol=2e-4, atol=2e-4)


def test_streaming_fit_equals_single_batch_fit():
    embeddings = gaussian_embeddings()
    whole = PatchGaussians().fit(embeddings)
    streamed = PatchGaussians()
    for batch in embeddings.split(37):
        streamed.partial_fit(batch)
    streamed.finalize()
    torch.testing.assert_close(streamed.mean, whole.mean, atol=1e-6, rtol=0)
    torch.testing.assert_close(streamed.whitener, whole.whitener, atol=1e-5, rtol=1e-5)


def test_distance_matches_scipy_mahalanobis():
    embeddings = gaussian_embeddings()
    model = PatchGaussians(ridge=0.01).fit(embeddings)
    queries = embeddings[:7] * 1.5 + 0.3
    distances = model.mahalanobis(queries).numpy()
    for patch in range(embeddings.shape[1]):
        samples = embeddings[:, patch].double().numpy()
        covariance = np.cov(samples, rowvar=False) + 0.01 * np.eye(samples.shape[1])
        inverse = np.linalg.inv(covariance)
        for row in range(queries.shape[0]):
            expected = mahalanobis(queries[row, patch].numpy(), samples.mean(axis=0), inverse)
            assert distances[row, patch] == pytest.approx(expected, rel=2e-4)


def test_distance_is_invariant_to_an_invertible_linear_map_without_ridge():
    embeddings = gaussian_embeddings(dims=4)
    transform = torch.tensor(
        [[2.0, 0.5, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.3, 0.0, 1.5, 0.2], [0.0, 0.0, 0.0, 0.7]]
    )
    original = PatchGaussians(ridge=0.0).fit(embeddings)
    mapped = PatchGaussians(ridge=0.0).fit(embeddings @ transform.T)
    queries = embeddings[:20] + 1.0
    torch.testing.assert_close(
        mapped.mahalanobis(queries @ transform.T), original.mahalanobis(queries), rtol=2e-3, atol=0
    )


def test_squared_distance_of_fresh_good_samples_averages_to_the_dimension():
    dims = 6
    samples = gaussian_embeddings(n_samples=6000, dims=dims, seed=1)
    model = PatchGaussians(ridge=0.0).fit(samples[:3000])
    squared = model.mahalanobis(samples[3000:]) ** 2  # chi-square with `dims` degrees of freedom
    assert squared.mean().item() == pytest.approx(dims, rel=0.03)
    assert squared.var().item() == pytest.approx(2 * dims, rel=0.10)


def test_fit_needs_two_samples():
    model = PatchGaussians()
    model.partial_fit(torch.zeros(1, 3, 4))
    with pytest.raises(RuntimeError, match="at least two samples"):
        model.finalize()


def test_channel_index_is_sorted_unique_and_reproducible():
    first = random_channel_index(448, 100, torch.Generator().manual_seed(3))
    again = random_channel_index(448, 100, torch.Generator().manual_seed(3))
    other = random_channel_index(448, 100, torch.Generator().manual_seed(4))
    assert torch.equal(first, again) and not torch.equal(first, other)
    assert first.unique().numel() == 100 and torch.equal(first, first.sort().values)
    assert first.min() >= 0 and first.max() < 448
    with pytest.raises(ValueError):
        random_channel_index(10, 11, torch.Generator())


def test_embeddings_stack_selected_channels_on_the_finest_grid():
    fine = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    coarse = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).view(2, 2, 2, 2) + 100
    coarsest = torch.full((2, 2, 1, 1), -7.0)
    index = torch.tensor([1, 3, 6])  # fine channel 1, coarse channel 0, coarsest channel 1
    embeddings = patch_embeddings((fine, coarse, coarsest), index)
    assert embeddings.shape == (2, 16, 3)
    grid = embeddings.transpose(1, 2).reshape(2, 3, 4, 4)
    repeated = coarse[:, 0].repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    torch.testing.assert_close(grid[:, 0], fine[:, 1])
    torch.testing.assert_close(grid[:, 1], repeated)
    assert torch.all(grid[:, 2] == -7.0)
