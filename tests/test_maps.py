"""Anomaly-map post-processing against SciPy."""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from vqgate.maps import anomaly_maps, gaussian_blur, gaussian_kernel1d


def test_blur_matches_scipy_gaussian_filter_with_mirrored_borders():
    maps = torch.rand(3, 48, 40, generator=torch.Generator().manual_seed(0))
    blurred = gaussian_blur(maps, sigma=4.0).numpy()
    for index in range(3):
        expected = gaussian_filter(maps[index].numpy(), sigma=4.0, mode="mirror", truncate=4.0)
        np.testing.assert_allclose(blurred[index], expected, atol=1e-5)


def test_kernel_is_normalised_symmetric_and_covers_four_sigma():
    kernel = gaussian_kernel1d(4.0)
    assert kernel.numel() == 33
    assert kernel.sum().item() == pytest.approx(1.0)
    torch.testing.assert_close(kernel, kernel.flip(0))


def test_blur_leaves_a_constant_map_unchanged():
    constant = torch.full((1, 40, 40), 2.5)
    torch.testing.assert_close(gaussian_blur(constant), constant, atol=1e-5, rtol=0)


def test_anomaly_map_has_requested_size_and_stays_within_the_score_range():
    patch_scores = torch.rand(2, 49, generator=torch.Generator().manual_seed(1))
    maps = anomaly_maps(patch_scores, out_side=56)
    assert maps.shape == (2, 56, 56)
    assert maps.min() >= patch_scores.min() - 1e-6 and maps.max() <= patch_scores.max() + 1e-6


def test_patch_count_must_form_a_square_grid():
    with pytest.raises(ValueError, match="square grid"):
        anomaly_maps(torch.zeros(1, 50), out_side=56)
