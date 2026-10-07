"""Turn a coarse grid of patch scores into a full-resolution anomaly map.

Both papers upsample the patch scores bilinearly to the input resolution and smooth them with
a Gaussian of sigma = 4 px (PaDiM Sec. III-C; PatchCore Sec. 3.3).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SMOOTHING_SIGMA = 4.0
KERNEL_TRUNCATE = 4.0


def gaussian_kernel1d(sigma: float, truncate: float = KERNEL_TRUNCATE) -> torch.Tensor:
    """Normalised samples of exp(-x^2 / (2 sigma^2)) on integer x in [-r, r], r = truncate*sigma."""
    radius = int(truncate * sigma + 0.5)
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur(maps: torch.Tensor, sigma: float = SMOOTHING_SIGMA) -> torch.Tensor:
    """Separable Gaussian filter of ``[N, H, W]`` maps with mirrored borders.

    Equivalent to ``scipy.ndimage.gaussian_filter(..., mode="mirror", truncate=4)``.
    """
    kernel = gaussian_kernel1d(sigma).to(maps)
    radius = kernel.numel() // 2
    blurred = maps.unsqueeze(1)
    blurred = F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1)
    )
    blurred = F.conv2d(
        F.pad(blurred, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1)
    )
    return blurred.squeeze(1)


def anomaly_maps(
    patch_scores: torch.Tensor, out_side: int, sigma: float = SMOOTHING_SIGMA
) -> torch.Tensor:
    """Bilinear upsampling of ``[N, P]`` scores on a square grid to ``[N, out, out]``, then blur."""
    n_images, n_patches = patch_scores.shape
    grid_side = round(n_patches**0.5)
    if grid_side * grid_side != n_patches:
        raise ValueError(f"{n_patches} patches do not form a square grid")
    grid = patch_scores.view(n_images, 1, grid_side, grid_side)
    upsampled = F.interpolate(grid, size=out_side, mode="bilinear", align_corners=False)
    return gaussian_blur(upsampled.squeeze(1), sigma)
