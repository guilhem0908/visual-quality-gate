"""PaDiM: one multivariate Gaussian per patch position, scored by Mahalanobis distance.

Defard, Setkov, Loesch, Audigier, "PaDiM: a Patch Distribution Modeling Framework for Anomaly
Detection and Localization", ICPR 2020 workshops (arXiv:2011.08785).

For a patch position (i, j) and N good images with embeddings x_k in R^d (Sec. III-B):

    mu_ij    = 1/N sum_k x_k
    Sigma_ij = 1/(N-1) sum_k (x_k - mu_ij)(x_k - mu_ij)^T + eps I        (eps = 0.01)
    M(x)     = sqrt( (x - mu_ij)^T Sigma_ij^-1 (x - mu_ij) )

With the Cholesky factor Sigma = L L^T the distance is the Euclidean norm of the whitened
residual, M(x) = || L^-1 (x - mu) ||, so the model stores W = L^-1 instead of Sigma^-1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vqgate.backbone import LayerMaps
from vqgate.maps import anomaly_maps

COVARIANCE_RIDGE = 0.01
EMBEDDING_DIMS = 100


def random_channel_index(n_channels: int, n_keep: int, generator: torch.Generator) -> torch.Tensor:
    """Sorted random subset of channel indices (the paper's "Rd" dimensionality reduction)."""
    if n_keep > n_channels:
        raise ValueError(f"cannot keep {n_keep} of {n_channels} channels")
    return torch.randperm(n_channels, generator=generator)[:n_keep].sort().values


def patch_embeddings(layers: LayerMaps, channel_index: torch.Tensor) -> torch.Tensor:
    """Concatenate the three layers on the finest grid and keep ``channel_index``.

    Coarser maps are repeated (nearest neighbour) so that every position of the 56 x 56 grid
    carries the activations of the three semantic levels (Sec. III-A). ``channel_index``
    addresses the concatenation [layer1 | layer2 | layer3] and must be sorted.
    Returns ``[N, P, d]`` with P = 56 * 56.
    """
    grid = layers[0].shape[-2:]
    parts = []
    offset = 0
    for layer in layers:
        width = layer.shape[1]
        in_layer = (channel_index >= offset) & (channel_index < offset + width)
        selected = layer[:, (channel_index[in_layer] - offset).to(layer.device)]
        if selected.shape[-2:] != grid:
            selected = F.interpolate(selected, size=grid, mode="nearest")
        parts.append(selected)
        offset += width
    return torch.cat(parts, dim=1).flatten(2).transpose(1, 2)


class PatchGaussians:
    """Per-position Gaussians fitted in one pass with running first and second moments."""

    def __init__(self, ridge: float = COVARIANCE_RIDGE) -> None:
        self.ridge = ridge
        self.n_samples = 0
        self._sum: torch.Tensor | None = None
        self._outer: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None
        self.whitener: torch.Tensor | None = None

    def partial_fit(self, embeddings: torch.Tensor) -> None:
        """Accumulate sum x and sum x x^T over a batch ``[N, P, d]`` (in float64)."""
        batch = embeddings.double()
        if self._sum is None:
            _, n_patches, dims = batch.shape
            self._sum = batch.new_zeros(n_patches, dims)
            self._outer = batch.new_zeros(n_patches, dims, dims)
        self.n_samples += batch.shape[0]
        self._sum += batch.sum(dim=0)
        self._outer += torch.einsum("npi,npj->pij", batch, batch)

    def finalize(self) -> None:
        """Turn the moments into mu and the whitening matrix W = L^-1, then drop the moments."""
        if self._sum is None or self.n_samples < 2:
            raise RuntimeError("at least two samples are needed to estimate a covariance")
        n = self.n_samples
        mean = self._sum / n
        covariance = (self._outer - n * mean.unsqueeze(2) * mean.unsqueeze(1)) / (n - 1)
        dims = mean.shape[1]
        covariance += self.ridge * torch.eye(dims, dtype=covariance.dtype, device=mean.device)
        cholesky = torch.linalg.cholesky(covariance)
        identity = torch.eye(dims, dtype=cholesky.dtype, device=cholesky.device).expand_as(cholesky)
        whitener = torch.linalg.solve_triangular(cholesky, identity, upper=False)
        self.mean = mean.float()
        self.whitener = whitener.float()
        self._sum = self._outer = None

    def fit(self, embeddings: torch.Tensor) -> PatchGaussians:
        self.partial_fit(embeddings)
        self.finalize()
        return self

    def mahalanobis(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Distance of every patch ``[N, P, d]`` to the Gaussian of its position: ``[N, P]``."""
        residual = embeddings - self.mean
        whitened = torch.einsum("pij,npj->npi", self.whitener, residual)
        return whitened.norm(dim=2)


class PaDiMDetector:
    """PaDiM on one backbone: channel subset + per-position Gaussians."""

    def __init__(self, channel_index: torch.Tensor, ridge: float = COVARIANCE_RIDGE) -> None:
        self.channel_index = channel_index
        self.gaussians = PatchGaussians(ridge)

    def embed(self, layers: LayerMaps) -> torch.Tensor:
        return patch_embeddings(layers, self.channel_index)

    def fit(self, embeddings: torch.Tensor) -> None:
        self.gaussians.fit(embeddings)

    def patch_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.gaussians.mahalanobis(embeddings)

    def score_maps(
        self, patch_scores: torch.Tensor, out_side: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Image score = maximum of the smoothed anomaly map (Sec. III-C)."""
        maps = anomaly_maps(patch_scores, out_side)
        return maps.flatten(1).amax(dim=1), maps

    def state(self) -> dict[str, torch.Tensor]:
        return {
            "channel_index": self.channel_index,
            "mean": self.gaussians.mean,
            "whitener": self.gaussians.whitener,
        }

    @classmethod
    def from_state(cls, state: dict[str, torch.Tensor]) -> PaDiMDetector:
        detector = cls(state["channel_index"])
        detector.gaussians.mean = state["mean"]
        detector.gaussians.whitener = state["whitener"]
        return detector
