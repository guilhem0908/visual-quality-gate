"""Greedy k-centre coreset selection, the memory-bank reduction step of PatchCore.

The k-centre problem asks for k centres C in a point set X minimising the coverage radius

    r(C) = max_{x in X} min_{c in C} || x - c ||.

Farthest-first traversal (Gonzalez, 1985) repeatedly adds the point farthest from the centres
chosen so far. It is a 2-approximation, r(C_greedy) <= 2 r(C_opt), and its picks are nested:
the first m picks of a run are exactly the m-centre solution of the same run. PatchCore
(Roth et al., 2022, Alg. 1) runs it on a random linear projection of the patch features,
which preserves pairwise distances up to a small distortion (Johnson and Lindenstrauss, 1984).
"""

from __future__ import annotations

import torch

PROJECTION_DIMS = 128
DISTANCE_CHUNK = 4096
PROJECTION_CHUNK = 1 << 16


def random_projection(
    features: torch.Tensor,
    out_dims: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Project ``[N, D]`` features to ``out_dims`` with a Gaussian matrix of variance 1/out_dims.

    With entries N(0, 1/out_dims) the squared norm is preserved in expectation:
    E || x R ||^2 = || x ||^2. The product is taken in chunks so that only the projected copy
    has to live on ``device``.
    """
    matrix = torch.randn(features.shape[1], out_dims, generator=generator) / out_dims**0.5
    matrix = matrix.to(device)
    return torch.cat([chunk.to(device) @ matrix for chunk in features.split(PROJECTION_CHUNK)])


def greedy_k_center(points: torch.Tensor, n_select: int, start: int) -> torch.Tensor:
    """Indices of ``n_select`` centres chosen by farthest-first traversal from ``start``.

    Each step costs one matrix-vector product: with squared norms cached,
    || x - c ||^2 = || x ||^2 + || c ||^2 - 2 x.c for all x at once. The running index stays a
    tensor so that a GPU run never waits for a host round trip inside the loop.
    """
    n_points = points.shape[0]
    if not 0 < n_select <= n_points:
        raise ValueError(f"cannot select {n_select} centres among {n_points} points")
    squared_norms = (points * points).sum(dim=1)
    nearest_sq = torch.full((n_points,), torch.inf, dtype=points.dtype, device=points.device)
    selected = torch.empty(n_select, dtype=torch.long, device=points.device)
    current = torch.as_tensor(start, device=points.device)
    for step in range(n_select):
        selected[step] = current
        to_current = torch.addmv(squared_norms, points, points[current], alpha=-2.0)
        to_current.add_(squared_norms[current]).clamp_(min=0.0)
        torch.minimum(nearest_sq, to_current, out=nearest_sq)
        nearest_sq[current] = -1.0  # below any distance: a chosen centre is never picked again
        current = torch.argmax(nearest_sq)
    return selected


def nearest_distance(queries: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from every query ``[Q, D]`` to its nearest reference ``[M, D]``."""
    reference_sq = (references * references).sum(dim=1)
    nearest = []
    for chunk in queries.split(DISTANCE_CHUNK):
        chunk_sq = (chunk * chunk).sum(dim=1, keepdim=True)
        squared = chunk_sq + reference_sq - 2.0 * (chunk @ references.T)
        nearest.append(squared.amin(dim=1).clamp_(min=0.0).sqrt())
    return torch.cat(nearest)


def coverage_radius(points: torch.Tensor, centers: torch.Tensor) -> float:
    """r(C): the largest distance from a point to its nearest centre."""
    return float(nearest_distance(points, centers).max())
