"""PatchCore: a coreset-reduced memory bank of good patch features, scored by nearest neighbour.

Roth, Pemula, Zepeda, Schoelkopf, Brox, Gehler, "Towards Total Recall in Industrial Anomaly
Detection", CVPR 2022 (arXiv:2106.08265).

Fit (Alg. 1): collect the locally aware patch features of every good image, then keep a
fraction of them with greedy k-centre selection. Score (Eq. 6): for a test patch feature m,

    s(m) = min_{m' in memory} || m - m' ||_2 ,      image score = max over the patches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vqgate.backbone import LayerMaps
from vqgate.coreset import PROJECTION_DIMS, greedy_k_center, nearest_distance, random_projection
from vqgate.maps import anomaly_maps

NEIGHBOURHOOD = 3


def patch_embeddings(layers: LayerMaps) -> torch.Tensor:
    """Locally aware features from layers 2 and 3 on the 28 x 28 grid of layer 2.

    Each activation is replaced by the mean of its 3 x 3 neighbourhood (Sec. 3.1), the coarser
    map is upsampled bilinearly, and the two are concatenated along channels.
    Returns ``[N, P, C2 + C3]`` with P = 28 * 28.
    """
    _, second, third = layers
    padding = NEIGHBOURHOOD // 2
    second = F.avg_pool2d(second, NEIGHBOURHOOD, stride=1, padding=padding)
    third = F.avg_pool2d(third, NEIGHBOURHOOD, stride=1, padding=padding)
    third = F.interpolate(third, size=second.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([second, third], dim=1).flatten(2).transpose(1, 2)


def coreset_order(
    features: torch.Tensor,
    n_select: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Greedy k-centre pick order over ``[K, D]`` features, run in a 128-d random projection.

    The returned indices are ordered: ``order[:m]`` is the m-centre coreset of the same run, so
    one run at the largest ratio serves every smaller ratio.
    """
    projected = random_projection(features, PROJECTION_DIMS, generator, device)
    start = int(torch.randint(features.shape[0], (1,), generator=generator))
    return greedy_k_center(projected, n_select, start).cpu()


class PatchCoreDetector:
    """PatchCore scoring against a fixed memory bank ``[M, D]``."""

    def __init__(self, memory: torch.Tensor) -> None:
        self.memory = memory

    @staticmethod
    def embed(layers: LayerMaps) -> torch.Tensor:
        return patch_embeddings(layers)

    def patch_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        n_images, n_patches, dims = embeddings.shape
        distances = nearest_distance(embeddings.reshape(-1, dims), self.memory)
        return distances.view(n_images, n_patches)

    def score_maps(
        self, patch_scores: torch.Tensor, out_side: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Image score = largest patch distance; the map is upsampled and smoothed (Sec. 3.3)."""
        return patch_scores.amax(dim=1), anomaly_maps(patch_scores, out_side)

    def state(self) -> dict[str, torch.Tensor]:
        return {"memory": self.memory}

    @classmethod
    def from_state(cls, state: dict[str, torch.Tensor]) -> PatchCoreDetector:
        return cls(state["memory"])
