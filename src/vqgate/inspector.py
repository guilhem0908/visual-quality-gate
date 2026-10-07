"""The object a quality station runs: frozen backbone + fitted detector, image in, score out."""

from __future__ import annotations

from pathlib import Path

import torch

from vqgate.backbone import FrozenBackbone
from vqgate.padim import PaDiMDetector
from vqgate.patchcore import PatchCoreDetector

Detector = PaDiMDetector | PatchCoreDetector
DETECTORS = {"padim": PaDiMDetector, "patchcore": PatchCoreDetector}


def method_of(detector: Detector) -> str:
    return next(name for name, kind in DETECTORS.items() if isinstance(detector, kind))


class Inspector:
    """Scores batches of preprocessed images ``[N, 3, S, S]`` with one fitted detector."""

    def __init__(self, backbone: FrozenBackbone, detector: Detector) -> None:
        self.backbone = backbone
        self.detector = detector

    @torch.inference_mode()
    def inspect(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the image scores ``[N]`` and the anomaly maps ``[N, S, S]``."""
        embeddings = self.detector.embed(self.backbone(images))
        patch_scores = self.detector.patch_scores(embeddings)
        return self.detector.score_maps(patch_scores, images.shape[-1])

    def detector_megabytes(self) -> float:
        """Size of the fitted statistics (Gaussians or memory bank), backbone excluded."""
        tensors = self.detector.state().values()
        return sum(t.numel() * t.element_size() for t in tensors) / 1e6

    def save(self, path: Path) -> None:
        """Write the detector statistics; the backbone is rebuilt from its name on load."""
        torch.save(
            {
                "method": method_of(self.detector),
                "backbone": self.backbone.name,
                "detector": self.detector.state(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, pretrained: bool = True) -> Inspector:
        payload = torch.load(path, weights_only=True)
        detector = DETECTORS[payload["method"]].from_state(payload["detector"])
        return cls(FrozenBackbone(payload["backbone"], pretrained=pretrained), detector)
