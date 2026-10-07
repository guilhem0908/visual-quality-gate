"""Frozen ImageNet backbone that returns the three intermediate feature maps.

Both detectors reuse a classification network without training it: PaDiM reads layers 1-3
(Defard et al., 2020, Sec. III-A), PatchCore reads layers 2-3 (Roth et al., 2022, Sec. 3.1).
Only the stem and the first three residual stages are kept; the last stage and the classifier
are never evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class BackboneSpec:
    builder_name: str
    weights_name: str
    channels: tuple[int, int, int]


BACKBONES = {
    "resnet18": BackboneSpec("resnet18", "ResNet18_Weights.IMAGENET1K_V1", (64, 128, 256)),
    "wide_resnet50_2": BackboneSpec(
        "wide_resnet50_2", "Wide_ResNet50_2_Weights.IMAGENET1K_V1", (256, 512, 1024)
    ),
}

LayerMaps = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class FrozenBackbone(nn.Module):
    """ResNet stem + layer1..layer3 in eval mode, gradients disabled.

    For a 224 x 224 input the maps have strides 4, 8 and 16: 56 x 56, 28 x 28 and 14 x 14.
    ``pretrained=False`` builds the same architecture with random weights (used by the tests,
    which must not download anything).
    """

    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        spec = BACKBONES[name]
        weights = models.get_weight(spec.weights_name) if pretrained else None
        network = getattr(models, spec.builder_name)(weights=weights)
        self.name = name
        self.channels = spec.channels
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.eval()
        self.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> LayerMaps:
        first = self.layer1(self.stem(images))
        second = self.layer2(first)
        third = self.layer3(second)
        return first, second, third
