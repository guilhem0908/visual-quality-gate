"""MVTec AD folder index and the image preprocessing shared by both detectors.

The dataset layout is the one published by MVTec (Bergmann et al., CVPR 2019)::

    <root>/<category>/train/good/000.png
    <root>/<category>/test/<defect>/000.png          (<defect> is "good" or a defect type)
    <root>/<category>/ground_truth/<defect>/000_mask.png

Preprocessing follows the PaDiM and PatchCore papers: resize the shorter side to 256 px,
centre-crop 224 x 224, normalise with the ImageNet statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RESIZE_SIDE = 256
CROP_SIDE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GOOD = "good"

CATEGORIES = ("bottle", "grid", "metal_nut", "screw", "zipper")


@dataclass(frozen=True)
class Sample:
    """One image of a category, with its ground-truth mask when it shows a defect."""

    category: str
    split: str
    defect: str
    image_path: Path
    mask_path: Path | None

    @property
    def is_defective(self) -> bool:
        return self.defect != GOOD

    @property
    def key(self) -> str:
        """Dataset-relative identifier, e.g. ``test/broken_large/000.png``."""
        return f"{self.split}/{self.defect}/{self.image_path.name}"


def index_category(root: Path, category: str) -> list[Sample]:
    """List every train and test image of ``category`` in a deterministic order."""
    base = Path(root) / category
    if not (base / "train" / GOOD).is_dir():
        raise FileNotFoundError(
            f"{base} is not an MVTec AD category folder; run scripts/fetch_mvtec.py first"
        )
    samples: list[Sample] = []
    for split in ("train", "test"):
        for defect_dir in sorted(p for p in (base / split).iterdir() if p.is_dir()):
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if split == "test" and defect_dir.name != GOOD:
                    mask_path = (
                        base / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
                    )
                samples.append(Sample(category, split, defect_dir.name, image_path, mask_path))
    return samples


def _resize_and_crop(image: Image.Image, resample: Image.Resampling) -> Image.Image:
    width, height = image.size
    scale = RESIZE_SIDE / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), resample)
    left = (resized.width - CROP_SIDE) // 2
    top = (resized.height - CROP_SIDE) // 2
    return resized.crop((left, top, left + CROP_SIDE, top + CROP_SIDE))


def load_display_image(path: Path) -> np.ndarray:
    """Resized and cropped RGB image as ``uint8 [224, 224, 3]`` (what the network sees)."""
    with Image.open(path) as image:
        return np.asarray(_resize_and_crop(image.convert("RGB"), Image.Resampling.BILINEAR))


def to_network_input(display_image: np.ndarray) -> torch.Tensor:
    """ImageNet-normalised ``float32 [3, 224, 224]`` tensor from a ``uint8`` RGB image."""
    pixels = torch.from_numpy(display_image.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (pixels - mean) / std


def load_network_input(path: Path) -> torch.Tensor:
    return to_network_input(load_display_image(path))


def load_mask(sample: Sample) -> np.ndarray:
    """Ground-truth defect mask as ``bool [224, 224]``; all False for a good part."""
    if sample.mask_path is None:
        return np.zeros((CROP_SIDE, CROP_SIDE), dtype=bool)
    with Image.open(sample.mask_path) as mask:
        return np.asarray(_resize_and_crop(mask.convert("L"), Image.Resampling.NEAREST)) > 0
