"""Synthetic parts shared by the tests: no dataset and no pretrained weights are needed."""

import numpy as np
import pytest
import torch
from PIL import Image

from vqgate.data import to_network_input

IMAGE_SIDE = 224
DEFECT_BOX = (120, 168, 40, 88)  # rows 120..168, columns 40..88


def synthetic_part(rng: np.random.Generator, defective: bool) -> np.ndarray:
    """A grey textured part; a defective one carries a bright square at ``DEFECT_BOX``."""
    stripes = 20.0 * np.sin(np.arange(IMAGE_SIDE) / 6.0)[None, :, None]
    image = 110.0 + stripes + rng.normal(0.0, 6.0, size=(IMAGE_SIDE, IMAGE_SIDE, 3))
    if defective:
        top, bottom, left, right = DEFECT_BOX
        image[top:bottom, left:right] = 245.0
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def synthetic_parts() -> dict[str, torch.Tensor]:
    """Network inputs: 24 good parts to fit on, 19 held out, 8 good and 8 defective to test."""
    rng = np.random.default_rng(0)

    def batch(count: int, defective: bool) -> torch.Tensor:
        return torch.stack([to_network_input(synthetic_part(rng, defective)) for _ in range(count)])

    return {
        "train": batch(24, False),
        "held_out": batch(19, False),
        "good": batch(8, False),
        "defective": batch(8, True),
    }


@pytest.fixture
def mvtec_like_root(tmp_path):
    """A tiny folder tree with the MVTec AD layout: 3 train, 2 test good, 2 test defective."""
    rng = np.random.default_rng(1)
    root = tmp_path / "dataset"

    def save(folder, name, array, mode):
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode).save(folder / name)

    for index in range(3):
        save(
            root / "widget" / "train" / "good",
            f"{index:03d}.png",
            synthetic_part(rng, False),
            "RGB",
        )
    for index in range(2):
        save(
            root / "widget" / "test" / "good", f"{index:03d}.png", synthetic_part(rng, False), "RGB"
        )
        save(
            root / "widget" / "test" / "dent", f"{index:03d}.png", synthetic_part(rng, True), "RGB"
        )
        mask = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
        top, bottom, left, right = DEFECT_BOX
        mask[top:bottom, left:right] = 255
        save(root / "widget" / "ground_truth" / "dent", f"{index:03d}_mask.png", mask, "L")
    return root
