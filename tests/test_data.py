"""Dataset index and preprocessing on a tiny MVTec-like folder tree."""

import numpy as np
import pytest
import torch
from PIL import Image

from vqgate.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    index_category,
    load_display_image,
    load_mask,
    load_network_input,
    to_network_input,
)


def test_index_lists_train_then_test_in_sorted_order_with_masks(mvtec_like_root):
    samples = index_category(mvtec_like_root, "widget")
    assert [s.key for s in samples] == [
        "train/good/000.png",
        "train/good/001.png",
        "train/good/002.png",
        "test/dent/000.png",
        "test/dent/001.png",
        "test/good/000.png",
        "test/good/001.png",
    ]
    assert [s.is_defective for s in samples] == [False, False, False, True, True, False, False]
    assert samples[3].mask_path.name == "000_mask.png" and samples[3].mask_path.exists()
    assert samples[0].mask_path is None and samples[5].mask_path is None


def test_missing_category_points_to_the_fetch_script(mvtec_like_root):
    with pytest.raises(FileNotFoundError, match="fetch_mvtec"):
        index_category(mvtec_like_root, "gear")


def test_mask_follows_the_same_resize_and_crop_as_the_image(mvtec_like_root):
    samples = index_category(mvtec_like_root, "widget")
    defective, good = samples[3], samples[5]
    mask = load_mask(defective)
    image = load_display_image(defective.image_path)
    assert mask.shape == (224, 224) and mask.dtype == bool and image.shape == (224, 224, 3)
    assert not load_mask(good).any()
    # the planted square is bright in the image exactly where the mask is set (borders aside)
    assert image[mask].mean() > 220 and image[~mask].mean() < 140
    assert mask.mean() == pytest.approx((48 * 48) * (256 / 224) ** 2 / 224**2, rel=0.1)


def test_rectangular_images_are_resized_on_the_short_side_then_centre_cropped(tmp_path):
    gradient = np.tile(np.linspace(0, 255, 600, dtype=np.uint8), (300, 1))
    Image.fromarray(gradient, "L").save(tmp_path / "wide.png")
    image = load_display_image(tmp_path / "wide.png")
    assert image.shape == (224, 224, 3)
    # 600 x 300 -> 512 x 256, centre crop keeps columns 144..368 of 512: the middle of the ramp
    assert image[:, 0].mean() == pytest.approx(255 * 144 / 512, abs=3)
    assert image[:, -1].mean() == pytest.approx(255 * 368 / 512, abs=3)


def test_network_input_is_imagenet_normalised(mvtec_like_root):
    sample = index_category(mvtec_like_root, "widget")[0]
    display = load_display_image(sample.image_path)
    tensor = load_network_input(sample.image_path)
    assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
    restored = tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(
        3, 1, 1
    )
    np.testing.assert_allclose(restored.permute(1, 2, 0).numpy() * 255.0, display, atol=1e-3)
    torch.testing.assert_close(tensor, to_network_input(display))
