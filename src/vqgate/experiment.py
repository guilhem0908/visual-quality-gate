"""Benchmark runner: fit every configuration on MVTec AD categories and score the parts.

Two protocols are run for every (configuration, category, seed):

* ``full``  - fit on all good training images, as in the papers. Used for the image-level and
  pixel-level AUROC that are compared with the published values.
* ``gate``  - fit on 80 % of the good training images and score the held-out 20 % together
  with the test set. The held-out scores set the accept/reject threshold (``vqgate.gate``).

A seed drives the fit/held-out split, PaDiM's random channel subset, and PatchCore's random
projection and first centre. Images are decoded once per category and backbone features are
extracted once per (backbone, category); every seed, protocol and configuration reuses them.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vqgate.backbone import FrozenBackbone, LayerMaps
from vqgate.data import CROP_SIDE, Sample, index_category, load_mask, load_network_input
from vqgate.metrics import auroc
from vqgate.padim import EMBEDDING_DIMS, PaDiMDetector, random_channel_index
from vqgate.patchcore import PatchCoreDetector, coreset_order, patch_embeddings

SEEDS = (0, 1, 2)
CALIBRATION_FRACTION = 0.2
PROTOCOLS = ("full", "gate")
BATCH_SIZE = 32
DECODE_THREADS = 8

Device = torch.device | str
Scored = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Config:
    name: str
    label: str
    method: str
    backbone: str
    coreset_ratio: float | None = None


CONFIGS = (
    Config("padim_r18", "PaDiM R18-Rd100", "padim", "resnet18"),
    Config("patchcore_r18_1", "PatchCore R18-1%", "patchcore", "resnet18", 0.01),
    Config("patchcore_wr50_1", "PatchCore WR50-1%", "patchcore", "wide_resnet50_2", 0.01),
    Config("patchcore_wr50_10", "PatchCore WR50-10%", "patchcore", "wide_resnet50_2", 0.10),
)
CONFIG_BY_NAME = {config.name: config for config in CONFIGS}


@dataclass(frozen=True)
class ScoreRow:
    """One image score, as written to ``results/scores/<config>.csv``."""

    config: str
    category: str
    seed: int
    protocol: str
    role: str
    defect: str
    image: str
    score: float


def calibration_split(n_train: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random split of the good training images into fit and held-out (calibration) indices."""
    permutation = np.random.default_rng(seed).permutation(n_train)
    n_calibration = math.ceil(CALIBRATION_FRACTION * n_train)
    return np.sort(permutation[n_calibration:]), np.sort(permutation[:n_calibration])


def coreset_size(n_features: int, ratio: float) -> int:
    return max(1, round(ratio * n_features))


def load_images(samples: list[Sample]) -> torch.Tensor:
    """Decode and preprocess the samples in parallel: ``[N, 3, 224, 224]``."""
    with ThreadPoolExecutor(DECODE_THREADS) as pool:
        return torch.stack(list(pool.map(lambda s: load_network_input(s.image_path), samples)))


@torch.inference_mode()
def extract_layers(backbone: FrozenBackbone, images: torch.Tensor, device: Device) -> LayerMaps:
    """Raw backbone maps of every image, gathered on the CPU (PaDiM picks channels per seed)."""
    batches = [
        [layer.cpu() for layer in backbone(batch.to(device))] for batch in images.split(BATCH_SIZE)
    ]
    return tuple(torch.cat(layers) for layers in zip(*batches, strict=True))


@torch.inference_mode()
def extract_patchcore_features(
    backbone: FrozenBackbone, images: torch.Tensor, device: Device
) -> torch.Tensor:
    """PatchCore patch features ``[N, P, D]`` of every image, gathered on the CPU."""
    return torch.cat(
        [patch_embeddings(backbone(batch.to(device))).cpu() for batch in images.split(BATCH_SIZE)]
    )


def fit_padim(layers: LayerMaps, seed: int, device: Device) -> PaDiMDetector:
    generator = torch.Generator().manual_seed(seed)
    n_channels = sum(layer.shape[1] for layer in layers)
    detector = PaDiMDetector(random_channel_index(n_channels, EMBEDDING_DIMS, generator))
    for start in range(0, layers[0].shape[0], BATCH_SIZE):
        batch = tuple(layer[start : start + BATCH_SIZE].to(device) for layer in layers)
        detector.gaussians.partial_fit(detector.embed(batch))
    detector.gaussians.finalize()
    return detector


def fit_patchcore(
    features: torch.Tensor, ratios: list[float], seed: int, device: Device
) -> list[PatchCoreDetector]:
    """One greedy run at the largest ratio; every smaller ratio is a prefix of its picks."""
    generator = torch.Generator().manual_seed(seed)
    sizes = [coreset_size(len(features), ratio) for ratio in ratios]
    order = coreset_order(features, max(sizes), generator, device)
    return [PatchCoreDetector(features[order[:size]].to(device)) for size in sizes]


@torch.inference_mode()
def score_padim(detector: PaDiMDetector, layers: LayerMaps, device: Device) -> Scored:
    """Image scores, anomaly maps and raw patch scores of PaDiM for a set of layer maps."""
    outputs = []
    for start in range(0, layers[0].shape[0], BATCH_SIZE):
        batch = tuple(layer[start : start + BATCH_SIZE].to(device) for layer in layers)
        outputs.append(_score_batch(detector, detector.embed(batch)))
    return _gather(outputs)


@torch.inference_mode()
def score_patchcore(detector: PatchCoreDetector, features: torch.Tensor, device: Device) -> Scored:
    """Image scores, anomaly maps and raw patch scores of PatchCore for ``[N, P, D]`` features."""
    return _gather(
        [_score_batch(detector, batch.to(device)) for batch in features.split(BATCH_SIZE)]
    )


def _score_batch(detector: PaDiMDetector | PatchCoreDetector, embeddings: torch.Tensor):
    patch_scores = detector.patch_scores(embeddings)
    scores, maps = detector.score_maps(patch_scores, CROP_SIDE)
    return scores.cpu(), maps.cpu(), patch_scores.cpu()


def _gather(outputs: list[tuple[torch.Tensor, ...]]) -> Scored:
    return tuple(torch.cat(parts).numpy() for parts in zip(*outputs, strict=True))


class CategoryRun:
    """Everything measured on one category: score rows, pixel AUROC, heat-map grids."""

    def __init__(self, category: str, data_root: Path, seeds: tuple[int, ...] = SEEDS) -> None:
        samples = index_category(data_root, category)
        self.category = category
        self.seeds = seeds
        self.train = [s for s in samples if s.split == "train"]
        self.test = [s for s in samples if s.split == "test"]
        self.train_images = load_images(self.train)
        self.test_images = load_images(self.test)
        self.pixel_labels = np.stack([load_mask(s) for s in self.test]).ravel()
        self.rows: list[ScoreRow] = []
        self.pixel_auroc: dict[str, list[float]] = {}
        self.gate_patch_scores: dict[str, np.ndarray] = {}

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train_good": len(self.train),
            "test_good": sum(not s.is_defective for s in self.test),
            "test_defective": sum(s.is_defective for s in self.test),
        }

    def _record(self, config, seed, protocol, role, samples, scores) -> None:
        for sample, score in zip(samples, scores, strict=True):
            self.rows.append(
                ScoreRow(
                    config.name,
                    self.category,
                    seed,
                    protocol,
                    role,
                    sample.defect,
                    sample.key,
                    float(score),
                )
            )

    def _record_run(self, config, seed, protocol, test_out, held_out_scores, held_out) -> None:
        """Store the rows of one fitted detector, plus what its protocol is used for."""
        scores, maps, patch_scores = test_out
        self._record(config, seed, protocol, "test", self.test, scores)
        if protocol == "full":
            pixel = auroc(maps.ravel(), self.pixel_labels)
            self.pixel_auroc.setdefault(config.name, []).append(round(pixel, 6))
            return
        self._record(config, seed, protocol, "calibration", held_out, held_out_scores)
        if seed == self.seeds[0]:
            self.gate_patch_scores[config.name] = patch_scores

    def run(self, backbone: FrozenBackbone, configs: list[Config], device: Device) -> None:
        """Fit and score every configuration that uses ``backbone``."""
        padim = [c for c in configs if c.method == "padim"]
        patchcore = [c for c in configs if c.method == "patchcore"]
        if padim:
            train_layers = extract_layers(backbone, self.train_images, device)
            test_layers = extract_layers(backbone, self.test_images, device)
        if patchcore:
            train_features = extract_patchcore_features(backbone, self.train_images, device)
            test_features = extract_patchcore_features(backbone, self.test_images, device)
        for seed in self.seeds:
            fit_index, calibration_index = calibration_split(len(self.train), seed)
            held_out = [self.train[i] for i in calibration_index]
            held_out_rows = torch.from_numpy(calibration_index)
            for protocol in PROTOCOLS:
                is_gate = protocol == "gate"
                fit_rows = torch.from_numpy(fit_index if is_gate else np.arange(len(self.train)))
                for config in padim:
                    detector = fit_padim(tuple(m[fit_rows] for m in train_layers), seed, device)
                    test_out = score_padim(detector, test_layers, device)
                    held_out_scores = None
                    if is_gate:
                        held_out_layers = tuple(m[held_out_rows] for m in train_layers)
                        held_out_scores = score_padim(detector, held_out_layers, device)[0]
                    self._record_run(config, seed, protocol, test_out, held_out_scores, held_out)
                if not patchcore:
                    continue
                ratios = [c.coreset_ratio for c in patchcore]
                fit_features = train_features[fit_rows].flatten(0, 1)
                detectors = fit_patchcore(fit_features, ratios, seed, device)
                for config, detector in zip(patchcore, detectors, strict=True):
                    test_out = score_patchcore(detector, test_features, device)
                    held_out_scores = None
                    if is_gate:
                        held_out_features = train_features[held_out_rows]
                        held_out_scores = score_patchcore(detector, held_out_features, device)[0]
                    self._record_run(config, seed, protocol, test_out, held_out_scores, held_out)


def run_category(
    category: str,
    data_root: Path,
    device: Device,
    configs: tuple[Config, ...] = CONFIGS,
    seeds: tuple[int, ...] = SEEDS,
) -> CategoryRun:
    """Run every configuration on one category, one backbone at a time."""
    run = CategoryRun(category, data_root, seeds)
    for backbone_name in dict.fromkeys(c.backbone for c in configs):
        backbone = FrozenBackbone(backbone_name).to(device)
        run.run(backbone, [c for c in configs if c.backbone == backbone_name], device)
    return run
