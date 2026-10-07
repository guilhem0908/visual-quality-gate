"""CPU cost of one configuration: fit time, per-image latency, detector size, peak memory.

Run as two short-lived processes so that the memory measured at inference is not inflated by
the training features held during the fit:

    python -m vqgate.latency fit   --config padim_r18 --category screw --model cache/m.pt
    python -m vqgate.latency infer --config padim_r18 --category screw --model cache/m.pt

Each prints one JSON object. Timings cover the model only: the fit is measured from decoded
images to fitted statistics, the latency from one preprocessed ``[1, 3, 224, 224]`` tensor to
its score and anomaly map. PNG decoding and resizing are excluded.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from vqgate.backbone import FrozenBackbone
from vqgate.data import Sample, index_category, load_network_input
from vqgate.experiment import (
    CONFIG_BY_NAME,
    Config,
    extract_layers,
    extract_patchcore_features,
    fit_padim,
    fit_patchcore,
    load_images,
)
from vqgate.inspector import Inspector

CPU_THREADS = 4
WARMUP_IMAGES = 10
FIT_SEED = 0


def peak_rss_megabytes() -> float:
    """Largest resident set size reached by this process so far."""
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset / 1e6
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def cpu_name() -> str:
    """Marketing name of the processor, for the caption of the timing table."""
    if sys.platform == "win32":
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def fit_on_cpu(config: Config, images: torch.Tensor) -> tuple[Inspector, float]:
    """Fit ``config`` on good images with the benchmark's own fit functions; return the time."""
    backbone = FrozenBackbone(config.backbone)
    started = time.perf_counter()
    if config.method == "padim":
        detector = fit_padim(extract_layers(backbone, images, "cpu"), FIT_SEED, "cpu")
    else:
        features = extract_patchcore_features(backbone, images, "cpu").flatten(0, 1)
        detector = fit_patchcore(features, [config.coreset_ratio], FIT_SEED, "cpu")[0]
    return Inspector(backbone, detector), time.perf_counter() - started


def time_inspections(inspector: Inspector, samples: list[Sample]) -> np.ndarray:
    """Milliseconds per single-image inspection, after a warm-up that is not counted.

    Images are decoded one at a time, outside the timed region, so that the process holds one
    image at once, as a station would.
    """
    elapsed = []
    for position, sample in enumerate(samples[:WARMUP_IMAGES] + samples):
        image = load_network_input(sample.image_path).unsqueeze(0)
        started = time.perf_counter()
        inspector.inspect(image)
        if position >= WARMUP_IMAGES:
            elapsed.append((time.perf_counter() - started) * 1e3)
    return np.asarray(elapsed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("fit", "infer"))
    parser.add_argument("--config", required=True, choices=sorted(CONFIG_BY_NAME))
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(CPU_THREADS)
    samples = index_category(args.data_root, args.category)
    if args.phase == "fit":
        images = load_images([s for s in samples if s.split == "train"])
        inspector, seconds = fit_on_cpu(CONFIG_BY_NAME[args.config], images)
        inspector.save(args.model)
        report = {
            "n_train": len(images),
            "fit_seconds": round(seconds, 1),
            "detector_megabytes": round(inspector.detector_megabytes(), 1),
        }
    else:
        inspector = Inspector.load(args.model)
        elapsed = time_inspections(inspector, [s for s in samples if s.split == "test"])
        report = {
            "n_images": len(elapsed),
            "median_ms": round(float(np.median(elapsed)), 1),
            "p95_ms": round(float(np.percentile(elapsed, 95)), 1),
            "peak_rss_megabytes": round(peak_rss_megabytes()),
        }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
