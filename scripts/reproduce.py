"""Regenerate every result file and figure of the repository.

Stages (default: all four, in this order):

  benchmark  fit and score all configurations on MVTec AD   -> results/scores/*.csv,
             (needs the dataset; cached per category)           results/benchmark.json
  latency    CPU fit time, latency, memory on one category   -> results/latency.json
             (needs the dataset)
  report     metrics and gate outcomes from the score files  -> results/summary.json,
             (no dataset needed)                                docs/tradeoff.png
  hero       animated inspection sheet                       -> docs/hero.gif,
             (needs the dataset and the benchmark cache)        results/hero.json

Usage:
  python scripts/reproduce.py                       # everything
  python scripts/reproduce.py --stages report       # recompute from the committed scores
  python scripts/reproduce.py --stages benchmark --categories bottle grid   # resumable
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, astuple
from pathlib import Path

import numpy as np
import psutil
import torch

from vqgate.data import CATEGORIES
from vqgate.experiment import CALIBRATION_FRACTION, CONFIGS, SEEDS, ScoreRow, run_category
from vqgate.figures import (
    defects_outside_crop,
    plot_tradeoff,
    render_hero,
    select_hero_frames,
)
from vqgate.latency import CPU_THREADS, WARMUP_IMAGES, cpu_name
from vqgate.report import TARGET_FALSE_REJECT, group_scores, read_scores, summarise, write_scores

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DOCS = REPO / "docs"
CACHE = REPO / "cache"
STAGES = ("benchmark", "latency", "report", "hero")
LATENCY_CATEGORY = "screw"
LOAD_SAMPLE_SECONDS = 2.0
HERO_CONFIG = "patchcore_wr50_10"
HERO_SEED = SEEDS[0]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def stage_benchmark(data_root: Path, device: str, categories: list[str], force: bool) -> None:
    """Run the requested categories, then assemble the results once all five are cached."""
    cache = CACHE / "benchmark"
    cache.mkdir(parents=True, exist_ok=True)
    for category in categories:
        if (cache / f"{category}.json").exists() and not force:
            print(f"{category}: cached")
            continue
        started = time.perf_counter()
        run = run_category(category, data_root, device)
        seconds = round(time.perf_counter() - started, 1)
        np.savez_compressed(cache / f"{category}.npz", **run.gate_patch_scores)
        write_json(
            cache / f"{category}.json",
            {
                "device": device,
                "seconds": seconds,
                "counts": run.counts,
                "pixel_auroc": run.pixel_auroc,
                "rows": [astuple(row) for row in run.rows],
            },
        )
        print(f"{category}: done in {seconds:.0f} s on {device}")
    missing = [c for c in CATEGORIES if not (cache / f"{c}.json").exists()]
    if missing:
        print(f"benchmark incomplete, still to run: {' '.join(missing)}")
        return
    cached = {category: read_json(cache / f"{category}.json") for category in CATEGORIES}
    rows = [ScoreRow(*row) for entry in cached.values() for row in entry["rows"]]
    write_scores(rows, RESULTS / "scores")
    write_json(
        RESULTS / "benchmark.json",
        {
            "device": sorted({entry["device"] for entry in cached.values()}),
            "torch": torch.__version__,
            "seeds": list(SEEDS),
            "calibration_fraction": CALIBRATION_FRACTION,
            "wall_clock_seconds": round(sum(entry["seconds"] for entry in cached.values()), 1),
            "counts": {category: entry["counts"] for category, entry in cached.items()},
            "pixel_auroc": {
                config.name: {c: entry["pixel_auroc"][config.name] for c, entry in cached.items()}
                for config in CONFIGS
            },
        },
    )


def run_worker(phase: str, config: str, data_root: Path, model: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "vqgate.latency",
        phase,
        "--config",
        config,
        "--category",
        LATENCY_CATEGORY,
        "--data-root",
        str(data_root),
        "--model",
        str(model),
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def stage_latency(data_root: Path) -> None:
    """Fit and time every configuration in fresh CPU processes.

    The CPU load caused by other programs is sampled before each configuration and stored
    with the timings, so that a busy machine is visible in the results.
    """
    models = CACHE / "models"
    models.mkdir(parents=True, exist_ok=True)
    configs = {}
    other_load = []
    for config in CONFIGS:
        other_load.append(psutil.cpu_percent(interval=LOAD_SAMPLE_SECONDS))
        model = models / f"{config.name}.pt"
        configs[config.name] = run_worker("fit", config.name, data_root, model) | run_worker(
            "infer", config.name, data_root, model
        )
        model.unlink()
        print(f"{config.name}: {configs[config.name]}", flush=True)
    write_json(
        RESULTS / "latency.json",
        {
            "category": LATENCY_CATEGORY,
            "cpu": cpu_name(),
            "threads": CPU_THREADS,
            "warmup_images": WARMUP_IMAGES,
            "torch": torch.__version__,
            "other_cpu_load_percent": round(sum(other_load) / len(other_load)),
            "configs": configs,
        },
    )


def stage_report() -> None:
    rows = read_scores(RESULTS / "scores")
    summary = summarise(
        rows, read_json(RESULTS / "benchmark.json"), read_json(RESULTS / "latency.json")
    )
    write_json(RESULTS / "summary.json", summary)
    DOCS.mkdir(exist_ok=True)
    plot_tradeoff(summary, DOCS / "tradeoff.png")


def stage_hero(data_root: Path) -> None:
    groups = group_scores(read_scores(RESULTS / "scores"))
    cropped_away = defects_outside_crop(data_root, list(CATEGORIES))
    frames = select_hero_frames(
        groups, list(CATEGORIES), HERO_CONFIG, HERO_SEED, TARGET_FALSE_REJECT, cropped_away
    )
    write_json(
        RESULTS / "hero.json",
        {
            "config": HERO_CONFIG,
            "seed": HERO_SEED,
            "target_false_reject": TARGET_FALSE_REJECT,
            "defects_outside_crop": sorted(cropped_away),
            "frames": [{**asdict(frame), "outcome": frame.outcome} for frame in frames],
        },
    )
    patch_scores = {}
    for category in {frame.category for frame in frames}:
        with np.load(CACHE / "benchmark" / f"{category}.npz") as grids:
            patch_scores[category] = grids[HERO_CONFIG]
    render_hero(frames, data_root, patch_scores, HERO_CONFIG, DOCS / "hero.gif")
    for frame in frames:
        print(f"{frame.role}: {frame.category} {frame.image} -> {frame.outcome}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "mvtec_ad")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--force", action="store_true", help="ignore the benchmark cache")
    args = parser.parse_args()
    if "benchmark" in args.stages:
        stage_benchmark(args.data_root, resolve_device(args.device), args.categories, args.force)
    if "latency" in args.stages:
        stage_latency(args.data_root)
    if "report" in args.stages:
        stage_report()
    if "hero" in args.stages:
        stage_hero(args.data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
