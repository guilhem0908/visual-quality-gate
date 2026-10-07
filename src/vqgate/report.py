"""From per-image score files to the summary and the tables shown in the README.

Everything here is recomputed from ``results/scores/*.csv`` without the dataset, except the
pixel-level AUROC and the timings, which are copied from ``results/benchmark.json`` and
``results/latency.json`` (they need the images).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from vqgate.data import GOOD
from vqgate.experiment import CONFIGS, ScoreRow
from vqgate.gate import GateCounts, conformal_threshold, gate_counts, promised_false_rejects
from vqgate.metrics import auroc
from vqgate.published import PUBLISHED

TARGET_FALSE_REJECT = 0.05
TRADEOFF_TARGETS = tuple(round(0.025 * step, 3) for step in range(1, 21))
SCORE_DECIMALS = 6
MISSING = "-"

GroupKey = tuple[str, str, int, str, str]
Groups = dict[GroupKey, "ScoreGroup"]


@dataclass(frozen=True)
class ScoreGroup:
    """Scores of one (config, category, seed, protocol, role) cell."""

    images: tuple[str, ...]
    scores: np.ndarray
    defective: np.ndarray


def write_scores(rows: list[ScoreRow], directory: Path) -> None:
    """One CSV per configuration, rows in a stable order, scores with six decimals."""
    directory.mkdir(parents=True, exist_ok=True)
    header = [f.name for f in fields(ScoreRow)]
    for config in dict.fromkeys(row.config for row in rows):
        selected = sorted(astuple(row) for row in rows if row.config == config)
        with (directory / f"{config}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            for *prefix, score in selected:
                writer.writerow([*prefix, f"{score:.{SCORE_DECIMALS}f}"])


def read_scores(directory: Path) -> list[ScoreRow]:
    rows = []
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                record["seed"] = int(record["seed"])
                record["score"] = float(record["score"])
                rows.append(ScoreRow(**record))
    return rows


def group_scores(rows: list[ScoreRow]) -> Groups:
    buckets: dict[GroupKey, list[ScoreRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.config, row.category, row.seed, row.protocol, row.role)].append(row)
    return {
        key: ScoreGroup(
            images=tuple(row.image for row in bucket),
            scores=np.array([row.score for row in bucket]),
            defective=np.array([row.defect != GOOD for row in bucket]),
        )
        for key, bucket in buckets.items()
    }


def gate_run(
    groups: Groups, config: str, category: str, seed: int, alpha: float
) -> tuple[float, GateCounts]:
    """Threshold from the held-out good parts, then its outcome on the test parts."""
    held_out = groups[(config, category, seed, "gate", "calibration")]
    test = groups[(config, category, seed, "gate", "test")]
    threshold = conformal_threshold(held_out.scores, alpha)
    return threshold, gate_counts(test.scores, test.defective, threshold)


def good_shift(groups: Groups, config: str, category: str, seed: int) -> float:
    """AUROC of test-good against held-out-good scores; 0.5 when they are indistinguishable."""
    held_out = groups[(config, category, seed, "gate", "calibration")].scores
    test = groups[(config, category, seed, "gate", "test")]
    test_good = test.scores[~test.defective]
    labels = np.r_[np.zeros(len(held_out)), np.ones(len(test_good))]
    return auroc(np.r_[held_out, test_good], labels)


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def _counts_dict(counts: GateCounts) -> dict[str, int | float]:
    return {
        "false_rejects": counts.false_rejects,
        "n_good": counts.n_good,
        "false_reject_rate": round(counts.false_reject_rate, 4),
        "escapes": counts.escapes,
        "n_defective": counts.n_defective,
        "escape_rate": round(counts.escape_rate, 4),
    }


def _accuracy(groups: Groups, benchmark: dict, name: str) -> dict:
    """Image AUROC under both protocols and pixel AUROC, per category and averaged."""
    accuracy = {}
    for category in benchmark["counts"]:
        image = {}
        for protocol in ("full", "gate"):
            cells = [groups[(name, category, s, protocol, "test")] for s in benchmark["seeds"]]
            image[protocol] = [auroc(cell.scores, cell.defective) for cell in cells]
        accuracy[category] = {
            "image_auroc": _spread(image["full"]),
            "pixel_auroc": _spread(benchmark["pixel_auroc"][name][category]),
            "image_auroc_gate_fit": _spread(image["gate"]),
        }
    accuracy["mean"] = {
        metric: round(float(np.mean([cell[metric]["mean"] for cell in accuracy.values()])), 4)
        for metric in ("image_auroc", "pixel_auroc", "image_auroc_gate_fit")
    }
    return accuracy


def _gate(groups: Groups, categories: list[str], seeds: list[int], name: str, alpha: float) -> dict:
    """Outcome of the gate at one target, per category, per seed and pooled.

    ``promised_per_seed`` is what the rule would deliver on exchangeable good parts: the
    expected number of false rejects in one seed (all categories) and its standard deviation.
    """
    empty = GateCounts(0, 0, 0, 0)
    per_seed = dict.fromkeys(seeds, empty)
    per_category = {}
    promised_mean = promised_variance = 0.0
    for category in categories:
        total = empty
        thresholds = []
        for seed in seeds:
            threshold, counts = gate_run(groups, name, category, seed, alpha)
            thresholds.append(round(threshold, SCORE_DECIMALS))
            total += counts
            per_seed[seed] += counts
        n_held_out = len(groups[(name, category, seeds[0], "gate", "calibration")].scores)
        mean, variance = promised_false_rejects(n_held_out, alpha, total.n_good // len(seeds))
        promised_mean += mean
        promised_variance += variance
        shift = float(np.mean([good_shift(groups, name, category, seed) for seed in seeds]))
        per_category[category] = _counts_dict(total) | {
            "thresholds": thresholds,
            "good_shift_auroc": round(shift, 4),
        }
    shifts = [cell["good_shift_auroc"] for cell in per_category.values()]
    return {
        "pooled": _counts_dict(sum(per_seed.values(), empty)),
        "promised_per_seed": {
            "false_rejects": round(promised_mean, 1),
            "standard_deviation": round(promised_variance**0.5, 1),
        },
        "good_shift_auroc": round(float(np.mean(shifts)), 4),
        "per_seed": [_counts_dict(counts) for counts in per_seed.values()],
        "per_category": per_category,
    }


def _tradeoff(groups: Groups, categories: list[str], seeds: list[int], name: str) -> list[dict]:
    """Pooled false-reject and escape rates for a sweep of target rates."""
    curve = []
    for alpha in TRADEOFF_TARGETS:
        pooled = GateCounts(0, 0, 0, 0)
        for category in categories:
            for seed in seeds:
                pooled += gate_run(groups, name, category, seed, alpha)[1]
        curve.append(
            {
                "target": alpha,
                "false_reject_rate": round(pooled.false_reject_rate, 4),
                "escape_rate": round(pooled.escape_rate, 4),
            }
        )
    return curve


def summarise(rows: list[ScoreRow], benchmark: dict, latency: dict) -> dict:
    """All numbers quoted in the README, as one JSON-ready dictionary."""
    groups = group_scores(rows)
    categories = list(benchmark["counts"])
    seeds = benchmark["seeds"]
    reference = CONFIGS[0].name
    target = TARGET_FALSE_REJECT
    return {
        "seeds": seeds,
        "calibration_fraction": benchmark["calibration_fraction"],
        "target_false_reject": target,
        "counts": benchmark["counts"],
        "held_out_good": {
            category: len(groups[(reference, category, seeds[0], "gate", "calibration")].scores)
            for category in categories
        },
        "benchmark": {key: benchmark[key] for key in ("device", "wall_clock_seconds")},
        "accuracy": {c.name: _accuracy(groups, benchmark, c.name) for c in CONFIGS},
        "gate": {c.name: _gate(groups, categories, seeds, c.name, target) for c in CONFIGS},
        "tradeoff": {c.name: _tradeoff(groups, categories, seeds, c.name) for c in CONFIGS},
        "cost": {c.name: latency["configs"][c.name] for c in CONFIGS},
        "cost_setup": {key: value for key, value in latency.items() if key != "configs"},
    }


def _pct(fraction: float, decimals: int = 1) -> str:
    return f"{100 * fraction:.{decimals}f}"


def _ratio(count: int, total: int) -> str:
    return f"{count} / {total} ({_pct(count / total)} %)"


def _published(config: str, metric: str, categories: list[str]) -> str:
    """Published value in %, averaged over ``categories``; a dash when the paper has none."""
    values = PUBLISHED.get(config, {}).get(metric)
    if values is None or not all(category in values for category in categories):
        return MISSING
    return f"{np.mean([values[c] for c in categories]):.1f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table: first column left-aligned, all others right-aligned."""
    align = [":--"] + ["--:"] * (len(header) - 1)
    return "\n".join("| " + " | ".join(line) + " |" for line in [header, align, *rows])


def accuracy_table(summary: dict) -> str:
    """Image / pixel AUROC of the ``full`` protocol next to the published values."""
    categories = list(summary["counts"])
    header = ["Category"]
    for config in CONFIGS:
        header += [config.label, "paper"] if config.name in PUBLISHED else [config.label]
    rows = []
    for category in [*categories, "mean"]:
        is_mean = category == "mean"
        row = ["**mean**" if is_mean else category]
        for config in CONFIGS:
            cell = summary["accuracy"][config.name][category]
            image = cell["image_auroc"] if is_mean else cell["image_auroc"]["mean"]
            pixel = cell["pixel_auroc"] if is_mean else cell["pixel_auroc"]["mean"]
            row.append(f"{_pct(image)} / {_pct(pixel)}")
            if config.name in PUBLISHED:
                over = categories if is_mean else [category]
                paper = [_published(config.name, metric, over) for metric in ("image", "pixel")]
                row.append(" / ".join(paper))
        rows.append(row)
    return _table(header, rows)


def _worst_category(gate: dict) -> tuple[str, dict]:
    return max(gate["per_category"].items(), key=lambda item: item[1]["escape_rate"])


def _seed_range(gate: dict, key: str) -> str:
    values = [seed[key] for seed in gate["per_seed"]]
    return f"{min(values)} to {max(values)}"


def gate_table(summary: dict) -> str:
    """Outcome of the 5 % gate: counts summed over the seeds, and their per-seed range."""
    header = [
        "Configuration",
        "False rejects (good parts refused)",
        "per seed",
        "Escapes (defective parts passed)",
        "per seed",
        "Worst category for escapes",
        "Shift (%)",
    ]
    reference = summary["gate"][CONFIGS[0].name]
    promised = reference["promised_per_seed"]
    n_good = reference["per_seed"][0]["n_good"]
    rows = [
        [
            "*what the rule promises*",
            f"{_pct(promised['false_rejects'] / n_good)} %",
            f"{promised['false_rejects']:.1f} ± {promised['standard_deviation']:.1f}",
            MISSING,
            MISSING,
            MISSING,
            "50.0",
        ]
    ]
    for config in CONFIGS:
        gate = summary["gate"][config.name]
        pooled = gate["pooled"]
        worst_name, worst = _worst_category(gate)
        rows.append(
            [
                config.label,
                _ratio(pooled["false_rejects"], pooled["n_good"]),
                _seed_range(gate, "false_rejects"),
                _ratio(pooled["escapes"], pooled["n_defective"]),
                _seed_range(gate, "escapes"),
                f"{worst_name}: {_ratio(worst['escapes'], worst['n_defective'])}",
                _pct(gate["good_shift_auroc"]),
            ]
        )
    return _table(header, rows)


def cost_table(summary: dict) -> str:
    """CPU cost of each configuration, with the measurement setup as a caption line."""
    setup = summary["cost_setup"]
    n_train = summary["counts"][setup["category"]]["train_good"]
    caption = (
        f"Measured on `{setup['category']}` ({n_train} good training images), "
        f"{setup['cpu']}, {setup['threads']} threads, PyTorch {setup['torch'].split('+')[0]}, "
        f"while other programs were using {setup['other_cpu_load_percent']} % of the CPU."
    )
    header = [
        "Configuration",
        "Mean image AUROC (%)",
        "Fit (s)",
        "Latency median (ms)",
        "Latency p95 (ms)",
        "Detector (MB)",
        "Peak RSS (MB)",
    ]
    rows = []
    for config in CONFIGS:
        cost = summary["cost"][config.name]
        rows.append(
            [
                config.label,
                _pct(summary["accuracy"][config.name]["mean"]["image_auroc"]),
                f"{cost['fit_seconds']:.1f}",
                f"{cost['median_ms']:.1f}",
                f"{cost['p95_ms']:.1f}",
                f"{cost['detector_megabytes']:.1f}",
                f"{cost['peak_rss_megabytes']}",
            ]
        )
    return caption + "\n\n" + _table(header, rows)


def data_table(summary: dict) -> str:
    header = ["Category", "Train good", "of which held out", "Test good", "Test defective"]
    rows = []
    for category, counts in summary["counts"].items():
        rows.append(
            [
                category,
                str(counts["train_good"]),
                str(summary["held_out_good"][category]),
                str(counts["test_good"]),
                str(counts["test_defective"]),
            ]
        )
    totals = [sum(int(row[column]) for row in rows) for column in range(1, 5)]
    rows.append(["**total**", *map(str, totals)])
    return _table(header, rows)


def _by_escapes(summary: dict) -> list:
    return sorted(CONFIGS, key=lambda c: summary["gate"][c.name]["pooled"]["escape_rate"])


def verdict(summary: dict) -> str:
    """The one-sentence result: best and worst gate at the 5 % target."""
    ranked = _by_escapes(summary)
    best, worst = ranked[0], ranked[-1]
    best_gate = summary["gate"][best.name]["pooled"]
    worst_gate = summary["gate"][worst.name]["pooled"]
    target = summary["target_false_reject"]
    return (
        f"**With a threshold set from held-out good parts only (target: {_pct(target, 0)} % false "
        f"rejects), {best.label} lets {_ratio(best_gate['escapes'], best_gate['n_defective'])} "
        f"defective parts through but refuses "
        f"{_ratio(best_gate['false_rejects'], best_gate['n_good'])} good parts, "
        f"{best_gate['false_reject_rate'] / target:.1f} times the target; {worst.label} stays at "
        f"{_ratio(worst_gate['false_rejects'], worst_gate['n_good'])} false rejects and lets "
        f"{_ratio(worst_gate['escapes'], worst_gate['n_defective'])} defective parts through.**"
    )


def findings(summary: dict) -> str:
    """Three findings stated with the numbers of the tables."""
    best = _by_escapes(summary)[0]
    gate = summary["gate"][best.name]
    worst_name, worst = _worst_category(gate)
    worst_auroc = summary["accuracy"][best.name][worst_name]["image_auroc"]["mean"]
    promised = gate["promised_per_seed"]
    limit = promised["false_rejects"] + 2 * promised["standard_deviation"]
    realised = []
    beyond = []
    for config in CONFIGS:
        config_gate = summary["gate"][config.name]
        realised.append(f"{config.label} {_pct(config_gate['pooled']['false_reject_rate'])} %")
        if min(seed["false_rejects"] for seed in config_gate["per_seed"]) > limit:
            beyond.append(config.label)
    gaps = [
        (summary["accuracy"][name][category][f"{metric}_auroc"]["mean"] * 100 - value, name)
        for name, metrics in PUBLISHED.items()
        for metric, values in metrics.items()
        for category, value in values.items()
        if category in summary["counts"]
    ]
    lines = [
        f"- **AUROC hides the operating point.** On {worst_name}, {best.label} reaches "
        f"{_pct(worst_auroc)} % image AUROC and still passes "
        f"{_ratio(worst['escapes'], worst['n_defective'])} defective parts at the "
        f"{_pct(summary['target_false_reject'], 0)} % target.",
        f"- **The false-reject promise is not always kept.** On exchangeable good parts the "
        f"rule would refuse {promised['false_rejects']:.1f} ± "
        f"{promised['standard_deviation']:.1f} of the {gate['per_seed'][0]['n_good']} good "
        f"test parts per seed. Realised rates: {', '.join(realised)}. Above the promise by "
        f"more than two standard deviations in every seed: {', '.join(beyond) or 'none'}.",
    ]
    if gaps:
        largest_gap, largest_name = min(gaps)
        labels = {c.name: c.label for c in CONFIGS}
        within_one = sum(abs(gap) <= 1.0 for gap, _ in gaps)
        lines.append(
            f"- **The reimplementation tracks the papers.** {within_one} of the {len(gaps)} "
            f"published per-category values are matched within 1 point; the largest gap is "
            f"{largest_gap:+.1f} points ({labels[largest_name]})."
        )
    return "\n".join(lines)


def readme_blocks(summary: dict) -> dict[str, str]:
    """Generated README fragments, keyed by the marker name that delimits them."""
    return {
        "verdict": verdict(summary),
        "findings": findings(summary),
        "accuracy": accuracy_table(summary),
        "gate": gate_table(summary),
        "cost": cost_table(summary),
        "data": data_table(summary),
    }
