"""Summary logic on hand-built scores, and consistency of the committed result files."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vqgate.experiment import CONFIGS, ScoreRow
from vqgate.figures import defects_outside_crop, select_hero_frames
from vqgate.report import (
    TARGET_FALSE_REJECT,
    group_scores,
    read_scores,
    readme_blocks,
    summarise,
    write_scores,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_HELD_OUT = 39


def toy_rows(config: str, seed: int, test_good: list[float], test_defective: list[float]):
    """Held-out good scores 1..39, so that the 5 % threshold is exactly 38."""
    rows = [
        ScoreRow(
            config, "widget", seed, "gate", "calibration", "good", f"train/good/{i:03d}.png", s
        )
        for i, s in enumerate(np.arange(1.0, N_HELD_OUT + 1))
    ]
    for protocol in ("full", "gate"):
        rows += [
            ScoreRow(config, "widget", seed, protocol, "test", "good", f"test/good/{i:03d}.png", s)
            for i, s in enumerate(test_good)
        ]
        rows += [
            ScoreRow(config, "widget", seed, protocol, "test", "dent", f"test/dent/{i:03d}.png", s)
            for i, s in enumerate(test_defective)
        ]
    return rows


@pytest.fixture
def toy_summary():
    good = [5.0, 20.0, 38.0, 39.5]  # 38.0 equals the threshold: accepted; 39.5: false reject
    defective = [10.0, 37.0, 50.0, 60.0, 70.0]  # two escapes
    rows = [r for c in CONFIGS for seed in (0, 1) for r in toy_rows(c.name, seed, good, defective)]
    benchmark = {
        "seeds": [0, 1],
        "calibration_fraction": 0.2,
        "counts": {"widget": {"train_good": 195, "test_good": 4, "test_defective": 5}},
        "pixel_auroc": {c.name: {"widget": [0.9, 0.92]} for c in CONFIGS},
        "device": ["cpu"],
        "wall_clock_seconds": 1.0,
    }
    cost = {
        "fit_seconds": 1.0,
        "median_ms": 2.0,
        "p95_ms": 3.0,
        "detector_megabytes": 4.0,
        "peak_rss_megabytes": 5,
    }
    latency = {
        "category": "widget",
        "cpu": "Test CPU",
        "threads": 4,
        "torch": "2.0.0+cpu",
        "other_cpu_load_percent": 7,
        "configs": {c.name: cost for c in CONFIGS},
    }
    return rows, summarise(rows, benchmark, latency)


def test_gate_counts_are_pooled_over_seeds_with_the_conformal_threshold(toy_summary):
    _, summary = toy_summary
    gate = summary["gate"][CONFIGS[0].name]
    assert gate["per_category"]["widget"]["thresholds"] == [38.0, 38.0]
    assert gate["pooled"]["false_rejects"] == 2 and gate["pooled"]["n_good"] == 8
    assert gate["pooled"]["escapes"] == 4 and gate["pooled"]["n_defective"] == 10
    assert gate["pooled"]["escape_rate"] == 0.4
    assert [seed["escapes"] for seed in gate["per_seed"]] == [2, 2]
    assert summary["held_out_good"] == {"widget": N_HELD_OUT}


def test_promise_and_shift_are_reported_next_to_the_realised_counts(toy_summary):
    _, summary = toy_summary
    gate = summary["gate"][CONFIGS[0].name]
    # k = 38 of 39: a new good part is refused with probability 2 / 40, so 0.2 of 4 on average
    assert gate["promised_per_seed"] == {"false_rejects": 0.2, "standard_deviation": 0.5}
    # test-good scores 5, 20, 38, 39.5 against held-out 1..39: (4.5 + 19.5 + 37.5 + 39) / 156
    assert gate["good_shift_auroc"] == pytest.approx(100.5 / 156, abs=1e-4)
    blocks = readme_blocks(summary)
    assert "| *what the rule promises* | 5.0 % | 0.2 ± 0.5 |" in blocks["gate"]
    assert (
        "Above the promise by more than two standard deviations in every seed: none."
        in (blocks["findings"])
    )


def test_auroc_and_tradeoff_follow_from_the_scores(toy_summary):
    _, summary = toy_summary
    accuracy = summary["accuracy"][CONFIGS[0].name]
    # 20 good/defective pairs, 5 wrongly ordered: 10 below 20, 38 and 39.5; 37 below 38 and 39.5
    assert accuracy["widget"]["image_auroc"]["mean"] == pytest.approx(1 - 5 / 20)
    assert accuracy["widget"]["pixel_auroc"] == {"mean": 0.91, "min": 0.9, "max": 0.92}
    curve = summary["tradeoff"][CONFIGS[0].name]
    assert [point["target"] for point in curve][:2] == [0.025, 0.05]
    escapes = [point["escape_rate"] for point in curve]
    false_rejects = [point["false_reject_rate"] for point in curve]
    assert escapes == sorted(escapes, reverse=True) and false_rejects == sorted(false_rejects)
    assert curve[1]["escape_rate"] == 0.4 and curve[1]["false_reject_rate"] == 0.25


def test_readme_tables_quote_raw_counts_next_to_percentages(toy_summary):
    _, summary = toy_summary
    blocks = readme_blocks(summary)
    assert "4 / 10 (40.0 %)" in blocks["gate"] and "2 / 8 (25.0 %)" in blocks["gate"]
    assert "| widget | 195 | 39 | 4 | 5 |" in blocks["data"]
    assert blocks["accuracy"].count("\n") == 3  # header, alignment, one category, mean
    assert "75.0 / 91.0" in blocks["accuracy"]
    assert blocks["cost"].startswith("Measured on `widget` (195 good training images), Test CPU")
    assert "| 75.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5 |" in blocks["cost"]


def test_hero_rule_picks_medians_and_extremes_of_the_score_to_threshold_ratio(toy_summary):
    rows, _ = toy_summary
    frames = select_hero_frames(
        group_scores(rows), ["widget"], CONFIGS[0].name, seed=0, alpha=TARGET_FALSE_REJECT
    )
    assert [frame.score for frame in frames] == [60.0, 20.0, 50.0, 10.0, 39.5]
    assert [frame.outcome for frame in frames] == [
        "correct reject",
        "correct accept",
        "correct reject",
        "ESCAPE: defect passed",
        "FALSE REJECT: good part refused",
    ]
    assert all(frame.threshold == 38.0 for frame in frames)


def test_hero_rule_skips_a_defect_that_the_crop_removed(toy_summary):
    rows, _ = toy_summary
    frames = select_hero_frames(
        group_scores(rows),
        ["widget"],
        CONFIGS[0].name,
        seed=0,
        alpha=TARGET_FALSE_REJECT,
        cropped_away=frozenset({("widget", "test/dent/000.png")}),
    )
    escape = next(frame for frame in frames if frame.role == "lowest-scoring defect")
    assert (escape.image, escape.score) == ("test/dent/001.png", 37.0)


def test_a_defect_with_an_empty_mask_is_reported_as_cropped_away(mvtec_like_root):
    assert defects_outside_crop(mvtec_like_root, ["widget"]) == frozenset()
    mask = mvtec_like_root / "widget" / "ground_truth" / "dent" / "001_mask.png"
    Image.fromarray(np.zeros((224, 224), dtype=np.uint8), "L").save(mask)
    assert defects_outside_crop(mvtec_like_root, ["widget"]) == {("widget", "test/dent/001.png")}


def test_scores_survive_a_round_trip_through_csv(toy_summary, tmp_path):
    rows, _ = toy_summary
    write_scores(rows, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{c.name}.csv" for c in CONFIGS)
    assert sorted(read_scores(tmp_path), key=repr) == sorted(rows, key=repr)


def test_committed_summary_is_reproduced_from_the_committed_scores():
    benchmark = json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))
    latency = json.loads((RESULTS / "latency.json").read_text(encoding="utf-8"))
    committed = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    recomputed = summarise(read_scores(RESULTS / "scores"), benchmark, latency)
    assert json.loads(json.dumps(recomputed)) == committed


def test_committed_hero_frames_follow_from_the_committed_scores():
    hero = json.loads((RESULTS / "hero.json").read_text(encoding="utf-8"))
    frames = select_hero_frames(
        group_scores(read_scores(RESULTS / "scores")),
        list(json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))["counts"]),
        hero["config"],
        hero["seed"],
        hero["target_false_reject"],
        frozenset(map(tuple, hero["defects_outside_crop"])),
    )
    assert [{**asdict(f), "outcome": f.outcome} for f in frames] == hero["frames"]
    assert hero["defects_outside_crop"] == [["grid", "test/glue/008.png"]]


def test_committed_scores_cover_every_configuration_category_seed_and_protocol():
    benchmark = json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))
    groups = group_scores(read_scores(RESULTS / "scores"))
    for config in CONFIGS:
        for category, counts in benchmark["counts"].items():
            n_test = counts["test_good"] + counts["test_defective"]
            for seed in benchmark["seeds"]:
                assert len(groups[(config.name, category, seed, "full", "test")].scores) == n_test
                assert len(groups[(config.name, category, seed, "gate", "test")].scores) == n_test
                held_out = groups[(config.name, category, seed, "gate", "calibration")]
                assert len(held_out.scores) == -(-counts["train_good"] // 5)
                assert not held_out.defective.any()
