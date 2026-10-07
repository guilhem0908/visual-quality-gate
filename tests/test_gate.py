"""Conformal threshold: order statistic, sample-size limit and simulated guarantee."""

import numpy as np
import pytest

from vqgate.gate import (
    GateCounts,
    conformal_rank,
    conformal_threshold,
    gate_counts,
    min_calibration_size,
    promised_false_rejects,
    rejected,
)


def test_threshold_is_the_kth_smallest_held_out_score():
    scores = np.random.default_rng(0).permutation(np.arange(1.0, 40.0))  # 39 scores: 1..39
    assert conformal_threshold(scores, alpha=0.05) == 38.0  # k = ceil(40 * 0.95) = 38
    assert conformal_threshold(scores, alpha=0.10) == 36.0  # k = ceil(40 * 0.90) = 36
    assert conformal_threshold(scores, alpha=0.50) == 20.0


def test_rank_does_not_overshoot_on_exact_products():
    assert conformal_rank(19, 0.05) == 19  # 20 * 0.95 is 19.000000000000004 in floating point
    assert conformal_rank(99, 0.01) == 99
    assert conformal_rank(42, 0.05) == 41


def test_too_few_held_out_parts_cannot_certify_the_target():
    assert min_calibration_size(0.05) == 19
    assert min_calibration_size(0.01) == 99
    assert conformal_threshold(np.arange(19.0), alpha=0.05) == 18.0
    with pytest.raises(ValueError, match="at least 19"):
        conformal_threshold(np.arange(18.0), alpha=0.05)
    with pytest.raises(ValueError):
        conformal_threshold(np.arange(50.0), alpha=1.5)


def test_simulated_false_reject_rate_equals_one_minus_k_over_n_plus_one():
    rng = np.random.default_rng(1)
    n_held_out, alpha, trials = 39, 0.05, 20_000
    held_out = rng.gamma(shape=3.0, size=(trials, n_held_out))
    new_good_part = rng.gamma(shape=3.0, size=trials)
    thresholds = np.array([conformal_threshold(row, alpha) for row in held_out])
    rate = np.mean(new_good_part > thresholds)
    assert rate == pytest.approx(1 - 38 / 40, abs=0.006)  # four standard errors


def test_promised_false_reject_count_matches_simulation_in_mean_and_variance():
    rng = np.random.default_rng(3)
    n_held_out, alpha, n_new, trials = 48, 0.05, 32, 40_000
    held_out = rng.normal(size=(trials, n_held_out))
    thresholds = np.sort(held_out, axis=1)[:, 47 - 1]  # k = ceil(49 * 0.95) = 47
    false_rejects = (rng.normal(size=(trials, n_new)) > thresholds[:, None]).sum(axis=1)
    mean, variance = promised_false_rejects(n_held_out, alpha, n_new)
    assert mean == pytest.approx(n_new * 2 / 49)
    assert false_rejects.mean() == pytest.approx(mean, rel=0.03)
    assert false_rejects.var() == pytest.approx(variance, rel=0.05)
    assert variance > mean * (1 - 2 / 49)  # wider than a binomial with the same mean


def test_stricter_target_never_lowers_the_threshold():
    scores = np.random.default_rng(2).normal(size=200)
    thresholds = [conformal_threshold(scores, alpha) for alpha in (0.5, 0.2, 0.1, 0.05, 0.01)]
    assert thresholds == sorted(thresholds)


def test_counts_use_a_strict_inequality_and_name_both_error_types():
    scores = np.array([0.2, 0.5, 0.9, 0.5, 0.7, 0.1])
    defective = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert rejected(scores, 0.5).tolist() == [False, False, True, False, True, False]
    counts = gate_counts(scores, defective, threshold=0.5)
    assert counts == GateCounts(n_good=3, n_defective=3, false_rejects=1, escapes=2)
    assert counts.false_reject_rate == pytest.approx(1 / 3)
    assert counts.escape_rate == pytest.approx(2 / 3)


def test_counts_add_across_categories():
    total = GateCounts(20, 60, 1, 3) + GateCounts(41, 119, 2, 30)
    assert total == GateCounts(61, 179, 3, 33)
