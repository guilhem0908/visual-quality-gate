"""The gate itself: an accept/reject threshold set from good parts only.

A line has plenty of good parts and almost no catalogued defects, so the threshold cannot be
tuned on defects. Split conformal prediction gives a distribution-free alternative
(Vovk, Gammerman and Shafer, 2005; Angelopoulos and Bates, 2023, Sec. 1): score n held-out good
parts that the detector never saw, sort them s_(1) <= ... <= s_(n), and take

    t = s_(k),    k = ceil( (n + 1) (1 - alpha) ).

If future good parts are exchangeable with the held-out ones, P(score > t) <= alpha, with
equality at 1 - k / (n + 1) for continuous scores. The rule needs k <= n, that is
n >= 1/alpha - 1: 19 held-out parts for alpha = 5 %, 99 for alpha = 1 %.

Vocabulary: a *false reject* is a good part sent to scrap or rework (false positive); an
*escape* is a defective part that passes the gate (false negative).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def min_calibration_size(alpha: float) -> int:
    """Smallest n for which the conformal threshold at level ``alpha`` is finite."""
    return math.ceil(1.0 / alpha) - 1


def conformal_rank(n_calibration: int, alpha: float) -> int:
    """k = ceil((n + 1)(1 - alpha)), computed so that floating point cannot overshoot."""
    return math.ceil(round((n_calibration + 1) * (1.0 - alpha), 9))


def conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Threshold t such that a new good part is rejected (score > t) with probability <= alpha."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(np.asarray(calibration_scores, dtype=np.float64).ravel())
    rank = conformal_rank(scores.size, alpha)
    if rank > scores.size:
        raise ValueError(
            f"{scores.size} held-out good parts cannot certify alpha = {alpha}; "
            f"at least {min_calibration_size(alpha)} are needed"
        )
    return float(scores[rank - 1])


def promised_false_rejects(n_held_out: int, alpha: float, n_new: int) -> tuple[float, float]:
    """Mean and variance of the false rejects among ``n_new`` good parts when the rule holds.

    For one held-out draw, the false-reject probability p of the threshold s_(k) follows a
    Beta(a, b) law with a = n + 1 - k and b = k (distribution of a uniform order statistic).
    The count among ``n_new`` exchangeable good parts is therefore Beta-Binomial:

        mean = n_new a / (a + b),
        variance = n_new a b (a + b + n_new) / ( (a + b)^2 (a + b + 1) ).
    """
    k = conformal_rank(n_held_out, alpha)
    a, b = n_held_out + 1 - k, k
    mean = n_new * a / (a + b)
    variance = n_new * a * b * (a + b + n_new) / ((a + b) ** 2 * (a + b + 1))
    return mean, variance


@dataclass(frozen=True)
class GateCounts:
    """Outcome of a gate on labelled parts."""

    n_good: int
    n_defective: int
    false_rejects: int
    escapes: int

    @property
    def false_reject_rate(self) -> float:
        return self.false_rejects / self.n_good

    @property
    def escape_rate(self) -> float:
        return self.escapes / self.n_defective

    def __add__(self, other: GateCounts) -> GateCounts:
        return GateCounts(
            self.n_good + other.n_good,
            self.n_defective + other.n_defective,
            self.false_rejects + other.false_rejects,
            self.escapes + other.escapes,
        )


def rejected(scores: np.ndarray, threshold: float) -> np.ndarray:
    """The verdict: NOK when the score is strictly above the threshold."""
    return np.asarray(scores) > threshold


def gate_counts(scores: np.ndarray, is_defective: np.ndarray, threshold: float) -> GateCounts:
    nok = rejected(scores, threshold)
    defective = np.asarray(is_defective).astype(bool)
    return GateCounts(
        n_good=int((~defective).sum()),
        n_defective=int(defective.sum()),
        false_rejects=int((nok & ~defective).sum()),
        escapes=int((~nok & defective).sum()),
    )
