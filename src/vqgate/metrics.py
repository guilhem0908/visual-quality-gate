"""Ranking metrics written from the definitions (cross-checked against scikit-learn in tests).

AUROC is the probability that a random defective sample scores higher than a random good
one, ties counting one half (Mann-Whitney U statistic, Hanley and McNeil 1982):

    AUROC = ( sum of the ranks of the positives - n_pos (n_pos + 1) / 2 ) / ( n_pos n_neg )

with average ranks for tied scores.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve; ``labels`` are truthy for defective samples."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(labels).ravel().astype(bool)
    n_positive = int(positive.sum())
    n_negative = positive.size - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC needs at least one good and one defective sample")
    ranks = rankdata(scores, method="average")
    u_statistic = ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
    return float(u_statistic / (n_positive * n_negative))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False-positive rate, true-positive rate and thresholds for the rule ``score >= t``.

    One point per distinct score, from the strictest threshold (nothing rejected) to the
    loosest (everything rejected). The first threshold is +inf.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(labels).ravel().astype(bool)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(sorted_scores)), scores.size - 1]
    true_positives = np.cumsum(positive[order])[last_of_tie]
    false_positives = (last_of_tie + 1) - true_positives
    tpr = np.r_[0.0, true_positives / positive.sum()]
    fpr = np.r_[0.0, false_positives / (~positive).sum()]
    thresholds = np.r_[np.inf, sorted_scores[last_of_tie]]
    return fpr, tpr, thresholds
