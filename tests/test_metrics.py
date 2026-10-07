"""AUROC and ROC curve cross-checked against scikit-learn."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve as sklearn_roc_curve

from vqgate.metrics import auroc, roc_curve


def scores_with_ties(seed, n=400):
    rng = np.random.default_rng(seed)
    labels = rng.random(n) < 0.3
    scores = np.round(rng.normal(size=n) + 0.8 * labels, 1)  # one decimal: many ties
    return scores, labels


@pytest.mark.parametrize("seed", range(5))
def test_auroc_matches_scikit_learn_with_tied_scores(seed):
    scores, labels = scores_with_ties(seed)
    assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_auroc_extremes():
    labels = np.array([0, 0, 0, 1, 1])
    assert auroc(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), labels) == 1.0
    assert auroc(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), labels) == 0.0
    assert auroc(np.zeros(5), labels) == 0.5


def test_auroc_is_invariant_to_a_strictly_increasing_transform():
    scores, labels = scores_with_ties(seed=7)
    assert auroc(np.exp(0.5 * scores) + 3.0, labels) == pytest.approx(auroc(scores, labels))


def test_auroc_needs_both_classes():
    with pytest.raises(ValueError):
        auroc(np.array([0.1, 0.2]), np.array([1, 1]))


@pytest.mark.parametrize("seed", range(3))
def test_roc_curve_matches_scikit_learn_and_integrates_to_auroc(seed):
    scores, labels = scores_with_ties(seed)
    fpr, tpr, thresholds = roc_curve(scores, labels)
    expected_fpr, expected_tpr, _ = sklearn_roc_curve(labels, scores, drop_intermediate=False)
    np.testing.assert_allclose(fpr, expected_fpr)
    np.testing.assert_allclose(tpr, expected_tpr)
    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0)
    assert area == pytest.approx(auroc(scores, labels))
    assert thresholds[0] == np.inf and np.all(np.diff(thresholds) < 0)


def test_each_roc_point_is_the_outcome_of_rejecting_scores_at_or_above_its_threshold():
    scores, labels = scores_with_ties(seed=11)
    fpr, tpr, thresholds = roc_curve(scores, labels)
    for position in (1, len(thresholds) // 2, len(thresholds) - 1):
        nok = scores >= thresholds[position]
        assert tpr[position] == pytest.approx(nok[labels].mean())
        assert fpr[position] == pytest.approx(nok[~labels].mean())
