"""Checks that need the MVTec AD subset and the pretrained weights (skipped without them)."""

from pathlib import Path

import numpy as np
import pytest

from vqgate.backbone import FrozenBackbone
from vqgate.experiment import CONFIG_BY_NAME, CategoryRun
from vqgate.metrics import auroc
from vqgate.report import group_scores, read_scores

pytestmark = pytest.mark.dataset

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "mvtec_ad"
CATEGORY = "bottle"


@pytest.fixture(scope="module")
def cpu_run():
    """PaDiM and PatchCore R18-1% refitted on bottle, seed 0, on the CPU."""
    if not (DATA_ROOT / CATEGORY).is_dir():
        pytest.skip("MVTec AD subset not downloaded (scripts/fetch_mvtec.py)")
    configs = [CONFIG_BY_NAME["padim_r18"], CONFIG_BY_NAME["patchcore_r18_1"]]
    run = CategoryRun(CATEGORY, DATA_ROOT, seeds=(0,))
    run.run(FrozenBackbone("resnet18"), configs, "cpu")
    return group_scores(run.rows)


@pytest.fixture(scope="module")
def committed():
    return group_scores(read_scores(REPO / "results" / "scores"))


@pytest.mark.parametrize("protocol", ["full", "gate"])
def test_padim_scores_on_cpu_match_the_committed_scores(cpu_run, committed, protocol):
    key = ("padim_r18", CATEGORY, 0, protocol, "test")
    assert cpu_run[key].images == committed[key].images
    np.testing.assert_allclose(cpu_run[key].scores, committed[key].scores, rtol=2e-3)


def test_patchcore_ranking_on_cpu_matches_the_committed_scores(cpu_run, committed):
    key = ("patchcore_r18_1", CATEGORY, 0, "full", "test")
    recomputed = auroc(cpu_run[key].scores, cpu_run[key].defective)
    reference = auroc(committed[key].scores, committed[key].defective)
    assert recomputed == pytest.approx(reference, abs=0.005)
    assert np.corrcoef(cpu_run[key].scores, committed[key].scores)[0, 1] > 0.99
