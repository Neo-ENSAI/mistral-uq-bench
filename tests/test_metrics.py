import numpy as np
import pytest
from mistral_eval.stats.metrics import compute_ece


def test_ece_perfectly_calibrated():
    """
    A perfectly calibrated predictor satisfies E[Accuracy | Confidence = p] = p.
    The theoretical ECE is exactly 0.0.
    """

    rng = np.random.default_rng(seed=42)
    n_per_bin = 10_000

    conf_levels = [0.2, 0.4, 0.6, 0.8]
    conf_list = []
    acc_list = []

    for p in conf_levels:
        conf_list.extend([p] * n_per_bin)
        acc_list.extend(rng.binomial(n=1, p=p, size=n_per_bin))

    confs = np.array(conf_list)
    corrects = np.array(acc_list)

    ece = compute_ece(confs, corrects, n_bins=10)

    assert ece == pytest.approx(0.0, abs=1e-2)


def test_ece_overconfident_always_wrong():
    """
    Model that always predicts with absolute certainty (confidence = 1.0)
    but is always wrong (accuracy = 0.0) yields a maximal ECE of 1.0.
    """

    n_samples = 1_000
    confidences = np.ones(n_samples, dtype=np.float64)
    accuracies = np.zeros(n_samples, dtype=np.float64)

    ece = compute_ece(confidences, accuracies, n_bins=10)

    assert ece == pytest.approx(1.0, abs=1e-6)
