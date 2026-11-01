import numpy as np

from mistral_eval.stats.testing import CoverageMartingaleTester


def test_martingale_stays_bounded_under_nominal_coverage():
    """
    1) Error rate under nominal coverage (true non-coverage = alpha = 0.05):
    The martingale drifts downwards on average (negative drift), stays strictly below
    the threshold (1/delta = 20) and does not trigger any regression.
    """

    rng = np.random.default_rng(seed=42)
    n_questions = 200
    alpha_nominal = 0.05
    alpha_alt = 0.12
    delta = 0.05

    tester = CoverageMartingaleTester(
        alpha=alpha_nominal,
        alpha_alt=alpha_alt,
        type1_error_delta=delta,
    )

    # Simulated under H0: the model is covered in 95% of cases (5% errors)
    is_covered = (
        rng.binomial(n=1, p=1.0 - alpha_nominal, size=n_questions) == 1
    ).tolist()

    result = tester.run_stream(is_covered)

    assert not result.regression_detected
    assert result.stopped_at_step == n_questions
    assert result.final_martingale_value < result.threshold
    assert result.final_martingale_value < 1.0


def test_martingale_detects_drift_and_stops_early():
    """
    2) Drift from a high error rate model (p = 0.30 >> alpha_alt):
    The martingale explodes geometrically, exceeds the threshold (threshold = 20.0)
    within a few dozen questions and stops the stream immediately.
    """

    rng = np.random.default_rng(seed=42)
    alpha_nominal = 0.05
    alpha_alt = 0.12
    p_error_drift = 0.30
    delta = 0.05

    tester = CoverageMartingaleTester(
        alpha=alpha_nominal,
        alpha_alt=alpha_alt,
        type1_error_delta=delta,
    )

    # Simulated under H1: only 70% of questions are covered (30% errors)
    is_covered_drift = (
        rng.binomial(n=1, p=1.0 - p_error_drift, size=200) == 1
    ).tolist()

    result = tester.run_stream(is_covered_drift)

    assert result.regression_detected
    assert result.final_martingale_value >= result.threshold
    assert result.stopped_at_step < 60, (
        f"Stopped too late: {result.stopped_at_step} steps (expected < 60)"
    )
