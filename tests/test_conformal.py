import numpy as np

from mistral_eval.stats.conformal import (
    CHOICES,
    SplitConformalCalibrator,
    evaluate_conformal_sets,
)


def test_split_conformal_guarantee_monte_carlo():

    rng = np.random.default_rng(12345)
    alpha = 0.10
    coverages = []

    for _ in range(100):
        logits = rng.normal(0, 1.5, size=(400, 4))
        exp_l = np.exp(logits)
        probs_mat = exp_l / exp_l.sum(axis=1, keepdims=True)

        targets = [CHOICES[rng.choice(4, p=probs_mat[i])] for i in range(400)]
        probs_dicts = [
            {CHOICES[k]: float(probs_mat[i, k]) for k in range(4)} for i in range(400)
        ]

        cal = SplitConformalCalibrator(alpha=alpha, method="lac")
        cal.fit(probs_dicts[:200], targets[:200])
        pred_sets = cal.predict_sets(probs_dicts[200:])
        metrics = evaluate_conformal_sets(pred_sets, targets[200:])
        coverages.append(metrics["empirical_coverage"])

    mean_cov = float(np.mean(coverages))
    assert mean_cov >= (1.0 - alpha - 0.015), f"Couverture trop faible : {mean_cov}"


def _confident_probs(true_idx: int, conf: float = 0.97) -> dict[str, float]:

    rest = (1.0 - conf) / 3.0
    return {CHOICES[i]: (conf if i == true_idx else rest) for i in range(4)}


def test_perfect_model_yields_singleton_sets():

    rng = np.random.default_rng(0)
    cal_idx, test_idx = rng.integers(0, 4, 100), rng.integers(0, 4, 100)

    cal = SplitConformalCalibrator(alpha=0.05, method="lac")
    cal.fit(
        [_confident_probs(int(i)) for i in cal_idx],
        [CHOICES[i] for i in cal_idx],
    )
    sets = cal.predict_sets([_confident_probs(int(i)) for i in test_idx])
    m = evaluate_conformal_sets(sets, [CHOICES[i] for i in test_idx])

    assert m["empirical_coverage"] == 1.0
    assert m["mean_set_size"] == 1.0
    assert m["singleton_ratio"] == 1.0


def test_random_model_saturates_sets():

    probs = [{c: 0.25 for c in CHOICES} for _ in range(120)]
    targets = [CHOICES[i % 4] for i in range(120)]

    for method in ("lac", "aps"):
        cal = SplitConformalCalibrator(alpha=0.05, method=method)
        cal.fit(probs[:100], targets[:100])
        sets = cal.predict_sets(probs[100:])
        m = evaluate_conformal_sets(sets, targets[100:])

        assert m["empirical_coverage"] == 1.0
        assert m["mean_set_size"] == 4.0, f"{method} devrait saturer: {m}"


def test_aps_coverage_guarantee_monte_carlo():

    rng = np.random.default_rng(12345)
    alpha, coverages = 0.10, []

    for _ in range(100):
        logits = rng.normal(0, 1.5, size=(400, 4))
        exp_l = np.exp(logits)
        probs_mat = exp_l / exp_l.sum(axis=1, keepdims=True)
        targets = [CHOICES[rng.choice(4, p=probs_mat[i])] for i in range(400)]
        probs_dicts = [
            {CHOICES[k]: float(probs_mat[i, k]) for k in range(4)} for i in range(400)
        ]
        cal = SplitConformalCalibrator(alpha=alpha, method="aps")
        cal.fit(probs_dicts[:200], targets[:200])
        m = evaluate_conformal_sets(
            cal.predict_sets(probs_dicts[200:]), targets[200:]
        )
        coverages.append(m["empirical_coverage"])

    mean_cov = float(np.mean(coverages))
    assert mean_cov >= (1.0 - alpha - 0.02), f"Couverture APS trop faible : {mean_cov}"
