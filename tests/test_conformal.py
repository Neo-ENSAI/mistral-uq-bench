import numpy as np
from mistral_eval.stats.conformal import SplitConformalCalibrator, evaluate_conformal_sets, CHOICES


def test_split_conformal_guarantee_monte_carlo():
    
    rng = np.random.default_rng(12345)
    alpha = 0.10
    coverages = []

    for _ in range(100):
        logits = rng.normal(0, 1.5, size=(400, 4))
        exp_l = np.exp(logits)
        probs_mat = exp_l / exp_l.sum(axis=1, keepdims=True)

        targets = [
            CHOICES[rng.choice(4, p=probs_mat[i])] for i in range(400)
        ]
        probs_dicts = [
            {CHOICES[k]: float(probs_mat[i, k]) for k in range(4)}
            for i in range(400)
        ]

        cal = SplitConformalCalibrator(alpha=alpha, method="lac")
        cal.fit(probs_dicts[:200], targets[:200])
        pred_sets = cal.predict_sets(probs_dicts[200:])
        metrics = evaluate_conformal_sets(pred_sets, targets[200:])
        coverages.append(metrics["empirical_coverage"])

    mean_cov = float(np.mean(coverages))
    assert mean_cov >= (1.0 - alpha - 0.015), f"Couverture trop faible : {mean_cov}"
