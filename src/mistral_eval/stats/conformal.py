import math
import numpy as np
from typing import List, Dict

CHOICES = ["A", "B", "C", "D"]


class SplitConformalCalibrator:
    """
    Implement Split Conformal Prediction (LAC and APS scores)
    """

    def __init__(self, alpha: float = 0.05, method: str = "lac"):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha should be in (0, 1).")
        if method not in ("lac", "aps"):
            raise ValueError("method should be 'lac' or 'aps'.")

        self.alpha = alpha
        self.method = method
        self.q_hat: float | None = None

    def fit(self, cal_probs: List[Dict[str, float]], cal_targets: List[str]) -> float:
        """
        Calibrate the corrected empirical quantile q_hat on the calibration set.
        """

        n = len(cal_probs)
        if n == 0 or len(cal_targets) != n:
            raise ValueError("Calibration set empty or invalid")

        scores = []
        for probs, y_true in zip(cal_probs, cal_targets):
            if self.method == "lac":
                scores.append(1.0 - float(probs[y_true]))
            else:
                scores.append(self._aps_score(probs, y_true))

        scores_arr = np.array(scores, dtype=np.float64)
        q_level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        self.q_hat = float(np.quantile(scores_arr, q_level, method="higher"))

        return self.q_hat

    @staticmethod
    def _aps_score(probs: Dict[str, float], target: str) -> float:
        """
        Adaptive Prediction Sets
        """

        sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        cum_prob = 0.0
        for letter, p in sorted_items:
            cum_prob += float(p)
            if letter == target:
                break

        return cum_prob

    def predict_sets(self, test_probs: List[Dict[str, float]]) -> List[List[str]]:
        """
        COnstruct the prediction set C(X) for each test example.
        """

        if self.q_hat is None:
            raise RuntimeError("Call .fit() before .predict_sets().")

        prediction_sets: List[List[str]] = []
        for probs in test_probs:
            if self.method == "lac":
                c_set = [
                    y for y in CHOICES
                    if (1.0 - float(probs[y])) <= self.q_hat
                ]
            else:
                c_set = [
                    y for y in CHOICES
                    if self._aps_score(probs, y) <= self.q_hat
                ]
            if not c_set:
                best_y = max(probs, key=probs.get)
                c_set = [best_y]
            prediction_sets.append(c_set)

        return prediction_sets


def evaluate_conformal_sets(prediction_sets: List[List[str]], test_targets: List[str]) -> Dict[str, float]:

    covered = [y in c_set for c_set, y in zip(prediction_sets, test_targets)]
    sizes = [len(c_set) for c_set in prediction_sets]
    singleton_rate = [1.0 if len(c_set) == 1 else 0.0 for c_set in prediction_sets]
    
    return {
        "empirical_coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean(sizes)),
        "singleton_ratio": float(np.mean(singleton_rate)),
    }
