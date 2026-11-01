import numpy as np
from scipy.special import logsumexp
from typing import Dict


CHOICES = ("A", "B", "C", "D")

def extract_choice_probabilities(top_logprobs: Dict[str, float]) -> Dict[str, float]:
    """
    Extract normalized probabilities for the choices A, B, C, D from the top_logprobs
    dictionary with logsumexp normalization
    """

    raw_logprobs = []
    for letter in CHOICES:
        candidates = [
            v for k, v in top_logprobs.items()
            if k.strip().upper() == letter
        ]
        if candidates:
            raw_logprobs.append(float(logsumexp(candidates)))
        else:
            raw_logprobs.append(-20.0)

    arr = np.array(raw_logprobs, dtype=np.float64)
    norm_logprobs = arr - logsumexp(arr)
    probs = np.exp(norm_logprobs)

    return {letter: float(probs[i]) for i, letter in enumerate(CHOICES)}


def compute_predictive_entropy(choice_probs: Dict[str, float]) -> float:
    """
    Compute the Shannon entropy of the choice probability distribution
    """

    p = np.array([choice_probs[c] for c in CHOICES], dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    
    return float(-np.sum(p * np.log(p)))


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10,) -> float:
    """
    Compute the Expected Calibration Error (ECE) over M equidistant bins in [0, 1].
    - confidences: array (n,) of predicted max probabilities max_y p(y|x_i)
    - accuracies: array (n,) binary (1.0 if prediction is correct, 0.0 otherwise)
    """

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece)


def compute_multiclass_brier_score(prob_matrix: np.ndarray, true_indices: np.ndarray) -> float:
    """
    Compute the multiclass Brier score: 1/n sum_i sum_k (p_{i,k} - 1_{y_i = k})^2.
    """

    n, k = prob_matrix.shape
    one_hot = np.zeros((n, k), dtype=np.float64)
    one_hot[np.arange(n), true_indices] = 1.0

    return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))
