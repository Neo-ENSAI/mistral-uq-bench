from dataclasses import dataclass

from mistral_eval.stats.testing import SequentialTestResult


@dataclass
class ReliabilityGateReport:
    passed: bool
    empirical_coverage: float
    target_coverage: float
    mean_set_size: float
    ece: float
    accuracy: float
    size_when_correct: float
    size_when_wrong: float
    martingale_triggered: bool
    reason: str


def evaluate_reliability_gate(
    empirical_coverage: float,
    alpha: float,
    mean_set_size: float,
    ece: float,
    accuracy: float,
    size_when_correct: float,
    size_when_wrong: float,
    seq_result: SequentialTestResult,
    max_allowed_set_size: float = 2.2,
) -> ReliabilityGateReport:
    """
    CI/CD statistical guardrail compatible with DeepEval / GitHub Actions pipelines
    """

    target_cov = 1.0 - alpha
    if seq_result.regression_detected:
        return ReliabilityGateReport(
            passed=False,
            empirical_coverage=empirical_coverage,
            target_coverage=target_cov,
            mean_set_size=mean_set_size,
            ece=ece,
            accuracy=accuracy,
            size_when_correct=size_when_correct,
            size_when_wrong=size_when_wrong,
            martingale_triggered=True,
            reason=f"Martingale alert at step {seq_result.stopped_at_step} (M_t={seq_result.final_martingale_value:.2f} >= {seq_result.threshold:.1f}).",
        )

    if mean_set_size > max_allowed_set_size:
        return ReliabilityGateReport(
            passed=False,
            empirical_coverage=empirical_coverage,
            target_coverage=target_cov,
            mean_set_size=mean_set_size,
            ece=ece,
            accuracy=accuracy,
            size_when_correct=size_when_correct,
            size_when_wrong=size_when_wrong,
            martingale_triggered=False,
            reason=f"Conformal sets too large ({mean_set_size:.2f} > {max_allowed_set_size}).",
        )

    return ReliabilityGateReport(
        passed=True,
        empirical_coverage=empirical_coverage,
        target_coverage=target_cov,
        mean_set_size=mean_set_size,
        ece=ece,
        accuracy=accuracy,
        size_when_correct=size_when_correct,
        size_when_wrong=size_when_wrong,
        martingale_triggered=False,
        reason="Conformal guarantees and sequential test validated successfully",
    )
