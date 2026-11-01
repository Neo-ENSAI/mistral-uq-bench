from dataclasses import dataclass


@dataclass
class SequentialTestResult:
    regression_detected: bool
    stopped_at_step: int
    final_martingale_value: float
    threshold: float
    empirical_error_rate: float
    trajectory: list[float]


class CoverageMartingaleTester:
    """
    Sequential Wald test / E-process based on a likelihood ratio supermartingale
    and Ville's inequality: P_{H0}(exists t >= 1 : M_t >= 1/delta) <= delta.
    Tests H0: non-coverage rate <= alpha against H1: non-coverage rate >= alpha_alt.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        alpha_alt: float = 0.12,
        type1_error_delta: float = 0.05,
    ):
        if not (0.0 < alpha < alpha_alt < 1.0):
            raise ValueError("We should have 0 < alpha < alpha_alt < 1.")
        self.alpha = alpha
        self.alpha_alt = alpha_alt
        self.delta = type1_error_delta
        self.threshold = 1.0 / type1_error_delta

    def run_stream(self, is_covered_stream: list[bool]) -> SequentialTestResult:
        m_t = 1.0
        trajectory = [1.0]
        stopped_step = len(is_covered_stream)
        detected = False
        errors = 0

        for step_idx, covered in enumerate(is_covered_stream, start=1):
            z_t = 0 if covered else 1
            errors += z_t
            if z_t == 1:
                e_t = self.alpha_alt / self.alpha
            else:
                e_t = (1.0 - self.alpha_alt) / (1.0 - self.alpha)

            m_t *= e_t
            trajectory.append(float(m_t))

            if m_t >= self.threshold:
                detected = True
                stopped_step = step_idx
                break

        err_rate = errors / max(stopped_step, 1)

        return SequentialTestResult(
            regression_detected=detected,
            stopped_at_step=stopped_step,
            final_martingale_value=float(m_t),
            threshold=float(self.threshold),
            empirical_error_rate=float(err_rate),
            trajectory=trajectory,
        )
