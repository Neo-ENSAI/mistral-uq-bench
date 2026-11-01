from pathlib import Path
from mistral_eval.ops.deepeval_metric import ReliabilityGateReport


def generate_markdown_report(
    model_name: str,
    engine_type: str,
    method: str,
    q_hat: float,
    throughput: float,
    gate: ReliabilityGateReport,
    output_path: str = "reports/latest_benchmark.md",
) -> str:
    status_badge = "✅ PASSED" if gate.passed else "❌ FAILED (REGRESSION)"
    md = f"""## ⚡ Mistral-UQ-Bench — Automated Statistical CI Report

**Status:** {status_badge}
**Reason:** {gate.reason}

| Metric | Value | Target / Threshold |
| :--- | :--- | :--- |
| **Model Backend** | `{model_name}` | `{engine_type}` |
| **Conformal Method** | `{method.upper()}` ($\\hat{{q}} = {q_hat:.4f}$) | Split Conformal |
| **Empirical Coverage** | **{gate.empirical_coverage * 100:.2f}%** | $\\ge {gate.target_coverage * 100:.1f}\\%$ |
| **Mean Prediction Set Size** | **{gate.mean_set_size:.2f}** | $\\le 2.20$ |
| **Expected Calibration Error (ECE)** | `{gate.ece:.4f}` | Lower is better |
| **Martingale Drift Alert** | `{gate.martingale_triggered}` | `False` ($M_t < 20$) |
| **Mean Throughput** | `{throughput:.1f} tok/s` | - |
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(md, encoding="utf-8")
    return md
