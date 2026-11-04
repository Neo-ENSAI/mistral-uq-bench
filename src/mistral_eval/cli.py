import asyncio
import json
import random
import sys
from pathlib import Path

import nest_asyncio
import numpy as np
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mistral_eval.dataset import split_calibration_test
from mistral_eval.ops.deepeval_metric import evaluate_reliability_gate
from mistral_eval.ops.langfuse_cb import LangfuseTracer
from mistral_eval.ops.reporter import generate_markdown_report
from mistral_eval.schemas import InferenceResult
from mistral_eval.stats.conformal import (
    CHOICES,
    SplitConformalCalibrator,
    evaluate_conformal_sets,
)
from mistral_eval.stats.metrics import compute_ece
from mistral_eval.stats.testing import CoverageMartingaleTester

load_dotenv()

app = typer.Typer(
    help="mistral-conformal-eval: Uncertainty Quantification & Sequential Non-Regression Pipeline"
)
console = Console()
nest_asyncio.apply()


class MockInferenceEngine:
    """
    Fast stochastic simulator for continuous integration and unit testing
    """

    def __init__(self, accuracy: float = 0.85, seed: int = 42):
        self.rng = random.Random(seed)
        self.accuracy = accuracy

    async def generate_batch(
        self, prompts: list[dict[str, str]]
    ) -> list[InferenceResult]:
        results = []
        for item in prompts:
            target = item.get("target", "A")
            is_correct = self.rng.random() < self.accuracy

            raw_scores = [self.rng.uniform(0.01, 0.15) for _ in CHOICES]
            target_idx = CHOICES.index(target)

            if is_correct:
                raw_scores[target_idx] += self.rng.uniform(0.65, 0.90)
            else:
                wrong_idx = self.rng.choice([i for i in range(4) if i != target_idx])
                raw_scores[wrong_idx] += self.rng.uniform(0.65, 0.90)

            total = sum(raw_scores)
            probs = {letter: raw_scores[i] / total for i, letter in enumerate(CHOICES)}
            pred = max(probs, key=probs.get)
            entropy = float(-sum(p * np.log(max(p, 1e-12)) for p in probs.values()))

            results.append(
                InferenceResult(
                    prompt_id=item["id"],
                    model_name="mock-simulator",
                    engine_type="mock",
                    raw_text=pred,
                    predicted_choice=pred,
                    choice_probs=probs,
                    predictive_entropy=entropy,
                    tokens=[],
                    latency_ms=self.rng.uniform(15.0, 35.0),
                    throughput_tok_per_sec=self.rng.uniform(90.0, 140.0),
                )
            )

        return results


def load_dataset_records(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Dataset file not found at: {file_path}"
        )
        raise typer.Exit(code=1)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@app.command()
def evaluate(
    data_path: str = typer.Option(
        "data/sample_eval.jsonl", "--data", "-d", help="Path to input JSONL dataset"
    ),
    engine: str = typer.Option(
        "mock", "--engine", "-e", help="Inference backend: 'mock', 'api', or 'vllm'"
    ),
    model_name: str = typer.Option(
        "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "--model",
        "-m",
        help="Mistral model identifier",
    ),
    quantization: str | None = typer.Option(
        None,
        "--quantization",
        "-q",
        help="Quantization method for vLLM (e.g. 'awq')",
    ),
    method: str = typer.Option(
        "lac", "--method", help="Conformal prediction algorithm: 'lac' or 'aps'"
    ),
    alpha: float = typer.Option(
        0.05,
        "--alpha",
        "-a",
        help="Nominal error rate (e.g., 0.05 for 95% target coverage)",
    ),
    alpha_alt: float = typer.Option(
        0.12,
        "--alpha-alt",
        help="Alternative error rate under H1 for sequential testing",
    ),
    delta: float = typer.Option(
        0.05,
        "--delta",
        help="Type-I error tolerance for Ville's inequality threshold (1/delta)",
    ),
    cal_ratio: float = typer.Option(
        0.5, "--cal-ratio", help="Proportion of the dataset allocated to calibration"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    report_path: str = typer.Option(
        "reports/latest_benchmark.md",
        "--report",
        help="Destination path for Markdown CI report",
    ),
):
    """
    Run the end-to-end conformal calibration and sequential non-regression evaluation pipeline
    """

    if engine == "vllm" and quantization is None and "awq" in model_name.lower():
        quantization = "awq"

    console.print(
        Panel.fit(
            f"[bold cyan]MISTRAL-UQ-BENCH[/bold cyan]\n"
            f"[dim]Engine:[/dim] [yellow]{engine}[/yellow] | [dim]Model:[/dim] {model_name}\n"
            f"[dim]Target Coverage (1 - α):[/dim] [green]{(1 - alpha) * 100:.1f}%[/green] | "
            f"[dim]Method:[/dim] [magenta]{method.upper()}[/magenta]",
            border_style="cyan",
        )
    )

    # Dataset loading
    records = load_dataset_records(data_path)
    d_cal, d_test = split_calibration_test(records, cal_ratio=cal_ratio, seed=seed)
    console.print(
        f"Dataset loaded: [bold]{len(d_cal)}[/bold] calibration samples | [bold]{len(d_test)}[/bold] evaluation samples"
    )

    # Engine initialization
    if engine == "mock":
        runner = MockInferenceEngine(seed=seed)
    elif engine == "api":
        from mistral_eval.engine.api_runner import MistralAPIEngine

        runner = MistralAPIEngine(model_name=model_name)
    elif engine == "vllm":
        from mistral_eval.engine.vllm_runner import VLLMInferenceEngine

        runner = VLLMInferenceEngine(
            model_name=model_name,
            quantization=quantization,
        )
    else:
        console.print(
            f"[bold red]Error:[/bold red] Unsupported engine type '{engine}'."
        )
        raise typer.Exit(code=1)

    tracer = LangfuseTracer()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        t_cal = progress.add_task(
            "[yellow]Running inference on calibration set (D_cal) ...", total=None
        )
        cal_results = asyncio.run(runner.generate_batch(d_cal))
        progress.update(t_cal, description="[green]Inference on D_cal completed")

        calibrator = SplitConformalCalibrator(alpha=alpha, method=method)
        cal_probs = [r.choice_probs for r in cal_results]
        cal_targets = [item["target"] for item in d_cal]
        q_hat = calibrator.fit(cal_probs, cal_targets)
        console.print(
            f"Calibrated non-conformity threshold: [bold magenta]q_hat = {q_hat:.4f}[/bold magenta]"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        t_test = progress.add_task(
            "[yellow]Running inference on evaluation set (D_test)...", total=None
        )
        test_results = asyncio.run(runner.generate_batch(d_test))
        progress.update(t_test, description="[green]Inference on D_test completed")

    test_probs = [r.choice_probs for r in test_results]
    test_targets = [item["target"] for item in d_test]
    prediction_sets = calibrator.predict_sets(test_probs)

    # Binary coverage stream for sequential martingale monitoring
    is_covered_stream = [
        target in c_set for target, c_set in zip(test_targets, prediction_sets)
    ]

    # Log evaluation steps to Langfuse
    for res, c_set, target in zip(test_results, prediction_sets, test_targets):
        tracer.log_evaluation_step(res, c_set, target)
    tracer.flush()

    # Sequential martingale testing
    martingale_tester = CoverageMartingaleTester(
        alpha=alpha, alpha_alt=alpha_alt, type1_error_delta=delta
    )
    seq_result = martingale_tester.run_stream(is_covered_stream)

    stopped_at = seq_result.stopped_at_step
    metrics = evaluate_conformal_sets(
        prediction_sets[:stopped_at], test_targets[:stopped_at]
    )

    # Compute additional metrics: ECE and mean throughput
    confidences = np.array([max(p.values()) for p in test_probs[:stopped_at]])
    accuracies = np.array(
        [
            1.0 if res.predicted_choice == target else 0.0
            for res, target in zip(test_results[:stopped_at], test_targets[:stopped_at])
        ]
    )
    ece_val = compute_ece(confidences, accuracies, n_bins=10)
    accuracy = float(np.mean(accuracies))
    sizes = [len(s) for s in prediction_sets[:stopped_at]]
    size_when_correct = float(np.mean([s for s, a in zip(sizes, accuracies) if a == 1.0]))
    size_when_wrong   = float(np.mean([s for s, a in zip(sizes, accuracies) if a == 0.0]))
    mean_throughput = float(np.mean([r.throughput_tok_per_sec for r in test_results]))


    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    # Generate Markdown report
    gate_report = evaluate_reliability_gate(
        empirical_coverage=metrics["empirical_coverage"],
        alpha=alpha,
        mean_set_size=metrics["mean_set_size"],
        ece=ece_val,
        seq_result=seq_result,
    )
    generate_markdown_report(
        model_name=model_name,
        engine_type=engine,
        method=method,
        q_hat=q_hat,
        throughput=mean_throughput,
        gate=gate_report,
        output_path=report_path,
    )

    # Console Summary Dashboard
    table = Table(
        title="Conformal Evaluation & Reliability Summary", border_style="cyan"
    )
    table.add_column("Evaluated Metric / Diagnostic", style="bold white")
    table.add_column("Observed Value", justify="right")
    table.add_column("Target / Bound", justify="right", style="dim")

    status_color = "red" if seq_result.regression_detected else "green"
    status_text = (
        "🚨 REGRESSION DETECTED" if seq_result.regression_detected else "✅ CONFORMAL"
    )

    table.add_row(
        "Martingale Status (Ville)",
        f"[{status_color}]{status_text}[/{status_color}]",
        f"Threshold B = {seq_result.threshold:.1f}",
    )
    table.add_row(
        "Stream Stopping Step",
        f"{stopped_at} / {len(d_test)} questions",
        "Max stream length",
    )
    table.add_row(
        "Final Martingale Value (M_t)",
        f"{seq_result.final_martingale_value:.3e}",
        f"< {seq_result.threshold:.1f}",
    )
    cov_color = "green" if metrics["empirical_coverage"] >= (1.0 - alpha) else "yellow"
    table.add_row(
        "Empirical Coverage",
        f"[{cov_color}]{metrics['empirical_coverage'] * 100:.2f}%[/{cov_color}]",
        f"≥ {(1.0 - alpha) * 100:.1f}%",
    )
    table.add_row(
        "Mean Set Size |C(X)|",
        f"{metrics['mean_set_size']:.2f} choices",
        "≤ 2.20 (sharpness)",
    )
    table.add_row(
        "Expected Calibration Error (ECE)", f"{ece_val:.4f}", "Lower is better"
    )
    table.add_row("Mean Inference Throughput", f"{mean_throughput:.1f} tok/s", "-")

    console.print("\n", table)
    console.print(
        f"📝 Markdown report saved to: [bold green]{report_path}[/bold green]\n"
    )

    if not gate_report.passed:
        console.print(
            Panel(
                f"[bold red]CI RELIABILITY GATE FAILED:[/bold red] {gate_report.reason}",
                border_style="red",
            )
        )
        sys.exit(1)
    else:
        console.print(
            "[bold green]SUCCESS: All statistical and operational guarantees are satisfied[/bold green]\n"
        )


if __name__ == "__main__":
    app()
