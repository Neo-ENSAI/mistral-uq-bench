from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MOCK_FILENAME_RE = re.compile(r"mock_acc_(\d+\.\d+)", re.IGNORECASE)
MODEL_ROW_RE = re.compile(r"\*\*Model Backend\*\* \| `([^`]+)` \| `([^`]+)`")
ACCURACY_RE = re.compile(
    r"Model Accuracy[^|]*\|\s*\*{0,2}(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE
)
SET_SIZE_RE = re.compile(
    r"Mean Prediction Set Size\*\* \| \*{0,2}(\d+(?:\.\d+)?)", re.IGNORECASE
)
COVERAGE_RE = re.compile(
    r"Empirical Coverage\*\* \| \*{0,2}(\d+(?:\.\d+)?)%", re.IGNORECASE
)
ECE_RE = re.compile(
    r"Expected Calibration Error \(ECE\)\*\* \| `(\d+(?:\.\d+)?)`", re.IGNORECASE
)


def parse_report(path: Path) -> dict | None:
    """
    Parse one markdown CI report into a metrics record
    """

    text = path.read_text(encoding="utf-8")
    model_row = MODEL_ROW_RE.search(text)
    if model_row is None:
        return None

    accuracy = float(ACCURACY_RE.search(text).group(1))
    if accuracy > 1.0:
        accuracy /= 100.0

    def grab(rx: re.Pattern[str]) -> float:
        return float(rx.search(text).group(1))

    return {
        "file": path.name,
        "model": model_row.group(1),
        "engine": model_row.group(2),
        "set_size": grab(SET_SIZE_RE),
        "coverage_pct": grab(COVERAGE_RE),
        "ece": grab(ECE_RE),
        "accuracy": accuracy,
    }


def collect_points(reports_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Split report records into mock points and real-model points
    """

    mock_points: list[dict] = []
    real_points: list[dict] = []

    for path in sorted(reports_dir.glob("*.md")):
        record = parse_report(path)
        if record is None:
            print(f"[skip] {path.name}: no parseable metrics", file=sys.stderr)
            continue

        mock_match = MOCK_FILENAME_RE.search(path.name)
        if mock_match or "mock" in record["model"].lower():
            record["accuracy"] = (
                float(mock_match.group(1)) if mock_match else record["accuracy"]
            )
            mock_points.append(record)
        else:
            real_points.append(record)

    return mock_points, real_points


def build_figure(mock_points: list[dict], real_points: list[dict], sharpness_target: float = 2.20, ):
    """
    Assemble the plotly sharpness-vs-accuracy figure
    """

    import plotly.graph_objects as go

    fig = go.Figure()

    if mock_points:
        ordered = sorted(mock_points, key=lambda p: p["accuracy"])
        fig.add_trace(
            go.Scatter(
                x=[p["accuracy"] for p in ordered],
                y=[p["set_size"] for p in ordered],
                mode="lines+markers",
                name="Calibration sandbox (mock engine)",
                line={"color": "#7aa2f7", "dash": "dash"},
                marker={"size": 9},
                hovertemplate=(
                    "simulated accuracy: %{x:.2f}<br>|C(X)|: %{y:.2f}"
                    "<extra>mock</extra>"
                ),
            )
        )

    for point in real_points:
        label = point["model"].split("/")[-1].replace("-AWQ", "").replace("-awq", "")
        details = (
            f"{label}<br>engine: {point['engine']}<br>"
            f"coverage: {point['coverage_pct']:.1f}%<br>ECE: {point['ece']:.4f}"
        )
        fig.add_trace(
            go.Scatter(
                x=[point["accuracy"]],
                y=[point["set_size"]],
                mode="markers",
                name=label,
                marker={"size": 13, "symbol": "diamond"},
                text=[details],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.add_hline(
        y=sharpness_target,
        line={"color": "#ff9e64", "dash": "dot"},
        annotation_text=f"Sharpness reference |C| = {sharpness_target:.2f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=(
            "Sharpness of conformal sets vs model accuracy"
            "<br><sup>95% target coverage, split conformal</sup>"
        ),
        xaxis_title="Model accuracy on D_test",
        yaxis_title="Mean prediction set size |C(X)|",
        xaxis_range=[0.0, 1.0],
        yaxis_range=[0.5, 4.2],
        template="plotly_white",
        width=880,
        height=560,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def save_figure(fig, output_path: Path) -> None:
    """
    Write the figure as HTML (default) or PNG via kaleido
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".png":
        fig.write_image(str(output_path), scale=2)
    else:
        fig.write_html(str(output_path), include_plotlyjs="cdn")


def main() -> None:
    """
    CLI entrypoint of the sharpness curve plotter
    """

    parser = argparse.ArgumentParser(
        description="Plot conformal set size vs model accuracy from CI reports."
    )
    parser.add_argument(
        "--reports", default="reports", help="Directory containing markdown reports"
    )
    parser.add_argument(
        "--out",
        default="reports/figures/sharpness_curve.html",
        help="Output figure path (.html or .png)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse and print points without plotting"
    )
    args = parser.parse_args()

    mock_points, real_points = collect_points(Path(args.reports))
    if not mock_points and not real_points:
        sys.exit(f"No parseable reports found in {args.reports}/")

    print(f"\n{'model':<44s} {'accuracy':>8s} {'|C(X)|':>7s} {'cov%':>6s} {'ECE':>7s}")
    print("-" * 78)
    for label, points in (("MOCK", mock_points), ("REAL", real_points)):
        for point in sorted(points, key=lambda p: p["accuracy"]):
            print(
                f"{label + ': ' + point['model']:<44s} {point['accuracy']:>8.2f} "
                f"{point['set_size']:>7.2f} {point['coverage_pct']:>6.1f} "
                f"{point['ece']:>7.4f}"
            )

    if args.dry_run:
        print("\n[dry-run] parsing OK, run without --dry-run to plot.")
        return

    fig = build_figure(mock_points, real_points)
    output_path = Path(args.out)
    save_figure(fig, output_path)
    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
