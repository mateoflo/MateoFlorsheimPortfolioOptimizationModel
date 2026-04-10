from __future__ import annotations

import csv
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
BACKTEST_CSV = BASE_DIR / "BackTestResults" / "backtest_results_liquid_10ticker_2024_01_01.csv"

WIDTH = 1600
HEIGHT = 960
MARGIN_LEFT = 120
MARGIN_RIGHT = 80
MARGIN_TOP = 180
MARGIN_BOTTOM = 120


def load_backtest_rows() -> list[dict]:
    with BACKTEST_CSV.open() as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "expected_return": float(row["expected_return"]),
                    "expected_mc_return": float(row["expected_mc_return"]),
                    "realized_return": float(row["realized_return"]),
                    "prediction_accurate": row["prediction_accurate"] == "True",
                    "realized_terminal_value": float(row["realized_terminal_value"]),
                }
            )
    return rows


def svg_header(width: int = WIDTH, height: int = HEIGHT) -> list[str]:
    return [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none" xmlns="http://www.w3.org/2000/svg">',
        f'  <rect width="{width}" height="{height}" fill="#F6F0E6"/>',
        f'  <rect x="48" y="48" width="{width - 96}" height="{height - 96}" rx="30" fill="#FFFDF8" stroke="#D9CDB8" stroke-width="2"/>',
    ]


def svg_footer() -> list[str]:
    return ["</svg>"]


def px(value: float) -> str:
    return f"{value:.2f}"


def x_map(value: float, min_value: float, max_value: float) -> float:
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    ratio = (value - min_value) / (max_value - min_value)
    return MARGIN_LEFT + ratio * plot_width


def y_map(value: float, min_value: float, max_value: float) -> float:
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    ratio = (value - min_value) / (max_value - min_value)
    return HEIGHT - MARGIN_BOTTOM - ratio * plot_height


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[index]


def write_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def generate_calibration_scatter(rows: list[dict]) -> None:
    x_min, x_max = -0.10, 0.55
    y_min, y_max = -0.30, 0.75
    lines = svg_header()
    lines += [
        '  <text x="90" y="120" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="40" font-weight="700">Figure 1. Forecast Calibration Scatter</text>',
        '  <text x="90" y="158" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="20">Empirical expected return versus realized one-year return for the 994 completed 10-ticker backtests</text>',
    ]

    for tick in [-0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50]:
        x = x_map(tick, x_min, x_max)
        lines.append(f'  <line x1="{px(x)}" y1="{MARGIN_TOP}" x2="{px(x)}" y2="{HEIGHT - MARGIN_BOTTOM}" stroke="#E3D9C8" stroke-width="2"/>')
        lines.append(f'  <text x="{px(x)}" y="{HEIGHT - 76}" text-anchor="middle" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="18">{tick*100:.0f}%</text>')
    for tick in [-0.20, 0.00, 0.20, 0.40, 0.60]:
        y = y_map(tick, y_min, y_max)
        lines.append(f'  <line x1="{MARGIN_LEFT}" y1="{px(y)}" x2="{WIDTH - MARGIN_RIGHT}" y2="{px(y)}" stroke="#E3D9C8" stroke-width="2"/>')
        lines.append(f'  <text x="88" y="{px(y + 6)}" text-anchor="end" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="18">{tick*100:.0f}%</text>')

    diag_start_x = x_map(max(x_min, y_min), x_min, x_max)
    diag_start_y = y_map(max(x_min, y_min), y_min, y_max)
    diag_end_x = x_map(min(x_max, y_max), x_min, x_max)
    diag_end_y = y_map(min(x_max, y_max), y_min, y_max)
    lines.append(f'  <line x1="{px(diag_start_x)}" y1="{px(diag_start_y)}" x2="{px(diag_end_x)}" y2="{px(diag_end_y)}" stroke="#183B2D" stroke-width="3" stroke-dasharray="10 10"/>')

    lines.append(f'  <text x="{WIDTH/2:.0f}" y="{HEIGHT - 26}" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700">Model expected return</text>')
    lines.append(f'  <text x="34" y="{HEIGHT/2:.0f}" transform="rotate(-90 34 {HEIGHT/2:.0f})" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700">Realized one-year return</text>')

    for row in rows:
        x = x_map(row["expected_return"], x_min, x_max)
        y = y_map(row["realized_return"], y_min, y_max)
        if row["prediction_accurate"]:
            color = "#1B7A5B"
        elif row["realized_return"] > 0:
            color = "#D98C3A"
        else:
            color = "#B33A3A"
        lines.append(f'  <circle cx="{px(x)}" cy="{px(y)}" r="4.8" fill="{color}" fill-opacity="0.68"/>')

    lines += [
        '  <rect x="1010" y="96" width="26" height="26" rx="6" fill="#1B7A5B"/>',
        '  <text x="1048" y="116" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700">Met or exceeded expected return</text>',
        '  <rect x="1010" y="136" width="26" height="26" rx="6" fill="#D98C3A"/>',
        '  <text x="1048" y="156" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700">Profitable but missed forecast</text>',
        '  <rect x="1010" y="176" width="26" height="26" rx="6" fill="#B33A3A"/>',
        '  <text x="1048" y="196" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700">Finished the year negative</text>',
        '  <text x="90" y="890" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="20">The dense amber region below the diagonal shows the model’s main weakness: many portfolios were profitable, but the expected-return hurdle was set too high.</text>',
    ]
    lines += svg_footer()
    write_svg(FIGURES_DIR / "figure_1_model_workflow.svg", lines)


def generate_forecast_error_histogram(rows: list[dict]) -> None:
    errors = [row["realized_return"] - row["expected_return"] for row in rows]
    min_v, max_v = -0.45, 0.55
    bins = 20
    bin_width = (max_v - min_v) / bins
    counts = [0 for _ in range(bins)]
    for value in errors:
        idx = int((value - min_v) / bin_width)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    max_count = max(counts)

    lines = svg_header()
    lines += [
        '  <text x="90" y="120" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="40" font-weight="700">Figure 5. Forecast Error Distribution</text>',
        '  <text x="90" y="158" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="20">Forecast error is defined as realized return minus expected return for the 10-ticker baseline backtest</text>',
    ]
    plot_left, plot_right = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    plot_top, plot_bottom = MARGIN_TOP, HEIGHT - MARGIN_BOTTOM

    for tick in [-0.40, -0.20, 0.00, 0.20, 0.40]:
        x = x_map(tick, min_v, max_v)
        lines.append(f'  <line x1="{px(x)}" y1="{plot_top}" x2="{px(x)}" y2="{plot_bottom}" stroke="#E3D9C8" stroke-width="2"/>')
        lines.append(f'  <text x="{px(x)}" y="{HEIGHT - 76}" text-anchor="middle" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="18">{tick*100:.0f}%</text>')
    for tick in [0, 50, 100, 150, 200]:
        y = y_map(tick, 0, 200)
        lines.append(f'  <line x1="{plot_left}" y1="{px(y)}" x2="{plot_right}" y2="{px(y)}" stroke="#E3D9C8" stroke-width="2"/>')
        lines.append(f'  <text x="94" y="{px(y + 6)}" text-anchor="end" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="18">{tick}</text>')

    lines.append(f'  <line x1="{px(x_map(0.0, min_v, max_v))}" y1="{plot_top}" x2="{px(x_map(0.0, min_v, max_v))}" y2="{plot_bottom}" stroke="#183B2D" stroke-width="3" stroke-dasharray="10 10"/>')

    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    bar_gap = 6
    bar_width = plot_width / bins - bar_gap
    for i, count in enumerate(counts):
        x = plot_left + i * (plot_width / bins) + bar_gap / 2
        y = plot_bottom - (count / max_count) * plot_height
        height = plot_bottom - y
        color = "#D98C3A" if i < bins // 2 else "#1B7A5B"
        lines.append(f'  <rect x="{px(x)}" y="{px(y)}" width="{px(bar_width)}" height="{px(height)}" rx="8" fill="{color}" fill-opacity="0.88"/>')

    mean_error = sum(errors) / len(errors)
    sd = math.sqrt(sum((value - mean_error) ** 2 for value in errors) / len(errors))
    lines += [
        f'  <text x="{WIDTH/2:.0f}" y="{HEIGHT - 26}" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700">Forecast error (realized return minus expected return)</text>',
        f'  <text x="34" y="{HEIGHT/2:.0f}" transform="rotate(-90 34 {HEIGHT/2:.0f})" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700">Number of portfolios</text>',
        f'  <text x="1050" y="120" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700">Mean forecast error: {mean_error*100:.2f}%</text>',
        f'  <text x="1050" y="152" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700">Std. deviation: {sd*100:.2f}%</text>',
        '  <text x="90" y="890" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="20">The left-heavy histogram confirms the baseline model was usually too optimistic, even though many portfolios still finished positive.</text>',
    ]
    lines += svg_footer()
    write_svg(FIGURES_DIR / "figure_5_forecast_error_distribution.svg", lines)


def generate_hmm_diagram() -> None:
    lines = svg_header()
    lines += [
        '  <text x="90" y="120" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="40" font-weight="700">Figure 6. HMM Regime-Switching Illustration</text>',
        '  <text x="90" y="158" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="20">Methodological figure showing how the model treats returns as switching across latent market states rather than following one stationary regime</text>',
        '  <rect x="150" y="240" width="300" height="420" rx="26" fill="#E6F1EC"/>',
        '  <rect x="650" y="180" width="300" height="480" rx="26" fill="#FCEAD9"/>',
        '  <rect x="1150" y="280" width="300" height="360" rx="26" fill="#EEF1FA"/>',
        '  <text x="300" y="308" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700">State A</text>',
        '  <text x="300" y="344" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="22">Low Vol / Stable Trend</text>',
        '  <text x="300" y="390" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">Lower variance</text>',
        '  <text x="300" y="420" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">Higher persistence</text>',
        '  <text x="300" y="450" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">More moderate tail risk</text>',
        '  <text x="800" y="248" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700">State B</text>',
        '  <text x="800" y="284" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="22">High Vol / Stress Regime</text>',
        '  <text x="800" y="330" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">Higher variance</text>',
        '  <text x="800" y="360" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">Wider outcome dispersion</text>',
        '  <text x="800" y="390" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">Potential correlation spikes</text>',
        '  <text x="1300" y="348" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700">Simulation Layer</text>',
        '  <text x="1300" y="384" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="22">Markov state transitions</text>',
        '  <text x="1300" y="430" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">Daily draws depend on the</text>',
        '  <text x="1300" y="458" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">current latent regime and its</text>',
        '  <text x="1300" y="486" text-anchor="middle" fill="#4C463D" font-family="Arial, Helvetica, sans-serif" font-size="20">transition probabilities</text>',
        '  <path d="M450 450 C540 390, 560 330, 650 320" stroke="#183B2D" stroke-width="5" fill="none"/>',
        '  <polygon points="650,320 632,314 638,332" fill="#183B2D"/>',
        '  <path d="M950 360 C1040 390, 1060 430, 1150 430" stroke="#183B2D" stroke-width="5" fill="none"/>',
        '  <polygon points="1150,430 1134,421 1134,439" fill="#183B2D"/>',
        '  <path d="M950 510 C860 590, 590 610, 450 520" stroke="#183B2D" stroke-width="5" fill="none" stroke-dasharray="12 10"/>',
        '  <polygon points="450,520 467,523 460,507" fill="#183B2D"/>',
        '  <text x="530" y="290" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">p(A→B)</text>',
        '  <text x="1010" y="332" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">state-specific return draws</text>',
        '  <text x="690" y="650" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">p(B→A)</text>',
        '  <text x="90" y="890" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="20">This is a methodological diagram rather than an empirical chart. It communicates why the Monte Carlo engine is regime-aware instead of assuming one fixed volatility environment.</text>',
    ]
    lines += svg_footer()
    write_svg(FIGURES_DIR / "figure_6_hmm_regime_switching.svg", lines)


def generate_monte_carlo_diagram() -> None:
    lines = svg_header()
    lines += [
        '  <text x="90" y="120" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="40" font-weight="700">Figure 7. HMM Monte Carlo Terminal Value Distribution</text>',
        '  <text x="90" y="158" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="20">Illustrative one-year terminal-value density showing how the report can visualize downside risk, central tendency, and tail outcomes</text>',
    ]

    plot_left, plot_right = 140, 1460
    plot_top, plot_bottom = 220, 760
    lines.append(f'  <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#183B2D" stroke-width="4"/>')
    lines.append(f'  <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#183B2D" stroke-width="4"/>')

    def x(value: float) -> float:
        return plot_left + (value - 70) / (190 - 70) * (plot_right - plot_left)

    def y(value: float) -> float:
        return plot_bottom - value * (plot_bottom - plot_top)

    values = []
    for i in range(121):
        xv = 70 + i
        peak1 = 0.72 * math.exp(-((xv - 108) ** 2) / (2 * 11 ** 2))
        peak2 = 0.95 * math.exp(-((xv - 131) ** 2) / (2 * 17 ** 2))
        peak3 = 0.28 * math.exp(-((xv - 165) ** 2) / (2 * 10 ** 2))
        values.append((xv, 0.15 + peak1 + peak2 + peak3))

    max_density = max(v for _, v in values)
    path = []
    for i, (xv, density) in enumerate(values):
        cmd = "M" if i == 0 else "L"
        path.append(f"{cmd} {px(x(xv))} {px(y(density / max_density * 0.82))}")
    area = " ".join(path) + f" L {px(x(values[-1][0]))} {plot_bottom} L {px(x(values[0][0]))} {plot_bottom} Z"
    lines.append(f'  <path d="{area}" fill="#D8E9E2" stroke="#1B7A5B" stroke-width="4"/>')

    markers = [
        (95, "#B33A3A", "5th percentile"),
        (118, "#183B2D", "Median"),
        (124, "#2C5E92", "Mean"),
        (100, "#8B5FBF", "Starting capital"),
    ]
    for value, color, label in markers:
        xpos = x(value)
        lines.append(f'  <line x1="{px(xpos)}" y1="{plot_top}" x2="{px(xpos)}" y2="{plot_bottom}" stroke="{color}" stroke-width="4" stroke-dasharray="10 10"/>')
        lines.append(f'  <text x="{px(xpos + 8)}" y="{plot_top + 36}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">{label}</text>')

    for tick in [80, 100, 120, 140, 160, 180]:
        xpos = x(tick)
        lines.append(f'  <line x1="{px(xpos)}" y1="{plot_bottom}" x2="{px(xpos)}" y2="{plot_bottom + 8}" stroke="#183B2D" stroke-width="3"/>')
        lines.append(f'  <text x="{px(xpos)}" y="{plot_bottom + 34}" text-anchor="middle" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="18">${tick}k</text>')

    lines += [
        f'  <text x="{WIDTH/2:.0f}" y="{HEIGHT - 44}" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700">Terminal portfolio value after one year</text>',
        '  <text x="34" y="470" transform="rotate(-90 34 470)" text-anchor="middle" fill="#183B2D" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700">Relative probability density</text>',
        '  <text x="930" y="860" fill="#6B6256" font-family="Arial, Helvetica, sans-serif" font-size="20">Illustrative methodology chart. Use this to explain how the report interprets median, mean, tail risk, and loss probability.</text>',
    ]
    lines += svg_footer()
    write_svg(FIGURES_DIR / "figure_7_monte_carlo_distribution.svg", lines)


def update_figures_doc() -> None:
    text = """# Project Figures

These figures were created as report-ready SVG assets so they can be placed directly into the project paper, website, or presentation deck.

## Files

- `figure_1_model_workflow.svg`
  - Caption:
    - `Forecast calibration scatter: expected return versus realized one-year return for the 10-ticker baseline backtest.`
- `figure_2_baseline_backtests.svg`
  - Caption:
    - `Original baseline backtest results showing that profitability was materially higher than forecast accuracy.`
- `figure_3_return_model_comparison.svg`
  - Caption:
    - `Comparison of expected-return estimation methods across the 10-ticker point-in-time backtest.`
- `figure_4_upgrade_roadmap.svg`
  - Caption:
    - `Research-based upgrade roadmap for improving expected-return calibration and future model accuracy.`
- `figure_5_forecast_error_distribution.svg`
  - Caption:
    - `Histogram of realized minus expected return across the 10-ticker baseline backtest.`
- `figure_6_hmm_regime_switching.svg`
  - Caption:
    - `Methodological HMM diagram showing latent regime transitions and state-specific return generation.`
- `figure_7_monte_carlo_distribution.svg`
  - Caption:
    - `Illustrative one-year terminal-value distribution for explaining Monte Carlo outputs in the report.`

## Suggested Usage In The Report

- Use `figure_1_model_workflow.svg` in the original-results section when explaining model calibration.
- Use `figure_2_baseline_backtests.svg` in the original-results section.
- Use `figure_3_return_model_comparison.svg` in the upgrade-testing section.
- Use `figure_4_upgrade_roadmap.svg` in the conclusion or future-work section.
- Use `figure_5_forecast_error_distribution.svg` in the model-calibration discussion.
- Use `figure_6_hmm_regime_switching.svg` in the methodology section.
- Use `figure_7_monte_carlo_distribution.svg` in the Monte Carlo explanation section.

## Notes

- Figures 1, 2, 3, and 5 are tied directly to the saved backtest data in this project.
- Figures 6 and 7 are methodological visuals meant to explain the HMM and Monte Carlo framework when raw path/state data is not stored in the current CSV outputs.
"""
    (FIGURES_DIR / "FIGURES.md").write_text(text)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    rows = load_backtest_rows()
    generate_calibration_scatter(rows)
    generate_forecast_error_histogram(rows)
    generate_hmm_diagram()
    generate_monte_carlo_diagram()
    update_figures_doc()
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
