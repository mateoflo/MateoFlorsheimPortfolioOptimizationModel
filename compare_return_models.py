"""Run a side-by-side backtest of four return-estimation methods and print a comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtester import BacktestConfig, load_ticker_universe_from_csv, run_random_universe_backtest
from optimizer import PortfolioConfig


def _parse_optional_pct(value: float | None) -> float | None:
    """Convert a whole-number percentage (e.g. 5.0) to a decimal fraction (0.05), or return None."""
    if value is None:
        return None
    return value / 100.0


def main() -> None:
    """Parse CLI arguments, run each return model through the same backtest, and print results."""
    parser = argparse.ArgumentParser(description="Compare expected-return estimators on the same random-universe backtest.")

    # Required arguments
    parser.add_argument("--universe-csv", required=True,
                        help="Path to a CSV file containing the list of stock tickers.")
    parser.add_argument("--formation-date", required=True,
                        help="The date we 'stand at' when building the portfolio (YYYY-MM-DD).")

    # Portfolio and backtest configuration
    parser.add_argument("--capital", type=float, default=100000.0,
                        help="Starting portfolio value in dollars (default: $100,000).")
    parser.add_argument("--universe-size", type=int, default=None,
                        help="How many tickers to randomly draw from the CSV. None means use all.")
    parser.add_argument("--portfolio-size", type=int, default=10,
                        help="Number of stocks to hold in each portfolio (default: 10).")
    parser.add_argument("--combination-count", type=int, default=100,
                        help="How many different random portfolios to build and test (default: 100).")
    parser.add_argument("--lookback-years", type=float, default=1.0,
                        help="Years of historical data to use when estimating returns (default: 1).")
    parser.add_argument("--forward-years", type=float, default=1.0,
                        help="Years of future data to measure realized performance against (default: 1).")

    # Cash allocation controls
    parser.add_argument("--min-cash-pct", type=float, default=5.0,
                        help="Minimum percentage of the portfolio kept in cash (default: 5%%).")
    parser.add_argument("--max-cash-pct", type=float, default=25.0,
                        help="Maximum percentage of the portfolio kept in cash (default: 25%%).")
    parser.add_argument("--cash-yield-pct", type=float, default=4.0,
                        help="Annual interest rate earned on cash (default: 4%%).")

    # Optional return / volatility targets
    parser.add_argument("--target-return-pct", type=float, default=None,
                        help="If set, optimizer targets this annualized return (as a whole-number %).")
    parser.add_argument("--target-vol-pct", type=float, default=None,
                        help="If set, optimizer targets this annualized volatility (as a whole-number %).")

    # Simulation and reproducibility
    parser.add_argument("--mc-paths", type=int, default=500,
                        help="Number of Monte Carlo simulation paths (default: 500). More paths = more accuracy but slower.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Seed for the random number generator so results are reproducible (default: 42).")
    parser.add_argument("--download-timeout-sec", type=float, default=5.0,
                        help="Seconds to wait for each price-data download before giving up (default: 5).")

    # Shrinkage pulls return estimates toward a neutral baseline to reduce noise sensitivity.
    parser.add_argument("--expected-return-shrinkage", type=float, default=0.50,
                        help="How much to shrink expected returns toward zero (0=none, 1=full shrinkage; default: 0.50).")

    # Output
    parser.add_argument("--summary-csv", default=None,
                        help="Optional path to save the comparison table as a CSV file.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output during the backtest.")

    args = parser.parse_args()

    universe = load_ticker_universe_from_csv(args.universe_csv)

    # Single backtest config shared across all methods — only the return model changes.
    backtest = BacktestConfig(
        formation_date=args.formation_date,
        lookback_years=args.lookback_years,
        forward_years=args.forward_years,
        universe_size=args.universe_size,
        portfolio_size=args.portfolio_size,
        combination_count=args.combination_count,
        random_seed=args.random_seed,
        progress=not args.quiet,
        download_timeout_sec=args.download_timeout_sec,
    )

    methods = ["historical_mean", "bayes_stein", "market_factor", "black_litterman"]

    summaries = []  # One result dictionary per method.

    for method in methods:
        # Build a fresh PortfolioConfig for each method; all settings are identical except the return model.
        config = PortfolioConfig(
            capital=args.capital,
            risk_aversion=4.0,           # Standard assumption for a moderately risk-averse investor.
            shrinkage=0.20,              # Pulls the covariance matrix toward a simpler structure.
            concentration_penalty=0.05,  # Penalizes over-weighting any single stock.
            min_cash_weight=args.min_cash_pct / 100.0,
            max_cash_weight=args.max_cash_pct / 100.0,
            cash_yield=args.cash_yield_pct / 100.0,
            treasury_bill_yield=None,       # Auto-fetch the current yield.
            auto_treasury_bill_yield=True,
            auto_max_allocation=True,
            auto_max_floor=0.02,   # No stock can be less than 2% of the portfolio.
            auto_max_ceiling=0.10, # No stock can be more than 10% of the portfolio.
            target_expected_return=_parse_optional_pct(args.target_return_pct),
            target_volatility=_parse_optional_pct(args.target_vol_pct),
            expected_return_method=method,  # The variable that differs between iterations.
            expected_return_shrinkage=args.expected_return_shrinkage,
            hmm_states=2,  # Bull and bear hidden states for regime detection.
            simulation_paths=args.mc_paths,
            simulation_horizon_years=args.forward_years,
        )

        print(f"\nRunning method: {method}", flush=True)

        summary = run_random_universe_backtest(universe, config, backtest)

        summaries.append(
            {
                "expected_return_method": method,
                "completed_combinations": summary["completed_combinations"],
                "failed_combinations": summary["failed_combinations"],
                "accuracy_rate": summary["accuracy_rate"],
                "mean_expected_return": summary["mean_expected_return"],
                "mean_expected_mc_return": summary["mean_expected_mc_return"],
                "mean_realized_return": summary["mean_realized_return"],
                "median_realized_return": summary["median_realized_return"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )

    frame = pd.DataFrame(summaries)

    if args.summary_csv:
        output_path = Path(args.summary_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create missing directories.
        frame.to_csv(output_path, index=False)
        print(f"\nWrote comparison summary to {output_path}")

    print("\n" + frame.to_string(index=False))


if __name__ == "__main__":
    main()
