from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtester import BacktestConfig, load_ticker_universe_from_csv, run_random_universe_backtest
from optimizer import PortfolioConfig


def _parse_optional_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare expected-return estimators on the same random-universe backtest.")
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--formation-date", required=True)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--universe-size", type=int, default=None)
    parser.add_argument("--portfolio-size", type=int, default=10)
    parser.add_argument("--combination-count", type=int, default=100)
    parser.add_argument("--lookback-years", type=float, default=1.0)
    parser.add_argument("--forward-years", type=float, default=1.0)
    parser.add_argument("--min-cash-pct", type=float, default=5.0)
    parser.add_argument("--max-cash-pct", type=float, default=25.0)
    parser.add_argument("--cash-yield-pct", type=float, default=4.0)
    parser.add_argument("--target-return-pct", type=float, default=None)
    parser.add_argument("--target-vol-pct", type=float, default=None)
    parser.add_argument("--mc-paths", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--download-timeout-sec", type=float, default=5.0)
    parser.add_argument("--expected-return-shrinkage", type=float, default=0.50)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    universe = load_ticker_universe_from_csv(args.universe_csv)
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
    summaries = []
    for method in methods:
        config = PortfolioConfig(
            capital=args.capital,
            risk_aversion=4.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=args.min_cash_pct / 100.0,
            max_cash_weight=args.max_cash_pct / 100.0,
            cash_yield=args.cash_yield_pct / 100.0,
            treasury_bill_yield=None,
            auto_max_allocation=True,
            auto_max_floor=0.02,
            auto_max_ceiling=0.10,
            auto_treasury_bill_yield=True,
            target_expected_return=_parse_optional_pct(args.target_return_pct),
            target_volatility=_parse_optional_pct(args.target_vol_pct),
            expected_return_method=method,
            expected_return_shrinkage=args.expected_return_shrinkage,
            hmm_states=2,
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"\nWrote comparison summary to {output_path}")
    print("\n" + frame.to_string(index=False))


if __name__ == "__main__":
    main()
