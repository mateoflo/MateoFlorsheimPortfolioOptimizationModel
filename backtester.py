"""Out-of-sample portfolio backtesting engine."""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from optimizer import (
    AssetInput,
    PortfolioConfig,
    estimate_asset_statistics_from_prices,
    optimize_portfolio,
    resolve_treasury_bill_yield,
    run_hmm_monte_carlo_projection,
)

# Default ticker universe CSV — resolved relative to this file to avoid cwd dependence.
DEFAULT_UNIVERSE_CSV = (
    Path(__file__).resolve().parent / "data" / "nyse_nasdaq_most_volatile_asof_2024_01_01.csv"
)


@dataclass
class BacktestConfig:
    """Settings for a backtesting run, bundled into a single object."""

    formation_date: str
    lookback_years: float = 1.0
    forward_years: float = 1.0
    universe_size: int | None = None
    portfolio_size: int = 20
    combination_count: int = 100
    random_seed: int = 42
    price_chunk_size: int = 100
    progress: bool = True
    download_timeout_sec: float = 5.0
    checkpoint_every: int = 10


def load_ticker_universe_from_csv(path: str | Path, ticker_column: str = "ticker") -> list[str]:
    """Load a deduplicated, uppercase list of tickers from a CSV file."""

    csv_path = Path(path)
    frame = pd.read_csv(csv_path)

    # Case-insensitive lookup of column names.
    normalized = {str(column).strip().lower(): column for column in frame.columns}

    # Try the requested column name, then fall back to 'symbol'.
    candidate_columns = [ticker_column.lower(), "symbol"]
    source_col = None
    for candidate in candidate_columns:
        if candidate in normalized:
            source_col = normalized[candidate]
            break
    if source_col is None:
        raise ValueError("Universe CSV must include a 'ticker' or 'symbol' column.")

    tickers = (
        frame[source_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan})  # treat empty strings as missing
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not tickers:
        raise ValueError("Universe CSV did not include any usable tickers.")
    return tickers


def sample_unique_portfolios(
    tickers: list[str],
    portfolio_size: int,
    combination_count: int,
    seed: int = 42,
) -> list[tuple[str, ...]]:
    """Randomly draw many unique portfolios from a pool of tickers."""

    if portfolio_size <= 0:
        raise ValueError("Portfolio size must be positive.")
    if combination_count <= 0:
        raise ValueError("Combination count must be positive.")
    if len(tickers) < portfolio_size:
        raise ValueError("Universe is smaller than the requested portfolio size.")

    rng = np.random.default_rng(seed)
    combos: set[tuple[str, ...]] = set()

    # Cap attempts to avoid an infinite loop on a near-exhausted universe.
    max_attempts = max(combination_count * 20, 1000)
    attempts = 0

    # Sort and deduplicate so sampling is deterministic.
    ordered = np.array(sorted(dict.fromkeys(tickers)), dtype=object)

    while len(combos) < combination_count and attempts < max_attempts:
        chosen = rng.choice(ordered, size=portfolio_size, replace=False)
        # Sort before storing so {'AAPL','MSFT'} and {'MSFT','AAPL'} hash identically.
        combos.add(tuple(sorted(str(ticker) for ticker in chosen.tolist())))
        attempts += 1

    if len(combos) < combination_count:
        raise ValueError(
            f"Unable to generate {combination_count} unique portfolios from the supplied universe."
        )
    return list(combos)


def _extract_close_prices(raw: pd.DataFrame | pd.Series, requested_tickers: list[str]) -> pd.DataFrame:
    """Extract closing prices from a raw yfinance download into a simple date × ticker DataFrame."""

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return pd.DataFrame(columns=requested_tickers)

    # Multi-ticker download has a MultiIndex on columns; single-ticker does not.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        prices = raw.rename(columns={"Close": requested_tickers[0] if len(requested_tickers) == 1 else "Close"})
        if "Close" in prices.columns:
            prices = prices[["Close"]].rename(columns={"Close": requested_tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=requested_tickers[0])

    prices = prices.dropna(how="all")
    return prices


def download_universe_prices(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    chunk_size: int = 100,
    timeout_sec: float = 5.0,
) -> pd.DataFrame:
    """Download daily closing prices for a list of tickers from Yahoo Finance in batches."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    # Split into smaller batches for reliability.
    chunks = [tickers[idx : idx + chunk_size] for idx in range(0, len(tickers), chunk_size)]
    frames: list[pd.DataFrame] = []

    for chunk in chunks:
        # Suppress yfinance's noisy terminal output.
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            raw = yf.download(
                chunk,
                start=str(pd.Timestamp(start).date()),
                end=str(pd.Timestamp(end).date()),
                interval="1d",
                auto_adjust=True,   # adjusts for splits and dividends
                progress=False,
                threads=False,      # single-threaded to avoid race conditions
                timeout=timeout_sec,
            )
        prices = _extract_close_prices(raw, chunk)
        if prices.empty:
            continue
        frames.append(prices)

    if not frames:
        raise ValueError("No price data could be downloaded for the universe.")

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()].sort_index()
    return combined


def build_random_stock_assets(tickers: list[str], max_weight: float = 1.0) -> list[AssetInput]:
    """Create placeholder AssetInput objects for a list of tickers with zeroed statistics."""

    return [
        AssetInput(
            ticker=ticker,
            price=0.0,            # filled in by the statistics estimator
            expected_return=0.0,  # filled in by the statistics estimator
            volatility=0.0,       # filled in by the statistics estimator
            max_weight=max_weight,
        )
        for ticker in tickers
    ]


def compute_realized_portfolio_return(
    optimization_result: dict,
    evaluation_prices: pd.DataFrame,
    capital: float,
    cash_yield: float,
    treasury_bill_yield: float,
    years: float,
) -> dict:
    """Compute the actual buy-and-hold return of an optimized portfolio over the evaluation window."""

    asset_rows = optimization_result["asset_rows"]
    if evaluation_prices.empty:
        raise ValueError("Evaluation window did not contain any price data.")

    # Forward-fill gaps, then use the last available price as the sell price.
    latest_prices = evaluation_prices.ffill().iloc[-1]

    stock_terminal_value = 0.0
    missing = []
    for row in asset_rows:
        ticker = row["ticker"]
        if ticker not in latest_prices or pd.isna(latest_prices[ticker]):
            missing.append(ticker)
            continue
        stock_terminal_value += int(row["recommended_shares"]) * float(latest_prices[ticker])

    if missing:
        raise ValueError(f"Missing evaluation-end prices for: {', '.join(missing)}")

    # Grow cash and T-bill allocations via continuous compounding: e^(rate × time).
    cash_terminal_value = float(optimization_result["cash_dollars"]) * math.exp(cash_yield * years)
    tbill_terminal_value = float(optimization_result["treasury_bill_dollars"]) * math.exp(treasury_bill_yield * years)

    terminal_value = stock_terminal_value + cash_terminal_value + tbill_terminal_value

    # Realized return = (ending value / starting value) - 1.
    realized_return = terminal_value / capital - 1.0

    return {
        "terminal_value": float(terminal_value),
        "realized_return": float(realized_return),
        "end_date": str(evaluation_prices.index[-1].date()),
    }


def _build_formation_and_evaluation_windows(
    prices: pd.DataFrame,
    tickers: list[str],
    formation_date: pd.Timestamp,
    lookback_years: float,
    forward_years: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice a price DataFrame into a training window and a testing window around a formation date."""

    # Convert years to days with a 120-day floor to ensure enough observations.
    lookback_start = formation_date - pd.DateOffset(days=max(int(round(365 * lookback_years)), 120))
    evaluation_end = formation_date + pd.DateOffset(days=max(int(round(365 * forward_years)), 120))

    # History includes the formation date so we can price initial purchases at that close.
    history = prices.loc[(prices.index >= lookback_start) & (prices.index <= formation_date), tickers].copy()

    # Future starts the day after formation — no lookahead.
    future = prices.loc[(prices.index > formation_date) & (prices.index <= evaluation_end), tickers].copy()

    if history.empty or future.empty:
        raise ValueError("Insufficient price history around the requested formation/evaluation dates.")
    return history, future


def run_single_portfolio_backtest(
    tickers: list[str],
    price_history: pd.DataFrame,
    config: PortfolioConfig,
    formation_date: str | pd.Timestamp,
    lookback_years: float = 1.0,
    forward_years: float = 1.0,
) -> dict:
    """Run a complete out-of-sample backtest for one portfolio and return results and forecast accuracy."""

    formation_ts = pd.Timestamp(formation_date)

    # Split prices into lookback (training) and forward (evaluation) windows.
    history_prices, future_prices = _build_formation_and_evaluation_windows(
        price_history,
        tickers,
        formation_ts,
        lookback_years=lookback_years,
        forward_years=forward_years,
    )

    # Resolve the T-bill yield once, then freeze it to avoid re-fetching mid-calculation.
    treasury_bill_yield, treasury_bill_source = resolve_treasury_bill_yield(config)
    effective_config = replace(
        config,
        treasury_bill_yield=treasury_bill_yield,
        auto_treasury_bill_yield=False,
        simulation_horizon_years=forward_years,
    )

    assets = build_random_stock_assets(tickers, max_weight=1.0)

    # Estimate returns, volatilities, and correlations from the lookback window only.
    estimated_assets, corr, sample_window, returns_np, denoise_info, regime_info, return_model_info = estimate_asset_statistics_from_prices(
        assets,
        history_prices,
        hmm_states=max(2, int(effective_config.hmm_states)),
        expected_return_method=effective_config.expected_return_method,
        expected_return_shrinkage=effective_config.expected_return_shrinkage,
    )

    # Optimize portfolio weights using only the lookback statistics.
    optimization_result = optimize_portfolio(estimated_assets, corr, effective_config)

    # Simulate thousands of future return paths using HMM regime dynamics.
    monte_carlo = run_hmm_monte_carlo_projection(
        capital=effective_config.capital,
        stock_weights=np.asarray(optimization_result["portfolio_weights"][: len(estimated_assets)], dtype=float),
        cash_weight=float(optimization_result["cash_weight"]),
        treasury_bill_weight=float(optimization_result["treasury_bill_weight"]),
        cash_yield=effective_config.cash_yield,
        treasury_bill_yield=treasury_bill_yield,
        regime_info=regime_info,
        years=forward_years,
        paths=effective_config.simulation_paths,
        seed=42,
    )

    # Measure how the portfolio actually performed over the evaluation window.
    realized = compute_realized_portfolio_return(
        optimization_result=optimization_result,
        evaluation_prices=future_prices,
        capital=effective_config.capital,
        cash_yield=effective_config.cash_yield,
        treasury_bill_yield=treasury_bill_yield,
        years=forward_years,
    )

    predicted_expected_return = float(optimization_result["expected_return"])
    # MC expected return: (mean terminal value / capital) - 1.
    predicted_mc_return = float(monte_carlo["expected_terminal_value"] / effective_config.capital - 1.0)

    return {
        "tickers": list(tickers),
        "formation_date": str(formation_ts.date()),
        "sample_window": sample_window,
        "evaluation_end_date": realized["end_date"],
        "expected_return_method": effective_config.expected_return_method,
        "expected_return": predicted_expected_return,
        "expected_mc_return": predicted_mc_return,
        "realized_return": realized["realized_return"],
        "realized_terminal_value": realized["terminal_value"],
        # Accurate if realized return meets or exceeds the optimizer's prediction.
        "prediction_accurate": bool(realized["realized_return"] >= predicted_expected_return),
        "monte_carlo": monte_carlo,
        "optimization": optimization_result,
        "denoise_info": denoise_info,
        "return_model_info": return_model_info,
        "regime_info": {
            "states": int(effective_config.hmm_states),
            "transition_matrix": regime_info["transition_matrix"],
        },
        "treasury_bill_yield": float(treasury_bill_yield),
        "treasury_bill_source": treasury_bill_source,
        "history_observations": int(returns_np.shape[0]),    # daily returns used for estimation
        "evaluation_observations": int(len(future_prices)),  # trading days in the test window
    }


def _format_seconds(seconds: float) -> str:
    """Convert a duration in seconds to a MM:SS (or H:MM:SS) string."""

    total = max(int(round(seconds)), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_results_frame(results: list[dict]) -> pd.DataFrame:
    """Convert a list of single-portfolio result dicts into a flat DataFrame for export."""

    return pd.DataFrame(
        {
            "formation_date": [result["formation_date"] for result in results],
            "evaluation_end_date": [result["evaluation_end_date"] for result in results],
            # Store tickers as a space-separated string for easy CSV round-tripping.
            "tickers": [" ".join(result["tickers"]) for result in results],
            "expected_return_method": [result.get("expected_return_method", "") for result in results],
            "expected_return": [result["expected_return"] for result in results],
            "expected_mc_return": [result["expected_mc_return"] for result in results],
            "realized_return": [result["realized_return"] for result in results],
            "prediction_accurate": [result["prediction_accurate"] for result in results],
            "realized_terminal_value": [result["realized_terminal_value"] for result in results],
            "history_observations": [result["history_observations"] for result in results],
            "evaluation_observations": [result["evaluation_observations"] for result in results],
            "treasury_bill_yield": [result["treasury_bill_yield"] for result in results],
            "treasury_bill_source": [result["treasury_bill_source"] for result in results],
        }
    )


def load_existing_results(results_csv: str | Path) -> list[dict]:
    """Load a previously saved backtest CSV so a long run can be resumed."""

    frame = pd.read_csv(results_csv)
    if frame.empty:
        return []

    required_columns = [
        "formation_date",
        "evaluation_end_date",
        "tickers",
        "expected_return",
        "expected_mc_return",
        "realized_return",
        "prediction_accurate",
        "realized_terminal_value",
        "history_observations",
        "evaluation_observations",
        "treasury_bill_yield",
        "treasury_bill_source",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Existing results CSV is missing required columns: {', '.join(missing)}")

    records = []
    for row in frame.to_dict(orient="records"):
        # Re-expand space-separated ticker string back into a list.
        row["tickers"] = str(row["tickers"]).split()
        # CSV stores booleans as 0/1 or "True"/"False" — cast explicitly.
        row["prediction_accurate"] = bool(row["prediction_accurate"])
        # Provide a default for files generated before this column was added.
        row.setdefault("expected_return_method", "historical_mean")
        records.append(row)
    return records


def run_random_universe_backtest(
    universe_tickers: list[str],
    config: PortfolioConfig,
    backtest: BacktestConfig,
    prices: pd.DataFrame | None = None,
    existing_results: list[dict] | None = None,
    checkpoint_csv: str | Path | None = None,
) -> dict:
    """Run a large-scale backtesting experiment across many randomly sampled portfolios and return aggregate results."""

    requested_universe_size = backtest.universe_size or len(universe_tickers)
    if requested_universe_size < backtest.portfolio_size:
        raise ValueError("Universe size cannot be smaller than portfolio size.")

    rng = np.random.default_rng(backtest.random_seed)

    # Deduplicate and clean the universe before sub-sampling.
    unique_universe = sorted(dict.fromkeys(str(ticker).strip().upper() for ticker in universe_tickers if str(ticker).strip()))
    if len(unique_universe) < requested_universe_size:
        raise ValueError("Supplied universe has fewer tickers than the requested universe size.")

    # Randomly draw the requested number of tickers from the full pool.
    selected_universe = rng.choice(np.array(unique_universe, dtype=object), size=requested_universe_size, replace=False)
    selected_universe = [str(ticker) for ticker in selected_universe.tolist()]

    # Compute the full date range needed: lookback_start through evaluation_end.
    formation_ts = pd.Timestamp(backtest.formation_date)
    lookback_start = formation_ts - pd.DateOffset(days=max(int(round(365 * backtest.lookback_years)), 120))
    evaluation_end = formation_ts + pd.DateOffset(days=max(int(round(365 * backtest.forward_years)), 120))

    # Download prices if not supplied by the caller.
    if prices is None:
        prices = download_universe_prices(
            selected_universe,
            start=lookback_start,
            end=evaluation_end + pd.DateOffset(days=5),  # small buffer to capture all evaluation data
            chunk_size=backtest.price_chunk_size,
            timeout_sec=backtest.download_timeout_sec,
        )

    prices = prices.copy().sort_index()

    # Drop tickers with no downloaded data.
    available_universe = [ticker for ticker in selected_universe if ticker in prices.columns]
    if len(available_universe) < backtest.portfolio_size:
        raise ValueError("Not enough downloaded tickers remain after filtering missing Yahoo price history.")

    # Require >= 60 lookback days and >= 2 evaluation days per ticker.
    history_slice = prices.loc[(prices.index >= lookback_start) & (prices.index <= formation_ts), available_universe]
    future_slice = prices.loc[(prices.index > formation_ts) & (prices.index <= evaluation_end), available_universe]
    historical_counts = history_slice.notna().sum(axis=0)
    future_counts = future_slice.notna().sum(axis=0)
    eligible = [
        ticker
        for ticker in available_universe
        if historical_counts.get(ticker, 0) >= 60 and future_counts.get(ticker, 0) >= 2
    ]
    if len(eligible) < backtest.portfolio_size:
        raise ValueError("Not enough eligible tickers with both formation and evaluation history.")

    combinations = sample_unique_portfolios(
        eligible,
        portfolio_size=backtest.portfolio_size,
        combination_count=backtest.combination_count,
        seed=backtest.random_seed,
    )

    # Build a set of already-completed portfolio keys to support resuming.
    results = list(existing_results or [])
    failures = []
    started_at = time.perf_counter()
    completed_keys = {
        " ".join(sorted(str(ticker) for ticker in result["tickers"]))
        for result in results
    }
    pending_combinations = [
        combo for combo in combinations if " ".join(sorted(str(ticker) for ticker in combo)) not in completed_keys
    ]

    total = len(combinations)
    if backtest.progress:
        print(
            f"Starting backtest: formation={formation_ts.date()} eligible={len(eligible)} "
            f"portfolios={total} size={backtest.portfolio_size}",
            flush=True,
        )
        if results:
            print(
                f"Resuming from existing results: already completed={len(results)} pending={len(pending_combinations)}",
                flush=True,
            )

    processed_count = len(results)

    for combo in pending_combinations:
        processed_count += 1
        try:
            outcome = run_single_portfolio_backtest(
                tickers=list(combo),
                price_history=prices,
                config=config,
                formation_date=formation_ts,
                lookback_years=backtest.lookback_years,
                forward_years=backtest.forward_years,
            )
            results.append(outcome)
        except Exception as exc:
            # Record failure without crashing the whole experiment.
            failures.append({"tickers": list(combo), "error": str(exc)})

        # Write a checkpoint every N successes to preserve progress.
        if checkpoint_csv and (len(results) % max(backtest.checkpoint_every, 1) == 0) and results:
            checkpoint_frame = build_results_frame(results)
            checkpoint_path = Path(checkpoint_csv)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_frame.to_csv(checkpoint_path, index=False)

        if backtest.progress:
            elapsed = time.perf_counter() - started_at
            # Base ETA on newly processed portfolios only, not pre-loaded ones.
            newly_processed = max(processed_count - len(existing_results or []), 1)
            avg_time = elapsed / newly_processed
            eta = avg_time * max(total - processed_count, 0)
            print(
                f"Processed {processed_count}/{total} portfolios | success={len(results)} "
                f"failed={len(failures)} | elapsed={_format_seconds(elapsed)} "
                f"| eta={_format_seconds(eta)}",
                flush=True,
            )

    if not results:
        raise ValueError("All sampled portfolios failed during backtesting.")

    result_frame = build_results_frame(results)
    elapsed_total = time.perf_counter() - started_at

    return {
        "formation_date": str(formation_ts.date()),
        "lookback_years": float(backtest.lookback_years),
        "forward_years": float(backtest.forward_years),
        "requested_universe_size": int(requested_universe_size),
        "eligible_universe_size": int(len(eligible)),
        "portfolio_size": int(backtest.portfolio_size),
        "requested_combinations": int(backtest.combination_count),
        "completed_combinations": int(len(results)),
        "failed_combinations": int(len(failures)),
        "elapsed_seconds": float(elapsed_total),
        # Fraction of portfolios where realized return >= predicted return.
        "accuracy_rate": float(result_frame["prediction_accurate"].mean()),
        "mean_expected_return": float(result_frame["expected_return"].mean()),
        "mean_expected_mc_return": float(result_frame["expected_mc_return"].mean()),
        "mean_realized_return": float(result_frame["realized_return"].mean()),
        "median_realized_return": float(result_frame["realized_return"].median()),
        "results": results,
        "failures": failures,
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the backtester command-line interface."""

    parser = argparse.ArgumentParser(description="Run a no-lookahead random-universe portfolio backtest.")
    parser.add_argument(
        "--universe-csv",
        default=str(DEFAULT_UNIVERSE_CSV),
        help="CSV file with a ticker or symbol column.",
    )
    parser.add_argument("--formation-date", required=True, help="Formation date in YYYY-MM-DD format.")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--universe-size", type=int, default=None)
    parser.add_argument("--portfolio-size", type=int, default=20)
    parser.add_argument("--combination-count", type=int, default=100)
    parser.add_argument("--lookback-years", type=float, default=1.0)
    parser.add_argument("--forward-years", type=float, default=1.0)
    parser.add_argument("--min-cash-pct", type=float, default=5.0)
    parser.add_argument("--max-cash-pct", type=float, default=25.0)
    parser.add_argument("--cash-yield-pct", type=float, default=4.0)
    parser.add_argument("--target-return-pct", type=float, default=None)
    parser.add_argument("--target-vol-pct", type=float, default=None)
    parser.add_argument(
        "--expected-return-method",
        default="historical_mean",
        choices=["historical_mean", "bayes_stein", "market_factor", "black_litterman"],
    )
    parser.add_argument("--expected-return-shrinkage", type=float, default=0.50)
    parser.add_argument("--mc-paths", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--results-csv", default=None, help="Optional path to write portfolio-level backtest results.")
    parser.add_argument("--resume-results-csv", default=None, help="Optional existing results CSV to resume from.")
    parser.add_argument("--download-timeout-sec", type=float, default=5.0)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Write partial results every N successful portfolios.")
    parser.add_argument("--quiet", action="store_true", help="Disable live progress updates.")
    return parser


def _parse_optional_pct(value: float | None) -> float | None:
    """Convert an optional CLI percentage (e.g. 12.5) to a decimal fraction (0.125)."""

    if value is None:
        return None
    return value / 100.0


def main() -> None:
    """Parse CLI arguments, run the backtest experiment, and write results to CSV."""

    parser = _build_cli_parser()
    args = parser.parse_args()

    universe = load_ticker_universe_from_csv(args.universe_csv)

    # Percentage inputs are divided by 100 to match the optimizer's decimal convention.
    config = PortfolioConfig(
        capital=args.capital,
        risk_aversion=4.0,           # moderate risk aversion
        shrinkage=0.20,              # pulls covariance toward a structured estimate
        concentration_penalty=0.05,  # discourages over-concentration in a single stock
        min_cash_weight=args.min_cash_pct / 100.0,
        max_cash_weight=args.max_cash_pct / 100.0,
        cash_yield=args.cash_yield_pct / 100.0,
        treasury_bill_yield=None,               # auto-fetched at runtime
        auto_max_allocation=True,               # automatically sets per-stock weight ceilings
        auto_max_floor=0.02,                    # minimum per-stock ceiling: 2%
        auto_max_ceiling=0.10,                  # maximum per-stock ceiling: 10%
        auto_treasury_bill_yield=True,          # fetch the live T-bill rate
        target_expected_return=_parse_optional_pct(args.target_return_pct),
        target_volatility=_parse_optional_pct(args.target_vol_pct),
        expected_return_method=args.expected_return_method,
        expected_return_shrinkage=args.expected_return_shrinkage,
        hmm_states=2,                           # two regimes: bull and bear
        simulation_paths=args.mc_paths,
        simulation_horizon_years=args.forward_years,
    )

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
        checkpoint_every=args.checkpoint_every,
    )

    # Load prior results if resuming an interrupted run.
    existing_results = load_existing_results(args.resume_results_csv) if args.resume_results_csv else None

    summary = run_random_universe_backtest(
        universe,
        config,
        backtest,
        existing_results=existing_results,
        checkpoint_csv=args.results_csv,  # also used for in-run checkpointing
    )

    if args.results_csv:
        results_frame = build_results_frame(summary["results"])
        output_path = Path(args.results_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_frame.to_csv(output_path, index=False)
        print(f"Wrote portfolio-level results to {output_path}")

    # Print summary stats, excluding large nested list fields.
    print(pd.Series({k: v for k, v in summary.items() if k not in {"results", "failures"}}).to_string())


if __name__ == "__main__":
    main()
