"""Build a CSV of the most volatile, liquid common stocks listed on NASDAQ and NYSE."""

from __future__ import annotations

import argparse
import contextlib
from io import StringIO
import io
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yfinance as yf


# NASDAQ Trader URLs for daily exchange listing files (pipe-delimited).
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Matches only standard common-stock tickers: one to five uppercase letters.
COMMON_STOCK_PATTERN = re.compile(r"^[A-Z]{1,5}$")


def _download_text(url: str) -> str:
    """Fetch a URL and return its content as a decoded UTF-8 string."""
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="ignore")


def load_exchange_tickers() -> tuple[list[str], list[str]]:
    """Download NASDAQ and NYSE listing files and return filtered common-stock ticker lists."""
    nasdaq_raw = _download_text(NASDAQ_LISTED_URL)
    other_raw  = _download_text(OTHER_LISTED_URL)

    nasdaq = pd.read_csv(StringIO(nasdaq_raw), sep="|")
    other  = pd.read_csv(StringIO(other_raw),  sep="|")

    # Remove metadata footer, test symbols, and ETFs from the NASDAQ file.
    nasdaq = nasdaq[nasdaq["Symbol"].notna()].copy()
    nasdaq = nasdaq[nasdaq["Symbol"] != "File Creation Time"]
    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
    nasdaq = nasdaq[nasdaq["ETF"] == "N"]

    # Apply the same filters to the NYSE file.
    other = other[other["ACT Symbol"].notna()].copy()
    other = other[other["ACT Symbol"] != "File Creation Time"]
    other = other[other["Test Issue"] == "N"]
    other = other[other["ETF"] == "N"]

    nyse = other[other["Exchange"] == "N"].copy()  # Exchange code "N" = NYSE.

    # Keep only tickers matching the common-stock pattern and sort alphabetically.
    nasdaq_tickers = sorted(
        ticker
        for ticker in nasdaq["Symbol"].astype(str).str.strip().str.upper().unique().tolist()
        if COMMON_STOCK_PATTERN.fullmatch(ticker)
    )
    nyse_tickers = sorted(
        ticker
        for ticker in nyse["ACT Symbol"].astype(str).str.strip().str.upper().unique().tolist()
        if COMMON_STOCK_PATTERN.fullmatch(ticker)
    )

    return nasdaq_tickers, nyse_tickers


def _extract_field(raw: pd.DataFrame | pd.Series, requested_tickers: list[str], field_name: str) -> pd.DataFrame:
    """Extract one named field from a yfinance download result, normalizing single- and multi-ticker layouts."""
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return pd.DataFrame(columns=requested_tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker layout: columns are (field_name, ticker).
        if field_name in raw.columns.get_level_values(0):
            prices = raw[field_name].copy()
        else:
            prices = raw.xs(field_name, axis=1, level=0, drop_level=True).copy()
    else:
        # Single-ticker layout: rename the field column to the ticker name.
        prices = raw.rename(columns={field_name: requested_tickers[0] if len(requested_tickers) == 1 else field_name})
        if field_name in prices.columns:
            prices = prices[[field_name]].rename(columns={field_name: requested_tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=requested_tickers[0])

    return prices


def _extract_close_prices(raw: pd.DataFrame | pd.Series, requested_tickers: list[str]) -> pd.DataFrame:
    """Extract daily closing prices from a raw yfinance download result."""
    return _extract_field(raw, requested_tickers, "Close")


def _extract_volume(raw: pd.DataFrame | pd.Series, requested_tickers: list[str]) -> pd.DataFrame:
    """Extract daily share volume from a raw yfinance download result."""
    return _extract_field(raw, requested_tickers, "Volume")


def compute_realized_volatility_ranks(
    tickers: list[str],
    exchange_name: str,
    top_n: int = 500,
    lookback_period: str = "1y",
    as_of_date: str | None = None,
    min_price: float = 5.0,
    min_avg_dollar_volume_63d: float = 5_000_000.0,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Download price history, compute annualized volatility, filter low-quality names, and return the top-N."""
    as_of_ts = pd.Timestamp(as_of_date) if as_of_date else None

    lookback_days = max(365, 252)  # Use at least one year of data for a stable estimate.

    if as_of_ts is not None:
        start_date = (as_of_ts - pd.DateOffset(days=lookback_days)).date()
        end_date = (as_of_ts + pd.DateOffset(days=5)).date()  # Small buffer to include the as-of day.

    records: list[dict] = []

    # Download in batches to avoid API rate limits.
    for idx in range(0, len(tickers), chunk_size):
        chunk = tickers[idx : idx + chunk_size]

        print(f"[{exchange_name}] Downloading chunk {idx // chunk_size + 1} / {(len(tickers) + chunk_size - 1) // chunk_size}")

        # Suppress yfinance's own progress output.
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            if as_of_ts is not None:
                raw = yf.download(
                    chunk,
                    start=str(start_date),
                    end=str(end_date),
                    interval="1d",
                    auto_adjust=True,  # Adjust for splits and dividends.
                    progress=False,
                    threads=False,
                )
            else:
                raw = yf.download(
                    chunk,
                    period=lookback_period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )

        prices  = _extract_close_prices(raw, chunk).sort_index()
        volumes = _extract_volume(raw, chunk).sort_index()

        if prices.empty:
            continue

        prices = prices.ffill()   # Forward-fill to handle non-trading days.
        volumes = volumes.fillna(0.0)  # No trading = zero volume.

        # Trim data to the as-of date to prevent look-ahead bias.
        if as_of_ts is not None:
            prices  = prices.loc[prices.index   <= as_of_ts]
            volumes = volumes.loc[volumes.index <= as_of_ts]

        # Log-returns are additive over time and statistically better-behaved than simple returns.
        returns = np.log(prices / prices.shift(1))

        for ticker in chunk:
            if ticker not in prices.columns:
                continue

            series        = prices[ticker].dropna()
            ret_series    = returns[ticker].dropna()   if ticker in returns.columns  else pd.Series(dtype=float)
            volume_series = volumes[ticker].reindex(series.index).fillna(0.0) if ticker in volumes.columns else pd.Series(dtype=float)

            # Require at least half a year of price data and 60 return observations.
            if len(series) < 126 or len(ret_series) < 60:
                continue

            # Skip stale series — likely delisted before the as-of date.
            if as_of_ts is not None:
                if pd.Timestamp(series.index[-1]).date() < (as_of_ts - pd.DateOffset(days=10)).date():
                    continue

            last_close = float(series.iloc[-1])

            if last_close < min_price:  # Filter out penny stocks.
                continue

            if volume_series.empty:
                continue

            # Average dollar volume over the last 63 trading days (~3 months).
            dollar_volume     = series * volume_series
            avg_dollar_volume = float(dollar_volume.tail(63).mean())

            if not np.isfinite(avg_dollar_volume) or avg_dollar_volume < min_avg_dollar_volume_63d:
                continue  # Too illiquid.

            # Annualize daily volatility using sqrt(252) — standard finance convention.
            annualized_vol = float(ret_series.std(ddof=1) * np.sqrt(252.0))

            records.append(
                {
                    "ticker": ticker,
                    "exchange": exchange_name,
                    "volatility_1y": annualized_vol,
                    "last_close": last_close,
                    "observations": int(len(ret_series)),
                    "lookback_period": lookback_period,
                    "as_of_date": str(as_of_ts.date()) if as_of_ts is not None else "",
                    "avg_dollar_volume_63d": avg_dollar_volume,
                }
            )

    frame = pd.DataFrame(records)

    if frame.empty:
        raise ValueError(f"No usable volatility history was downloaded for {exchange_name}.")

    # Sort by descending volatility, using observation count as a tiebreaker.
    frame = (
        frame
        .sort_values(["volatility_1y", "observations"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )

    frame["rank_within_exchange"] = np.arange(1, len(frame) + 1)

    return frame


def build_top_volatile_universe(
    top_n_per_exchange: int = 500,
    lookback_period: str = "1y",
    as_of_date: str | None = None,
    min_price: float = 5.0,
    min_avg_dollar_volume_63d: float = 5_000_000.0,
) -> pd.DataFrame:
    """Combine NASDAQ and NYSE top-volatile rankings into a single DataFrame with a global rank."""
    nasdaq_tickers, nyse_tickers = load_exchange_tickers()

    nasdaq_top = compute_realized_volatility_ranks(
        nasdaq_tickers,
        exchange_name="NASDAQ",
        top_n=top_n_per_exchange,
        lookback_period=lookback_period,
        as_of_date=as_of_date,
        min_price=min_price,
        min_avg_dollar_volume_63d=min_avg_dollar_volume_63d,
    )

    nyse_top = compute_realized_volatility_ranks(
        nyse_tickers,
        exchange_name="NYSE",
        top_n=top_n_per_exchange,
        lookback_period=lookback_period,
        as_of_date=as_of_date,
        min_price=min_price,
        min_avg_dollar_volume_63d=min_avg_dollar_volume_63d,
    )

    combined = pd.concat([nasdaq_top, nyse_top], ignore_index=True)

    # Rank all stocks across both exchanges by volatility (1 = most volatile overall).
    combined["global_rank"] = (
        combined["volatility_1y"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    combined = combined.sort_values(["exchange", "rank_within_exchange"]).reset_index(drop=True)

    return combined


def main() -> None:
    """Parse CLI arguments and write the volatile-universe CSV to disk."""
    parser = argparse.ArgumentParser(
        description="Generate a NASDAQ and NYSE most-volatile universe CSV, optionally as of a specific date."
    )

    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "nyse_nasdaq_most_volatile_asof_2024_01_01.csv"
        ),
        help="Output CSV path (default: data/nyse_nasdaq_most_volatile_asof_2024_01_01.csv).",
    )
    parser.add_argument("--top-n-per-exchange", type=int, default=500,
                        help="Number of most-volatile tickers to keep per exchange (default: 500).")
    parser.add_argument("--lookback-period", default="1y",
                        help="yfinance period string for the history window (default: '1y').")
    parser.add_argument("--as-of-date", default=None,
                        help="Optional YYYY-MM-DD date to make the universe point-in-time valid.")
    parser.add_argument("--min-price", type=float, default=5.0,
                        help="Minimum stock price filter in dollars (default: $5.00).")
    parser.add_argument("--min-avg-dollar-volume-63d", type=float, default=5_000_000.0,
                        help="Minimum 63-day average daily dollar volume filter (default: $5,000,000).")

    args = parser.parse_args()

    frame = build_top_volatile_universe(
        top_n_per_exchange=args.top_n_per_exchange,
        lookback_period=args.lookback_period,
        as_of_date=args.as_of_date,
        min_price=args.min_price,
        min_avg_dollar_volume_63d=args.min_avg_dollar_volume_63d,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create missing parent directories.
    frame.to_csv(output_path, index=False)

    print(f"Wrote {len(frame)} rows to {output_path}")


if __name__ == "__main__":
    main()
