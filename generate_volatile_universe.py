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


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
COMMON_STOCK_PATTERN = re.compile(r"^[A-Z]{1,5}$")


def _download_text(url: str) -> str:
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="ignore")


def load_exchange_tickers() -> tuple[list[str], list[str]]:
    nasdaq_raw = _download_text(NASDAQ_LISTED_URL)
    other_raw = _download_text(OTHER_LISTED_URL)

    nasdaq = pd.read_csv(StringIO(nasdaq_raw), sep="|")
    other = pd.read_csv(StringIO(other_raw), sep="|")

    nasdaq = nasdaq[nasdaq["Symbol"].notna()].copy()
    nasdaq = nasdaq[nasdaq["Symbol"] != "File Creation Time"]
    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
    nasdaq = nasdaq[nasdaq["ETF"] == "N"]

    other = other[other["ACT Symbol"].notna()].copy()
    other = other[other["ACT Symbol"] != "File Creation Time"]
    other = other[other["Test Issue"] == "N"]
    other = other[other["ETF"] == "N"]

    nyse = other[other["Exchange"] == "N"].copy()
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
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return pd.DataFrame(columns=requested_tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        if field_name in raw.columns.get_level_values(0):
            prices = raw[field_name].copy()
        else:
            prices = raw.xs(field_name, axis=1, level=0, drop_level=True).copy()
    else:
        prices = raw.rename(columns={field_name: requested_tickers[0] if len(requested_tickers) == 1 else field_name})
        if field_name in prices.columns:
            prices = prices[[field_name]].rename(columns={field_name: requested_tickers[0]})
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=requested_tickers[0])
    return prices


def _extract_close_prices(raw: pd.DataFrame | pd.Series, requested_tickers: list[str]) -> pd.DataFrame:
    return _extract_field(raw, requested_tickers, "Close")


def _extract_volume(raw: pd.DataFrame | pd.Series, requested_tickers: list[str]) -> pd.DataFrame:
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
    as_of_ts = pd.Timestamp(as_of_date) if as_of_date else None
    lookback_days = max(365, 252)
    if as_of_ts is not None:
        start_date = (as_of_ts - pd.DateOffset(days=lookback_days)).date()
        end_date = (as_of_ts + pd.DateOffset(days=5)).date()
    records: list[dict] = []
    for idx in range(0, len(tickers), chunk_size):
        chunk = tickers[idx : idx + chunk_size]
        print(f"[{exchange_name}] Downloading chunk {idx // chunk_size + 1} / {(len(tickers) + chunk_size - 1) // chunk_size}")
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            if as_of_ts is not None:
                raw = yf.download(
                    chunk,
                    start=str(start_date),
                    end=str(end_date),
                    interval="1d",
                    auto_adjust=True,
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
        prices = _extract_close_prices(raw, chunk).sort_index()
        volumes = _extract_volume(raw, chunk).sort_index()
        if prices.empty:
            continue
        prices = prices.ffill()
        volumes = volumes.fillna(0.0)
        if as_of_ts is not None:
            prices = prices.loc[prices.index <= as_of_ts]
            volumes = volumes.loc[volumes.index <= as_of_ts]
        returns = np.log(prices / prices.shift(1))
        for ticker in chunk:
            if ticker not in prices.columns:
                continue
            series = prices[ticker].dropna()
            ret_series = returns[ticker].dropna() if ticker in returns.columns else pd.Series(dtype=float)
            volume_series = volumes[ticker].reindex(series.index).fillna(0.0) if ticker in volumes.columns else pd.Series(dtype=float)
            if len(series) < 126 or len(ret_series) < 60:
                continue
            if as_of_ts is not None:
                if pd.Timestamp(series.index[-1]).date() < (as_of_ts - pd.DateOffset(days=10)).date():
                    continue
            last_close = float(series.iloc[-1])
            if last_close < min_price:
                continue
            if volume_series.empty:
                continue
            dollar_volume = series * volume_series
            avg_dollar_volume = float(dollar_volume.tail(63).mean())
            if not np.isfinite(avg_dollar_volume) or avg_dollar_volume < min_avg_dollar_volume_63d:
                continue
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
    frame = frame.sort_values(["volatility_1y", "observations"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    frame["rank_within_exchange"] = np.arange(1, len(frame) + 1)
    return frame


def build_top_volatile_universe(
    top_n_per_exchange: int = 500,
    lookback_period: str = "1y",
    as_of_date: str | None = None,
    min_price: float = 5.0,
    min_avg_dollar_volume_63d: float = 5_000_000.0,
) -> pd.DataFrame:
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
    combined["global_rank"] = combined["volatility_1y"].rank(method="first", ascending=False).astype(int)
    combined = combined.sort_values(["exchange", "rank_within_exchange"]).reset_index(drop=True)
    return combined


def main() -> None:
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
        help="Output CSV path.",
    )
    parser.add_argument("--top-n-per-exchange", type=int, default=500)
    parser.add_argument("--lookback-period", default="1y")
    parser.add_argument("--as-of-date", default=None, help="Optional YYYY-MM-DD date to make the universe point-in-time valid.")
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-avg-dollar-volume-63d", type=float, default=5_000_000.0)
    args = parser.parse_args()

    frame = build_top_volatile_universe(
        top_n_per_exchange=args.top_n_per_exchange,
        lookback_period=args.lookback_period,
        as_of_date=args.as_of_date,
        min_price=args.min_price,
        min_avg_dollar_volume_63d=args.min_avg_dollar_volume_63d,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Wrote {len(frame)} rows to {output_path}")


if __name__ == "__main__":
    main()
