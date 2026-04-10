# Portfolio Optimizer GUI

This folder contains three related tools:

- `app.py`: a desktop portfolio optimizer for live portfolio construction
- `webapp.py`: a browser-based HTML frontend backed by the same Python optimizer
- `backtester.py`: a no-lookahead random-universe backtester for testing whether the model's one-year expected return was actually met

## Optimizer Summary

The optimizer:

- estimates stock return and volatility from historical daily log returns
- denoises the stock correlation matrix with Marchenko-Pastur filtering
- builds a constrained long-only allocation across:
  - stocks
  - cash
  - Treasury bills
- can auto-fetch the 1-year Treasury bill yield
- runs HMM regime-switching Monte Carlo for a one-year distribution of outcomes

The defensive sleeve is:

- `cash + Treasury bills`

So `Minimum Cash %` and `Maximum Cash %` apply to the combined defensive sleeve, not idle cash alone.

## GUI Inputs

The main GUI takes:

- stock tickers
- max allocation mode:
  - `Manual`
  - `Auto`
- if `Manual`: per-stock `Max Weight %`
- if `Auto`: the model generates stock-specific caps from volatility and correlation
- capital
- lookback years
- minimum cash %
- maximum cash %
- cash yield %
- optional target return %
- optional target volatility %
- Monte Carlo paths
- Monte Carlo horizon years

All percentage fields accept whole-number percents:

- `1` means `1%`
- `5` means `5%`
- `25` means `25%`

## Run The Desktop GUI

```bash
cd /Users/mateoflo/Desktop/florsheim_capital/PortfolioOptimizerGUI
python3 app.py
```

## Run The HTML Frontend

```bash
cd /Users/mateoflo/Desktop/florsheim_capital/PortfolioOptimizerGUI
python3 webapp.py
```

Then open:

```text
http://127.0.0.1:8080
```

The HTML frontend exposes:

- `GET /`
- `GET /health`
- `POST /api/optimize`

## Random-Universe Backtester

`backtester.py` is built to avoid lookahead bias.

For each sampled portfolio, it:

1. chooses a formation date
2. uses only the trailing lookback window ending on that formation date
3. estimates expected return, volatility, denoised correlation, and HMM regimes from that historical window only
4. optimizes the portfolio at that point in time
5. runs one-year Monte Carlo from the point-in-time estimated model
6. holds the portfolio through the next one-year evaluation window
7. checks whether realized return was at least as high as the model's expected return

The backtester reports:

- expected return
- Monte Carlo implied expected return
- realized one-year return
- terminal portfolio value
- whether the prediction was accurate
- overall accuracy rate across all sampled portfolios

## Backtester Inputs

The backtester expects a universe CSV with a `ticker` column. This is the practical way to supply your own screened NYSE/NASDAQ universe of roughly 2,000 names.

Key settings:

- formation date
- universe size
- portfolio size
- number of random unique combinations
- lookback years
- forward years
- capital
- minimum and maximum defensive sleeve %
- cash yield %
- optional target return %
- optional target volatility %

## Run The Backtester

If you generated the built-in volatile universe, the backtester defaults to:

- `data/nyse_nasdaq_most_volatile_asof_2024_01_01.csv`

Example:

```bash
cd /Users/mateoflo/Desktop/florsheim_capital/PortfolioOptimizerGUI
python3 backtester.py \
  --formation-date 2024-01-01 \
  --universe-size 1000 \
  --portfolio-size 20 \
  --combination-count 100 \
  --lookback-years 1 \
  --forward-years 1 \
  --min-cash-pct 5 \
  --max-cash-pct 25 \
  --cash-yield-pct 4 \
  --mc-paths 1000
```

Backtest CSV outputs (including the return model comparison) are stored in `BackTestResults/`, so point any reporting scripts there.

Notes:

- the backtester auto-enables auto max allocation
- it reuses the same optimizer logic as the GUI
- it needs market data access to run on live tickers
- `10,000` combinations is computationally heavy and should be treated as a batch job

## Notes on the code layout

- Scripts expect `BackTestResults/` to house the derived backtest CSVs and `data/` to hold the ticker universes, so the current folder structure should not be rearranged.
- Every public function in `optimizer.py`, `backtester.py`, `app.py`, and `webapp.py` now has a one-line “high school freshman friendly” comment describing its purpose, which should help new contributors understand the flow more quickly before diving into the details.

## Verification

```bash
python3 -m py_compile optimizer.py backtester.py test_optimizer.py test_backtester.py
python3 -m unittest test_optimizer.py test_backtester.py
```
