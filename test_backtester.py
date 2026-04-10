import unittest

import numpy as np
import pandas as pd

from backtester import (
    BacktestConfig,
    run_random_universe_backtest,
    run_single_portfolio_backtest,
    sample_unique_portfolios,
)
from optimizer import PortfolioConfig


def make_synthetic_prices(tickers: list[str], periods: int = 540) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    market = rng.normal(0.00035, 0.009, size=periods)
    data = {}
    for idx, ticker in enumerate(tickers):
        alpha = 0.00005 + idx * 0.00001
        beta = 0.7 + (idx % 5) * 0.08
        noise_scale = 0.006 + (idx % 4) * 0.001
        returns = alpha + beta * market + rng.normal(0.0, noise_scale, size=periods)
        prices = 50.0 + idx * 2.0
        series = prices * np.exp(np.cumsum(returns))
        data[ticker] = series
    return pd.DataFrame(data, index=dates)


class BacktesterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tickers = [f"TK{idx:02d}" for idx in range(15)]
        self.prices = make_synthetic_prices(self.tickers)
        self.config = PortfolioConfig(
            capital=100000.0,
            risk_aversion=4.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=0.05,
            max_cash_weight=0.25,
            cash_yield=0.04,
            treasury_bill_yield=0.045,
            auto_max_allocation=True,
            auto_max_floor=0.02,
            auto_max_ceiling=0.35,
            auto_treasury_bill_yield=False,
            target_expected_return=None,
            target_volatility=None,
            hmm_states=2,
            simulation_paths=250,
            simulation_horizon_years=1.0,
        )

    def test_sample_unique_portfolios_returns_unique_sorted_combinations(self) -> None:
        combos = sample_unique_portfolios(self.tickers, portfolio_size=4, combination_count=25, seed=7)
        self.assertEqual(len(combos), 25)
        self.assertEqual(len(set(combos)), 25)
        self.assertTrue(all(tuple(sorted(combo)) == combo for combo in combos))

    def test_run_single_portfolio_backtest_reports_prediction_accuracy(self) -> None:
        result = run_single_portfolio_backtest(
            tickers=self.tickers[:5],
            price_history=self.prices,
            config=self.config,
            formation_date="2024-01-02",
            lookback_years=1.0,
            forward_years=1.0,
        )
        self.assertEqual(len(result["tickers"]), 5)
        self.assertIn("expected_return", result)
        self.assertIn("realized_return", result)
        self.assertIn("prediction_accurate", result)
        self.assertGreater(result["history_observations"], 100)
        self.assertGreater(result["evaluation_observations"], 100)
        self.assertGreater(result["realized_terminal_value"], 0.0)

    def test_run_random_universe_backtest_summarizes_accuracy(self) -> None:
        backtest = BacktestConfig(
            formation_date="2024-01-02",
            lookback_years=1.0,
            forward_years=1.0,
            universe_size=10,
            portfolio_size=5,
            combination_count=8,
            random_seed=11,
        )
        summary = run_random_universe_backtest(
            universe_tickers=self.tickers,
            config=self.config,
            backtest=backtest,
            prices=self.prices,
        )
        self.assertEqual(summary["requested_combinations"], 8)
        self.assertEqual(summary["completed_combinations"], 8)
        self.assertGreaterEqual(summary["accuracy_rate"], 0.0)
        self.assertLessEqual(summary["accuracy_rate"], 1.0)
        self.assertEqual(len(summary["results"]), 8)


if __name__ == "__main__":
    unittest.main()
