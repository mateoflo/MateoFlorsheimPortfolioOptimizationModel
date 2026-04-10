import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from optimizer import (
    AssetInput,
    PortfolioConfig,
    compute_auto_max_weights,
    denoise_correlation_matrix_mp,
    estimate_expected_returns,
    estimate_asset_statistics,
    estimate_regime_statistics,
    fetch_official_1y_tbill_yield,
    optimize_portfolio,
    optimize_portfolio_from_tickers,
    resolve_treasury_bill_yield,
    run_hmm_monte_carlo_projection,
)


class OptimizerTests(unittest.TestCase):
    def test_optimizer_respects_cash_and_tbill_constraints(self) -> None:
        assets = [
            AssetInput("SPY", 500.0, 0.10, 0.18, 0.50),
            AssetInput("QQQ", 450.0, 0.12, 0.24, 0.35),
            AssetInput("XLF", 42.0, 0.08, 0.20, 0.30),
            AssetInput("XLV", 145.0, 0.07, 0.15, 0.30),
        ]
        corr = np.array(
            [
                [1.0, 0.86, 0.72, 0.64],
                [0.86, 1.0, 0.68, 0.61],
                [0.72, 0.68, 1.0, 0.58],
                [0.64, 0.61, 0.58, 1.0],
            ]
        )
        config = PortfolioConfig(
            capital=100000.0,
            risk_aversion=4.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=0.05,
            max_cash_weight=0.30,
            cash_yield=0.04,
            treasury_bill_yield=0.045,
            auto_max_allocation=False,
            auto_max_floor=0.02,
            auto_max_ceiling=0.10,
            auto_treasury_bill_yield=False,
            target_expected_return=None,
            target_volatility=0.12,
        )

        result = optimize_portfolio(assets, corr, config)
        self.assertGreaterEqual(result["defensive_weight"], 0.05 - 1e-6)
        self.assertLessEqual(result["defensive_weight"], 0.305)
        self.assertGreaterEqual(result["treasury_bill_weight"], 0.0)
        self.assertAlmostEqual(
            sum(row["invested_dollars"] for row in result["asset_rows"]) + result["cash_dollars"] + result["treasury_bill_dollars"],
            100000.0,
            places=4,
        )

    @patch("optimizer.yf.download")
    def test_estimate_asset_statistics_from_price_history(self, mock_download) -> None:
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        close = pd.DataFrame(
            {
                ("Close", "SPY"): np.linspace(100, 120, len(dates)),
                ("Close", "IEF"): np.linspace(90, 95, len(dates)),
            },
            index=dates,
        )
        close.columns = pd.MultiIndex.from_tuples(close.columns)
        mock_download.return_value = close

        assets = [
            AssetInput("SPY", 0.0, 0.0, 0.0, 0.60),
            AssetInput("IEF", 0.0, 0.0, 0.0, 0.50),
        ]
        estimated_assets, corr, sample_window, _log_returns, denoise_info, regime_info, return_model_info = estimate_asset_statistics(
            assets, 1.0, hmm_states=2
        )

        self.assertEqual(len(estimated_assets), 2)
        self.assertEqual(corr.shape, (2, 2))
        self.assertGreater(estimated_assets[0].price, 0)
        self.assertIn("2024-01-01", sample_window)
        self.assertIn("signal_eigenvalues", denoise_info)
        self.assertIn("transition_matrix", regime_info)
        self.assertIn("method", return_model_info)

    def test_hmm_monte_carlo_projection_outputs_valid_stats(self) -> None:
        regime_info = {
            "transition_matrix": np.array([[0.92, 0.08], [0.15, 0.85]]),
            "last_state_probabilities": np.array([0.7, 0.3]),
            "regime_means": np.array([[0.0004, 0.0002], [-0.0002, -0.0004]]),
            "regime_covariances": np.array(
                [
                    [[0.0001, 0.00002], [0.00002, 0.00009]],
                    [[0.0004, 0.00005], [0.00005, 0.00035]],
                ]
            ),
        }
        result = run_hmm_monte_carlo_projection(
            capital=100000.0,
            stock_weights=np.array([0.55, 0.25]),
            cash_weight=0.10,
            treasury_bill_weight=0.10,
            cash_yield=0.04,
            treasury_bill_yield=0.045,
            regime_info=regime_info,
            years=1.0,
            paths=5000,
            seed=7,
        )
        self.assertEqual(result["paths"], 5000)
        self.assertGreater(result["expected_terminal_value"], 0)
        self.assertGreaterEqual(result["probability_of_loss"], 0.0)
        self.assertLessEqual(result["probability_of_loss"], 1.0)
        self.assertEqual(result["method"], "hmm_regime_monte_carlo")

    @patch("optimizer.yf.download")
    def test_optimize_from_tickers_includes_monte_carlo(self, mock_download) -> None:
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        close = pd.DataFrame(
            {
                ("Close", "SPY"): np.linspace(100, 120, len(dates)),
                ("Close", "IEF"): np.linspace(90, 95, len(dates)),
            },
            index=dates,
        )
        close.columns = pd.MultiIndex.from_tuples(close.columns)
        mock_download.return_value = close

        assets = [
            AssetInput("SPY", 0.0, 0.0, 0.0, 0.60),
            AssetInput("IEF", 0.0, 0.0, 0.0, 0.50),
        ]
        config = PortfolioConfig(
            capital=100000.0,
            risk_aversion=4.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=0.05,
            max_cash_weight=0.30,
            cash_yield=0.04,
            treasury_bill_yield=0.045,
            auto_max_allocation=False,
            auto_max_floor=0.02,
            auto_max_ceiling=0.10,
            auto_treasury_bill_yield=False,
            target_expected_return=None,
            target_volatility=0.12,
            hmm_states=2,
            simulation_paths=2000,
            simulation_horizon_years=1.0,
        )
        result = optimize_portfolio_from_tickers(assets, config, lookback_years=1.0)
        self.assertIn("monte_carlo", result)
        self.assertEqual(result["monte_carlo"]["paths"], 2000)
        self.assertEqual(result["monte_carlo"]["method"], "hmm_regime_monte_carlo")
        self.assertIn("denoise_info", result)
        self.assertEqual(result["regime_info"]["states"], 2)
        self.assertIn("risk_score", result)
        self.assertIn("risk_label", result)

    @patch("optimizer.yf.download")
    def test_auto_treasury_bill_yield_is_used_when_available(self, mock_download) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        irx = pd.DataFrame({"Close": [5.10, 5.05, 5.00]}, index=dates)
        mock_download.return_value = irx
        config = PortfolioConfig(
            capital=100000.0,
            risk_aversion=4.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=0.05,
            max_cash_weight=0.30,
            cash_yield=0.04,
            treasury_bill_yield=0.045,
            auto_max_allocation=False,
            auto_max_floor=0.02,
            auto_max_ceiling=0.10,
            auto_treasury_bill_yield=True,
            target_expected_return=None,
        )
        value, source = resolve_treasury_bill_yield(config)
        self.assertAlmostEqual(value, 0.05, places=6)
        self.assertIn("^IRX", source)

    @patch("optimizer.urlopen")
    def test_fetch_official_1y_tbill_yield_from_treasury_csv(self, mock_urlopen) -> None:
        csv_text = "\n".join(
            [
                "Date,4 WEEKS BANK DISCOUNT,4 WEEKS COUPON EQUIVALENT,52 WEEKS BANK DISCOUNT,52 WEEKS COUPON EQUIVALENT",
                "03/24/2026,4.20,4.27,3.90,4.08",
                "03/25/2026,4.18,4.25,3.88,4.05",
            ]
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return csv_text.encode("utf-8")

        mock_urlopen.return_value = FakeResponse()
        value = fetch_official_1y_tbill_yield()
        self.assertAlmostEqual(value, 0.0405, places=6)

    @patch("optimizer.urlopen", side_effect=RuntimeError("offline"))
    def test_blank_tbill_field_still_attempts_auto_fetch(self, _mock_urlopen) -> None:
        config = PortfolioConfig(
            capital=100000.0,
            risk_aversion=4.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=0.05,
            max_cash_weight=0.30,
            cash_yield=0.04,
            treasury_bill_yield=None,
            auto_max_allocation=False,
            auto_max_floor=0.02,
            auto_max_ceiling=0.10,
            auto_treasury_bill_yield=False,
            target_expected_return=None,
        )
        with self.assertRaises(ValueError):
            resolve_treasury_bill_yield(config)

    def test_marchenko_pastur_denoising_preserves_correlation_shape(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=(300, 3))
        noise = rng.normal(scale=0.3, size=(300, 2))
        returns = np.column_stack([base[:, 0], base[:, 0] * 0.8 + noise[:, 0], base[:, 1], base[:, 2], noise[:, 1]])
        denoised_corr, info = denoise_correlation_matrix_mp(returns)
        self.assertEqual(denoised_corr.shape, (5, 5))
        self.assertTrue(np.allclose(np.diag(denoised_corr), 1.0))
        self.assertEqual(info["method"], "marchenko_pastur")

    def test_hmm_regime_estimation_returns_two_states(self) -> None:
        rng = np.random.default_rng(2)
        low = rng.normal(0.0005, 0.005, size=(120, 3))
        high = rng.normal(-0.0003, 0.02, size=(120, 3))
        returns = np.vstack([low, high])
        regime_info = estimate_regime_statistics(returns, n_states=2)
        self.assertEqual(regime_info["transition_matrix"].shape, (2, 2))
        self.assertEqual(regime_info["regime_means"].shape[0], 2)

    def test_auto_max_weights_respect_floor_and_ceiling(self) -> None:
        assets = [
            AssetInput("LOWVOL", 100.0, 0.07, 0.10, 0.50),
            AssetInput("MIDVOL", 100.0, 0.08, 0.20, 0.50),
            AssetInput("HIVOL", 100.0, 0.10, 0.40, 0.50),
        ]
        corr = np.array(
            [
                [1.0, 0.20, 0.10],
                [0.20, 1.0, 0.30],
                [0.10, 0.30, 1.0],
            ]
        )
        config = PortfolioConfig(
            capital=100000.0,
            risk_aversion=4.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=0.05,
            max_cash_weight=0.30,
            cash_yield=0.04,
            treasury_bill_yield=0.045,
            auto_max_allocation=True,
            auto_max_floor=0.02,
            auto_max_ceiling=0.20,
            auto_treasury_bill_yield=False,
            target_expected_return=None,
        )
        caps = compute_auto_max_weights(assets, corr, config)
        self.assertEqual(len(caps), 3)
        self.assertTrue(all(0.02 <= cap <= 0.20 for cap in caps))
        self.assertGreater(caps[0], caps[2])

    def test_expected_return_estimators_produce_valid_shapes(self) -> None:
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        returns = pd.DataFrame(
            {
                "A": np.linspace(0.0001, 0.0006, len(dates)),
                "B": np.linspace(0.0002, 0.0004, len(dates)),
                "C": np.linspace(-0.0001, 0.0005, len(dates)),
            },
            index=dates,
        )
        annual_cov = returns.cov().to_numpy(dtype=float) * 252.0
        for method in ["historical_mean", "bayes_stein", "market_factor", "black_litterman"]:
            mu, info = estimate_expected_returns(returns, annual_cov, method=method, shrinkage=0.5)
            self.assertEqual(mu.shape, (3,))
            self.assertIn("method", info)

    def test_target_expected_return_constraint_is_respected(self) -> None:
        assets = [
            AssetInput("SPY", 500.0, 0.12, 0.18, 0.60),
            AssetInput("QQQ", 450.0, 0.15, 0.24, 0.50),
            AssetInput("XLV", 145.0, 0.08, 0.15, 0.40),
        ]
        corr = np.array(
            [
                [1.0, 0.85, 0.60],
                [0.85, 1.0, 0.55],
                [0.60, 0.55, 1.0],
            ]
        )
        config = PortfolioConfig(
            capital=100000.0,
            risk_aversion=3.0,
            shrinkage=0.20,
            concentration_penalty=0.05,
            min_cash_weight=0.05,
            max_cash_weight=0.40,
            cash_yield=0.04,
            treasury_bill_yield=0.045,
            auto_max_allocation=False,
            auto_max_floor=0.02,
            auto_max_ceiling=0.20,
            auto_treasury_bill_yield=False,
            target_expected_return=0.08,
            target_volatility=None,
        )
        result = optimize_portfolio(assets, corr, config)
        self.assertGreaterEqual(result["expected_return"], 0.08 - 1e-6)


if __name__ == "__main__":
    unittest.main()
