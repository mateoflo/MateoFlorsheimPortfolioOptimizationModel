import unittest

from webapp import parse_request_payload, serialize_result


class WebAppTests(unittest.TestCase):
    def test_parse_request_payload_supports_whole_number_percentages(self) -> None:
        payload = {
            "settings": {
                "capital": "100000",
                "lookback_years": "3",
                "min_cash_weight": "5",
                "max_cash_weight": "30",
                "cash_yield": "4",
                "treasury_bill_yield": "4.5",
                "target_expected_return": "10",
                "target_volatility": "12",
                "auto_max_floor": "2",
                "auto_max_ceiling": "8",
                "expected_return_method": "black_litterman",
                "expected_return_shrinkage": "50",
                "simulation_paths": "1000",
                "simulation_horizon_years": "1",
                "max_allocation_mode": "Auto",
                "auto_treasury_bill_yield": True,
            },
            "assets": [
                {"ticker": "SPY", "max_weight": "45"},
                {"ticker": "QQQ", "max_weight": "35"},
            ],
        }

        assets, config, lookback_years = parse_request_payload(payload)

        self.assertEqual(len(assets), 2)
        self.assertEqual(assets[0].ticker, "SPY")
        self.assertAlmostEqual(assets[0].max_weight, 0.45)
        self.assertAlmostEqual(config.min_cash_weight, 0.05)
        self.assertAlmostEqual(config.max_cash_weight, 0.30)
        self.assertAlmostEqual(config.cash_yield, 0.04)
        self.assertAlmostEqual(config.treasury_bill_yield, 0.045)
        self.assertAlmostEqual(config.target_expected_return, 0.10)
        self.assertAlmostEqual(config.target_volatility, 0.12)
        self.assertEqual(config.expected_return_method, "black_litterman")
        self.assertTrue(config.auto_max_allocation)
        self.assertEqual(lookback_years, 3.0)

    def test_serialize_result_formats_summary_and_table(self) -> None:
        result = {
            "expected_return": 0.12,
            "expected_volatility": 0.18,
            "risk_label": "Moderate",
            "risk_score": 42,
            "cash_dollars": 12000.0,
            "cash_weight": 0.12,
            "treasury_bill_dollars": 8000.0,
            "treasury_bill_weight": 0.08,
            "defensive_dollars": 20000.0,
            "defensive_weight": 0.20,
            "treasury_bill_yield": 0.045,
            "treasury_bill_source": "auto",
            "sample_window": "2023-01-03 to 2024-01-02",
            "monte_carlo": {
                "expected_terminal_value": 108000.0,
                "median_terminal_value": 106500.0,
                "value_at_5pct": 91000.0,
                "probability_of_loss": 0.18,
                "paths": 1000,
                "horizon_years": 1.0,
            },
            "return_model_info": {
                "method": "black_litterman",
                "description": "Blends equilibrium and sample estimates.",
            },
            "asset_rows": [
                {
                    "ticker": "SPY",
                    "price": 500.0,
                    "expected_return": 0.11,
                    "volatility": 0.17,
                    "continuous_weight": 0.35,
                    "max_weight": 0.45,
                    "recommended_shares": 70,
                    "invested_dollars": 35000.0,
                    "realized_weight": 0.35,
                }
            ],
        }

        payload = serialize_result(result)

        self.assertEqual(payload["summary"]["expected_return"], "12.00%")
        self.assertEqual(payload["summary"]["risk_level"], "Moderate (42/100)")
        self.assertEqual(payload["summary"]["tbill_source"], "auto")
        self.assertEqual(payload["summary"]["expected_return_method"], "black_litterman")
        self.assertEqual(payload["asset_rows"][0]["ticker"], "SPY")
        self.assertEqual(payload["asset_rows"][0]["recommended_shares"], 70)


if __name__ == "__main__":
    unittest.main()
