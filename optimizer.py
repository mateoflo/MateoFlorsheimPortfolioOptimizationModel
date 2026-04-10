from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from urllib.request import urlopen
import csv
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class AssetInput:
    ticker: str
    price: float
    expected_return: float
    volatility: float
    max_weight: float


@dataclass
class PortfolioConfig:
    capital: float
    risk_aversion: float
    shrinkage: float
    concentration_penalty: float
    min_cash_weight: float
    max_cash_weight: float | None
    cash_yield: float
    treasury_bill_yield: float | None
    auto_max_allocation: bool = False
    auto_max_floor: float = 0.02
    auto_max_ceiling: float = 0.10
    auto_treasury_bill_yield: bool = True
    target_expected_return: float | None = None
    target_volatility: float | None = None
    expected_return_method: str = "historical_mean"
    expected_return_shrinkage: float = 0.50
    hmm_states: int = 2
    simulation_paths: int = 10000
    simulation_horizon_years: float = 1.0


# This turns a bunch of returns into a grid so we know how each asset moves with others.
def build_covariance_matrix(
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    vols = np.asarray(volatilities, dtype=float)
    corr = np.asarray(correlation_matrix, dtype=float)
    base_cov = np.outer(vols, vols) * corr
    diagonal = np.diag(np.diag(base_cov))
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    return (1.0 - shrinkage) * base_cov + shrinkage * diagonal


# This changes the covariance grid into correlations so we can compare pairs on the same scale.
def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    std = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    corr = covariance / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


# This guesses how much money each asset might make based on history so we have a target.
def estimate_expected_returns(
    log_returns: pd.DataFrame,
    annual_covariance: np.ndarray,
    method: str = "historical_mean",
    shrinkage: float = 0.50,
) -> tuple[np.ndarray, dict]:
    returns = log_returns.to_numpy(dtype=float)
    sample_mean = log_returns.mean().to_numpy(dtype=float) * 252.0
    n_assets = returns.shape[1]
    method_key = (method or "historical_mean").strip().lower()
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    if method_key == "historical_mean":
        return sample_mean, {
            "method": "historical_mean",
            "shrinkage": 0.0,
            "description": "Annualized sample mean of daily log returns.",
        }

    if method_key == "bayes_stein":
        target = float(np.mean(sample_mean))
        variances = np.clip(np.diag(annual_covariance), 1e-8, None)
        t_obs = max(len(log_returns), 1)
        asset_shrink = variances / (variances + np.square(sample_mean - target) * t_obs + 1e-8)
        posterior = (1.0 - asset_shrink) * sample_mean + asset_shrink * target
        return posterior, {
            "method": "bayes_stein",
            "grand_mean_target": target,
            "average_shrinkage": float(np.mean(asset_shrink)),
            "description": "Bayes-Stein shrinkage toward the cross-sectional grand mean.",
        }

    market_factor = log_returns.mean(axis=1).to_numpy(dtype=float)
    market_mean = float(np.mean(market_factor) * 252.0)
    market_var = float(np.var(market_factor, ddof=1) * 252.0) if len(market_factor) > 1 else 0.0
    asset_cov_with_market = np.cov(returns, market_factor, rowvar=False)[:n_assets, n_assets]
    betas = asset_cov_with_market / max(market_var, 1e-8)
    factor_implied = betas * market_mean

    if method_key in {"market_factor", "factor", "capm"}:
        return factor_implied, {
            "method": "market_factor",
            "market_mean": market_mean,
            "market_variance": market_var,
            "description": "Market-factor-implied expected returns using asset beta to the equal-weight market factor.",
        }

    if method_key in {"black_litterman", "black_litterman_blend", "bl"}:
        posterior = (1.0 - shrinkage) * factor_implied + shrinkage * sample_mean
        return posterior, {
            "method": "black_litterman_blend",
            "prior_method": "market_factor",
            "view_method": "historical_mean",
            "blend_weight_on_sample_views": shrinkage,
            "description": "Black-Litterman-style blend of factor-implied prior returns with historical sample-return views.",
        }

    raise ValueError(
        "Unknown expected return method. Use one of: historical_mean, bayes_stein, market_factor, black_litterman."
    )


# This cleans the noisy correlation matrix so we trust the real signals before optimizing.
def denoise_correlation_matrix_mp(log_returns: np.ndarray) -> tuple[np.ndarray, dict]:
    returns = np.asarray(log_returns, dtype=float)
    t_obs, n_assets = returns.shape
    if t_obs < max(20, n_assets + 2):
        corr = np.corrcoef(returns, rowvar=False)
        return corr, {
            "method": "sample_correlation",
            "q_ratio": float(t_obs / max(n_assets, 1)),
            "lambda_plus": None,
            "signal_eigenvalues": int(n_assets),
            "noise_eigenvalues": 0,
        }

    sample_corr = np.corrcoef(returns, rowvar=False)
    q_ratio = t_obs / n_assets
    lambda_plus = (1.0 + 1.0 / np.sqrt(q_ratio)) ** 2

    eigvals, eigvecs = np.linalg.eigh(sample_corr)
    signal_mask = eigvals > lambda_plus
    signal_count = int(np.sum(signal_mask))
    noise_count = int(len(eigvals) - signal_count)

    if signal_count == 0 or noise_count == 0:
        denoised = sample_corr
    else:
        avg_noise = float(np.mean(eigvals[~signal_mask]))
        adjusted_eigvals = eigvals.copy()
        adjusted_eigvals[~signal_mask] = avg_noise
        denoised = eigvecs @ np.diag(adjusted_eigvals) @ eigvecs.T
        denoised = correlation_from_covariance(denoised)

    return denoised, {
        "method": "marchenko_pastur",
        "q_ratio": float(q_ratio),
        "lambda_plus": float(lambda_plus),
        "signal_eigenvalues": signal_count,
        "noise_eigenvalues": noise_count,
    }


# This checks the assets, correlation, and config so nothing breaks during optimization.
def validate_inputs(assets: Iterable[AssetInput], corr: np.ndarray, config: PortfolioConfig) -> None:
    assets = list(assets)
    if not assets:
        raise ValueError("At least one asset is required.")
    if config.capital <= 0:
        raise ValueError("Capital must be positive.")
    if config.min_cash_weight < 0:
        raise ValueError("Minimum cash weight must be non-negative.")
    if config.max_cash_weight is not None:
        if not 0 <= config.max_cash_weight <= 1:
            raise ValueError("Maximum cash weight must be between 0% and 100%.")
        if config.max_cash_weight < config.min_cash_weight:
            raise ValueError("Maximum cash weight cannot be below minimum cash weight.")
    if not 0 < config.auto_max_floor <= 1:
        raise ValueError("Auto max floor must be between 0% and 100%.")
    if not 0 < config.auto_max_ceiling <= 1:
        raise ValueError("Auto max ceiling must be between 0% and 100%.")
    if config.auto_max_floor > config.auto_max_ceiling:
        raise ValueError("Auto max floor cannot exceed auto max ceiling.")

    n = len(assets)
    if corr.shape != (n, n):
        raise ValueError("Correlation matrix dimensions do not match the asset list.")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(corr), np.ones(n), atol=1e-8):
        raise ValueError("Correlation matrix diagonal must be 1.0.")

    for asset in assets:
        if not asset.ticker.strip():
            raise ValueError("Each asset requires a ticker.")
        if asset.price <= 0:
            raise ValueError(f"{asset.ticker}: price must be positive.")
        if asset.volatility < 0:
            raise ValueError(f"{asset.ticker}: volatility cannot be negative.")
        if not 0 <= asset.max_weight <= 1:
            raise ValueError(f"{asset.ticker}: max weight must be between 0 and 1.")


# This runs math to split the capital across stocks, cash, and T-bills to best fit the goals.
def optimize_portfolio(
    assets: list[AssetInput],
    correlation_matrix: np.ndarray,
    config: PortfolioConfig,
) -> dict:
    validate_inputs(assets, correlation_matrix, config)

    expected_returns = np.array([asset.expected_return for asset in assets], dtype=float)
    volatilities = np.array([asset.volatility for asset in assets], dtype=float)
    covariance = build_covariance_matrix(volatilities, correlation_matrix, config.shrinkage)

    n = len(assets)
    cash_index = n
    tbill_index = n + 1
    mu = np.array(list(expected_returns) + [config.cash_yield, config.treasury_bill_yield], dtype=float)
    cov_augmented = np.zeros((n + 2, n + 2), dtype=float)
    cov_augmented[:n, :n] = covariance
    effective_asset_caps = np.array(
        compute_auto_max_weights(assets, correlation_matrix, config)
        if config.auto_max_allocation
        else [asset.max_weight for asset in assets],
        dtype=float,
    )
    max_weights = np.array(
        list(effective_asset_caps) + [1.0, 1.0],
        dtype=float,
    )
    if np.sum(max_weights[:n]) + max_weights[cash_index] + max_weights[tbill_index] < 0.999999:
        raise ValueError("Asset max weights are too restrictive to allocate the portfolio.")

    def objective(weights: np.ndarray) -> float:
        gross_return = float(mu @ weights)
        variance = float(weights @ cov_augmented @ weights)
        concentration = float(np.sum(np.square(weights[:n])))
        utility = gross_return - 0.5 * config.risk_aversion * variance - config.concentration_penalty * concentration
        return -utility

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w[cash_index] + w[tbill_index] - config.min_cash_weight},
    ]

    if config.max_cash_weight is not None:
        constraints.append({"type": "ineq", "fun": lambda w: config.max_cash_weight - (w[cash_index] + w[tbill_index])})
    if config.target_expected_return is not None:
        constraints.append({"type": "ineq", "fun": lambda w: float(mu @ w) - config.target_expected_return})

    if config.target_volatility is not None and config.target_volatility > 0:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: config.target_volatility**2 - float(w @ cov_augmented @ w),
            }
        )

    bounds = [(0.0, max_weights[i]) for i in range(n + 2)]
    initial = np.array(
        [min(asset.max_weight, 1.0 / max(n + 2, 1)) for asset in assets] + [config.min_cash_weight, 0.10],
        dtype=float,
    )
    initial_sum = initial.sum()
    if initial_sum <= 0:
        initial = np.full(n + 2, 1.0 / (n + 2))
    else:
        initial = initial / initial_sum

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    continuous_weights = np.clip(result.x, 0.0, 1.0)
    continuous_weights = continuous_weights / continuous_weights.sum()
    discrete = discrete_share_allocator(
        assets,
        stock_target_weights=continuous_weights[:n],
        capital=config.capital,
        min_cash_weight=config.min_cash_weight,
        target_cash_weight=float(continuous_weights[cash_index]),
        target_tbill_weight=float(continuous_weights[tbill_index]),
    )

    invested_weights = np.array(discrete["invested_dollars_by_asset"], dtype=float) / config.capital
    total_weights = np.array(
        list(invested_weights)
        + [discrete["cash_dollars"] / config.capital, discrete["treasury_bill_dollars"] / config.capital],
        dtype=float,
    )
    variance = float(total_weights @ cov_augmented @ total_weights)

    asset_rows = []
    for idx, asset in enumerate(assets):
        asset_rows.append(
            {
                "ticker": asset.ticker,
                "price": asset.price,
                "expected_return": asset.expected_return,
                "volatility": asset.volatility,
                "max_weight": float(effective_asset_caps[idx]),
                "continuous_weight": float(continuous_weights[idx]),
                "recommended_shares": int(discrete["shares"][idx]),
                "invested_dollars": float(discrete["invested_dollars_by_asset"][idx]),
                "realized_weight": float(total_weights[idx]),
            }
        )

    return {
        "asset_rows": asset_rows,
        "cash_weight": float(total_weights[cash_index]),
        "cash_dollars": float(discrete["cash_dollars"]),
        "treasury_bill_weight": float(total_weights[tbill_index]),
        "treasury_bill_dollars": float(discrete["treasury_bill_dollars"]),
        "defensive_weight": float(total_weights[cash_index] + total_weights[tbill_index]),
        "defensive_dollars": float(discrete["cash_dollars"] + discrete["treasury_bill_dollars"]),
        "expected_return": float(mu @ total_weights),
        "expected_volatility": float(np.sqrt(max(variance, 0.0))),
        "utility_score": float(-objective(total_weights)),
        "covariance_matrix": covariance,
        "correlation_matrix": correlation_matrix,
        "portfolio_weights": total_weights,
        "portfolio_mu_vector": mu,
        "portfolio_covariance": cov_augmented,
        "auto_max_allocation": bool(config.auto_max_allocation),
    }


# This assigns a simple label to the portfolio risk so we know how nervous the strategy is.
def classify_portfolio_risk(expected_volatility: float, probability_of_loss: float) -> dict:
    vol_component = min((expected_volatility * 100.0) / 30.0, 1.0)
    loss_component = min(probability_of_loss / 0.50, 1.0)
    risk_score = int(round((0.65 * vol_component + 0.35 * loss_component) * 100))
    if risk_score < 30:
        risk_label = "Low"
    elif risk_score < 55:
        risk_label = "Moderate"
    elif risk_score < 75:
        risk_label = "Elevated"
    else:
        risk_label = "High"
    return {"risk_score": risk_score, "risk_label": risk_label}


# This figures out how big each stock can be when the model is choosing max weights automatically.
def compute_auto_max_weights(
    assets: list[AssetInput],
    correlation_matrix: np.ndarray,
    config: PortfolioConfig,
) -> list[float]:
    vol = np.array([max(asset.volatility, 1e-6) for asset in assets], dtype=float)
    corr = np.asarray(correlation_matrix, dtype=float)
    avg_abs_corr = np.mean(np.abs(corr - np.eye(len(assets))), axis=1)
    inv_vol_score = 1.0 / vol
    diversification_score = 1.0 / (1.0 + avg_abs_corr)
    raw_score = inv_vol_score * diversification_score
    raw_score = raw_score / np.mean(raw_score)
    equal_weight = 1.0 / max(len(assets), 1)
    suggested = np.clip(equal_weight * raw_score, config.auto_max_floor, config.auto_max_ceiling)
    return suggested.tolist()


# This teaches the model to see hidden regimes in the returns so the Monte Carlo is more realistic.
def fit_gaussian_hmm_1d(
    observations: np.ndarray,
    n_states: int = 2,
    n_iter: int = 50,
    tol: float = 1e-4,
) -> dict:
    x = np.asarray(observations, dtype=float).reshape(-1)
    t_obs = len(x)
    if t_obs < max(30, n_states * 10):
        raise ValueError("Not enough observations to fit the HMM.")

    quantiles = np.quantile(x, np.linspace(0.0, 1.0, n_states + 2)[1:-1])
    means = np.array(quantiles[:n_states], dtype=float)
    if len(means) < n_states:
        means = np.linspace(np.min(x), np.max(x), n_states)
    variances = np.full(n_states, max(float(np.var(x)), 1e-6), dtype=float)
    trans = np.full((n_states, n_states), 1.0 / n_states, dtype=float)
    start = np.full(n_states, 1.0 / n_states, dtype=float)

    def emission_log_probs() -> np.ndarray:
        eps = 1e-8
        centered = x[:, None] - means[None, :]
        return -0.5 * (
            np.log(2.0 * np.pi * np.maximum(variances, eps))[None, :]
            + (centered**2) / np.maximum(variances, eps)[None, :]
        )

    prev_loglik = None
    for _ in range(n_iter):
        log_emit = emission_log_probs()
        log_start = np.log(np.clip(start, 1e-12, None))
        log_trans = np.log(np.clip(trans, 1e-12, None))

        alpha = np.zeros((t_obs, n_states), dtype=float)
        alpha[0] = log_start + log_emit[0]
        for t in range(1, t_obs):
            alpha[t] = log_emit[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)

        beta = np.zeros((t_obs, n_states), dtype=float)
        for t in range(t_obs - 2, -1, -1):
            beta[t] = logsumexp(log_trans + log_emit[t + 1][None, :] + beta[t + 1][None, :], axis=1)

        loglik = float(logsumexp(alpha[-1]))
        gamma = np.exp(alpha + beta - loglik)

        xi = np.zeros((t_obs - 1, n_states, n_states), dtype=float)
        for t in range(t_obs - 1):
            xi[t] = np.exp(
                alpha[t][:, None] + log_trans + log_emit[t + 1][None, :] + beta[t + 1][None, :] - loglik
            )

        start = gamma[0] / np.sum(gamma[0])
        trans = np.sum(xi, axis=0)
        trans = trans / np.clip(np.sum(trans, axis=1, keepdims=True), 1e-12, None)
        weights = np.clip(np.sum(gamma, axis=0), 1e-12, None)
        means = np.sum(gamma * x[:, None], axis=0) / weights
        centered = x[:, None] - means[None, :]
        variances = np.sum(gamma * centered**2, axis=0) / weights
        variances = np.maximum(variances, 1e-8)

        if prev_loglik is not None and abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik

    order = np.argsort(variances)
    start = start[order]
    trans = trans[order][:, order]
    means = means[order]
    variances = variances[order]
    gamma = gamma[:, order]

    return {
        "start_probabilities": start,
        "transition_matrix": trans,
        "means": means,
        "variances": variances,
        "posterior_probabilities": gamma,
        "most_likely_states": np.argmax(gamma, axis=1),
        "last_state_probabilities": gamma[-1],
    }


# This turns the HMM output into mean and variance for each hidden state to guide Monte Carlo.
def estimate_regime_statistics(
    log_returns: np.ndarray,
    n_states: int = 2,
) -> dict:
    returns = np.asarray(log_returns, dtype=float)
    market_factor = np.mean(returns, axis=1)
    hmm = fit_gaussian_hmm_1d(market_factor, n_states=n_states)
    states = hmm["most_likely_states"]

    regime_means = []
    regime_covariances = []
    for state in range(n_states):
        subset = returns[states == state]
        if len(subset) < max(10, returns.shape[1] + 1):
            subset = returns
        regime_mean = np.mean(subset, axis=0)
        regime_cov = np.cov(subset, rowvar=False)
        if regime_cov.ndim == 0:
            regime_cov = np.array([[float(regime_cov)]], dtype=float)
        regime_cov += np.eye(regime_cov.shape[0]) * 1e-8
        regime_means.append(regime_mean)
        regime_covariances.append(regime_cov)

    return {
        "transition_matrix": hmm["transition_matrix"],
        "start_probabilities": hmm["start_probabilities"],
        "last_state_probabilities": hmm["last_state_probabilities"],
        "regime_means": np.asarray(regime_means, dtype=float),
        "regime_covariances": np.asarray(regime_covariances, dtype=float),
        "regime_market_means": np.asarray(hmm["means"], dtype=float),
        "regime_market_vols": np.sqrt(np.asarray(hmm["variances"], dtype=float)),
        "state_assignments": states,
    }


# This grabs adjusted close prices so we can safely build log returns later on.
def extract_adjusted_close_prices(raw: pd.DataFrame | pd.Series, tickers: list[str]) -> pd.DataFrame:
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Unable to download price history. In this environment, live market data appears unavailable.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        prices = raw.rename(columns={"Close": tickers[0] if len(tickers) == 1 else "Close"})
        if "Close" in prices.columns:
            prices = prices[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices


# This makes sure the price table is clean and has columns for each ticker before more math.
def prepare_prices_dataframe(prices: pd.DataFrame | pd.Series, tickers: list[str]) -> pd.DataFrame:
    if prices is None or (hasattr(prices, "empty") and prices.empty):
        raise ValueError("Unable to download price history. In this environment, live market data appears unavailable.")

    prepared = prices.copy()
    if isinstance(prepared, pd.Series):
        prepared = prepared.to_frame(name=tickers[0])
    prepared = prepared.dropna(how="all").ffill().dropna()
    if prepared.empty or len(prepared) < 60:
        raise ValueError("Not enough historical data to estimate the portfolio.")

    missing = [ticker for ticker in tickers if ticker not in prepared.columns]
    if missing:
        raise ValueError(f"Missing downloaded price history for: {', '.join(missing)}")
    return prepared[tickers].copy()


# This reads price history and calculates expected returns and volatility per ticker.
def estimate_asset_statistics_from_prices(
    assets: list[AssetInput],
    prices: pd.DataFrame | pd.Series,
    hmm_states: int = 2,
    expected_return_method: str = "historical_mean",
    expected_return_shrinkage: float = 0.50,
) -> tuple[list[AssetInput], np.ndarray, str, np.ndarray, dict, dict, dict]:
    tickers = [asset.ticker for asset in assets]
    prices_df = prepare_prices_dataframe(prices, tickers)
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Price history returned no usable returns.")

    returns_np = log_returns.to_numpy(dtype=float)
    denoised_corr, denoise_info = denoise_correlation_matrix_mp(returns_np)
    annual_cov = log_returns.cov().to_numpy(dtype=float) * 252.0
    annual_cov = build_covariance_matrix(np.sqrt(np.clip(np.diag(annual_cov), 0.0, None)), denoised_corr, 0.0)
    annual_mean, return_model_info = estimate_expected_returns(
        log_returns,
        annual_covariance=annual_cov,
        method=expected_return_method,
        shrinkage=expected_return_shrinkage,
    )
    annual_vol = np.sqrt(np.clip(np.diag(annual_cov), 0.0, None))
    last_prices = prices_df.iloc[-1].to_dict()
    regime_info = estimate_regime_statistics(returns_np, n_states=max(2, int(hmm_states)))

    estimated_assets: list[AssetInput] = []
    for idx, asset in enumerate(assets):
        estimated_assets.append(
            AssetInput(
                ticker=asset.ticker,
                price=float(last_prices[asset.ticker]),
                expected_return=float(annual_mean[idx]),
                volatility=float(annual_vol[idx]),
                max_weight=asset.max_weight,
            )
        )

    sample_window = f"{prices_df.index.min().date()} to {prices_df.index.max().date()}"
    return estimated_assets, denoised_corr, sample_window, returns_np, denoise_info, regime_info, return_model_info


# This digs into tickers, fetches prices, and returns statistics plus the clean correlation matrix.
def estimate_asset_statistics(
    assets: list[AssetInput],
    lookback_years: float,
    hmm_states: int = 2,
    expected_return_method: str = "historical_mean",
    expected_return_shrinkage: float = 0.50,
) -> tuple[list[AssetInput], np.ndarray, str, np.ndarray, dict, dict, dict]:
    tickers = [asset.ticker for asset in assets]
    if lookback_years <= 0:
        raise ValueError("Lookback years must be positive.")

    period_days = max(int(round(365 * lookback_years)), 120)
    raw = yf.download(
        tickers,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    prices = extract_adjusted_close_prices(raw, tickers)
    return estimate_asset_statistics_from_prices(
        assets,
        prices,
        hmm_states=max(2, int(hmm_states)),
        expected_return_method=expected_return_method,
        expected_return_shrinkage=expected_return_shrinkage,
    )


# This wraps estimate_asset_statistics and then runs the optimizer so we can input tickers only.
def optimize_portfolio_from_tickers(
    assets: list[AssetInput],
    config: PortfolioConfig,
    lookback_years: float = 3.0,
) -> dict:
    effective_hmm_states = max(2, int(config.hmm_states))
    estimated_assets, corr, sample_window, log_returns, denoise_info, regime_info, return_model_info = estimate_asset_statistics(
        assets,
        lookback_years,
        hmm_states=effective_hmm_states,
        expected_return_method=config.expected_return_method,
        expected_return_shrinkage=config.expected_return_shrinkage,
    )
    treasury_bill_yield, treasury_bill_source = resolve_treasury_bill_yield(config)
    effective_config = replace(
        config,
        treasury_bill_yield=treasury_bill_yield,
        auto_treasury_bill_yield=False,
        hmm_states=effective_hmm_states,
    )
    result = optimize_portfolio(estimated_assets, corr, effective_config)
    simulation = run_hmm_monte_carlo_projection(
        capital=effective_config.capital,
        stock_weights=np.asarray(result["portfolio_weights"][: len(estimated_assets)], dtype=float),
        cash_weight=float(result["cash_weight"]),
        treasury_bill_weight=float(result["treasury_bill_weight"]),
        cash_yield=effective_config.cash_yield,
        treasury_bill_yield=treasury_bill_yield,
        regime_info=regime_info,
        years=effective_config.simulation_horizon_years,
        paths=effective_config.simulation_paths,
        seed=42,
    )
    result["sample_window"] = sample_window
    result["monte_carlo"] = simulation
    result["treasury_bill_yield"] = treasury_bill_yield
    result["treasury_bill_source"] = treasury_bill_source
    result["denoise_info"] = denoise_info
    result["return_model_info"] = return_model_info
    result["regime_info"] = {
        "states": int(effective_config.hmm_states),
        "transition_matrix": regime_info["transition_matrix"],
        "regime_market_means": regime_info["regime_market_means"],
        "regime_market_vols": regime_info["regime_market_vols"],
    }
    result.update(classify_portfolio_risk(result["expected_volatility"], simulation["probability_of_loss"]))
    return result


# This decides whether to use a user-yield or auto-fetch the 1-year Treasury bill rate.
def resolve_treasury_bill_yield(config: PortfolioConfig) -> tuple[float, str]:
    if config.auto_treasury_bill_yield or config.treasury_bill_yield is None:
        try:
            yield_value = fetch_official_1y_tbill_yield()
            if yield_value is not None and yield_value > 0:
                return yield_value, "Auto (U.S. Treasury 52-week bill coupon equivalent)"
        except Exception:
            pass

    if config.auto_treasury_bill_yield:
        try:
            raw = yf.download("^IRX", period="7d", interval="1d", auto_adjust=False, progress=False, threads=False)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    series = raw["Close"].iloc[:, 0]
                else:
                    series = raw["Close"]
                series = series.dropna()
                if not series.empty:
                    latest_percent = float(series.iloc[-1])
                    if latest_percent > 0:
                        return latest_percent / 100.0, "Auto (^IRX 13-week T-bill proxy)"
        except Exception:
            pass

    if config.treasury_bill_yield is not None:
        return float(config.treasury_bill_yield), "Manual fallback"
    raise ValueError("Unable to fetch a 1-year Treasury bill yield and no manual fallback yield was provided.")


# This tries to download the current 1-year Treasury bill yield when the internet works.
def fetch_official_1y_tbill_yield() -> float | None:
    current_year = datetime.now(UTC).year
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/all/{current_year}?type=daily_treasury_bill_rates"
    )
    with urlopen(url, timeout=10) as response:
        raw = response.read().decode("utf-8", errors="ignore").splitlines()
    reader = csv.DictReader(raw)
    rows = [row for row in reader if row.get("52 WEEKS COUPON EQUIVALENT")]
    if not rows:
        raise ValueError("Treasury bill dataset did not include a 52-week coupon equivalent series.")
    latest = rows[-1]["52 WEEKS COUPON EQUIVALENT"].strip()
    if not latest or latest == "N/A":
        raise ValueError("Latest 52-week Treasury bill yield is unavailable.")
    return float(latest) / 100.0


# This builds many possible future paths using the HMM-regime info so we can see ranges.
def run_hmm_monte_carlo_projection(
    capital: float,
    stock_weights: np.ndarray,
    cash_weight: float,
    treasury_bill_weight: float,
    cash_yield: float,
    treasury_bill_yield: float,
    regime_info: dict,
    years: float = 1.0,
    paths: int = 10000,
    seed: int | None = None,
) -> dict:
    if capital <= 0:
        raise ValueError("Capital must be positive for Monte Carlo simulation.")
    if paths <= 0:
        raise ValueError("Monte Carlo path count must be positive.")
    if years <= 0:
        raise ValueError("Monte Carlo horizon must be positive.")

    rng = np.random.default_rng(seed)
    stock_weights = np.asarray(stock_weights, dtype=float)
    transition = np.asarray(regime_info["transition_matrix"], dtype=float)
    last_state_probabilities = np.asarray(regime_info["last_state_probabilities"], dtype=float)
    regime_means = np.asarray(regime_info["regime_means"], dtype=float)
    regime_covariances = np.asarray(regime_info["regime_covariances"], dtype=float)
    n_states = transition.shape[0]
    steps = max(1, int(round(252 * years)))
    daily_cash_return = np.exp(cash_yield / 252.0) - 1.0
    daily_tbill_return = np.exp(treasury_bill_yield / 252.0) - 1.0

    terminal_values = np.full(paths, capital, dtype=float)
    current_states = np.array(
        [rng.choice(n_states, p=last_state_probabilities / np.sum(last_state_probabilities)) for _ in range(paths)],
        dtype=int,
    )

    for _ in range(steps):
        next_states = np.empty(paths, dtype=int)
        for state in range(n_states):
            mask = current_states == state
            count = int(np.sum(mask))
            if count == 0:
                continue
            next_states[mask] = rng.choice(n_states, size=count, p=transition[state])
        current_states = next_states

        portfolio_simple_returns = np.empty(paths, dtype=float)
        for state in range(n_states):
            mask = current_states == state
            count = int(np.sum(mask))
            if count == 0:
                continue
            draws = rng.multivariate_normal(regime_means[state], regime_covariances[state], size=count)
            stock_simple_returns = np.exp(draws) - 1.0
            stock_portfolio_returns = stock_simple_returns @ stock_weights
            portfolio_simple_returns[mask] = (
                stock_portfolio_returns
                + cash_weight * daily_cash_return
                + treasury_bill_weight * daily_tbill_return
            )
        terminal_values *= 1.0 + portfolio_simple_returns

    pnl = terminal_values - capital

    return {
        "method": "hmm_regime_monte_carlo",
        "paths": int(paths),
        "horizon_years": float(years),
        "expected_terminal_value": float(np.mean(terminal_values)),
        "median_terminal_value": float(np.median(terminal_values)),
        "value_at_5pct": float(np.percentile(terminal_values, 5)),
        "value_at_25pct": float(np.percentile(terminal_values, 25)),
        "value_at_75pct": float(np.percentile(terminal_values, 75)),
        "value_at_95pct": float(np.percentile(terminal_values, 95)),
        "probability_of_loss": float(np.mean(terminal_values < capital)),
        "expected_pnl": float(np.mean(pnl)),
        "states": int(n_states),
    }


# This rounds the weight plan into actual share counts so we know what to buy.
def discrete_share_allocator(
    assets: list[AssetInput],
    stock_target_weights: np.ndarray,
    capital: float,
    min_cash_weight: float,
    target_cash_weight: float,
    target_tbill_weight: float,
) -> dict:
    prices = np.array([asset.price for asset in assets], dtype=float)
    target_dollars = capital * np.asarray(stock_target_weights, dtype=float)
    shares = np.floor(target_dollars / prices).astype(int)
    invested = shares * prices
    min_cash_dollars = capital * min_cash_weight
    reserved_cash = capital * max(target_cash_weight, 0.0)
    treasury_bill_dollars = capital * max(target_tbill_weight, 0.0)
    cash = capital - float(np.sum(invested)) - treasury_bill_dollars

    if cash + treasury_bill_dollars < min_cash_dollars - 1e-8:
        shortfall = min_cash_dollars - (cash + treasury_bill_dollars)
        cash += shortfall
        treasury_bill_dollars = max(0.0, treasury_bill_dollars - shortfall)

    if cash < 0:
        shortfall = -cash
        treasury_bill_dollars = max(0.0, treasury_bill_dollars - shortfall)
        cash = capital - float(np.sum(invested)) - treasury_bill_dollars

    if cash + treasury_bill_dollars < min_cash_dollars - 1e-8:
        raise ValueError("Discrete allocation cannot satisfy the minimum defensive sleeve requirement.")

    safety_counter = 0
    while safety_counter < 10000:
        gaps = target_dollars - invested
        best_idx = None
        best_score = -np.inf
        for idx, asset in enumerate(assets):
            if cash - asset.price < reserved_cash - 1e-8:
                continue
            score = gaps[idx] / max(asset.price, 1e-9)
            if score > best_score and score > 0:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        shares[best_idx] += 1
        invested[best_idx] += prices[best_idx]
        cash -= prices[best_idx]
        safety_counter += 1

    if not isfinite(cash):
        raise ValueError("Cash balance became invalid during discrete allocation.")

    return {
        "shares": shares.tolist(),
        "invested_dollars_by_asset": invested.tolist(),
        "cash_dollars": float(cash),
        "treasury_bill_dollars": float(treasury_bill_dollars),
    }
