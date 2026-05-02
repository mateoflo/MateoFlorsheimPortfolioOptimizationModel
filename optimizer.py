"""Core portfolio optimization engine: statistics estimation, optimization, simulation, and risk classification."""

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


# Data classes

@dataclass
class AssetInput:
    """Holds price, return, volatility, and weight cap for a single investable asset."""

    ticker: str
    price: float
    expected_return: float
    volatility: float
    max_weight: float


@dataclass
class PortfolioConfig:
    """Holds all user-facing settings that control optimizer behavior."""

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


# Covariance and correlation helpers

def build_covariance_matrix(
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Construct an annualized covariance matrix from per-asset volatilities, a correlation matrix, and a shrinkage factor."""
    vols = np.asarray(volatilities, dtype=float)
    corr = np.asarray(correlation_matrix, dtype=float)

    # Cov(i,j) = sigma_i * sigma_j * rho(i,j)
    base_cov = np.outer(vols, vols) * corr

    # Diagonal-only target treats every pair of assets as uncorrelated
    diagonal = np.diag(np.diag(base_cov))

    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    # Blend full sample covariance with diagonal based on shrinkage intensity
    return (1.0 - shrinkage) * base_cov + shrinkage * diagonal


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into a correlation matrix with 1s on the diagonal."""
    covariance = np.asarray(covariance, dtype=float)

    # Clip to a tiny floor to avoid division by zero
    std = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))

    # corr[i,j] = cov[i,j] / (std[i] * std[j])
    corr = covariance / np.outer(std, std)

    # Floating-point arithmetic can push values slightly outside [-1, 1]
    corr = np.clip(corr, -1.0, 1.0)

    np.fill_diagonal(corr, 1.0)
    return corr


# Expected return estimation

def estimate_expected_returns(
    log_returns: pd.DataFrame,
    annual_covariance: np.ndarray,
    method: str = "historical_mean",
    shrinkage: float = 0.50,
) -> tuple[np.ndarray, dict]:
    """Estimate annualized expected returns for each asset using one of four statistical methods."""
    returns = log_returns.to_numpy(dtype=float)

    # Annualize by multiplying daily mean by 252 trading days per year
    sample_mean = log_returns.mean().to_numpy(dtype=float) * 252.0
    n_assets = returns.shape[1]

    # Normalize method string so alternate spellings all map to the same branch
    method_key = (method or "historical_mean").strip().lower()
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    # Method 1: plain annualized sample mean
    if method_key == "historical_mean":
        return sample_mean, {
            "method": "historical_mean",
            "shrinkage": 0.0,
            "description": "Annualized sample mean of daily log returns.",
        }

    # Method 2: Bayes-Stein — shrink each asset's mean toward the cross-sectional grand mean
    if method_key == "bayes_stein":
        target = float(np.mean(sample_mean))  # grand mean across all assets

        variances = np.clip(np.diag(annual_covariance), 1e-8, None)
        t_obs = max(len(log_returns), 1)

        # Per-asset shrinkage is larger when the historical mean deviates far from the target
        asset_shrink = variances / (variances + np.square(sample_mean - target) * t_obs + 1e-8)

        posterior = (1.0 - asset_shrink) * sample_mean + asset_shrink * target
        return posterior, {
            "method": "bayes_stein",
            "grand_mean_target": target,
            "average_shrinkage": float(np.mean(asset_shrink)),
            "description": "Bayes-Stein shrinkage toward the cross-sectional grand mean.",
        }

    # Shared: build an equal-weight market factor and compute per-asset betas
    market_factor = log_returns.mean(axis=1).to_numpy(dtype=float)
    market_mean = float(np.mean(market_factor) * 252.0)
    market_var = float(np.var(market_factor, ddof=1) * 252.0) if len(market_factor) > 1 else 0.0

    # Last column of the cov matrix is each asset's covariance with the market factor
    asset_cov_with_market = np.cov(returns, market_factor, rowvar=False)[:n_assets, n_assets]

    # Beta = Cov(asset, market) / Var(market)
    betas = asset_cov_with_market / max(market_var, 1e-8)

    factor_implied = betas * market_mean  # CAPM-style implied return

    # Method 3: pure market factor / CAPM
    if method_key in {"market_factor", "factor", "capm"}:
        return factor_implied, {
            "method": "market_factor",
            "market_mean": market_mean,
            "market_variance": market_var,
            "description": "Market-factor-implied expected returns using asset beta to the equal-weight market factor.",
        }

    # Method 4: Black-Litterman blend of factor-implied prior and historical views
    if method_key in {"black_litterman", "black_litterman_blend", "bl"}:
        # shrinkage=0 → full prior; shrinkage=1 → full historical sample
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


# Marchenko-Pastur denoising

def denoise_correlation_matrix_mp(log_returns: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove statistical noise from the sample correlation matrix using the Marchenko-Pastur distribution."""
    returns = np.asarray(log_returns, dtype=float)
    t_obs, n_assets = returns.shape

    # Fall back to plain sample correlation when there is not enough data
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

    # q = T/N; larger q means a tighter noise band
    q_ratio = t_obs / n_assets

    # Marchenko-Pastur upper edge: eigenvalues above this are genuine signal
    lambda_plus = (1.0 + 1.0 / np.sqrt(q_ratio)) ** 2

    # Symmetric matrix → use eigh for numerical stability
    eigvals, eigvecs = np.linalg.eigh(sample_corr)

    signal_mask = eigvals > lambda_plus
    signal_count = int(np.sum(signal_mask))
    noise_count = int(len(eigvals) - signal_count)

    # If everything is signal or everything is noise, leave the matrix unchanged
    if signal_count == 0 or noise_count == 0:
        denoised = sample_corr
    else:
        # Replace noise eigenvalues with their average to remove spurious directional structure
        avg_noise = float(np.mean(eigvals[~signal_mask]))
        adjusted_eigvals = eigvals.copy()
        adjusted_eigvals[~signal_mask] = avg_noise

        # Reconstruct: M = V * diag(adjusted_eigvals) * V^T
        denoised = eigvecs @ np.diag(adjusted_eigvals) @ eigvecs.T

        # Convert reconstructed matrix back to a proper correlation matrix
        denoised = correlation_from_covariance(denoised)

    return denoised, {
        "method": "marchenko_pastur",
        "q_ratio": float(q_ratio),
        "lambda_plus": float(lambda_plus),
        "signal_eigenvalues": signal_count,
        "noise_eigenvalues": noise_count,
    }


# Input validation

def validate_inputs(assets: Iterable[AssetInput], corr: np.ndarray, config: PortfolioConfig) -> None:
    """Validate that all inputs are self-consistent and within valid ranges before running the optimizer."""
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

    # Correlation matrix must be symmetric
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")

    # Diagonal must be 1 — a stock is perfectly correlated with itself
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


# Main optimizer

def optimize_portfolio(
    assets: list[AssetInput],
    correlation_matrix: np.ndarray,
    config: PortfolioConfig,
) -> dict:
    """Find the optimal allocation of capital across stocks, cash, and T-bills by maximizing a risk-adjusted utility function."""
    validate_inputs(assets, correlation_matrix, config)

    expected_returns = np.array([asset.expected_return for asset in assets], dtype=float)
    volatilities = np.array([asset.volatility for asset in assets], dtype=float)

    covariance = build_covariance_matrix(volatilities, correlation_matrix, config.shrinkage)

    n = len(assets)

    # Cash and T-bills are appended as two extra zero-volatility "assets"
    cash_index = n
    tbill_index = n + 1

    # Full expected return vector: stocks, cash, T-bills
    mu = np.array(list(expected_returns) + [config.cash_yield, config.treasury_bill_yield], dtype=float)

    # Augmented covariance: stock block top-left, zeros elsewhere for cash/T-bills
    cov_augmented = np.zeros((n + 2, n + 2), dtype=float)
    cov_augmented[:n, :n] = covariance

    # Per-asset weight caps — auto-calculated or user-supplied
    effective_asset_caps = np.array(
        compute_auto_max_weights(assets, correlation_matrix, config)
        if config.auto_max_allocation
        else [asset.max_weight for asset in assets],
        dtype=float,
    )

    # Full cap vector: per-asset caps for stocks, then uncapped cash and T-bills
    max_weights = np.array(
        list(effective_asset_caps) + [1.0, 1.0],
        dtype=float,
    )

    if np.sum(max_weights[:n]) + max_weights[cash_index] + max_weights[tbill_index] < 0.999999:
        raise ValueError("Asset max weights are too restrictive to allocate the portfolio.")

    def objective(weights: np.ndarray) -> float:
        """Return negative utility so scipy.minimize effectively maximizes it."""
        gross_return = float(mu @ weights)                          # Expected portfolio return
        variance = float(weights @ cov_augmented @ weights)         # Portfolio variance
        concentration = float(np.sum(np.square(weights[:n])))       # Herfindahl concentration index
        utility = (
            gross_return
            - 0.5 * config.risk_aversion * variance
            - config.concentration_penalty * concentration
        )
        return -utility

    # scipy constraint format: "eq" → must equal zero, "ineq" → must be >= 0
    constraints = [
        # Weights must sum to exactly 1 (fully invested)
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        # Combined defensive weight must meet the minimum floor
        {"type": "ineq", "fun": lambda w: w[cash_index] + w[tbill_index] - config.min_cash_weight},
    ]

    if config.max_cash_weight is not None:
        constraints.append({"type": "ineq", "fun": lambda w: config.max_cash_weight - (w[cash_index] + w[tbill_index])})

    if config.target_expected_return is not None:
        constraints.append({"type": "ineq", "fun": lambda w: float(mu @ w) - config.target_expected_return})

    # Compare variance (vol²) to avoid a square-root inside the constraint function
    if config.target_volatility is not None and config.target_volatility > 0:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: config.target_volatility**2 - float(w @ cov_augmented @ w),
            }
        )

    bounds = [(0.0, max_weights[i]) for i in range(n + 2)]

    # Initial guess: equal-weight subject to per-asset caps
    initial = np.array(
        [min(asset.max_weight, 1.0 / max(n + 2, 1)) for asset in assets] + [config.min_cash_weight, 0.10],
        dtype=float,
    )
    initial_sum = initial.sum()
    if initial_sum <= 0:
        initial = np.full(n + 2, 1.0 / (n + 2))  # degenerate fallback
    else:
        initial = initial / initial_sum  # normalize to sum to 1

    # SLSQP handles equality and inequality constraints well for portfolio problems
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

    # Clip tiny negative floating-point residuals and renormalize
    continuous_weights = np.clip(result.x, 0.0, 1.0)
    continuous_weights = continuous_weights / continuous_weights.sum()

    # Convert continuous weights to whole share counts
    discrete = discrete_share_allocator(
        assets,
        stock_target_weights=continuous_weights[:n],
        capital=config.capital,
        min_cash_weight=config.min_cash_weight,
        target_cash_weight=float(continuous_weights[cash_index]),
        target_tbill_weight=float(continuous_weights[tbill_index]),
    )

    # Realized weights reflect the discrete (rounded) allocation actually purchased
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
        "defensive_weight": float(total_weights[cash_index] + total_weights[tbill_index]),  # cash + T-bills combined
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


# Risk classification

def classify_portfolio_risk(expected_volatility: float, probability_of_loss: float) -> dict:
    """Assign a human-readable risk label and 0–100 risk score based on volatility and simulated loss probability."""
    # Normalize each component to [0, 1] against reference benchmarks
    vol_component = min((expected_volatility * 100.0) / 30.0, 1.0)   # 30% vol = max reference
    loss_component = min(probability_of_loss / 0.50, 1.0)             # 50% loss prob = max reference

    # Volatility carries more weight as a direct and stable risk measure
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


# Auto max-weight calculation

def compute_auto_max_weights(
    assets: list[AssetInput],
    correlation_matrix: np.ndarray,
    config: PortfolioConfig,
) -> list[float]:
    """Compute per-asset weight caps based on each asset's volatility and diversification value."""
    # Floor volatility to avoid division by zero
    vol = np.array([max(asset.volatility, 1e-6) for asset in assets], dtype=float)
    corr = np.asarray(correlation_matrix, dtype=float)

    # Average absolute off-diagonal correlation — how correlated is this stock with the others?
    avg_abs_corr = np.mean(np.abs(corr - np.eye(len(assets))), axis=1)

    inv_vol_score = 1.0 / vol                         # low vol → higher score → larger cap
    diversification_score = 1.0 / (1.0 + avg_abs_corr)  # low correlation → higher score

    # Combined score rewards assets that are both low-vol and add diversification
    raw_score = inv_vol_score * diversification_score

    # Normalize around 1 so suggested caps are anchored near the equal-weight baseline
    raw_score = raw_score / np.mean(raw_score)

    equal_weight = 1.0 / max(len(assets), 1)

    suggested = np.clip(equal_weight * raw_score, config.auto_max_floor, config.auto_max_ceiling)
    return suggested.tolist()


# Hidden Markov Model (HMM) for regime detection

def fit_gaussian_hmm_1d(
    observations: np.ndarray,
    n_states: int = 2,
    n_iter: int = 50,
    tol: float = 1e-4,
) -> dict:
    """Fit a Gaussian HMM to scalar observations using the Baum-Welch EM algorithm."""
    x = np.asarray(observations, dtype=float).reshape(-1)
    t_obs = len(x)

    if t_obs < max(30, n_states * 10):
        raise ValueError("Not enough observations to fit the HMM.")

    # Initialize means spread across quantiles so each state starts in a distinct region
    quantiles = np.quantile(x, np.linspace(0.0, 1.0, n_states + 2)[1:-1])
    means = np.array(quantiles[:n_states], dtype=float)
    if len(means) < n_states:
        means = np.linspace(np.min(x), np.max(x), n_states)  # fallback: linearly spaced

    variances = np.full(n_states, max(float(np.var(x)), 1e-6), dtype=float)  # shared initial variance
    trans = np.full((n_states, n_states), 1.0 / n_states, dtype=float)       # uniform transitions
    start = np.full(n_states, 1.0 / n_states, dtype=float)                   # uniform initial state

    def emission_log_probs() -> np.ndarray:
        """Compute log Gaussian emission probabilities for every observation and state."""
        eps = 1e-8
        centered = x[:, None] - means[None, :]  # shape (T, n_states)
        # Log Gaussian PDF: -0.5 * [log(2π σ²) + (x-μ)²/σ²]
        return -0.5 * (
            np.log(2.0 * np.pi * np.maximum(variances, eps))[None, :]
            + (centered**2) / np.maximum(variances, eps)[None, :]
        )

    prev_loglik = None

    for _ in range(n_iter):
        log_emit = emission_log_probs()
        log_start = np.log(np.clip(start, 1e-12, None))
        log_trans = np.log(np.clip(trans, 1e-12, None))

        # Forward pass: alpha[t, k] = log P(x_1..x_t, state_t=k)
        alpha = np.zeros((t_obs, n_states), dtype=float)
        alpha[0] = log_start + log_emit[0]
        for t in range(1, t_obs):
            # logsumexp gives log(sum(exp(...))) — numerically stable probability addition
            alpha[t] = log_emit[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)

        # Backward pass: beta[t, k] = log P(x_{t+1}..x_T | state_t=k)
        beta = np.zeros((t_obs, n_states), dtype=float)
        for t in range(t_obs - 2, -1, -1):
            beta[t] = logsumexp(log_trans + log_emit[t + 1][None, :] + beta[t + 1][None, :], axis=1)

        loglik = float(logsumexp(alpha[-1]))  # total log-likelihood of the observation sequence

        # Posterior state probabilities: gamma[t, k] = P(state_t=k | all observations)
        gamma = np.exp(alpha + beta - loglik)

        # Pairwise posteriors: xi[t, j, k] = P(state_t=j, state_{t+1}=k | all observations)
        xi = np.zeros((t_obs - 1, n_states, n_states), dtype=float)
        for t in range(t_obs - 1):
            xi[t] = np.exp(
                alpha[t][:, None] + log_trans + log_emit[t + 1][None, :] + beta[t + 1][None, :] - loglik
            )

        # M-step: update start probabilities from time-0 posterior
        start = gamma[0] / np.sum(gamma[0])

        # Update transition matrix by normalizing summed xi row sums
        trans = np.sum(xi, axis=0)
        trans = trans / np.clip(np.sum(trans, axis=1, keepdims=True), 1e-12, None)

        weights = np.clip(np.sum(gamma, axis=0), 1e-12, None)  # effective observations per state

        means = np.sum(gamma * x[:, None], axis=0) / weights  # weighted mean per state

        centered = x[:, None] - means[None, :]
        variances = np.sum(gamma * centered**2, axis=0) / weights
        variances = np.maximum(variances, 1e-8)  # floor to prevent zero variance

        if prev_loglik is not None and abs(loglik - prev_loglik) < tol:
            break  # converged
        prev_loglik = loglik

    # Sort states by ascending variance so state 0 is always the calm regime
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


def estimate_regime_statistics(
    log_returns: np.ndarray,
    n_states: int = 2,
) -> dict:
    """Fit an HMM and return per-regime mean vectors and covariance matrices for use in Monte Carlo simulation."""
    returns = np.asarray(log_returns, dtype=float)

    # Train HMM on the equal-weight market factor for stability
    market_factor = np.mean(returns, axis=1)
    hmm = fit_gaussian_hmm_1d(market_factor, n_states=n_states)
    states = hmm["most_likely_states"]

    regime_means = []
    regime_covariances = []
    for state in range(n_states):
        subset = returns[states == state]

        # Fall back to full history if too few days were assigned to this regime
        if len(subset) < max(10, returns.shape[1] + 1):
            subset = returns

        regime_mean = np.mean(subset, axis=0)
        regime_cov = np.cov(subset, rowvar=False)

        # np.cov returns a scalar for a single asset; wrap it in a 2-D array
        if regime_cov.ndim == 0:
            regime_cov = np.array([[float(regime_cov)]], dtype=float)

        # Small diagonal regularization ensures positive-definiteness for sampling
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


# Price download and preprocessing helpers

def extract_adjusted_close_prices(raw: pd.DataFrame | pd.Series, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close prices from the raw yfinance download, handling both single- and multi-ticker formats."""
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Unable to download price history. In this environment, live market data appears unavailable.")

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker downloads use a two-level column index: ("Close", "AAPL")
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        # Single-ticker download: flat columns like "Close", "Open", etc.
        prices = raw.rename(columns={"Close": tickers[0] if len(tickers) == 1 else "Close"})
        if "Close" in prices.columns:
            prices = prices[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")  # remove fully-missing rows

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices


def prepare_prices_dataframe(prices: pd.DataFrame | pd.Series, tickers: list[str]) -> pd.DataFrame:
    """Forward-fill gaps, drop remaining NaNs, and verify at least 60 rows remain for each requested ticker."""
    if prices is None or (hasattr(prices, "empty") and prices.empty):
        raise ValueError("Unable to download price history. In this environment, live market data appears unavailable.")

    prepared = prices.copy()
    if isinstance(prepared, pd.Series):
        prepared = prepared.to_frame(name=tickers[0])

    # Forward-fill isolated gaps (e.g., a holiday) then drop any remaining NaN rows
    prepared = prepared.dropna(how="all").ffill().dropna()

    if prepared.empty or len(prepared) < 60:
        raise ValueError("Not enough historical data to estimate the portfolio.")

    missing = [ticker for ticker in tickers if ticker not in prepared.columns]
    if missing:
        raise ValueError(f"Missing downloaded price history for: {', '.join(missing)}")

    return prepared[tickers].copy()


# Integrated statistics estimation

def estimate_asset_statistics_from_prices(
    assets: list[AssetInput],
    prices: pd.DataFrame | pd.Series,
    hmm_states: int = 2,
    expected_return_method: str = "historical_mean",
    expected_return_shrinkage: float = 0.50,
) -> tuple[list[AssetInput], np.ndarray, str, np.ndarray, dict, dict, dict]:
    """Compute all optimizer inputs from a prepared price DataFrame without fetching new data."""
    tickers = [asset.ticker for asset in assets]
    prices_df = prepare_prices_dataframe(prices, tickers)

    # Log returns are time-additive and scale-invariant across price levels
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Price history returned no usable returns.")

    returns_np = log_returns.to_numpy(dtype=float)

    denoised_corr, denoise_info = denoise_correlation_matrix_mp(returns_np)

    # Annualize daily covariance by multiplying by 252 trading days
    annual_cov = log_returns.cov().to_numpy(dtype=float) * 252.0

    # Rebuild covariance using denoised correlations; shrinkage is applied later in optimize_portfolio
    annual_cov = build_covariance_matrix(np.sqrt(np.clip(np.diag(annual_cov), 0.0, None)), denoised_corr, 0.0)

    annual_mean, return_model_info = estimate_expected_returns(
        log_returns,
        annual_covariance=annual_cov,
        method=expected_return_method,
        shrinkage=expected_return_shrinkage,
    )
    annual_vol = np.sqrt(np.clip(np.diag(annual_cov), 0.0, None))

    last_prices = prices_df.iloc[-1].to_dict()  # most recent closing price per ticker

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


def estimate_asset_statistics(
    assets: list[AssetInput],
    lookback_years: float,
    hmm_states: int = 2,
    expected_return_method: str = "historical_mean",
    expected_return_shrinkage: float = 0.50,
) -> tuple[list[AssetInput], np.ndarray, str, np.ndarray, dict, dict, dict]:
    """Download price history from Yahoo Finance and delegate to estimate_asset_statistics_from_prices."""
    tickers = [asset.ticker for asset in assets]
    if lookback_years <= 0:
        raise ValueError("Lookback years must be positive.")

    # Minimum 120 days to ensure enough data even for short lookback windows
    period_days = max(int(round(365 * lookback_years)), 120)
    raw = yf.download(
        tickers,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=True,   # adjusted prices account for splits and dividends
        progress=False,     # suppress download progress bar
        threads=False,      # single-threaded to avoid rate-limit issues
    )
    prices = extract_adjusted_close_prices(raw, tickers)
    return estimate_asset_statistics_from_prices(
        assets,
        prices,
        hmm_states=max(2, int(hmm_states)),
        expected_return_method=expected_return_method,
        expected_return_shrinkage=expected_return_shrinkage,
    )


# End-to-end convenience function

def optimize_portfolio_from_tickers(
    assets: list[AssetInput],
    config: PortfolioConfig,
    lookback_years: float = 3.0,
) -> dict:
    """Run the full pipeline — download, estimate, optimize, simulate, and classify risk — in one call."""
    effective_hmm_states = max(2, int(config.hmm_states))

    # Steps 1–2: download prices and estimate all statistics
    estimated_assets, corr, sample_window, log_returns, denoise_info, regime_info, return_model_info = estimate_asset_statistics(
        assets,
        lookback_years,
        hmm_states=effective_hmm_states,
        expected_return_method=config.expected_return_method,
        expected_return_shrinkage=config.expected_return_shrinkage,
    )

    # Step 3: resolve T-bill yield from live source or manual fallback
    treasury_bill_yield, treasury_bill_source = resolve_treasury_bill_yield(config)

    # Bake resolved T-bill yield into a new config so optimize_portfolio doesn't re-fetch it
    effective_config = replace(
        config,
        treasury_bill_yield=treasury_bill_yield,
        auto_treasury_bill_yield=False,
        hmm_states=effective_hmm_states,
    )

    # Step 4: run the optimizer
    result = optimize_portfolio(estimated_assets, corr, effective_config)

    # Step 5: run HMM-regime Monte Carlo projection
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
        seed=42,  # fixed seed so identical inputs produce identical simulations
    )

    # Step 6: attach all metadata to the result
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

    # Step 7: classify risk and merge the label into the result
    result.update(classify_portfolio_risk(result["expected_volatility"], simulation["probability_of_loss"]))
    return result


# Treasury bill yield resolution

def resolve_treasury_bill_yield(config: PortfolioConfig) -> tuple[float, str]:
    """Resolve the T-bill yield using a priority waterfall: Treasury website → Yahoo ^IRX → manual fallback."""
    # Priority 1: official U.S. Treasury data (most authoritative source)
    if config.auto_treasury_bill_yield or config.treasury_bill_yield is None:
        try:
            yield_value = fetch_official_1y_tbill_yield()
            if yield_value is not None and yield_value > 0:
                return yield_value, "Auto (U.S. Treasury 52-week bill coupon equivalent)"
        except Exception:
            pass  # fall through to next source on any network error

    # Priority 2: Yahoo Finance ^IRX as a 13-week T-bill proxy
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
                    latest_percent = float(series.iloc[-1])  # ^IRX is quoted in percent
                    if latest_percent > 0:
                        return latest_percent / 100.0, "Auto (^IRX 13-week T-bill proxy)"
        except Exception:
            pass  # fall through to manual value

    # Priority 3: user-supplied manual value
    if config.treasury_bill_yield is not None:
        return float(config.treasury_bill_yield), "Manual fallback"

    raise ValueError("Unable to fetch a 1-year Treasury bill yield and no manual fallback yield was provided.")


def fetch_official_1y_tbill_yield() -> float | None:
    """Fetch the most recent 52-week T-bill coupon-equivalent yield from the U.S. Treasury website."""
    current_year = datetime.now(UTC).year
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/all/{current_year}?type=daily_treasury_bill_rates"
    )

    # Timeout after 10 seconds to avoid hanging on slow network
    with urlopen(url, timeout=10) as response:
        raw = response.read().decode("utf-8", errors="ignore").splitlines()

    reader = csv.DictReader(raw)

    # Keep only rows with a populated 52-week coupon equivalent value
    rows = [row for row in reader if row.get("52 WEEKS COUPON EQUIVALENT")]
    if not rows:
        raise ValueError("Treasury bill dataset did not include a 52-week coupon equivalent series.")

    latest = rows[-1]["52 WEEKS COUPON EQUIVALENT"].strip()  # last row = most recent trading day
    if not latest or latest == "N/A":
        raise ValueError("Latest 52-week Treasury bill yield is unavailable.")

    return float(latest) / 100.0  # CSV value is in percent, convert to decimal


# Monte Carlo simulation

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
    """Simulate portfolio trajectories over the specified horizon using an HMM-driven regime-switching model."""
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

    steps = max(1, int(round(252 * years)))  # horizon in trading days

    # Continuously compounded daily returns for the deterministic defensive assets
    daily_cash_return = np.exp(cash_yield / 252.0) - 1.0
    daily_tbill_return = np.exp(treasury_bill_yield / 252.0) - 1.0

    terminal_values = np.full(paths, capital, dtype=float)  # all paths start at capital

    # Seed each path's starting regime from the last observed state distribution
    current_states = np.array(
        [rng.choice(n_states, p=last_state_probabilities / np.sum(last_state_probabilities)) for _ in range(paths)],
        dtype=int,
    )

    # Main simulation loop: one trading day per iteration
    for _ in range(steps):

        # Step 1: transition each path to a new Markov state
        next_states = np.empty(paths, dtype=int)
        for state in range(n_states):
            mask = current_states == state
            count = int(np.sum(mask))
            if count == 0:
                continue
            next_states[mask] = rng.choice(n_states, size=count, p=transition[state])
        current_states = next_states

        # Step 2: draw correlated daily stock returns from each regime's distribution
        portfolio_simple_returns = np.empty(paths, dtype=float)
        for state in range(n_states):
            mask = current_states == state
            count = int(np.sum(mask))
            if count == 0:
                continue

            draws = rng.multivariate_normal(regime_means[state], regime_covariances[state], size=count)  # shape (count, n_assets)

            # Convert log returns to simple returns for portfolio arithmetic
            stock_simple_returns = np.exp(draws) - 1.0

            stock_portfolio_returns = stock_simple_returns @ stock_weights  # weighted sum

            # Add deterministic cash and T-bill contributions
            portfolio_simple_returns[mask] = (
                stock_portfolio_returns
                + cash_weight * daily_cash_return
                + treasury_bill_weight * daily_tbill_return
            )

        # Step 3: compound the portfolio value by one day's return
        terminal_values *= 1.0 + portfolio_simple_returns

    pnl = terminal_values - capital  # positive = profit, negative = loss

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
        "probability_of_loss": float(np.mean(terminal_values < capital)),  # fraction of paths below starting capital
        "expected_pnl": float(np.mean(pnl)),
        "states": int(n_states),
    }


# Discrete share allocation

def discrete_share_allocator(
    assets: list[AssetInput],
    stock_target_weights: np.ndarray,
    capital: float,
    min_cash_weight: float,
    target_cash_weight: float,
    target_tbill_weight: float,
) -> dict:
    """Convert continuous optimizer weights into whole share counts, then greedily fill remaining cash toward targets."""
    prices = np.array([asset.price for asset in assets], dtype=float)

    target_dollars = capital * np.asarray(stock_target_weights, dtype=float)

    # Floor division: always buy fewer shares to avoid over-investing
    shares = np.floor(target_dollars / prices).astype(int)
    invested = shares * prices  # actual dollars spent per stock

    min_cash_dollars = capital * min_cash_weight
    reserved_cash = capital * max(target_cash_weight, 0.0)
    treasury_bill_dollars = capital * max(target_tbill_weight, 0.0)

    cash = capital - float(np.sum(invested)) - treasury_bill_dollars  # remaining cash after stocks and T-bills

    # If cash is short, claw back from T-bill allocation to meet the minimum floor
    if cash + treasury_bill_dollars < min_cash_dollars - 1e-8:
        shortfall = min_cash_dollars - (cash + treasury_bill_dollars)
        cash += shortfall
        treasury_bill_dollars = max(0.0, treasury_bill_dollars - shortfall)

    # If cash is still negative, reduce T-bills further
    if cash < 0:
        shortfall = -cash
        treasury_bill_dollars = max(0.0, treasury_bill_dollars - shortfall)
        cash = capital - float(np.sum(invested)) - treasury_bill_dollars

    if cash + treasury_bill_dollars < min_cash_dollars - 1e-8:
        raise ValueError("Discrete allocation cannot satisfy the minimum defensive sleeve requirement.")

    # Greedy loop: use leftover cash to buy one share of the stock furthest below its target
    safety_counter = 0
    while safety_counter < 10000:
        gaps = target_dollars - invested  # dollar gap to target per stock
        best_idx = None
        best_score = -np.inf

        for idx, asset in enumerate(assets):
            # Skip if buying one more share would drop cash below the reserved floor
            if cash - asset.price < reserved_cash - 1e-8:
                continue

            # Score by how many share-price units of gap remain; prefer the largest gap
            score = gaps[idx] / max(asset.price, 1e-9)
            if score > best_score and score > 0:
                best_idx = idx
                best_score = score

        if best_idx is None:
            break  # no affordable improvement exists

        # Buy one more share of the best candidate
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
