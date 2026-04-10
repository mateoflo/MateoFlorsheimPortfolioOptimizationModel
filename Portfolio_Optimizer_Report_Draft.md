# Portfolio Optimization Model And Monte Carlo Framework

## Research Foundation

This portfolio optimizer is built from ideas that come directly from established quantitative finance, random matrix theory, and regime-switching time-series research.

### Mean-Variance Portfolio Optimization

Harry Markowitz, *Portfolio Selection* (1952)

- Link: https://www.jstor.org/stable/2975974
- Why it matters:
  - provides the foundational expected return versus variance framework
  - motivates the portfolio variance term `w^T Sigma w`
  - gives the theoretical basis for solving for optimal portfolio weights under constraints

### Random Matrix Theory

V. A. Marchenko and L. A. Pastur, *Distribution of eigenvalues for some sets of random matrices* (1967)

- Link: https://www.mathnet.ru/eng/sm/v114/i4/p507
- Why it matters:
  - gives the eigenvalue bulk used to separate signal from noise
  - underpins denoising of empirical correlation matrices
  - helps prevent overfitting to unstable historical correlation noise

### Financial Correlation Matrix Filtering

Laloux, Cizeau, Bouchaud, Potters, *Noise Dressing of Financial Correlation Matrices* (1999)

- Link: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.83.1467
- Why it matters:
  - applies random matrix theory directly to financial correlation matrices
  - supports the practical use of eigenvalue filtering before portfolio optimization
  - motivates cleaning the correlation structure before solving for portfolio weights

### Regime-Switching / Hidden Markov Modeling

James D. Hamilton, *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle* (1989)

- Link: https://www.econometricsociety.org/publications/econometrica/1989/03/01/new-approach-economic-analysis-nonstationary-time-series-and
- Why it matters:
  - provides the conceptual basis for regime-switching time series
  - supports the idea that return and volatility dynamics are not stationary
  - underpins the HMM regime layer used in the Monte Carlo engine

## Mathematical Character Of The Model

This model uses the types of mathematics commonly associated with quantitative portfolio construction:

- `Linear algebra`
  - covariance matrices
  - correlation matrices
  - eigenvalues and eigenvectors
  - quadratic forms such as `w^T Sigma w`
- `Probability and stochastic modeling`
  - Monte Carlo simulation
  - Hidden Markov Models
  - multivariate random return generation
- `Multivariable optimization`
  - constrained optimization over the vector of portfolio weights
- `Calculus`
  - the objective is a differentiable multivariable function
  - the implementation solves the problem numerically rather than coding symbolic partial derivatives directly

So the model does use the kind of linear algebra, stochastic math, and multivariable calculus structure that you were thinking about. The current implementation simply solves the optimization numerically rather than by hand-derived closed-form first-order conditions.

## Overview

This application is a long-only quantitative portfolio optimizer for stock portfolios with two internal defensive sleeves:

- cash
- Treasury bills

The user enters a stock universe and practical allocation constraints. The system then:

1. Downloads historical stock prices.
2. Estimates expected return and volatility from historical returns.
3. Denoises the stock correlation matrix using Marchenko-Pastur random matrix filtering.
4. Builds a cleaned covariance structure.
5. Solves a constrained mean-variance optimization problem.
6. Converts the continuous solution into exact share counts.
7. Recommends how much capital to hold in cash.
8. Recommends how much capital to move into Treasury bills.
9. Fits a Hidden Markov Model to the market return environment.
10. Runs an HMM regime-switching Monte Carlo projection to estimate the portfolio’s one-year distribution of outcomes.

This means the output is not just an abstract weight vector. It is an implementation-ready portfolio with:

- exact stock share counts
- stock dollar allocations
- recommended cash allocation
- recommended Treasury-bill allocation
- estimated return
- estimated volatility
- portfolio risk score
- portfolio risk classification
- Monte Carlo terminal value estimates
- probability of loss over the selected horizon

## What The User Inputs

The GUI is intentionally narrow. The user does not manually enter stock return assumptions, volatility assumptions, or a correlation matrix.

All percentage-style fields are intended to be entered as whole-number percents. For example:

- `1` means `1%`
- `5` means `5%`
- `25` means `25%`

### Stock-level inputs

For each stock, the user enters:

- `Ticker`
- `Max Weight %`

### Portfolio-level inputs

The user also enters:

- `Capital ($)`
- `Lookback Years`
- `Max Allocation Mode`
- `Auto Max Floor %`
- `Auto Max Ceiling %`
- `Minimum Cash %`
- `Maximum Cash %`
- `Cash Yield %`
- `Fallback T-bill %`
- `Auto-fetch T-bill yield when online`
- `Target Return % (optional)`
- `Target Vol % (optional)`
- `Monte Carlo Paths`
- `Monte Carlo Horizon Years`

## What Each Input Does

### `Ticker`

Defines the stock universe. The optimizer downloads historical adjusted-close prices for these symbols and uses them to estimate return, volatility, covariance, and correlation.

### `Max Weight %`

Defines the maximum allowed portfolio weight for an individual stock when `Max Allocation Mode` is set to `Manual`.

### `Capital ($)`

Sets the total size of the portfolio. After optimization, the model converts continuous weights into exact discrete share counts using this capital amount.

### `Lookback Years`

Defines how much historical market data is used when estimating expected return and risk. A longer window generally stabilizes estimates. A shorter window reacts more quickly to recent market conditions.

### `Max Allocation Mode`

Lets the user choose how stock concentration caps are determined.

- `Manual`: the optimizer uses the user-entered `Max Weight %` for each stock
- `Auto`: the optimizer generates stock-specific max weights from estimated risk and diversification structure

### `Auto Max Floor %`

Defines the minimum allowable cap for a stock when `Max Allocation Mode` is set to `Auto`.

### `Auto Max Ceiling %`

Defines the maximum allowable cap for a stock when `Max Allocation Mode` is set to `Auto`.

### `Minimum Cash %`

Sets the minimum size of the defensive sleeve. In the current model, this floor applies to:

```text
cash + Treasury bills
```

not just idle cash.

### `Maximum Cash %`

Caps the total size of the defensive sleeve:

```text
cash + Treasury bills
```

This means the maximum cash setting is interpreted as a cap on combined cash-equivalent exposure, not just idle cash.

### `Cash Yield %`

Assigns an annualized return to idle cash. This allows cash to compete with risky assets inside the optimizer rather than being treated as a zero-return residual.

### `Fallback T-bill %`

Defines the Treasury-bill yield used if the application cannot fetch a live 1-year Treasury bill rate from the internet.

### `Auto-fetch T-bill yield when online`

When enabled, the application attempts to fetch the official 1-year Treasury bill rate first. If that fetch fails, it can fall back to the manual T-bill yield input.

### `Target Return % (optional)`

Adds a minimum expected-return constraint. If supplied, the optimized portfolio must meet or exceed the chosen expected annual return.

### `Target Vol % (optional)`

Adds an annualized volatility cap. If supplied, the optimized portfolio must remain below the chosen volatility threshold.

### `Monte Carlo Paths`

Sets how many random simulations are used in the Monte Carlo projection. More paths generally improve stability of the projected distribution.

### `Monte Carlo Horizon Years`

Defines the forward projection horizon for the Monte Carlo engine. For a one-year outlook, this is set to `1`.

## What The Model Outputs About Risk

The GUI no longer asks the user to input a risk-aversion number. Instead, the application reports how risky the final optimized portfolio is.

The model now outputs:

- expected annual volatility
- probability of loss from the Monte Carlo engine
- a portfolio risk score
- a portfolio risk label

The portfolio risk label is currently categorized as:

- `Low`
- `Moderate`
- `Elevated`
- `High`

The risk score is derived from a blend of:

- expected portfolio volatility
- simulated probability of loss

This design is more intuitive for advisors because it reports portfolio risk in output form rather than requiring them to choose an abstract optimization coefficient.

## How Estimated Return Is Calculated

The optimizer estimates stock return directly from historical adjusted-close prices.

Let:

- `P_t` = adjusted close price at time `t`
- `r_t` = daily log return

Daily log return is computed as:

```text
r_t = ln(P_t / P_{t-1})
```

For each stock `i`, the expected annual return is estimated as:

```text
mu_i = mean(r_i) * 252
```

where:

- `mu_i` is the annualized expected return for stock `i`
- `252` is the standard approximation for trading days per year

This means the expected return is the annualized average of historical daily log returns over the selected lookback window.

## How Risk Is Estimated

From the same historical daily log returns, the model estimates:

### Raw covariance matrix

```text
Sigma = cov(r) * 252
```

### Annualized volatility

For each stock `i`:

```text
sigma_i = sqrt(Sigma_ii)
```

### Correlation matrix

The correlation matrix is computed from the same return history. Correlation is what allows the optimizer to measure diversification effects across stocks instead of evaluating each position independently.

## Correlation Matrix Denoising With Marchenko-Pastur

The optimizer does not use the raw empirical correlation matrix directly. Instead, it denoises the matrix so the optimizer is not overfitting spurious correlation structure.

### Why denoising is needed

Empirical correlation matrices are noisy, especially when the number of assets is not small relative to the number of observations. Mean-variance optimization is highly sensitive to that noise, which can produce unstable or unrealistic allocations.

### Marchenko-Pastur framework

Let:

- `T` = number of return observations
- `N` = number of assets
- `Q = T / N`

Under the random matrix null, the upper edge of the noise eigenvalue bulk is:

```text
lambda_plus = (1 + 1 / sqrt(Q))^2
```

The optimizer:

1. Computes the eigenvalues and eigenvectors of the empirical stock correlation matrix.
2. Compares the empirical eigenvalues to the Marchenko-Pastur upper bound.
3. Treats eigenvalues above `lambda_plus` as signal.
4. Treats eigenvalues inside the Marchenko-Pastur bulk as noise.
5. Replaces the noisy bulk with an averaged noise eigenvalue.
6. Reconstructs a cleaner correlation matrix from the adjusted eigen-spectrum.

This preserves the dominant structure in the data while suppressing unstable noise.

### Cleaned covariance matrix

After denoising the correlation matrix, the model rebuilds the covariance matrix using the estimated stock volatilities:

```text
Sigma_clean = D * Corr_clean * D
```

where:

- `D` is the diagonal matrix of stock volatilities
- `Corr_clean` is the denoised correlation matrix

This cleaned covariance matrix is what the optimizer uses.

## How The Optimization Works

The optimizer uses a constrained mean-variance framework with cleaned covariance inputs and a concentration penalty.

### Core objective

The model maximizes:

```text
U(w) = mu^T w - 0.5 * lambda * w^T Sigma w - eta * ||w_stocks||^2
```

where:

- `w` = full portfolio weights
- `mu` = annualized expected return vector
- `Sigma` = annualized covariance matrix
- `lambda` = risk-aversion parameter
- `eta` = concentration penalty

Interpretation:

- `mu^T w` rewards expected return
- `0.5 * lambda * w^T Sigma w` penalizes portfolio variance
- `eta * ||w_stocks||^2` discourages excessive concentration in the stock sleeve

This is a multivariable optimization problem in the portfolio weight vector `w`. In theoretical form, the optimizer can be viewed through a Lagrangian framework with equality and inequality constraints. In implementation, the application solves the problem numerically using constrained nonlinear optimization rather than explicitly coding symbolic partial derivatives.

The current application keeps the risk-aversion coefficient internal rather than exposing it as a GUI input, which makes the interface more advisor-friendly while preserving the structure of the mean-variance objective.

### Portfolio structure

The optimizer allocates across:

- user-entered stocks
- cash
- Treasury bills

Cash and Treasury bills are modeled as internal sleeves. Stocks are estimated from market history. Cash uses the user-entered cash yield. Treasury bills use either the official 1-year Treasury bill rate, a fallback value, or a proxy depending on data availability.

### Manual versus automatic max allocation

The optimizer supports two concentration-cap modes.

#### Manual mode

In manual mode, each stock uses the user-entered `Max Weight %` directly.

#### Auto mode

In auto mode, the optimizer generates a stock-specific cap from:

- estimated stock volatility
- average correlation to the rest of the universe

The intuition is:

- lower-volatility stocks can support larger caps
- more diversifying stocks can support larger caps
- higher-volatility or highly entangled stocks should receive lower caps

The auto-cap score is based on:

```text
score_i = (1 / sigma_i) * (1 / (1 + avg_abs_corr_i))
```

where:

- `sigma_i` is the estimated stock volatility
- `avg_abs_corr_i` is the stock’s average absolute correlation to the rest of the stock universe

That score is normalized and applied to the equal-weight baseline:

```text
cap_i_raw = (1 / N) * normalized_score_i
```

Then the final cap is clipped to the user-chosen floor and ceiling:

```text
cap_i = clip(cap_i_raw, auto_floor, auto_ceiling)
```

This produces a practical hybrid:

- diversified low-volatility names can receive slightly higher caps
- volatile or more redundant names get tighter caps

### Constraints

The optimization is solved subject to:

```text
sum(w) = 1
```

```text
w_i >= 0
```

```text
w_i <= max_weight_i
```

```text
cash + Treasury bills >= minimum defensive sleeve
```

```text
w_cash + w_tbill <= max_cash
```

Optional return constraint:

```text
mu^T w >= target_return
```

Optional volatility constraint:

```text
w^T Sigma w <= sigma_target^2
```

### How The Final Allocation Is Implemented

The optimizer first solves for continuous target weights. Those weights are then translated into implementable share counts:

1. Convert each stock target weight into target dollars.
2. Floor each stock position to whole shares.
3. Reserve the required defensive sleeve.
4. Split that defensive sleeve across cash and Treasury bills.
5. Use remaining dollars to increment stock positions where the gap between target and realized dollars is largest.

This makes the final stock allocation executable in practice while preserving the defensive sleeves.

## What The Treasury-bill Sleeve Does

The Treasury-bill sleeve gives the optimizer a low-risk alternative to leaving all non-stock capital as idle cash.

Conceptually:

- cash is highly liquid operational reserve
- Treasury bills are short-duration capital parking with a positive yield

When the optimizer sees that the expected risk-adjusted payoff from additional stock exposure is not attractive enough, it can allocate part of the portfolio to Treasury bills instead.

## How The HMM Monte Carlo Layer Works

The Monte Carlo engine is not a single-regime Gaussian simulation. It is a regime-switching Monte Carlo process driven by a Hidden Markov Model.

### Step 1: Fit a Hidden Markov Model

The application computes a market factor from the average cross-sectional stock return series and fits a Gaussian Hidden Markov Model to that factor.

In practical terms, the HMM attempts to infer latent market regimes such as:

- lower-volatility / calmer market states
- higher-volatility / stressed market states

The HMM estimates:

- regime transition probabilities
- regime-specific means
- regime-specific variances
- posterior probabilities for the most recent regime

### Step 2: Estimate regime-specific stock return distributions

After regime assignment, the model computes state-conditional stock statistics:

- regime-specific stock mean vectors
- regime-specific stock covariance matrices

This means each regime has its own return environment rather than forcing the entire market into one stationary covariance structure.

### Step 3: Simulate regime transitions through time

For each Monte Carlo path, the simulation:

1. Starts from the latest inferred regime probabilities.
2. Draws the next regime using the Markov transition matrix.
3. Simulates stock returns from that regime’s multivariate distribution.
4. Repeats this process across daily steps over the selected horizon.

### Step 4: Combine stocks, cash, and Treasury bills

Within each daily step:

- stocks evolve according to the simulated regime-specific return draw
- cash accrues its configured yield
- Treasury bills accrue the selected T-bill yield

The full portfolio value is updated path-by-path over time.

### Monte Carlo outputs

The application reports:

- expected terminal portfolio value
- median terminal portfolio value
- 5th percentile terminal value
- 25th percentile terminal value
- 75th percentile terminal value
- 95th percentile terminal value
- expected profit/loss
- probability of ending below starting capital

### Why Monte Carlo Matters

The optimization engine gives a point estimate for return and volatility. HMM Monte Carlo turns that into a distribution of possible outcomes under changing market regimes. That makes the output more realistic and more intuitive for investors because it answers questions like:

- What is a reasonable bad-case outcome?
- What is the probability of losing money over the next year?
- What does the median one-year result look like?
- How wide is the distribution of likely outcomes?

The key advantage of the HMM extension is that it avoids assuming a single stationary return environment. Instead, the model allows return and covariance behavior to evolve through inferred market regimes, which is closer to how real financial markets behave.

## Research Experiment Backtest Framing

This project should be described as a research experiment rather than a finalized production model. That framing is important because the goal was not to claim that the first version of the optimizer was already perfect. The goal was to:

1. build an academically grounded portfolio construction engine
2. test it honestly in an out-of-sample setting
3. observe where prediction quality was weaker than expected
4. iterate on the methodology with more robust quantitative techniques

That is a stronger and more credible research narrative than presenting the model as if every stage immediately worked as intended.

For this report, the original baseline results of the current model should be defined as:

- the earlier `100`-portfolio baseline run
- the later `1,000`-portfolio / `10`-ticker baseline run
- the later large-sample liquid-universe run

Any additional backtests performed after further tuning, filtering changes, or methodological updates should be labeled separately as post-upgrade or post-tweak results.

## Original Out-Of-Sample Backtest Design

The original backtest was structured to avoid lookahead bias.

For a formation date of `2024-01-01`, the experiment:

1. used the trailing `2023` history to estimate expected return, volatility, correlation, and regime structure
2. built the optimized portfolio on `2024-01-01`
3. projected one-year forward behavior with the model
4. advanced to the next one-year evaluation point around `2025-01-01`
5. computed realized portfolio return from the actual subsequent price path
6. marked each portfolio prediction as:
   - `Accurate` if realized return was greater than or equal to the model’s predicted expected return
   - `Not Accurate` if realized return fell short of the model’s predicted expected return

The experiment used:

- a point-in-time volatile-stock universe as of `2024-01-01`
- randomly selected stock portfolios drawn from the eligible ticker list
- `20` stocks per portfolio in the original baseline design
- `100` unique random portfolios in the initial baseline run
- `1` year lookback
- `1` year forward evaluation horizon

Prediction accuracy was then defined as:

```text
accuracy_rate = accurate_predictions / total_tested_portfolios
```

## Original Backtest Results

The initial research run produced the following summary statistics:

- Formation Date: `2024-01-01`
- Lookback Window: `1 year`
- Forward Horizon: `1 year`
- Requested Universe Size: `710`
- Eligible Universe Size: `710`
- Portfolio Size: `20`
- Random Portfolio Iterations: `100`
- Completed Backtests: `100`
- Failed Backtests: `0`
- Elapsed Time: `980.45 seconds`
- Prediction Accuracy Rate: `59%`
- Mean Expected Return: `5.85%`
- Mean Monte Carlo Expected Return: `42.28%`
- Mean Realized Return: `11.63%`
- Median Realized Return: `9.67%`

Out of the `100` one-year portfolio experiments:

- `59` portfolio tests met or exceeded the model's estimated return
- `41` portfolio tests did not meet the model's estimated return
- of those `41` that did not meet the estimated return, `28` still produced a profitable realized return
- only `13` both missed the model's estimated return and finished the year with a non-positive realized return

This distinction is important. In this research framework, a portfolio can still have a profitable realized year while being classified as `Not Accurate` if its realized return remains below the model's own forecast threshold.

## Expanded Original Baseline Results

After the initial `100`-portfolio baseline run, the same current model was tested again on a larger point-in-time liquid-and-volatile universe. This larger run should still be classified as part of the original model baseline because it was performed before further model changes.

### Large-sample baseline run configuration

- Formation Date: `2024-01-01`
- Lookback Window: `1 year`
- Forward Horizon: `1 year`
- Universe Definition: liquid and volatile NYSE/NASDAQ names as of `2024-01-01`
- Requested Universe Size: `691`
- Eligible Universe Size: `591`
- Portfolio Construction Rule: randomly selected `20` tickers from the eligible ticker list for each portfolio
- Requested Portfolio Iterations: `10,000`
- Completed Portfolio Iterations: `9,997`
- Failed Portfolio Iterations: `3`

### Large-sample baseline run results

- Elapsed Time: `99,154.12 seconds`
- Prediction Accuracy Rate: `9.16%`
- Mean Expected Return: `24.00%`
- Mean Monte Carlo Expected Return: `69.98%`
- Mean Realized Return: `10.96%`
- Median Realized Return: `10.10%`

Out of the `9,997` completed portfolio tests:

- `916` met or exceeded the model's estimated return
- `9,081` did not meet the model's estimated return
- `9,020` still produced a profitable realized return
- of the `9,081` that missed the expected return threshold, `8,104` still produced a profitable realized return
- only `977` both missed the expected return threshold and finished with a non-positive realized return

This larger baseline run is especially informative because it shows that the original version of the model often generated portfolios with positive realized returns, but it substantially overstated the level of return it expected those portfolios to achieve.

## Third Original Baseline Test

To further evaluate the same current model without changing the core methodology, a third baseline experiment was run using smaller portfolios and a lower portfolio count than the `10,000`-iteration test, but still on the same point-in-time liquid-and-volatile universe.

### Third baseline run configuration

- Formation Date: `2024-01-01`
- Lookback Window: `1 year`
- Forward Horizon: `1 year`
- Universe Definition: liquid and volatile NYSE/NASDAQ names as of `2024-01-01`
- Requested Universe Size: `691`
- Eligible Universe Size: `691`
- Portfolio Construction Rule: randomly selected `10` tickers from the eligible ticker list for each portfolio
- Requested Portfolio Iterations: `1,000`
- Completed Portfolio Iterations: `994`
- Failed Portfolio Iterations: `6`

### Third baseline run results

- Elapsed Time: `4,237.98 seconds`
- Prediction Accuracy Rate: `23.44%`
- Mean Expected Return: `19.33%`
- Mean Monte Carlo Expected Return: `62.65%`
- Mean Realized Return: `9.78%`
- Median Realized Return: `8.84%`
- Standard Deviation Of Forecast Error `(realized return - expected return)`: `13.62%`
- Mean Forecast Error `(realized return - expected return)`: `-9.55%`
- Mean Absolute Forecast Error: `13.38%`

Out of the `994` completed portfolio tests:

- `233` met or exceeded the model's estimated return
- `761` did not meet the model's estimated return
- `562` did not meet the model's estimated return but still produced a profitable realized return
- `199` both missed the estimated return threshold and finished with a non-positive realized return

That means:

- `56.54%` of all completed portfolios were profitable but still failed to reach the model's expected return
- `73.85%` of the portfolios that missed the expected return threshold still produced a positive realized return

This third baseline run is useful because it shows that changing portfolio breadth from `20` names to `10` names improved the formal prediction-accuracy rate relative to the large `10,000`-portfolio test, but the same calibration issue still remained: expected returns were materially higher than realized returns on average.

## Interpretation Of The Initial Results

These results were informative, but they were not as strong as originally hoped.

The main takeaway is not that the project failed. The takeaway is that the first research version exposed clear model-behavior gaps:

- the one-year binary prediction accuracy of `59%` was only modestly above chance
- `41` out of `100` portfolios failed to reach the model's estimated return threshold
- some of those `41` portfolios can still be interpreted as profitable years that simply underperformed the model's forecast
- the Monte Carlo expected return estimate was materially more optimistic than the realized average return
- the model did produce positive realized return on average, but prediction calibration was weaker than desired

That is exactly the kind of outcome that should motivate methodological upgrades in a serious quantitative research project.

The large-sample `9,997`-portfolio baseline reinforced the same conclusion more strongly:

- the realized average return remained positive
- most portfolios were still profitable in absolute terms
- but the model's expected-return hurdle was far too optimistic
- that optimism translated into a very low formal prediction-accuracy rate despite many portfolios still earning positive returns

So the original model can reasonably be described as:

- often profitable in directional terms
- but not yet well calibrated as a return-prediction engine

Taken together, the first three baseline experiments yielded a consistent pattern:

- the model often produced portfolios with positive realized returns
- but the model's expected-return estimates were systematically too optimistic
- prediction accuracy deteriorated sharply as the test scale increased and the expected-return hurdle became harder to clear
- changing the portfolio width from `20` names to `10` names helped somewhat, but did not fully solve the calibration problem

## How To Describe The Upgrade Path In The Paper

A strong report should explain that the project evolved in stages.

### Stage 1: Original Research Prototype

The original prototype established the core portfolio-construction pipeline:

- historical return estimation
- constrained mean-variance optimization
- cash and Treasury-bill defensive sleeves
- implementation-aware share allocation

This stage was important because it created a working baseline that could be tested honestly.

### Stage 2: Correlation Cleaning Upgrade

After observing the instability that raw covariance estimation can create, the model was upgraded with Marchenko-Pastur random matrix filtering. The purpose of that upgrade was to reduce optimizer sensitivity to noisy empirical correlation structure.

### Stage 3: Regime-Aware Forward Simulation Upgrade

Because a single stationary Gaussian outlook was too simplistic, the model was then upgraded to include Hidden Markov Model regime-switching Monte Carlo. The purpose of that upgrade was to allow the forward distribution to reflect shifting market regimes rather than one fixed volatility state.

### Stage 4: Portfolio Construction Controls

The model was also upgraded with:

- automatic stock-specific max allocation logic
- point-in-time backtesting logic
- defensive sleeve constraints for cash plus Treasury bills
- advisor-facing target return and volatility controls
- output risk scoring instead of requiring the user to input an abstract risk-aversion number

### Stage 5: Calibration And Backtest Improvement Phase

After the original baseline tests, the next upgrade phase should be described as a calibration and robustness-improvement phase. The purpose of this phase is not to rewrite the original results, but to improve forecast realism and out-of-sample accuracy in later tests.

The key upgrades in this phase can be described as:

- tightening the universe definition toward more liquid and more tradable names
- improving point-in-time universe construction so the test set better matches the actual investable opportunity set at formation
- improving data-download resilience with shorter timeouts and resumable backtest execution
- refining return-calibration logic so expected returns are not materially overstated relative to realized outcomes
- stress-testing narrower and broader portfolio constructions such as `10`-name versus `20`-name portfolios
- evaluating whether market-neutral construction should be added as a future extension

### Stage 6: Expected-Return Estimation Upgrade

The most important quantitative upgrade after the original baseline tests is the expected-return layer itself. The original backtests showed that the framework often generated profitable portfolios, but the expected-return forecasts were too high and therefore produced weak formal prediction accuracy.

To address that issue, the upgraded model now supports multiple expected-return estimators:

#### Historical Mean

This remains the baseline estimator:

```text
mu_i = mean(r_i) * 252
```

It is retained as the control specification for later comparisons.

#### Bayes-Stein Shrinkage

This estimator shrinks raw sample means toward the cross-sectional grand mean, reducing the effect of noisy extreme return estimates.

The purpose of this upgrade is to improve calibration by pulling implausibly large expected-return estimates back toward more stable levels.

#### Market-Factor-Implied Expected Returns

This estimator infers expected return from each asset's beta to the equal-weight market factor rather than relying only on the raw sample mean.

The purpose of this upgrade is to create a more structural expected-return estimate tied to factor exposure.

#### Black-Litterman-Style Prior Blend

This estimator blends:

- a factor-implied prior expected return
- the historical sample-return estimate

The purpose is to anchor the expected-return forecast to a more stable prior while still allowing historical information to influence the posterior estimate.

### Comparison Framework For The Upgraded Return Models

The project now includes a direct comparison workflow across:

- `historical_mean`
- `bayes_stein`
- `market_factor`
- `black_litterman`

This comparison framework is designed to evaluate each return estimator on the same:

- formation date
- investable universe definition
- portfolio size
- random portfolio sampling procedure
- backtest horizon

That allows the research process to compare:

- prediction accuracy
- profitable-return frequency
- mean forecast error
- mean absolute forecast error
- standard deviation of forecast error

### Initial Comparison Results Across Return Estimators

Using the same point-in-time universe and the same `10`-ticker / `1,000`-portfolio backtest design, the initial estimator comparison produced the following results:

#### Historical Mean

- Completed Combinations: `994`
- Failed Combinations: `6`
- Prediction Accuracy Rate: `23.44%`
- Mean Expected Return: `19.34%`
- Mean Monte Carlo Expected Return: `62.23%`
- Mean Realized Return: `9.78%`
- Median Realized Return: `8.84%`
- Elapsed Time: `9,839.84 seconds`

#### Bayes-Stein

- Completed Combinations: `994`
- Failed Combinations: `6`
- Prediction Accuracy Rate: `23.24%`
- Mean Expected Return: `19.30%`
- Mean Monte Carlo Expected Return: `62.28%`
- Mean Realized Return: `9.79%`
- Median Realized Return: `8.84%`
- Elapsed Time: `9,874.55 seconds`

#### Market Factor

- Completed Combinations: `994`
- Failed Combinations: `6`
- Prediction Accuracy Rate: `76.76%`
- Mean Expected Return: `1.04%`
- Mean Monte Carlo Expected Return: `47.36%`
- Mean Realized Return: `9.12%`
- Median Realized Return: `8.49%`
- Elapsed Time: `9,824.85 seconds`

#### Black-Litterman-Style Blend

- Completed Combinations: `994`
- Failed Combinations: `6`
- Prediction Accuracy Rate: `46.48%`
- Mean Expected Return: `10.14%`
- Mean Monte Carlo Expected Return: `61.24%`
- Mean Realized Return: `9.77%`
- Median Realized Return: `8.83%`
- Elapsed Time: `9,862.15 seconds`

### Interpretation Of The Estimator Comparison

These results suggest a clear hierarchy.

- `Bayes-Stein` did not materially improve the model relative to the original historical-mean baseline
- `Black-Litterman` produced a meaningful improvement in prediction accuracy by reducing the expected-return hurdle from about `19.34%` to about `10.14%`
- `Market Factor` produced the strongest formal prediction accuracy, but it appears to do so largely by collapsing the expected-return hurdle to a very low level of about `1.04%`

That means the comparison should not be read too simplistically.

- `Market Factor` appears to be the most conservative and best calibrated in a strict forecasting sense
- `Black-Litterman` appears to be the most balanced upgrade because it materially improves accuracy while still preserving a more economically meaningful expected-return estimate
- `Historical Mean` and `Bayes-Stein` remain too optimistic relative to realized outcomes on this test design

In research-report terms, the current evidence suggests that:

- the original historical-mean framework was overestimating expected return
- the market-factor estimator may be too conservative for practical portfolio targeting
- the Black-Litterman-style blend is currently the most promising compromise between return realism and predictive accuracy

### Exact Comparison Conclusion

The estimator comparison can be summarized as follows:

- all four expected-return methods still produced positive mean realized returns
- `historical_mean` and `bayes_stein` remained too optimistic and did not materially improve calibration
- `market_factor` produced the highest formal prediction accuracy, but did so by setting a very low expected-return hurdle
- `black_litterman` produced the best overall balance between predictive improvement and economically meaningful expected-return estimates

Stated more directly:

- `historical_mean` was too aggressive
- `bayes_stein` was not a large enough improvement
- `market_factor` was the most conservative
- `black_litterman` was the most balanced

That makes the Black-Litterman-style blend the strongest current candidate for the next main version of the model.

### Recommended Upgrade Path

Based on both the baseline backtests and the return-estimator comparison, the most defensible next-stage upgrade path is:

1. Promote `black_litterman` to the leading expected-return framework.
2. Improve the prior used inside that framework so it is based on stronger factor structure rather than only a simple market-factor proxy.
3. Add a rule that excludes long-only portfolios with negative expected return from the candidate set.
4. Continue to use denoised covariance estimation and HMM regime-aware simulation as the risk layer.
5. Test a more robust Bayesian allocation framework after the Black-Litterman layer is stabilized.
6. Evaluate a market-neutral extension as a separate research branch rather than mixing it into the original long-only baseline.

### Recommended Research References For The Upgrade Phase

The most relevant references for the next iteration of the model are:

- Black and Litterman, *Global Portfolio Optimization*  
  - supports using a prior-and-views framework for more stable expected-return estimation
- Fama and French, *Common Risk Factors in the Returns on Stocks and Bonds*  
  - supports using factor structure to inform expected-return priors
- Meucci, *Robust Bayesian Allocation*  
  - supports a more robust Bayesian allocation framework once the expected-return layer is upgraded
- Jagannathan and Ma, *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps*  
  - supports the practical importance of constraints when estimation error is large
- DeMiguel, Garlappi, and Uppal, *Optimal Versus Naive Diversification*  
  - supports benchmarking optimized portfolios against simpler baselines so the model is held to a realistic standard

### How To Talk About This In The Report

A strong way to present this in the upgrade section is:

> The original portfolio-optimization framework often generated profitable realized outcomes, but its expected-return forecasts were too optimistic. To improve calibration, multiple expected-return estimators were tested on the same point-in-time universe and backtest design. Historical-mean and Bayes-Stein approaches remained too aggressive, while a pure market-factor approach improved formal prediction accuracy primarily by lowering the expected-return hurdle substantially. The Black-Litterman-style blend produced the most balanced result, improving predictive accuracy while retaining a more economically meaningful expected-return estimate. Based on those findings, the next model-upgrade phase should emphasize Black-Litterman-style expected-return estimation, stronger factor-informed priors, and more robust Bayesian portfolio construction.

### Publishable Upgrade Conclusion

For a final project report, the most defensible conclusion is:

- all tested expected-return methods produced positive mean realized returns
- the original `historical_mean` approach and the `bayes_stein` variant remained too optimistic
- the `market_factor` method achieved the highest formal accuracy, but likely did so by setting an expected-return hurdle that was too low for practical portfolio targeting
- the `black_litterman` method provided the best overall tradeoff between calibration improvement and economically meaningful expected-return levels

Therefore, the strongest recommendation for the next production-style version of the model is:

1. use a Black-Litterman-style expected-return framework as the primary return engine
2. strengthen the prior with better factor-informed return inputs
3. retain denoised covariance estimation and HMM regime-aware simulation as the risk layer
4. evaluate robust Bayesian allocation and market-neutral construction as later-stage research branches

### Source-Cited Research Basis For The Recommended Upgrade

The recommendation to prioritize a Black-Litterman-style return engine is supported by:

- Black, F., and Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal.  
  - Link: https://doi.org/10.2469/faj.v48.n5.28
  - Relevance: supports the use of a prior-and-views framework to stabilize expected-return estimation in portfolio construction.

- Fama, E. F., and French, K. R. (1993). *Common Risk Factors in the Returns on Stocks and Bonds*. Journal of Financial Economics.  
  - Link: https://doi.org/10.1016/0304-405X(93)90023-5
  - Relevance: supports building stronger factor-based priors rather than relying only on noisy raw historical means.

- Meucci, A. (2005). *Risk and Asset Allocation* / robust Bayesian allocation research stream.  
  - SSRN Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=681553
  - Relevance: supports the next-step transition toward a more robust Bayesian portfolio-construction framework after the expected-return layer is improved.

- Jagannathan, R., and Ma, T. (2002). *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps*. Journal of Finance / NBER working paper version.  
  - Link: https://www.nber.org/papers/w8922
  - Relevance: supports keeping practical constraints in the optimizer because constraints can reduce the damage caused by estimation error.

- DeMiguel, V., Garlappi, L., and Uppal, R. (2009). *Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?* Review of Financial Studies.  
  - SSRN Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199
  - Relevance: supports benchmarking the upgraded model against simple allocation baselines so any claimed improvement is empirically defensible.

### Potential Future Extension: Resampled Optimization

Another upgrade under consideration is resampled optimization. This is conceptually different from expected-return estimation because it addresses optimizer sensitivity to estimation error rather than directly changing the return forecast itself.

It should therefore be described as a later-stage robustness extension rather than as the first calibration fix.

### Potential Future Extension: Market-Neutral Variant

One possible next-stage enhancement is a market-neutral version of the framework.

That extension could:

- hedge out broad market beta
- shift the research objective from absolute return forecasting to relative alpha forecasting
- reduce the influence of directional market moves on the backtest accuracy statistic

In a research report, this should be described as a future extension under consideration rather than as part of the original baseline model.

## Recommended Paper Language

An honest and professional way to describe the project is:

> The original out-of-sample tests did not produce the level of predictive consistency initially expected. That result was treated as a research finding rather than a failure. In response, the model was upgraded with correlation-matrix denoising using Marchenko-Pastur theory, regime-aware Monte Carlo simulation using a Hidden Markov Model, improved portfolio-construction controls, and a stricter point-in-time backtesting framework. The project therefore evolved as an iterative quantitative research process in which empirical test results guided successive methodological improvements.

## Why This Framing Matters

This framing makes the report stronger because it shows:

- the model was evaluated honestly
- underperformance in the first research pass was acknowledged directly
- later enhancements were motivated by observed weaknesses rather than added arbitrarily
- the project reflects actual quantitative research practice rather than a one-shot polished demo

## Example Portfolio Universe: 20 Stocks

Below is an example stock universe a user could enter into the GUI.

1. `AAPL` with max weight `6%`
2. `MSFT` with max weight `6%`
3. `NVDA` with max weight `5%`
4. `AMZN` with max weight `5%`
5. `GOOGL` with max weight `5%`
6. `META` with max weight `5%`
7. `BRK-B` with max weight `5%`
8. `JPM` with max weight `4%`
9. `V` with max weight `4%`
10. `MA` with max weight `4%`
11. `XOM` with max weight `4%`
12. `LLY` with max weight `4%`
13. `UNH` with max weight `4%`
14. `COST` with max weight `4%`
15. `PG` with max weight `4%`
16. `HD` with max weight `4%`
17. `AVGO` with max weight `4%`
18. `ADBE` with max weight `4%`
19. `PEP` with max weight `4%`
20. `KO` with max weight `4%`

### Example portfolio-level settings

- Capital: `$1,000,000`
- Lookback Years: `3`
- Max Allocation Mode: `Auto`
- Auto Max Floor: `2%`
- Auto Max Ceiling: `8%`
- Minimum Cash: `3%`
- Maximum Cash: `12%`
- Cash Yield: `4.0%`
- Auto T-bill Yield: `On`
- Fallback T-bill Yield: `4.5%`
- Target Return: `9%`
- Target Volatility: `12%`
- Monte Carlo Paths: `25,000`
- Monte Carlo Horizon: `1 year`

### What happens when this runs

The application will:

1. Download historical adjusted-close prices for all 20 stocks.
2. Estimate annualized expected return, volatility, covariance, and correlation from the selected lookback window.
3. Denoise the empirical stock correlation matrix using Marchenko-Pastur eigenvalue filtering.
4. Rebuild the stock covariance matrix from the cleaned correlation structure.
5. If auto max allocation is enabled, generate stock-specific concentration caps from volatility and diversification scores.
6. Attempt to fetch the official 1-year Treasury bill rate.
7. Fall back to the manual T-bill yield if the official fetch fails and a fallback yield is provided.
8. Solve for the portfolio weights that maximize risk-adjusted expected utility under the chosen constraints, including any target return requirement.
9. Convert the stock sleeve into exact share counts.
10. Recommend a cash allocation.
11. Recommend a Treasury-bill allocation.
12. Fit an HMM to the market factor return series.
13. Run regime-switching Monte Carlo on the full optimized portfolio.
14. Compute the portfolio risk score and risk label from volatility and downside simulation metrics.
15. Return both the implementable allocation and the projected distribution of one-year outcomes.

## Practical Interpretation

This framework combines two complementary perspectives:

- optimization answers: "What is the best allocation given the denoised opportunity set and my constraints?"
- HMM Monte Carlo answers: "If I hold that optimized portfolio under changing market regimes, what could the next year plausibly look like?"

That combination makes the tool more useful than a static efficient-frontier output because it produces both:

- a practical implementation plan
- a probabilistic forward-looking risk view

## Key Caveats

- Historical returns do not guarantee future returns.
- Correlations can change materially during market stress.
- Marchenko-Pastur filtering reduces noise, but it does not eliminate model risk.
- HMM Monte Carlo results are only as good as the inferred regime structure and regime-specific return assumptions.
- The Treasury-bill sleeve attempts to use the official 1-year Treasury bill rate first, but may still fall back to a proxy or manual value depending on data availability.
- Auto max allocation is a model-based concentration heuristic, not a substitute for portfolio-manager judgment.
- A strict target return may make the optimization infeasible if the available opportunity set cannot meet it under the chosen constraints.
- The model is long-only and does not currently include taxes, slippage, commissions, or transaction costs.
- Discrete share rounding can slightly change the realized portfolio relative to the continuous optimizer output.

## Academic Framing For A Project Report

If this project is being presented in a research-style or professional report, the methodology can be described as:

1. A Markowitz-style constrained portfolio optimization framework.
2. Enhanced with random matrix denoising of the empirical correlation matrix using Marchenko-Pastur theory.
3. Extended with an HMM regime-switching Monte Carlo engine for forward portfolio simulation.
4. Combined with implementation-aware portfolio construction logic:
   - discrete share rounding
   - defensive cash and Treasury-bill sleeves
   - optional automatic stock-specific concentration caps
   - portfolio risk scoring in the output layer

That framing is accurate, detailed, and defensible for a serious project report.

**Figure 4** visually summarizes the upgrade roadmap outlined above: Black-Litterman anchors the return layer, followed by Fama–French factor priors to stabilize expectations, Meucci’s robust Bayesian allocation to damp estimation noise, and finally benchmarked constraint tuning plus volatility/regime-neutral experimentation tied to Jagannathan & Ma and DeMiguel et al. Each stage references the cited papers so the roadmap can be quoted directly in the methodology section.

## Suggested Citation Section

You can cite the model foundation in a report using language like this:

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance.
- Marchenko, V. A., and Pastur, L. A. (1967). *Distribution of eigenvalues for some sets of random matrices*.
- Laloux, L., Cizeau, P., Bouchaud, J.-P., and Potters, M. (1999). *Noise Dressing of Financial Correlation Matrices*. Physical Review Letters.
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica.

## Additional References For The Report

The following references are also useful for strengthening the methodology section, estimation-error discussion, and future-upgrade section of the paper.

### Covariance And Correlation Estimation

Ledoit, O., and Wolf, M. *Honey, I Shrunk the Sample Covariance Matrix*.

- Why it matters:
  - supports the argument that raw sample covariance estimation is unstable in portfolio optimization
  - provides a strong justification for using shrinkage or other regularization methods when building a covariance matrix
  - is especially useful if the report discusses future covariance upgrades beyond Marchenko-Pastur denoising
- Link:
  - https://www.ledoit.net/honey.pdf

Jagannathan, R., and Ma, T. *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps*.

- Why it matters:
  - supports the practical finding that constraints can improve out-of-sample portfolio behavior
  - gives academic backing to the use of max weights, defensive sleeve constraints, and other realistic portfolio restrictions
- Link:
  - https://www.nber.org/papers/w8922

### Expected Return Estimation

Jorion, P. *Bayes-Stein Estimation for Portfolio Analysis*.

- Why it matters:
  - directly supports the use of mean-return shrinkage in portfolio estimation
  - is useful for the section of the report where multiple expected-return models are compared
  - helps justify why raw historical mean return estimates are often too noisy
- Link:
  - https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/bayesstein-estimation-for-portfolio-analysis/B7D5C6C54432BDE3F8E3B107E68B0E1E

Black, F., and Litterman, R. *Global Portfolio Optimization*.

- Why it matters:
  - is the core reference for the Black-Litterman framework
  - supports the report’s conclusion that Black-Litterman produced a better balance between realism and predictive performance than the raw historical-mean approach
- Link:
  - https://rpc.cfainstitute.org/research/financial-analysts-journal/1992/faj-v48-n5-28

Fama, E. F., and French, K. R. *Common Risk Factors in the Returns on Stocks and Bonds*.

- Why it matters:
  - provides academic support for factor-based expected-return priors
  - strengthens the future-work section if the next model version is described as moving toward more formal factor-driven return estimation
- Link:
  - https://doi.org/10.1016/0304-405X(93)90023-5

### Estimation Error And Benchmarking

DeMiguel, V., Garlappi, L., and Uppal, R. *Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?*

- Why it matters:
  - is a strong reference for the danger of overfitting optimized portfolios when inputs are noisy
  - supports the idea that optimized portfolios should always be benchmarked against simpler allocation rules
  - is useful for framing the project’s backtesting results honestly
- Link:
  - https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901

### Regime-Aware And Volatility-Aware Extensions

Hamilton, J. D. *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*.

- Why it matters:
  - is the foundational reference for regime-switching models
  - supports the HMM logic already used in the project’s Monte Carlo engine
- Standard citation:
  - Econometrica, 1989

Moreira, A., and Muir, T. *Volatility Managed Portfolios*.

- Why it matters:
  - is useful if the project report discusses future volatility-scaling or volatility-managed extensions
  - provides support for dynamically adjusting exposure as volatility conditions change
- Link:
  - https://www.nber.org/papers/w22208

### Robust Bayesian Allocation

Meucci, A. *Risk and Asset Allocation* and associated robust Bayesian allocation work.

- Why it matters:
  - supports the report’s future-upgrade section if the next model version is framed as moving from basic return shrinkage toward more robust Bayesian portfolio construction
  - is especially relevant if the project narrative emphasizes reducing sensitivity to noisy expected-return inputs
- Link:
  - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=681553

## How These References Support The Upgrade Narrative

If the final report includes a section explaining how the model evolved, these references support a clean research progression:

1. Start with Markowitz-style optimization on historical estimates.
2. Improve covariance quality with denoising and shrinkage.
3. Improve expected-return estimation with Bayes-Stein, factor priors, and Black-Litterman.
4. Benchmark the optimizer against simpler allocations to avoid overstating performance.
5. Extend the risk model with regime-aware and volatility-aware methods.
6. Move toward robust Bayesian allocation for a more stable production framework.

## Suggested Professional Summary

This portfolio engine estimates expected return and volatility directly from market history, denoises the correlation matrix with Marchenko-Pastur filtering, optionally generates stock-specific concentration caps automatically, optimizes a constrained long-only stock portfolio with internal cash and Treasury-bill sleeves, converts the result into exact share counts, and then stress-tests the final allocation with Hidden Markov Model regime-switching Monte Carlo simulation. The result is a practical framework for building implementation-ready portfolios while also understanding the probable distribution of one-year outcomes under changing market conditions.
