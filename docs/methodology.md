---
layout: page
title: Methodology
permalink: /methodology/
---

# Methodology

## Theoretical Foundation

Our framework combines **Sparse Bayesian Learning (SBL)** with **regime-switching models** to create a robust and adaptive quantitative finance system. This approach addresses key challenges in traditional factor models by incorporating:

1. **Non-stationarity** through regime detection
2. **Feature selection** via sparse priors
3. **Uncertainty quantification** through Bayesian inference
4. **Dynamic adaptation** to changing market conditions

## Mathematical Framework

### 1. Sparse Bayesian Learning for Alpha Signals

The core alpha generation model follows a hierarchical Bayesian structure:

$$y_t = \boldsymbol{X}_t \boldsymbol{\beta}_{s_t} + \epsilon_t$$

Where:
- $y_t$ represents the target return at time $t$
- $\boldsymbol{X}_t$ is the feature matrix (technical, fundamental, macro)
- $\boldsymbol{\beta}_{s_t}$ are regime-specific coefficients for regime $s_t$
- $\epsilon_t \sim \mathcal{N}(0, \sigma^2_{s_t})$ is regime-dependent noise

#### Sparse Priors

We employ automatic relevance determination (ARD) priors:

$$\beta_i | \alpha_i \sim \mathcal{N}(0, \alpha_i^{-1})$$
$$\alpha_i \sim \text{Gamma}(a, b)$$

This hierarchical structure automatically determines feature relevance, with large $\alpha_i$ effectively removing irrelevant features.

### 2. Regime Detection Framework

#### Hidden Markov Model (HMM)

The regime state $s_t$ follows a first-order Markov chain:

$$P(s_t = j | s_{t-1} = i) = A_{ij}$$

With observation model:
$$P(\boldsymbol{o}_t | s_t = j) = \mathcal{N}(\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$$

Where $\boldsymbol{o}_t$ includes market volatility, correlation, and momentum indicators.

#### Markov-Switching Vector Autoregression (MS-VAR)

For capturing regime-dependent dynamics:

$$\boldsymbol{y}_t = \boldsymbol{\nu}_{s_t} + \sum_{i=1}^{p} \boldsymbol{A}_{i,s_t} \boldsymbol{y}_{t-i} + \boldsymbol{\epsilon}_t$$

Where regime-specific parameters $\{\boldsymbol{\nu}_{s_t}, \boldsymbol{A}_{i,s_t}\}$ capture different market dynamics.

#### Ensemble Regime Detection

We combine multiple methods using:

$$P(\text{regime}_j | \text{data}) = \sum_{k=1}^{K} w_k P_k(\text{regime}_j | \text{data})$$

Where $w_k$ are model weights determined by out-of-sample performance.

### 3. Portfolio Optimization

#### Regime-Aware Expected Returns

Expected returns incorporate regime probabilities:

$$\boldsymbol{\mu}_t = \sum_{j=1}^{J} P(s_t = j | \mathcal{I}_{t-1}) \boldsymbol{\mu}_j$$

Where $\mathcal{I}_{t-1}$ is the information set at time $t-1$.

#### Multi-Objective Optimization

Our optimization framework solves:

$$\max_{\boldsymbol{w}} \boldsymbol{w}^T \boldsymbol{\mu}_t - \frac{\lambda}{2} \boldsymbol{w}^T \boldsymbol{\Sigma}_t \boldsymbol{w} - \gamma \|\boldsymbol{w} - \boldsymbol{w}_{t-1}\|_1$$

Subject to:
- $\sum_i w_i = 1$ (fully invested)
- $|\boldsymbol{w}|_1 \leq L$ (leverage constraint)
- $|w_i| \leq w_{\max}$ (position size limits)

The penalty term controls turnover, reducing transaction costs.

## Feature Engineering Pipeline

### 1. Technical Features

- **Momentum Indicators**: RSI, MACD, Price momentum across multiple timeframes
- **Volatility Measures**: ATR, Bollinger Bands, GARCH volatility
- **Volume Analysis**: OBV, Volume-weighted prices, Accumulation/Distribution
- **Pattern Recognition**: Candlestick patterns, Support/Resistance levels

### 2. Fundamental Features

- **Valuation Ratios**: P/E, P/B, EV/EBITDA with sector adjustments
- **Quality Metrics**: ROE, ROA, Debt ratios, Interest coverage
- **Growth Indicators**: Sales growth, Earnings growth, Cash flow growth
- **Efficiency Measures**: Asset turnover, Inventory turnover

### 3. Macroeconomic Features

- **Interest Rates**: Term structure, Credit spreads, Real rates
- **Economic Indicators**: GDP growth, Inflation, Employment data
- **Market Sentiment**: VIX, Put/Call ratios, Insider trading
- **Currency Effects**: Exchange rates, Carry trade indicators

### 4. Cross-Sectional Features

- **Sector Relative**: Performance vs sector benchmarks
- **Size Effects**: Market cap quintiles, Size factor loadings
- **Momentum Ranks**: Cross-sectional momentum percentiles
- **Quality Scores**: Composite quality rankings

## Risk Management Framework

### 1. Factor Risk Model

We decompose portfolio risk using:

$$\boldsymbol{\Sigma} = \boldsymbol{B} \boldsymbol{F} \boldsymbol{B}^T + \boldsymbol{D}$$

Where:
- $\boldsymbol{B}$ is the factor loading matrix
- $\boldsymbol{F}$ is the factor covariance matrix
- $\boldsymbol{D}$ is the specific risk matrix

### 2. Regime-Conditional Risk

Risk parameters vary by regime:

$$\boldsymbol{F}_{s_t} = \boldsymbol{V}_{s_t} \boldsymbol{\Lambda}_{s_t} \boldsymbol{V}_{s_t}^T$$

Where eigenvalues $\boldsymbol{\Lambda}_{s_t}$ capture regime-specific risk levels.

### 3. Transaction Cost Model

We model implementation costs as:

$$TC = \sum_i |w_i - w_{i,t-1}| \cdot (\text{bid-ask}_i + \text{impact}_i + \text{commission}_i)$$

With regime-dependent market impact parameters.

## Backtesting Methodology

### 1. Walk-Forward Analysis

- **Training Window**: Rolling 252-day window for model fitting
- **Rebalancing**: Monthly portfolio updates
- **Out-of-Sample**: Strict temporal separation to avoid look-ahead bias
- **Model Updates**: Periodic retraining of all components

### 2. Performance Attribution

Returns are decomposed as:

$$r_{p,t} = \sum_j w_{j,t-1} r_{j,t} = \sum_f \beta_{f,t-1} r_{f,t} + r_{\text{specific},t}$$

Where factor exposures $\beta_{f,t-1}$ are calculated at portfolio construction.

### 3. Statistical Testing

- **Sharpe Ratio Significance**: Jobson-Korkie test with HAC adjustments
- **Alpha Significance**: t-tests with Newey-West standard errors
- **Regime Stability**: Chow tests for structural breaks
- **Bootstrap Confidence**: Non-parametric confidence intervals

## Implementation Considerations

### 1. Computational Efficiency

- **Parallel Processing**: Multi-threaded feature calculation
- **Incremental Updates**: Efficient regime probability updates
- **Memory Management**: Streaming data processing for large universes
- **Caching**: Intermediate result storage for faster recomputation

### 2. Robustness Checks

- **Parameter Sensitivity**: Monte Carlo analysis of key parameters
- **Regime Misclassification**: Performance under regime uncertainty
- **Transaction Cost Impact**: Sensitivity to cost assumptions
- **Data Quality**: Handling of missing data and outliers

### 3. Real-World Constraints

- **Liquidity Filters**: Minimum trading volume requirements
- **Capacity Constraints**: Position size limits based on ADV
- **Regulatory Compliance**: Concentration limits and reporting requirements
- **Operational Risk**: Settlement and custody considerations

## Model Validation

### 1. In-Sample Validation

- **Cross-Validation**: Time series CV with expanding windows
- **Information Criteria**: AIC/BIC for model selection
- **Residual Analysis**: Autocorrelation and heteroscedasticity tests
- **Feature Stability**: Coefficient significance over time

### 2. Out-of-Sample Testing

- **Paper Trading**: Simulated real-time implementation
- **Benchmark Comparison**: Performance vs relevant indices
- **Risk-Adjusted Metrics**: Sharpe, Sortino, Calmar ratios
- **Regime Performance**: Analysis across different market conditions

### 3. Stress Testing

- **Historical Scenarios**: Performance during major market events
- **Monte Carlo Stress**: Simulated extreme scenarios
- **Factor Shock**: Impact of specific risk factor movements
- **Liquidity Stress**: Performance under constrained liquidity

## Future Research Directions

- Theoretical convergence analysis for regime-switching variational inference
- Extension to high-frequency intraday regime detection
- Causal inference applications in regime-conditional settings

This methodology provides a comprehensive framework for generating robust alpha signals while managing risk in dynamic market environments. The combination of advanced machine learning, regime awareness, and rigorous backtesting ensures practical applicability in real-world investment contexts.
