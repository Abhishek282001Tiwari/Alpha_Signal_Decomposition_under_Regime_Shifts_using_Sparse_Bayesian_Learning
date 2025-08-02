---
layout: page
title: API Documentation
permalink: /documentation/
---

# API Documentation

## Overview

This documentation provides comprehensive API reference for the Alpha Signal Decomposition framework. The system is designed with a modular architecture that allows easy integration and customization of components.

## Installation and Setup

### Requirements

```bash
# Install core dependencies
pip install numpy pandas scipy scikit-learn
pip install yfinance fredapi
pip install sqlalchemy psycopg2-binary
pip install matplotlib seaborn plotly

# Optional dependencies for advanced features
pip install cvxpy  # For advanced optimization
pip install hmmlearn  # For HMM regime detection
pip install statsmodels  # For time series analysis
pip install pykalman  # For Kalman filtering
```

### Basic Setup

```python
import sys
sys.path.append('src/')

from src.data.collectors.market_data_collector import MarketDataCollector
from src.models.regime_detection.enhanced_regime_detector import EnhancedRegimeDetector
from src.portfolio.portfolio_optimizer import AdvancedPortfolioOptimizer
from src.backtesting.backtester import ComprehensiveBacktester
```

## Data Collection APIs

### MarketDataCollector

Collects and processes market data from various sources.

#### Class Definition

```python
class MarketDataCollector:
    def __init__(self, data_source: str = 'yfinance', cache_data: bool = True)
```

#### Methods

##### `collect_data(symbols, start_date, end_date, **kwargs)`

Collects historical market data for specified symbols.

**Parameters:**
- `symbols` (List[str]): List of ticker symbols
- `start_date` (str/datetime): Start date for data collection
- `end_date` (str/datetime): End date for data collection
- `include_technical_indicators` (bool): Include technical indicators
- `include_dividends` (bool): Include dividend data

**Returns:**
- `Dict[str, pd.DataFrame]`: Dictionary mapping symbols to price data

**Example:**
```python
collector = MarketDataCollector(data_source='yfinance')
data = collector.collect_data(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    include_technical_indicators=True
)
```

##### `get_real_time_data(symbols)`

Retrieves real-time market data.

**Parameters:**
- `symbols` (List[str]): List of ticker symbols

**Returns:**
- `pd.DataFrame`: Real-time price data

##### `calculate_technical_indicators(data, indicators)`

Calculates technical indicators for price data.

**Parameters:**
- `data` (pd.DataFrame): Price data
- `indicators` (List[str]): List of indicators to calculate

**Returns:**
- `pd.DataFrame`: Data with technical indicators

### FundamentalDataCollector

Collects fundamental data including financial ratios and company metrics.

#### Class Definition

```python
class FundamentalDataCollector:
    def __init__(self, data_source: str = 'yfinance')
```

#### Methods

##### `collect_fundamental_data(symbols, metrics)`

Collects fundamental data for specified symbols.

**Parameters:**
- `symbols` (List[str]): List of ticker symbols
- `metrics` (List[str]): Fundamental metrics to collect

**Returns:**
- `Dict[str, Dict]`: Nested dictionary with fundamental data

**Example:**
```python
fundamental_collector = FundamentalDataCollector()
fundamental_data = fundamental_collector.collect_fundamental_data(
    symbols=['AAPL', 'GOOGL'],
    metrics=['pe_ratio', 'debt_to_equity', 'roe', 'revenue_growth']
)
```

### MacroDataCollector

Collects macroeconomic data from various sources.

#### Class Definition

```python
class MacroDataCollector:
    def __init__(self, fred_api_key: Optional[str] = None)
```

#### Methods

##### `collect_fred_data(series_ids, start_date, end_date)`

Collects data from FRED (Federal Reserve Economic Data).

**Parameters:**
- `series_ids` (List[str]): FRED series identifiers
- `start_date` (str/datetime): Start date
- `end_date` (str/datetime): End date

**Returns:**
- `pd.DataFrame`: Macroeconomic time series data

## Feature Engineering APIs

### AdvancedFeatureEngineer

Comprehensive feature engineering pipeline.

#### Class Definition

```python
class AdvancedFeatureEngineer:
    def __init__(self, config: FeatureConfig = None)
```

#### Configuration

```python
@dataclass
class FeatureConfig:
    lookback_periods: List[int] = None
    include_technical: bool = True
    include_fundamental: bool = True
    include_macro: bool = True
    include_cross_sectional: bool = True
    normalization_method: str = 'standard'
    feature_selection_method: str = 'mutual_info'
    max_features: Optional[int] = None
```

#### Methods

##### `engineer_features(market_data, fundamental_data, macro_data)`

Engineers comprehensive features from multiple data sources.

**Parameters:**
- `market_data` (Dict[str, pd.DataFrame]): Market price data
- `fundamental_data` (Optional[Dict]): Fundamental data
- `macro_data` (Optional[pd.DataFrame]): Macroeconomic data

**Returns:**
- `pd.DataFrame`: Engineered features matrix

**Example:**
```python
from src.data.processors.feature_engineer import AdvancedFeatureEngineer, FeatureConfig

config = FeatureConfig(
    lookback_periods=[5, 10, 20, 50],
    include_technical=True,
    include_fundamental=True,
    max_features=100
)

feature_engineer = AdvancedFeatureEngineer(config)
features = feature_engineer.engineer_features(
    market_data=market_data,
    fundamental_data=fundamental_data,
    macro_data=macro_data
)
```

##### `get_feature_importance()`

Returns feature importance scores.

**Returns:**
- `Dict[str, float]`: Feature importance mapping

##### `get_feature_statistics(features)`

Calculates comprehensive feature statistics.

**Parameters:**
- `features` (pd.DataFrame): Feature matrix

**Returns:**
- `Dict[str, Dict]`: Feature statistics

## Regime Detection APIs

### EnhancedRegimeDetector

Advanced regime detection using multiple methodologies.

#### Class Definition

```python
class EnhancedRegimeDetector:
    def __init__(self, 
                 n_regimes: int = 3,
                 methods: List[str] = ['hmm', 'ms_var'],
                 ensemble_weights: Optional[List[float]] = None)
```

#### Methods

##### `fit_regime_models(features, price_data)`

Fits regime detection models to historical data.

**Parameters:**
- `features` (pd.DataFrame): Feature matrix
- `price_data` (Dict[str, pd.DataFrame]): Price data

**Returns:**
- `Dict[str, Any]`: Fitted model results

**Example:**
```python
regime_detector = EnhancedRegimeDetector(
    n_regimes=3,
    methods=['hmm', 'ms_var', 'gmm'],
    ensemble_weights=[0.4, 0.4, 0.2]
)

regime_results = regime_detector.fit_regime_models(
    features=features,
    price_data=market_data
)
```

##### `predict_regimes(data)`

Predicts current regime probabilities.

**Parameters:**
- `data` (pd.DataFrame): Input features

**Returns:**
- `pd.DataFrame`: Regime probabilities

##### `get_regime_statistics()`

Returns comprehensive regime statistics.

**Returns:**
- `Dict[str, Any]`: Regime analysis results

## Portfolio Optimization APIs

### AdvancedPortfolioOptimizer

Multi-method portfolio optimization with risk management.

#### Class Definition

```python
class AdvancedPortfolioOptimizer:
    def __init__(self, config: OptimizationConfig = None)
```

#### Configuration

```python
@dataclass
class OptimizationConfig:
    method: str = 'mean_variance'
    regime_aware: bool = True
    risk_aversion: float = 1.0
    max_leverage: float = 1.0
    max_position_size: float = 0.1
    min_position_size: float = 0.0
    turnover_penalty: float = 0.01
    transaction_cost_bps: float = 5.0
    confidence_level: float = 0.05
    rebalance_frequency: str = 'monthly'
    use_robust_covariance: bool = True
```

#### Methods

##### `optimize_portfolio(alpha_signals, regime_probabilities, returns_data, current_positions)`

Optimizes portfolio weights given signals and constraints.

**Parameters:**
- `alpha_signals` (pd.DataFrame): Alpha signals by asset
- `regime_probabilities` (pd.DataFrame): Regime probabilities
- `returns_data` (pd.DataFrame): Historical returns
- `current_positions` (Optional[pd.Series]): Current portfolio positions

**Returns:**
- `Dict[str, Any]`: Optimization results

**Example:**
```python
from src.portfolio.portfolio_optimizer import AdvancedPortfolioOptimizer, OptimizationConfig

config = OptimizationConfig(
    method='black_litterman',
    regime_aware=True,
    risk_aversion=1.5,
    max_position_size=0.1
)

optimizer = AdvancedPortfolioOptimizer(config)
result = optimizer.optimize_portfolio(
    alpha_signals=alpha_signals,
    regime_probabilities=regime_probs,
    returns_data=returns_data
)

optimal_weights = result['weights']
expected_return = result['expected_return']
```

##### `generate_optimization_report(optimization_result)`

Generates comprehensive optimization report.

**Parameters:**
- `optimization_result` (Dict): Optimization results

**Returns:**
- `str`: Formatted optimization report

## Backtesting APIs

### ComprehensiveBacktester

Advanced backtesting framework with walk-forward analysis.

#### Class Definition

```python
class ComprehensiveBacktester:
    def __init__(self, config: BacktestConfig = None)
```

#### Configuration

```python
@dataclass
class BacktestConfig:
    start_date: datetime = None
    end_date: datetime = None
    initial_capital: float = 1000000.0
    rebalance_frequency: str = 'monthly'
    lookback_period: int = 252
    walk_forward_step: int = 21
    transaction_cost_bps: float = 5.0
    benchmark_symbol: str = 'SPY'
    risk_free_rate: float = 0.02
    confidence_levels: List[float] = None
    enable_regime_analysis: bool = True
    max_drawdown_limit: float = 0.20
    n_bootstrap_samples: int = 1000
```

#### Methods

##### `run_comprehensive_backtest(alpha_model, regime_detector, portfolio_optimizer, market_data, ...)`

Runs comprehensive backtesting with multiple analysis dimensions.

**Parameters:**
- `alpha_model`: Alpha signal generation model
- `regime_detector`: Regime detection model
- `portfolio_optimizer`: Portfolio optimization engine
- `market_data` (Dict[str, pd.DataFrame]): Historical market data
- `fundamental_data` (Optional[Dict]): Fundamental data
- `macro_data` (Optional[pd.DataFrame]): Macroeconomic data
- `benchmark_data` (Optional[pd.DataFrame]): Benchmark data

**Returns:**
- `Dict[str, Any]`: Comprehensive backtesting results

**Example:**
```python
from src.backtesting.backtester import ComprehensiveBacktester, BacktestConfig
from datetime import datetime

config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=1000000.0,
    rebalance_frequency='monthly',
    transaction_cost_bps=5.0
)

backtester = ComprehensiveBacktester(config)
results = backtester.run_comprehensive_backtest(
    alpha_model=alpha_model,
    regime_detector=regime_detector,
    portfolio_optimizer=optimizer,
    market_data=market_data
)

# Access results
portfolio_returns = results['backtest_results']['returns']
performance_metrics = results['performance_metrics']
```

##### `generate_performance_report()`

Generates comprehensive performance report.

**Returns:**
- `str`: Formatted performance report

##### `save_results(filepath)`

Saves backtesting results to file.

**Parameters:**
- `filepath` (str): Path to save results

## Database Management APIs

### DatabaseManager

Comprehensive database management for storing and retrieving data.

#### Class Definition

```python
class DatabaseManager:
    def __init__(self, config: DatabaseConfig)
```

#### Configuration

```python
@dataclass
class DatabaseConfig:
    db_type: str = 'sqlite'
    db_path: str = 'data/alpha_signals.db'
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
```

#### Methods

##### `store_market_data(data, batch_size)`

Stores market data in the database.

**Parameters:**
- `data` (Dict[str, pd.DataFrame]): Market data by symbol
- `batch_size` (int): Batch size for insertion

##### `store_features(features, batch_size)`

Stores engineered features.

**Parameters:**
- `features` (pd.DataFrame): Feature matrix
- `batch_size` (int): Batch size for insertion

##### `get_market_data(symbols, start_date, end_date)`

Retrieves market data from database.

**Parameters:**
- `symbols` (Optional[List[str]]): Symbols to retrieve
- `start_date` (Optional[datetime]): Start date
- `end_date` (Optional[datetime]): End date

**Returns:**
- `Dict[str, pd.DataFrame]`: Retrieved market data

##### `get_database_summary()`

Returns summary of database contents.

**Returns:**
- `Dict[str, Any]`: Database summary statistics

## Utility Functions

### Data Validation

```python
from src.data.processors.data_validator import DataValidator

validator = DataValidator()
validation_report = validator.validate_data(
    data=market_data,
    checks=['completeness', 'consistency', 'outliers']
)
```

### Performance Metrics

```python
from src.utils.performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics()
sharpe_ratio = metrics.calculate_sharpe_ratio(returns, risk_free_rate)
max_drawdown = metrics.calculate_max_drawdown(returns)
information_ratio = metrics.calculate_information_ratio(portfolio_returns, benchmark_returns)
```

### Plotting and Visualization

```python
from src.utils.plotting import PerformancePlotter

plotter = PerformancePlotter()
plotter.plot_cumulative_returns(portfolio_returns, benchmark_returns)
plotter.plot_drawdown_analysis(returns)
plotter.plot_regime_probabilities(regime_probs)
```

## Error Handling

The framework includes comprehensive error handling and logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Example with error handling
try:
    results = backtester.run_comprehensive_backtest(...)
except DataValidationError as e:
    logging.error(f"Data validation failed: {e}")
except OptimizationError as e:
    logging.error(f"Portfolio optimization failed: {e}")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
```

## Advanced Configuration

### Environment Variables

```bash
# Database configuration
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=username
export DB_PASSWORD=password

# API keys
export FRED_API_KEY=your_fred_api_key
export BLOOMBERG_API_KEY=your_bloomberg_api_key

# Computation settings
export N_CORES=8
export MEMORY_LIMIT=16GB
```

### Configuration Files

Create `config/settings.yml`:

```yaml
# Global settings
computation:
  n_cores: 8
  memory_limit: "16GB"
  parallel_processing: true

data_sources:
  market_data:
    primary: "bloomberg"
    fallback: "yfinance"
  
  fundamental_data:
    source: "factset"
    update_frequency: "daily"

models:
  regime_detector:
    n_regimes: 3
    methods: ["hmm", "ms_var", "gmm"]
    ensemble_weights: [0.4, 0.4, 0.2]
  
  feature_engineering:
    max_features: 100
    normalization_method: "robust"
    selection_method: "mutual_info"

backtesting:
  initial_capital: 10000000
  transaction_costs: 0.0005
  rebalance_frequency: "monthly"
```

## Examples and Tutorials

### Complete Workflow Example

```python
from datetime import datetime
import pandas as pd

# 1. Data Collection
market_collector = MarketDataCollector()
fundamental_collector = FundamentalDataCollector()
macro_collector = MacroDataCollector()

# Collect data
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

market_data = market_collector.collect_data(symbols, start_date, end_date)
fundamental_data = fundamental_collector.collect_fundamental_data(symbols)
macro_data = macro_collector.collect_fred_data(['DGS10', 'UNRATE'], start_date, end_date)

# 2. Feature Engineering
feature_engineer = AdvancedFeatureEngineer()
features = feature_engineer.engineer_features(
    market_data=market_data,
    fundamental_data=fundamental_data,
    macro_data=macro_data
)

# 3. Regime Detection
regime_detector = EnhancedRegimeDetector(n_regimes=3)
regime_results = regime_detector.fit_regime_models(features, market_data)
regime_probs = regime_detector.predict_regimes(features)

# 4. Generate Alpha Signals (simplified)
alpha_signals = features.filter(like='momentum').mean(axis=1).to_frame('alpha')

# 5. Portfolio Optimization
optimizer = AdvancedPortfolioOptimizer()
portfolio_result = optimizer.optimize_portfolio(
    alpha_signals=alpha_signals,
    regime_probabilities=regime_probs,
    returns_data=pd.DataFrame({sym: data['Close'].pct_change() 
                              for sym, data in market_data.items()})
)

# 6. Backtesting
backtester = ComprehensiveBacktester()
backtest_results = backtester.run_comprehensive_backtest(
    alpha_model=None,  # Use pre-computed signals
    regime_detector=regime_detector,
    portfolio_optimizer=optimizer,
    market_data=market_data
)

# 7. Results Analysis
print(backtester.generate_performance_report())
```

This documentation provides a comprehensive reference for using the Alpha Signal Decomposition framework. Each component is designed to be modular and extensible, allowing for easy customization and integration into existing workflows.