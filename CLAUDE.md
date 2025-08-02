# Claude Configuration for Alpha Signal Decomposition Project

This file contains configuration settings and context for Claude to work effectively with the Alpha Signal Decomposition under Regime Shifts using Sparse Bayesian Learning project.

## Project Overview

This is a comprehensive quantitative finance framework that combines Sparse Bayesian Learning (SBL) with regime-aware modeling for robust alpha signal generation and portfolio optimization.

## Project Structure

```
Alpha_Signal_Decomposition_under_Regime_Shifts_using_Sparse_Bayesian_Learning/
├── src/
│   ├── data/
│   │   ├── collectors/           # Data collection modules
│   │   │   ├── market_data_collector.py
│   │   │   ├── fundamental_data_collector.py
│   │   │   └── macro_data_collector.py
│   │   ├── processors/           # Data processing and validation
│   │   │   ├── data_validator.py
│   │   │   └── feature_engineer.py
│   │   └── storage/              # Database management
│   │       └── database_manager.py
│   ├── models/
│   │   └── regime_detection/     # Regime detection models
│   │       └── enhanced_regime_detector.py
│   ├── portfolio/               # Portfolio optimization
│   │   └── portfolio_optimizer_clean.py
│   ├── backtesting/             # Backtesting framework
│   │   └── backtester.py
│   └── utils/                   # Utility functions
├── config/                      # Configuration files
├── docs/                        # Jekyll documentation site
│   ├── _config.yml
│   ├── index.md
│   ├── methodology.md
│   ├── implementation.md
│   ├── results.md
│   ├── documentation.md
│   ├── Gemfile
│   └── .gitignore
├── requirements.txt
└── CLAUDE.md                    # This file
```

## Key Technologies and Libraries

### Core Dependencies
- **NumPy & Pandas**: Data manipulation and numerical computing
- **SciPy**: Advanced mathematical functions and optimization
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Statsmodels**: Statistical modeling and time series analysis

### Financial Data
- **yfinance**: Yahoo Finance API for market data
- **fredapi**: Federal Reserve Economic Data API

### Advanced Features (Optional)
- **cvxpy**: Convex optimization for portfolio construction
- **hmmlearn**: Hidden Markov Models for regime detection
- **pykalman**: Kalman filtering for state estimation

### Database
- **SQLAlchemy**: Database ORM for data storage
- **psycopg2**: PostgreSQL adapter (for production)

### Visualization
- **matplotlib**: Basic plotting functionality
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations

## Key Components Implemented

### 1. Data Collection Pipeline
- **MarketDataCollector**: Comprehensive market data collection with technical indicators
- **FundamentalDataCollector**: Financial ratios and company metrics
- **MacroDataCollector**: Economic indicators from FRED API

### 2. Feature Engineering
- **AdvancedFeatureEngineer**: 100+ technical, fundamental, and cross-sectional features
- **DataValidator**: Statistical validation and anomaly detection

### 3. Regime Detection
- **EnhancedRegimeDetector**: Multiple methods (HMM, MS-VAR, GMM) with ensemble approach

### 4. Portfolio Optimization
- **AdvancedPortfolioOptimizer**: Multiple optimization methods:
  - Mean-Variance
  - Black-Litterman
  - Risk Parity
  - Hierarchical Risk Parity
  - Robust Optimization

### 5. Backtesting Framework
- **ComprehensiveBacktester**: Walk-forward analysis with:
  - Regime-aware performance attribution
  - Statistical significance testing
  - Bootstrap confidence intervals
  - Comprehensive risk metrics

### 6. Database Management
- **DatabaseManager**: Multi-database support (SQLite, PostgreSQL, MySQL)
- Optimized schema for financial time series data

### 7. Documentation Site
- **Jekyll**: Static site generator for project documentation
- Comprehensive methodology, implementation, and results documentation

## Configuration Guidelines

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables for API keys
export FRED_API_KEY=your_fred_api_key
export BLOOMBERG_API_KEY=your_bloomberg_api_key  # Optional

# Database configuration
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=username
export DB_PASSWORD=password
```

### Data Sources
- Primary: yfinance for market data (free, reliable)
- Secondary: FRED for macroeconomic data
- Optional: Bloomberg/FactSet for institutional-grade data

### Testing and Validation
When working with this codebase:
1. Always run data validation before model training
2. Use walk-forward backtesting to avoid look-ahead bias
3. Test with multiple market regimes
4. Validate statistical significance of results

## Common Commands and Workflows

### Lint and Type Check Commands
- **Linting**: `flake8 src/` (install with `pip install flake8`)
- **Type Checking**: `mypy src/` (install with `pip install mypy`)
- **Code Formatting**: `black src/` (install with `pip install black`)

### Running Tests
```bash
# Unit tests (if implemented)
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Backtesting validation
python src/backtesting/validate_backtest.py
```

### Documentation
```bash
# Serve Jekyll documentation locally
cd docs/
bundle install
bundle exec jekyll serve
# View at http://localhost:4000
```

## Important Notes for Claude

### Code Quality Standards
- Follow PEP 8 style guidelines
- Use type hints for function parameters and returns
- Include comprehensive docstrings for all classes and methods
- Handle errors gracefully with appropriate logging
- Write modular, testable code

### Financial Domain Considerations
- Always validate data quality before model training
- Be aware of look-ahead bias in backtesting
- Consider transaction costs and market impact
- Handle missing data appropriately
- Account for regime changes in model validation

### Performance Considerations
- Use vectorized operations with NumPy/Pandas
- Implement caching for expensive computations
- Consider memory usage for large datasets
- Optimize database queries for time series data

### Security Considerations
- Never commit API keys or passwords
- Use environment variables for sensitive configuration
- Implement proper error handling to avoid information leakage
- Validate all external data inputs

## Troubleshooting Common Issues

### Data Collection Issues
- API rate limits: Implement proper delays and retry logic
- Missing data: Use forward/backward fill or interpolation
- Data quality: Run validation checks after collection

### Model Training Issues
- Insufficient data: Ensure minimum training window requirements
- Convergence issues: Adjust model parameters or initialization
- Memory issues: Use batch processing for large datasets

### Optimization Issues
- Infeasible solutions: Check constraint compatibility
- Poor performance: Validate alpha signals and risk model
- High turnover: Adjust transaction cost penalties

This configuration provides Claude with comprehensive context about the project structure, technologies, and best practices for working with this quantitative finance framework.