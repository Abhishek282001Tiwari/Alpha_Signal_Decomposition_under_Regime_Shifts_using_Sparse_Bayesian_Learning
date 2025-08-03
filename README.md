# Alpha Signal Decomposition under Regime Shifts using Sparse Bayesian Learning

## Overview

A sophisticated quantitative finance system that decomposes alpha signals into regime-dependent components using advanced sparse Bayesian learning techniques. This institutional-grade platform combines cutting-edge machine learning with robust financial engineering for comprehensive alpha generation and portfolio optimization.

## Technical Architecture

### Core Components
- Regime-Conditional Sparse Bayesian Learning Engine (1,300+ lines)
- Multi-method regime detection (HMM, MS-VAR, GMM)
- Advanced portfolio optimization with multiple methods
- Comprehensive risk management framework
- Real-time data processing pipeline

### Key Features
- Dynamic feature selection based on regime transitions
- Hierarchical Bayesian parameter linking across regimes
- 100+ engineered features (technical, fundamental, macro)
- Multi-database support (SQLite, PostgreSQL, MySQL)
- Production-ready deployment with Docker/Kubernetes
- Advanced analytics with bootstrap confidence intervals

## Mathematical Framework

### Sparse Bayesian Learning
- Variational Bayesian inference with automatic relevance determination
- Regime-specific feature selection and sparsity patterns
- Cross-regime regularization for consistency
- Uncertainty quantification through Bayesian posteriors

### Regime Detection
- Hidden Markov Models for volatility regime identification
- Markov-Switching Vector Autoregression for multi-factor regimes
- Ensemble approach with weighted regime probabilities
- Dynamic regime transition detection

## Research Contributions

### Methodological Innovations
- Regime-conditional sparse Bayesian learning for non-stationary financial time series
- Hierarchical parameter linking across market regimes with cross-regime regularization
- Dynamic feature selection with regime-aware importance weighting
- Uncertainty quantification through variational Bayesian inference

### Theoretical Framework
- Extends classical sparse Bayesian learning to regime-switching environments
- Provides mathematical foundation for regime-dependent regularization
- Establishes convergence properties for variational inference under non-stationarity

### Operations Research Applications
- Stochastic optimization under regime uncertainty
- Multi-stage decision making with regime transition costs
- Robust portfolio optimization with parameter uncertainty
- Applied probability models for financial regime detection

## Installation

### Prerequisites
```bash
Python 3.8+
NumPy >= 1.21.0
Pandas >= 1.3.0
SciPy >= 1.7.0
Scikit-learn >= 1.0.0
TensorFlow Probability >= 0.15.0
```

### Setup
```bash
git clone https://github.com/Abhishek282001Tiwari/Alpha_Signal_Decomposition_under_Regime_Shifts_using_Sparse_Bayesian_Learning.git
cd Alpha_Signal_Decomposition_under_Regime_Shifts_using_Sparse_Bayesian_Learning
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.models.regime_conditional_sbl import RegimeConditionalSBL
from src.data.collectors.market_data_collector import MarketDataCollector
from src.regimes.regime_detector import EnhancedRegimeDetector

# Initialize components
regime_detector = EnhancedRegimeDetector()
sbl_model = RegimeConditionalSBL(n_regimes=3)
data_collector = MarketDataCollector()

# Process data and generate signals
market_data = data_collector.collect_data([
    # 25-Stock Universe for Alpha Signal Decomposition System
    # Diversified across sectors, market caps, and volatility profiles
    
    # Technology (Large Cap)
    'AAPL',   # Apple Inc. - Consumer Electronics
    'MSFT',   # Microsoft Corp. - Software
    'GOOGL',  # Alphabet Inc. - Internet/Search
    'NVDA',   # NVIDIA Corp. - Semiconductors
    'META',   # Meta Platforms - Social Media
    
    # Technology (Growth)
    'TSLA',   # Tesla Inc. - Electric Vehicles
    'CRM',    # Salesforce Inc. - Cloud Software
    'ADBE',   # Adobe Inc. - Creative Software
    
    # Financial Services
    'JPM',    # JPMorgan Chase - Banking
    'BAC',    # Bank of America - Banking
    'GS',     # Goldman Sachs - Investment Banking
    'V',      # Visa Inc. - Payment Processing

    # Healthcare/Pharmaceuticals
    'JNJ',    # Johnson & Johnson - Healthcare
    'PFE',    # Pfizer Inc. - Pharmaceuticals
    'UNH',    # UnitedHealth Group - Health Insurance
    'ABBV',   # AbbVie Inc. - Biotechnology
    
    # Consumer Discretionary
    'AMZN',   # Amazon.com - E-commerce/Cloud
    'HD',     # Home Depot - Retail
    'NKE',    # Nike Inc. - Apparel
    
    # Consumer Staples
    'PG',     # Procter & Gamble - Consumer Goods
    'KO',     # Coca-Cola - Beverages
    
    # Energy
    'XOM',    # Exxon Mobil - Oil & Gas
    'CVX',    # Chevron Corp. - Oil & Gas
    
    # Industrials
    'BA',     # Boeing Co. - Aerospace
    'CAT'     # Caterpillar Inc. - Heavy Machinery
])

# Sector allocation for regime analysis
SECTOR_MAPPING = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'CRM', 'ADBE'],
    'Financials': ['JPM', 'BAC', 'GS', 'V'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV'],
    'Consumer_Discretionary': ['AMZN', 'HD', 'NKE'],
    'Consumer_Staples': ['PG', 'KO'],
    'Energy': ['XOM', 'CVX'],
    'Industrials': ['BA', 'CAT']
}

# Market cap categories for factor analysis
MARKET_CAP_TIERS = {
    'Mega_Cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    'Large_Cap': ['JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'BAC', 'ABBV', 'XOM', 'CVX'],
    'Growth_Focus': ['CRM', 'ADBE', 'NKE', 'PFE', 'GS', 'KO', 'BA', 'CAT']
}

# Volatility profiles for regime-specific modeling
VOLATILITY_PROFILES = {
    'High_Volatility': ['TSLA', 'NVDA', 'META', 'CRM', 'BA'],
    'Medium_Volatility': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'ADBE', 'JPM', 'GS', 'NKE'],
    'Low_Volatility': ['JNJ', 'PG', 'KO', 'V', 'UNH', 'HD', 'PFE', 'ABBV', 'XOM', 'CVX', 'BAC', 'CAT']
}

# Generate regime-aware alpha signals
regimes = regime_detector.detect_regimes(market_data)
sbl_model.fit(market_data, regimes)
alpha_signals = sbl_model.predict(new_data)
```

### Configuration
```python
# Configure regime detection
regime_config = {
    'n_regimes': 3,
    'covariance_type': 'full',
    'max_iter': 1000,
    'tol': 1e-6
}

# Configure sparse Bayesian learning
sbl_config = {
    'alpha_threshold': 1e-6,
    'beta_threshold': 1e-6,
    'max_iterations': 500,
    'convergence_threshold': 1e-8
}
```

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── regime_conditional_sbl.py      # Core SBL implementation
│   │   └── sparse_bayesian_learner.py     # Base SBL algorithms
│   ├── data/
│   │   ├── collectors/                     # Data collection modules
│   │   ├── processors/                     # Data processing and validation
│   │   └── storage/                        # Database management
│   ├── regimes/                           # Regime identification methods
│   ├── portfolio/                         # Portfolio optimization
│   ├── risk/                             # Risk management framework
│   ├── backtesting/                      # Performance evaluation
│   └── utils/                            # Utility functions
├── tests/                                # Test suite
├── docs/                                 # Documentation and Jekyll site
├── config/                               # Configuration files
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

## Core Algorithms

### Regime-Conditional SBL Implementation

- Separate sparse Bayesian models for each market regime
- Hierarchical parameter structure linking regime-specific models
- Dynamic feature selection with regime-aware importance
- Advanced regularization techniques (adaptive, cross-regime)

### Risk Management

- Factor-based risk models with regime adjustments
- Value-at-Risk and Conditional Value-at-Risk calculations
- Dynamic hedging strategies for regime transitions
- Transaction cost modeling and optimization

### Performance Analytics

- Walk-forward analysis with regime-aware attribution
- Monte Carlo stress testing capabilities
- Bootstrap confidence intervals for metrics
- Statistical significance testing for alpha generation

## Testing Framework

### Unit Tests
```bash
python -m pytest tests/unit/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Performance Tests
```bash
python -m pytest tests/performance/
```

## Deployment

### Docker Deployment
```bash
docker build -t alpha-signal-system .
docker run -p 8000:8000 alpha-signal-system
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Performance Metrics

### Backtesting Results

- Regime-specific Sharpe ratios and risk-adjusted returns
- Feature importance evolution across market cycles
- Portfolio turnover and transaction cost analysis
- Maximum drawdown and recovery statistics

### Risk Analytics

- Factor exposure analysis across regimes
- Regime transition prediction accuracy
- Stress testing under extreme market conditions
- Model stability and performance consistency

## Contributing

### Development Guidelines

- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new functionality
- Update documentation for API changes

## Research Applications

### Academic Research

- Publication-quality mathematical implementation
- Rigorous statistical testing framework
- Comprehensive empirical analysis capabilities
- Reproducible research methodology

### Industry Applications

- Institutional-grade alpha generation
- Portfolio optimization and risk management
- Real-time trading system integration
- Regulatory compliance and reporting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{alpha_signal_decomposition,
  title={Alpha Signal Decomposition under Regime Shifts using Sparse Bayesian Learning},
  author={Abhishek Tiwari},
  year={2025},
  url={https://github.com/Abhishek282001Tiwari/Alpha_Signal_Decomposition_under_Regime_Shifts_using_Sparse_Bayesian_Learning}
}
```

## Contact

For technical support, collabration: 

- **Author**: Abhishek Tiwari
- **Email**: abhishekt282001@gmail.com
- **Repository**: https://github.com/Abhishek282001Tiwari/Alpha_Signal_Decomposition_under_Regime_Shifts_using_Sparse_Bayesian_Learning
- **Documentation**: https://abhishek282001tiwari.github.io/Alpha_Signal_Decomposition_under_Regime_Shifts_using_Sparse_Bayesian_Learning
