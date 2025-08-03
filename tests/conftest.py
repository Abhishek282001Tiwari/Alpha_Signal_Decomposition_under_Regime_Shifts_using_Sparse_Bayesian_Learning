# pytest configuration and fixtures for Alpha Signal Decomposition tests

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    data = {}
    for symbol in symbols:
        np.random.seed(42)  # For reproducible tests
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data[symbol] = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    return data


@pytest.fixture
def sample_regime_data():
    """Generate sample regime probability data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_regimes = 3
    
    # Generate smooth regime transitions
    regime_probs = np.random.dirichlet([1, 1, 1], len(dates))
    
    return pd.DataFrame(
        regime_probs,
        index=dates,
        columns=[f'Regime_{i}' for i in range(n_regimes)]
    )


@pytest.fixture
def sample_features():
    """Generate sample feature matrix for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    features_list = []
    for symbol in symbols:
        n_features = 20
        feature_data = np.random.randn(len(dates), n_features)
        
        feature_df = pd.DataFrame(
            feature_data,
            index=dates,
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        feature_df['Symbol'] = symbol
        features_list.append(feature_df)
    
    return pd.concat(features_list, axis=0)