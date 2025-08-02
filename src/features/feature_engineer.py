import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for financial signals.
    Generates technical indicators, cross-sectional signals, and fundamental features.
    """
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50, 100]):
        """
        Initialize feature engineer with default lookback periods.
        
        Args:
            lookback_periods: List of lookback periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators for a single asset.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators added
        """
        result = df.copy()
        
        # Price-based indicators
        result['returns'] = df['Close'].pct_change()
        result['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for period in self.lookback_periods:
            result[f'sma_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
            result[f'ema_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
            result[f'price_to_sma_{period}'] = df['Close'] / result[f'sma_{period}']
            
        # Momentum indicators
        result['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
        result['rsi_30'] = ta.momentum.rsi(df['Close'], window=30)
        result['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        result['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        result['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # MACD
        result['macd'] = ta.trend.macd_diff(df['Close'])
        result['macd_signal'] = ta.trend.macd_signal(df['Close'])
        result['macd_histogram'] = ta.trend.macd_diff(df['Close'])
        
        # Bollinger Bands
        result['bb_upper'] = ta.volatility.bollinger_hband(df['Close'])
        result['bb_lower'] = ta.volatility.bollinger_lband(df['Close'])
        result['bb_middle'] = ta.volatility.bollinger_mavg(df['Close'])
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_position'] = (df['Close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # Volatility indicators
        result['atr_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        for period in [5, 10, 20]:
            result[f'realized_vol_{period}'] = result['returns'].rolling(period).std() * np.sqrt(252)
            result[f'parkinson_vol_{period}'] = self._parkinson_volatility(df, period)
        
        # Volume indicators
        if 'Volume' in df.columns:
            result['volume_sma_20'] = ta.volume.sma_ease_of_movement(df['High'], df['Low'], df['Volume'])
            result['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            result['price_volume'] = df['Close'] * df['Volume']
            result['vwap'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # On-Balance Volume
            result['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            result['obv_signal'] = result['obv'].rolling(10).mean()
        
        # Support and Resistance
        for period in [20, 50]:
            result[f'high_{period}'] = df['High'].rolling(period).max()
            result[f'low_{period}'] = df['Low'].rolling(period).min()
            result[f'high_distance_{period}'] = (df['Close'] - result[f'high_{period}']) / result[f'high_{period}']
            result[f'low_distance_{period}'] = (df['Close'] - result[f'low_{period}']) / result[f'low_{period}']
        
        # Momentum signals
        for period in self.lookback_periods:
            result[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            result[f'roc_{period}'] = ta.momentum.roc(df['Close'], window=period)
        
        return result
    
    def _parkinson_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        hl_ratio = np.log(df['High'] / df['Low'])
        return np.sqrt(hl_ratio.rolling(window).mean() * 252 / (4 * np.log(2)))
    
    def create_cross_sectional_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create cross-sectional signals across multiple assets.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with price data
        
        Returns:
            Dictionary with cross-sectional features added
        """
        # Combine all close prices
        close_prices = pd.DataFrame({symbol: df['Close'] for symbol, df in data.items()})
        returns = close_prices.pct_change()
        
        result_data = {}
        
        for symbol in data.keys():
            df = data[symbol].copy()
            
            # Relative strength signals
            for period in [10, 20, 50]:
                # Relative performance vs universe
                universe_return = returns.mean(axis=1).rolling(period).mean()
                asset_return = returns[symbol].rolling(period).mean()
                df[f'relative_strength_{period}'] = asset_return - universe_return
                
                # Percentile ranking
                rolling_returns = returns.rolling(period).mean()
                df[f'percentile_rank_{period}'] = rolling_returns[symbol].rolling(period).rank(pct=True)
            
            # Cross-sectional momentum
            for period in [1, 5, 10]:
                momentum = returns.rolling(period).mean()
                df[f'cs_momentum_{period}'] = momentum[symbol].rank(pct=True)
            
            # Sector rotation signals (simplified - assumes all assets in same sector)
            df['sector_momentum'] = returns[symbol].rolling(20).mean().rank(pct=True)
            
            # Mean reversion signals
            for period in [5, 10]:
                z_score = (returns[symbol] - returns[symbol].rolling(period).mean()) / returns[symbol].rolling(period).std()
                df[f'mean_reversion_{period}'] = -z_score  # Negative for mean reversion
            
            result_data[symbol] = df
        
        return result_data
    
    def create_fundamental_signals(self, 
                                 price_data: pd.DataFrame, 
                                 fundamental_data: Dict) -> pd.DataFrame:
        """
        Create fundamental analysis signals.
        
        Args:
            price_data: DataFrame with price data
            fundamental_data: Dictionary with fundamental data
        
        Returns:
            DataFrame with fundamental signals
        """
        result = price_data.copy()
        
        if not fundamental_data:
            return result
        
        try:
            overview = fundamental_data.get('overview', pd.DataFrame())
            income_stmt = fundamental_data.get('income_statement', pd.DataFrame())
            balance_sheet = fundamental_data.get('balance_sheet', pd.DataFrame())
            
            if not overview.empty:
                # Valuation ratios
                pe_ratio = pd.to_numeric(overview.get('PERatio', np.nan), errors='coerce')
                pb_ratio = pd.to_numeric(overview.get('PriceToBookRatio', np.nan), errors='coerce')
                ps_ratio = pd.to_numeric(overview.get('PriceToSalesRatioTTM', np.nan), errors='coerce')
                
                result['pe_ratio'] = pe_ratio
                result['pb_ratio'] = pb_ratio
                result['ps_ratio'] = ps_ratio
                
                # Quality metrics
                roe = pd.to_numeric(overview.get('ReturnOnEquityTTM', np.nan), errors='coerce')
                roa = pd.to_numeric(overview.get('ReturnOnAssetsTTM', np.nan), errors='coerce')
                
                result['roe'] = roe
                result['roa'] = roa
                
                # Dividend metrics
                dividend_yield = pd.to_numeric(overview.get('DividendYield', np.nan), errors='coerce')
                result['dividend_yield'] = dividend_yield
            
            # Growth signals from income statement
            if not income_stmt.empty and len(income_stmt) > 1:
                # Revenue growth
                revenues = pd.to_numeric(income_stmt['totalRevenue'], errors='coerce')
                if len(revenues) >= 2:
                    revenue_growth = (revenues.iloc[-1] - revenues.iloc[-2]) / revenues.iloc[-2]
                    result['revenue_growth'] = revenue_growth
                
                # Earnings growth
                earnings = pd.to_numeric(income_stmt['netIncome'], errors='coerce')
                if len(earnings) >= 2:
                    earnings_growth = (earnings.iloc[-1] - earnings.iloc[-2]) / abs(earnings.iloc[-2])
                    result['earnings_growth'] = earnings_growth
            
        except Exception as e:
            self.logger.warning(f"Error processing fundamental data: {str(e)}")
        
        return result
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced volatility-based features.
        
        Args:
            df: DataFrame with returns data
        
        Returns:
            DataFrame with volatility features
        """
        result = df.copy()
        returns = result['returns'] if 'returns' in result.columns else result['Close'].pct_change()
        
        # GARCH-like features
        for window in [10, 20, 50]:
            # Exponentially weighted volatility
            result[f'ewm_vol_{window}'] = returns.ewm(span=window).std() * np.sqrt(252)
            
            # Volatility of volatility
            vol = returns.rolling(window).std()
            result[f'vol_of_vol_{window}'] = vol.rolling(window).std()
            
            # Skewness and Kurtosis
            result[f'skewness_{window}'] = returns.rolling(window).skew()
            result[f'kurtosis_{window}'] = returns.rolling(window).kurt()
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            result[f'downside_vol_{window}'] = np.sqrt(
                downside_returns.rolling(window).var() * 252
            )
        
        # VaR and CVaR approximations
        for confidence in [0.05, 0.01]:
            for window in [20, 50]:
                rolling_returns = returns.rolling(window)
                result[f'var_{int(confidence*100)}_{window}'] = rolling_returns.quantile(confidence)
                
                # Conditional VaR (Expected Shortfall)
                var_level = rolling_returns.quantile(confidence)
                cvar = rolling_returns.apply(
                    lambda x: x[x <= x.quantile(confidence)].mean()
                )
                result[f'cvar_{int(confidence*100)}_{window}'] = cvar
        
        return result
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create microstructure and liquidity features.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with microstructure features
        """
        result = df.copy()
        
        # Bid-ask spread proxy (High-Low spread)
        result['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        result['hl_spread_ma'] = result['hl_spread'].rolling(20).mean()
        
        if 'Volume' in df.columns:
            # Volume-based liquidity measures
            result['volume_volatility'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()
            
            # Amihud illiquidity measure
            result['amihud_illiq'] = abs(result['returns']) / (df['Volume'] * df['Close'])
            result['amihud_illiq_ma'] = result['amihud_illiq'].rolling(20).mean()
            
            # Price impact measures
            result['price_impact'] = result['returns'] / (df['Volume'] + 1e-8)
            
            # Roll's spread estimator
            cov_returns = result['returns'].rolling(2).cov().iloc[1::2]
            result['roll_spread'] = 2 * np.sqrt(-cov_returns.clip(upper=0))
        
        # Intraday patterns
        result['high_low_ratio'] = df['High'] / df['Low']
        result['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return result
    
    def calculate_information_coefficient(self, 
                                       signals: pd.DataFrame, 
                                       forward_returns: pd.Series,
                                       method: str = 'spearman') -> pd.Series:
        """
        Calculate information coefficient for signal evaluation.
        
        Args:
            signals: DataFrame with signal values
            forward_returns: Series with forward returns
            method: Correlation method ('spearman' or 'pearson')
        
        Returns:
            Series with IC values for each signal
        """
        ic_values = {}
        
        for column in signals.columns:
            if method == 'spearman':
                ic, _ = stats.spearmanr(signals[column].dropna(), 
                                      forward_returns[signals[column].dropna().index])
            else:
                ic = signals[column].corr(forward_returns)
            
            ic_values[column] = ic if not np.isnan(ic) else 0
        
        return pd.Series(ic_values)
    
    def standardize_features(self, 
                           df: pd.DataFrame, 
                           method: str = 'zscore',
                           window: Optional[int] = None) -> pd.DataFrame:
        """
        Standardize features to prevent look-ahead bias.
        
        Args:
            df: DataFrame with features
            method: Standardization method ('zscore', 'robust', 'minmax')
            window: Rolling window for standardization (None for expanding)
        
        Returns:
            Standardized DataFrame
        """
        result = df.copy()
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'zscore':
                if window:
                    rolling_mean = result[column].rolling(window, min_periods=10).mean()
                    rolling_std = result[column].rolling(window, min_periods=10).std()
                    result[column] = (result[column] - rolling_mean) / rolling_std
                else:
                    expanding_mean = result[column].expanding(min_periods=10).mean()
                    expanding_std = result[column].expanding(min_periods=10).std()
                    result[column] = (result[column] - expanding_mean) / expanding_std
            
            elif method == 'robust':
                if window:
                    rolling_median = result[column].rolling(window, min_periods=10).median()
                    rolling_mad = result[column].rolling(window, min_periods=10).apply(
                        lambda x: np.median(np.abs(x - np.median(x)))
                    )
                    result[column] = (result[column] - rolling_median) / rolling_mad
                else:
                    expanding_median = result[column].expanding(min_periods=10).median()
                    expanding_mad = result[column].expanding(min_periods=10).apply(
                        lambda x: np.median(np.abs(x - np.median(x)))
                    )
                    result[column] = (result[column] - expanding_median) / expanding_mad
        
        return result
    
    def feature_selection_by_ic(self, 
                               features: pd.DataFrame, 
                               returns: pd.Series,
                               min_ic: float = 0.02,
                               lookback: int = 252) -> List[str]:
        """
        Select features based on information coefficient stability.
        
        Args:
            features: DataFrame with feature values
            returns: Series with forward returns
            min_ic: Minimum absolute IC threshold
            lookback: Lookback period for IC calculation
        
        Returns:
            List of selected feature names
        """
        selected_features = []
        
        for feature in features.columns:
            ic_series = []
            
            # Calculate rolling IC
            for i in range(lookback, len(features)):
                start_idx = i - lookback
                end_idx = i
                
                feature_subset = features[feature].iloc[start_idx:end_idx]
                returns_subset = returns.iloc[start_idx:end_idx]
                
                # Align indices
                common_idx = feature_subset.dropna().index.intersection(returns_subset.dropna().index)
                if len(common_idx) > 10:
                    ic, _ = stats.spearmanr(feature_subset[common_idx], returns_subset[common_idx])
                    if not np.isnan(ic):
                        ic_series.append(ic)
            
            if ic_series:
                ic_mean = np.mean(ic_series)
                ic_std = np.std(ic_series)
                ic_stability = abs(ic_mean) / (ic_std + 1e-8)
                
                if abs(ic_mean) >= min_ic and ic_stability > 1.0:
                    selected_features.append(feature)
                    self.logger.info(f"Selected feature {feature}: IC={ic_mean:.3f}, Stability={ic_stability:.3f}")
        
        return selected_features