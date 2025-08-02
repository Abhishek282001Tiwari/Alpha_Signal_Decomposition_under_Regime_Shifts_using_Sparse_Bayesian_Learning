import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    lookback_periods: List[int] = None
    include_technical: bool = True
    include_fundamental: bool = True
    include_macro: bool = True
    include_cross_sectional: bool = True
    normalization_method: str = 'standard'  # 'standard', 'robust', 'quantile'
    feature_selection_method: str = 'mutual_info'  # 'f_test', 'mutual_info', 'pca'
    max_features: Optional[int] = None

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering pipeline for regime-conditional alpha signals
    with comprehensive technical, fundamental, macro, and cross-sectional features.
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        if self.config.lookback_periods is None:
            self.config.lookback_periods = [5, 10, 20, 50, 100, 252]
        
        self.logger = logging.getLogger(__name__)
        self.fitted_scalers = {}
        self.feature_importance = {}
        self.feature_stats = {}
        
        # Initialize scalers based on config
        self.scaler = self._initialize_scaler()
        
    def _initialize_scaler(self):
        """Initialize scaler based on configuration."""
        if self.config.normalization_method == 'robust':
            return RobustScaler()
        elif self.config.normalization_method == 'quantile':
            return QuantileTransformer(output_distribution='normal')
        else:
            return StandardScaler()
    
    def engineer_features(self, 
                         market_data: Dict[str, pd.DataFrame],
                         fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
                         macro_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Engineer comprehensive features from multiple data sources.
        
        Args:
            market_data: Dictionary of market data by symbol
            fundamental_data: Optional fundamental data by symbol
            macro_data: Optional macroeconomic data
        
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting comprehensive feature engineering")
        
        all_features = []
        
        # Process each symbol
        for symbol, data in market_data.items():
            try:
                self.logger.info(f"Engineering features for {symbol}")
                
                # Technical features
                if self.config.include_technical:
                    technical_features = self._create_technical_features(data, symbol)
                    all_features.append(technical_features)
                
                # Fundamental features
                if self.config.include_fundamental and fundamental_data and symbol in fundamental_data:
                    fundamental_features = self._create_fundamental_features(
                        fundamental_data[symbol], symbol
                    )
                    if not fundamental_features.empty:
                        all_features.append(fundamental_features)
                
            except Exception as e:
                self.logger.error(f"Error engineering features for {symbol}: {str(e)}")
                continue
        
        if not all_features:
            self.logger.warning("No features were generated")
            return pd.DataFrame()
        
        # Combine all features
        combined_features = pd.concat(all_features, axis=0, sort=False)
        
        # Add macro features
        if self.config.include_macro and macro_data is not None:
            combined_features = self._add_macro_features(combined_features, macro_data)
        
        # Add cross-sectional features
        if self.config.include_cross_sectional:
            combined_features = self._create_cross_sectional_features(combined_features)
        
        # Feature selection and normalization
        combined_features = self._postprocess_features(combined_features)
        
        self.logger.info(f"Feature engineering completed. Generated {len(combined_features.columns)} features")
        return combined_features
    
    def _create_technical_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive technical features."""
        try:
            features = pd.DataFrame(index=data.index)
            features['Symbol'] = symbol
            
            # Basic price features
            features = self._add_price_features(features, data)
            
            # Momentum features
            features = self._add_momentum_features(features, data)
            
            # Volatility features
            features = self._add_volatility_features(features, data)
            
            # Volume features
            features = self._add_volume_features(features, data)
            
            # Pattern recognition features
            features = self._add_pattern_features(features, data)
            
            # Statistical features
            features = self._add_statistical_features(features, data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating technical features for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_price_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price ratios
        if all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
            features['high_low_ratio'] = data['High'] / data['Low']
            features['close_open_ratio'] = data['Close'] / data['Open']
            features['high_close_ratio'] = data['High'] / data['Close']
            features['low_close_ratio'] = data['Low'] / data['Close']
        
        # Moving averages and ratios
        for period in self.config.lookback_periods:
            sma = data['Close'].rolling(window=period).mean()
            ema = data['Close'].ewm(span=period).mean()
            
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            features[f'price_to_sma_{period}'] = data['Close'] / sma
            features[f'price_to_ema_{period}'] = data['Close'] / ema
            features[f'sma_slope_{period}'] = sma.pct_change(periods=5)
            
            # Bollinger Bands
            bb_std = data['Close'].rolling(window=period).std()
            bb_upper = sma + (2 * bb_std)
            bb_lower = sma - (2 * bb_std)
            features[f'bb_position_{period}'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
            features[f'bb_width_{period}'] = (bb_upper - bb_lower) / sma
        
        # Price channels
        for period in [20, 50]:
            high_channel = data['High'].rolling(window=period).max()
            low_channel = data['Low'].rolling(window=period).min()
            features[f'price_channel_position_{period}'] = (data['Close'] - low_channel) / (high_channel - low_channel)
        
        return features
    
    def _add_momentum_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        # RSI
        for period in [14, 30]:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            features[f'roc_{period}'] = data['Close'].pct_change(periods=period)
        
        # Williams %R
        for period in [14, 28]:
            high_n = data['High'].rolling(window=period).max()
            low_n = data['Low'].rolling(window=period).min()
            features[f'williams_r_{period}'] = -100 * (high_n - data['Close']) / (high_n - low_n)
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_n = data['Low'].rolling(window=period).min()
            high_n = data['High'].rolling(window=period).max()
            features[f'stoch_k_{period}'] = 100 * (data['Close'] - low_n) / (high_n - low_n)
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(window=3).mean()
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # Historical volatility
        for period in self.config.lookback_periods:
            returns = data['Close'].pct_change()
            features[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / features[f'volatility_{period}'].rolling(window=period*2).mean()
        
        # GARCH-like volatility
        returns = data['Close'].pct_change()
        features['volatility_garch'] = returns.rolling(window=20).apply(
            lambda x: np.sqrt(0.94 * x.var() + 0.06 * x.iloc[-1]**2)
        )
        
        # True Range and ATR
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            for period in [14, 21]:
                features[f'atr_{period}'] = true_range.rolling(window=period).mean()
                features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / data['Close']
        
        # Volatility clustering
        returns_sq = returns ** 2
        features['volatility_clustering'] = returns_sq.rolling(window=20).corr(returns_sq.shift(1))
        
        return features
    
    def _add_volume_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'Volume' not in data.columns:
            return features
        
        # Volume moving averages
        for period in [10, 20, 50]:
            vol_ma = data['Volume'].rolling(window=period).mean()
            features[f'volume_ma_{period}'] = vol_ma
            features[f'volume_ratio_{period}'] = data['Volume'] / vol_ma
        
        # On Balance Volume (OBV)
        obv = np.zeros(len(data))
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - data['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        features['obv'] = obv
        features['obv_ma'] = pd.Series(obv, index=data.index).rolling(window=20).mean()
        
        # Volume Price Trend (VPT)
        returns = data['Close'].pct_change()
        features['vpt'] = (returns * data['Volume']).cumsum()
        
        # Accumulation/Distribution Line
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
            clv = clv.fillna(0)
            features['ad_line'] = (clv * data['Volume']).cumsum()
        
        # Volume-weighted prices
        features['vwap_20'] = (data['Close'] * data['Volume']).rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()
        
        return features
    
    def _add_pattern_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            return features
        
        # Candlestick patterns (simplified)
        body = data['Close'] - data['Open']
        upper_shadow = data['High'] - np.maximum(data['Open'], data['Close'])
        lower_shadow = np.minimum(data['Open'], data['Close']) - data['Low']
        
        features['body_size'] = np.abs(body) / (data['High'] - data['Low'])
        features['upper_shadow_ratio'] = upper_shadow / (data['High'] - data['Low'])
        features['lower_shadow_ratio'] = lower_shadow / (data['High'] - data['Low'])
        
        # Doji pattern
        features['doji'] = (np.abs(body) / (data['High'] - data['Low']) < 0.1).astype(int)
        
        # Hammer/Hanging Man
        features['hammer'] = (
            (lower_shadow > 2 * np.abs(body)) & 
            (upper_shadow < 0.1 * (data['High'] - data['Low']))
        ).astype(int)
        
        # Engulfing patterns
        prev_body = body.shift(1)
        features['bullish_engulfing'] = (
            (body > 0) & (prev_body < 0) & 
            (data['Close'] > data['Open'].shift(1)) & 
            (data['Open'] < data['Close'].shift(1))
        ).astype(int)
        
        features['bearish_engulfing'] = (
            (body < 0) & (prev_body > 0) & 
            (data['Close'] < data['Open'].shift(1)) & 
            (data['Open'] > data['Close'].shift(1))
        ).astype(int)
        
        # Gap analysis
        features['gap_up'] = (data['Open'] > data['High'].shift(1)).astype(int)
        features['gap_down'] = (data['Open'] < data['Low'].shift(1)).astype(int)
        
        return features
    
    def _add_statistical_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        returns = data['Close'].pct_change()
        
        # Rolling statistics
        for period in [20, 50]:
            features[f'skewness_{period}'] = returns.rolling(window=period).skew()
            features[f'kurtosis_{period}'] = returns.rolling(window=period).kurt()
            features[f'var_95_{period}'] = returns.rolling(window=period).quantile(0.05)
            features[f'var_99_{period}'] = returns.rolling(window=period).quantile(0.01)
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            features[f'autocorr_lag_{lag}'] = returns.rolling(window=50).apply(
                lambda x: x.autocorr(lag=lag)
            )
        
        # Hurst exponent (simplified)
        def hurst_exponent(ts, max_lag=20):
            if len(ts) < max_lag * 2:
                return np.nan
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        features['hurst_exponent'] = returns.rolling(window=100).apply(
            lambda x: hurst_exponent(x.values)
        )
        
        return features
    
    def _create_fundamental_features(self, fundamental_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create fundamental features."""
        try:
            # This is a simplified version - in practice, you'd extract from financial statements
            features = pd.DataFrame(index=fundamental_data.index)
            features['Symbol'] = symbol
            
            # Key ratios
            ratio_columns = [col for col in fundamental_data.columns if any(
                x in col.lower() for x in ['ratio', 'margin', 'roe', 'roa', 'eps', 'pe', 'pb']
            )]
            
            for col in ratio_columns:
                if col in fundamental_data.columns:
                    features[f'fundamental_{col}'] = fundamental_data[col]
                    
                    # Trend analysis
                    if len(fundamental_data) > 1:
                        features[f'fundamental_{col}_trend'] = fundamental_data[col].pct_change()
                        features[f'fundamental_{col}_rank'] = fundamental_data[col].rank(pct=True)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating fundamental features for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_macro_features(self, features: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic features."""
        try:
            # Resample macro data to match features frequency
            if isinstance(features.index, pd.DatetimeIndex) and isinstance(macro_data.index, pd.DatetimeIndex):
                # Forward fill macro data to daily frequency
                macro_daily = macro_data.resample('D').ffill()
                
                # Merge with features
                for col in macro_daily.columns:
                    if col not in features.columns:
                        merged = features.join(macro_daily[col], how='left')
                        features[f'macro_{col}'] = merged[col]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding macro features: {str(e)}")
            return features
    
    def _create_cross_sectional_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create cross-sectional features."""
        try:
            if 'Symbol' not in features.columns:
                return features
            
            # Group by date for cross-sectional analysis
            date_groups = features.groupby(features.index)
            
            # Calculate cross-sectional ranks and z-scores
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in features.columns and features[col].notna().sum() > 0:
                    # Cross-sectional rank
                    features[f'{col}_cs_rank'] = date_groups[col].rank(pct=True, method='min')
                    
                    # Cross-sectional z-score
                    features[f'{col}_cs_zscore'] = date_groups[col].apply(
                        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                    )
            
            # Sector relative features (if sector data available)
            if hasattr(features, 'Sector') and 'Sector' in features.columns:
                for col in numeric_columns[:10]:  # Limit to avoid too many features
                    if col in features.columns:
                        features[f'{col}_sector_rank'] = features.groupby(['Sector', features.index])[col].rank(pct=True)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating cross-sectional features: {str(e)}")
            return features
    
    def _postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Post-process features with selection and normalization."""
        try:
            # Remove non-feature columns for processing
            feature_cols = [col for col in features.columns if col not in ['Symbol', 'Sector']]
            
            if not feature_cols:
                return features
            
            # Handle infinite and missing values
            features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan)
            
            # Forward fill and backward fill to handle some missing values
            features[feature_cols] = features[feature_cols].fillna(method='ffill').fillna(method='bfill')
            
            # Remove features with too many missing values
            missing_threshold = 0.5  # 50% missing
            valid_features = []
            
            for col in feature_cols:
                missing_pct = features[col].isnull().sum() / len(features)
                if missing_pct < missing_threshold:
                    valid_features.append(col)
                else:
                    self.logger.warning(f"Removing feature {col} due to {missing_pct:.2%} missing values")
            
            if not valid_features:
                self.logger.error("No valid features remaining after missing value filter")
                return features
            
            # Feature selection
            if self.config.max_features and len(valid_features) > self.config.max_features:
                valid_features = self._select_features(features[valid_features], valid_features)
            
            # Keep only valid features plus metadata columns
            metadata_cols = [col for col in features.columns if col not in feature_cols]
            final_features = metadata_cols + valid_features
            features = features[final_features]
            
            # Normalization
            if valid_features:
                features[valid_features] = self._normalize_features(features[valid_features])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in feature post-processing: {str(e)}")
            return features
    
    def _select_features(self, feature_data: pd.DataFrame, feature_names: List[str]) -> List[str]:
        """Select most informative features."""
        try:
            # For feature selection, we need a target variable
            # As a proxy, use forward returns if available
            if 'returns' in feature_data.columns:
                target = feature_data['returns'].shift(-1)  # Next period return
                feature_subset = feature_data.drop(['returns'], axis=1)
            else:
                # If no returns, use PCA for feature selection
                return self._select_features_pca(feature_data, feature_names)
            
            # Remove rows with missing target
            valid_idx = target.notna()
            if valid_idx.sum() < len(feature_data) * 0.1:  # Less than 10% valid data
                return feature_names[:self.config.max_features]
            
            X = feature_subset[valid_idx].fillna(0)
            y = target[valid_idx]
            
            if self.config.feature_selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.max_features)
            else:
                selector = SelectKBest(score_func=f_regression, k=self.config.max_features)
            
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store feature importance
            self.feature_importance = dict(zip(X.columns, selector.scores_))
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            return feature_names[:self.config.max_features] if self.config.max_features else feature_names
    
    def _select_features_pca(self, feature_data: pd.DataFrame, feature_names: List[str]) -> List[str]:
        """Select features using PCA."""
        try:
            X = feature_data.fillna(0)
            
            # Use PCA to find most important features
            pca = PCA(n_components=min(self.config.max_features, len(feature_names)))
            pca.fit(X)
            
            # Get feature importance from principal components
            feature_importance = np.abs(pca.components_).mean(axis=0)
            
            # Select top features
            top_indices = np.argsort(feature_importance)[-self.config.max_features:]
            selected_features = [feature_names[i] for i in top_indices]
            
            self.feature_importance = dict(zip(feature_names, feature_importance))
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in PCA feature selection: {str(e)}")
            return feature_names[:self.config.max_features]
    
    def _normalize_features(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features."""
        try:
            # Fit scaler if not already fitted
            scaler_key = 'main_scaler'
            if scaler_key not in self.fitted_scalers:
                valid_data = feature_data.dropna()
                if len(valid_data) > 0:
                    self.fitted_scalers[scaler_key] = self.scaler.fit(valid_data)
                else:
                    return feature_data
            
            # Transform features
            normalized_data = feature_data.copy()
            valid_idx = feature_data.notna().all(axis=1)
            
            if valid_idx.sum() > 0:
                normalized_data.loc[valid_idx] = self.fitted_scalers[scaler_key].transform(
                    feature_data.loc[valid_idx]
                )
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Error in feature normalization: {str(e)}")
            return feature_data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.copy()
    
    def get_feature_statistics(self, features: pd.DataFrame) -> Dict[str, Dict]:
        """Get comprehensive feature statistics."""
        stats = {}
        
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in features.columns:
                series = features[col].dropna()
                if len(series) > 0:
                    stats[col] = {
                        'count': len(series),
                        'mean': series.mean(),
                        'std': series.std(),
                        'min': series.min(),
                        'max': series.max(),
                        'median': series.median(),
                        'skewness': series.skew(),
                        'kurtosis': series.kurt(),
                        'missing_pct': features[col].isnull().sum() / len(features)
                    }
        
        return stats