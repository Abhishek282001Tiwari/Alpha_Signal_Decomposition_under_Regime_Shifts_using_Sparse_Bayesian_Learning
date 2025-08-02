import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats, linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import Ridge
import logging
import warnings

warnings.filterwarnings('ignore')

class AlphaDecomposer:
    """
    Multi-factor alpha decomposition system for regime-dependent alpha signal analysis.
    Performs attribution analysis, factor orthogonalization, and alpha extraction.
    """
    
    def __init__(self, 
                 factor_models: List[str] = ['fama_french', 'momentum', 'quality', 'volatility'],
                 custom_factors: Optional[Dict[str, pd.Series]] = None,
                 regime_dependent: bool = True,
                 orthogonalize_factors: bool = True):
        """
        Initialize Alpha Decomposer.
        
        Args:
            factor_models: List of factor models to include
            custom_factors: Dictionary of custom factor time series
            regime_dependent: Whether to perform regime-dependent decomposition
            orthogonalize_factors: Whether to orthogonalize factor loadings
        """
        self.factor_models = factor_models
        self.custom_factors = custom_factors or {}
        self.regime_dependent = regime_dependent
        self.orthogonalize_factors = orthogonalize_factors
        
        # Factor data and loadings
        self.factor_data = {}
        self.factor_loadings = {}
        self.regime_factor_loadings = {}
        
        # Alpha components
        self.alpha_components = {}
        self.regime_alpha_components = {}
        self.residual_alpha = None
        
        # Attribution analysis
        self.attribution_results = {}
        self.performance_attribution = {}
        
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def fit_decomposition(self, 
                         returns: pd.DataFrame,
                         regime_probabilities: Optional[pd.DataFrame] = None,
                         market_data: Optional[pd.DataFrame] = None) -> 'AlphaDecomposer':
        """
        Fit alpha decomposition model to return data.
        
        Args:
            returns: DataFrame with asset returns (assets as columns, dates as index)
            regime_probabilities: Optional regime probability DataFrame
            market_data: Optional market data for factor construction
        
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting alpha decomposition model")
        
        # Validate input data
        returns = self._validate_returns_data(returns)
        
        # Construct factor data
        self._construct_factor_data(returns, market_data)
        
        # Align data
        aligned_data = self._align_data(returns, regime_probabilities)
        returns_aligned = aligned_data['returns']
        regime_probs_aligned = aligned_data.get('regime_probabilities')
        factors_aligned = aligned_data['factors']
        
        # Fit factor models
        if self.regime_dependent and regime_probs_aligned is not None:
            self._fit_regime_dependent_factors(returns_aligned, factors_aligned, regime_probs_aligned)
        else:
            self._fit_global_factors(returns_aligned, factors_aligned)
        
        # Extract alpha components
        self._extract_alpha_components(returns_aligned, factors_aligned, regime_probs_aligned)
        
        # Perform attribution analysis
        self._perform_attribution_analysis(returns_aligned, factors_aligned, regime_probs_aligned)
        
        self.logger.info("Alpha decomposition completed")
        return self
    
    def _validate_returns_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean returns data."""
        # Remove assets with insufficient data
        min_observations = max(252, len(returns) // 4)  # At least 1 year or 25% of data
        valid_assets = returns.count() >= min_observations
        returns_clean = returns.loc[:, valid_assets]
        
        # Handle missing values
        returns_clean = returns_clean.fillna(0)
        
        self.logger.info(f"Using {len(returns_clean.columns)} assets with sufficient data")
        return returns_clean
    
    def _construct_factor_data(self, returns: pd.DataFrame, market_data: Optional[pd.DataFrame]):
        """Construct factor time series data."""
        
        # Market factor (equal-weighted or cap-weighted portfolio)
        market_return = returns.mean(axis=1)  # Simple equal-weighted market
        self.factor_data['market'] = market_return
        
        # Fama-French factors
        if 'fama_french' in self.factor_models:
            ff_factors = self._construct_fama_french_factors(returns, market_data)
            self.factor_data.update(ff_factors)
        
        # Momentum factors
        if 'momentum' in self.factor_models:
            momentum_factors = self._construct_momentum_factors(returns)
            self.factor_data.update(momentum_factors)
        
        # Quality factors
        if 'quality' in self.factor_models:
            quality_factors = self._construct_quality_factors(returns, market_data)
            self.factor_data.update(quality_factors)
        
        # Volatility factors
        if 'volatility' in self.factor_models:
            volatility_factors = self._construct_volatility_factors(returns)
            self.factor_data.update(volatility_factors)
        
        # Add custom factors
        for factor_name, factor_series in self.custom_factors.items():
            if isinstance(factor_series, pd.Series):
                self.factor_data[f'custom_{factor_name}'] = factor_series
        
        # Create factor DataFrame
        factor_df = pd.DataFrame(self.factor_data)
        factor_df = factor_df.reindex(returns.index).fillna(0)
        
        # Orthogonalize factors if requested
        if self.orthogonalize_factors:
            factor_df = self._orthogonalize_factors(factor_df)
        
        self.factor_data = {col: factor_df[col] for col in factor_df.columns}
    
    def _construct_fama_french_factors(self, returns: pd.DataFrame, 
                                     market_data: Optional[pd.DataFrame]) -> Dict[str, pd.Series]:
        """Construct Fama-French factors (simplified implementation)."""
        factors = {}
        
        # SMB (Small Minus Big) - size factor
        # Use cross-sectional dispersion as proxy
        size_proxy = returns.std(axis=1)  # Cross-sectional volatility as size proxy
        factors['SMB'] = -size_proxy  # Negative because higher vol often means smaller cap
        
        # HML (High Minus Low) - value factor  
        # Use momentum reversal as value proxy
        short_term_mom = returns.rolling(window=21).mean()  # 1-month momentum
        long_term_mom = returns.rolling(window=252).mean()   # 1-year momentum
        
        # Value factor: mean reversion tendency
        value_proxy = -(short_term_mom.mean(axis=1) - long_term_mom.mean(axis=1))
        factors['HML'] = value_proxy
        
        return factors
    
    def _construct_momentum_factors(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """Construct momentum factors."""
        factors = {}
        
        # Short-term momentum (1-month)
        short_momentum = returns.rolling(window=21).mean().mean(axis=1)
        factors['momentum_1m'] = short_momentum
        
        # Medium-term momentum (3-month)
        medium_momentum = returns.rolling(window=63).mean().mean(axis=1)
        factors['momentum_3m'] = medium_momentum
        
        # Long-term momentum (12-month)
        long_momentum = returns.rolling(window=252).mean().mean(axis=1)
        factors['momentum_12m'] = long_momentum
        
        # Momentum factor (12-1 month)
        factors['momentum'] = long_momentum - short_momentum
        
        return factors
    
    def _construct_quality_factors(self, returns: pd.DataFrame,
                                 market_data: Optional[pd.DataFrame]) -> Dict[str, pd.Series]:
        """Construct quality factors."""
        factors = {}
        
        # Profitability proxy: consistency of returns
        return_consistency = -returns.rolling(window=63).std().mean(axis=1)
        factors['profitability'] = return_consistency
        
        # Investment quality: negative autocorrelation (mean reversion)
        autocorr = returns.rolling(window=21).apply(lambda x: x.autocorr(lag=1)).mean(axis=1)
        factors['investment_quality'] = -autocorr
        
        return factors
    
    def _construct_volatility_factors(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """Construct volatility factors."""
        factors = {}
        
        # Volatility factor
        volatility = returns.rolling(window=21).std().mean(axis=1)
        factors['volatility'] = volatility
        
        # Low volatility factor (negative of volatility)
        factors['low_volatility'] = -volatility
        
        # Volatility risk premium
        vol_premium = volatility - volatility.rolling(window=252).mean()
        factors['volatility_risk_premium'] = vol_premium
        
        return factors
    
    def _orthogonalize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Orthogonalize factors using Gram-Schmidt process."""
        
        # Start with market factor as base
        orthogonal_factors = factor_df.copy()
        factor_names = list(factor_df.columns)
        
        if 'market' in factor_names:
            # Market factor remains unchanged
            base_factors = ['market']
            remaining_factors = [f for f in factor_names if f != 'market']
        else:
            base_factors = [factor_names[0]]
            remaining_factors = factor_names[1:]
        
        # Orthogonalize remaining factors
        for factor in remaining_factors:
            factor_series = factor_df[factor].dropna()
            
            # Regress against all previous factors
            base_factor_data = factor_df[base_factors].dropna()
            common_index = factor_series.index.intersection(base_factor_data.index)
            
            if len(common_index) > 10:
                X = base_factor_data.loc[common_index]
                y = factor_series.loc[common_index]
                
                # Ridge regression for stability
                ridge = Ridge(alpha=0.01)
                ridge.fit(X, y)
                
                # Subtract projection onto base factors
                predicted = ridge.predict(X)
                residual = y - predicted
                
                # Update factor series with orthogonalized version
                orthogonal_factors.loc[common_index, factor] = residual
            
            # Add this factor to base factors for next iteration
            base_factors.append(factor)
        
        return orthogonal_factors
    
    def _align_data(self, returns: pd.DataFrame, 
                   regime_probabilities: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align returns, factors, and regime data."""
        
        # Create factor DataFrame
        factor_df = pd.DataFrame(self.factor_data)
        
        # Find common index
        common_index = returns.index
        if regime_probabilities is not None:
            common_index = common_index.intersection(regime_probabilities.index)
        common_index = common_index.intersection(factor_df.index)
        
        # Align all data
        aligned_data = {
            'returns': returns.loc[common_index],
            'factors': factor_df.loc[common_index]
        }
        
        if regime_probabilities is not None:
            aligned_data['regime_probabilities'] = regime_probabilities.loc[common_index]
        
        return aligned_data
    
    def _fit_global_factors(self, returns: pd.DataFrame, factors: pd.DataFrame):
        """Fit global factor model across all periods."""
        
        n_assets = len(returns.columns)
        n_factors = len(factors.columns)
        
        self.factor_loadings = {}
        
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            
            # Align factor data
            common_index = asset_returns.index.intersection(factors.index)
            if len(common_index) < 50:  # Minimum observations
                continue
            
            y = asset_returns.loc[common_index]
            X = factors.loc[common_index]
            
            # Ridge regression for factor loadings
            ridge = Ridge(alpha=0.01)
            ridge.fit(X, y)
            
            self.factor_loadings[asset] = {
                'loadings': dict(zip(X.columns, ridge.coef_)),
                'alpha': ridge.intercept_,
                'r_squared': ridge.score(X, y)
            }
    
    def _fit_regime_dependent_factors(self, returns: pd.DataFrame, 
                                    factors: pd.DataFrame,
                                    regime_probabilities: pd.DataFrame):
        """Fit regime-dependent factor models."""
        
        n_regimes = regime_probabilities.shape[1]
        self.regime_factor_loadings = {}
        
        for regime in range(n_regimes):
            regime_loadings = {}
            regime_weights = regime_probabilities.iloc[:, regime]
            
            for asset in returns.columns:
                asset_returns = returns[asset].dropna()
                
                # Align data
                common_index = asset_returns.index.intersection(factors.index)
                common_index = common_index.intersection(regime_weights.index)
                
                if len(common_index) < 30:  # Minimum observations
                    continue
                
                y = asset_returns.loc[common_index]
                X = factors.loc[common_index]
                weights = regime_weights.loc[common_index]
                
                # Weighted ridge regression
                weighted_loadings = self._fit_weighted_regression(X, y, weights)
                regime_loadings[asset] = weighted_loadings
            
            self.regime_factor_loadings[regime] = regime_loadings
    
    def _fit_weighted_regression(self, X: pd.DataFrame, y: pd.Series, 
                               weights: pd.Series) -> Dict:
        """Fit weighted ridge regression."""
        
        # Weight the data
        w_sqrt = np.sqrt(weights + 1e-8)  # Add small constant for stability
        X_weighted = X.multiply(w_sqrt, axis=0)
        y_weighted = y * w_sqrt
        
        # Ridge regression
        ridge = Ridge(alpha=0.01)
        ridge.fit(X_weighted, y_weighted)
        
        # Calculate weighted R-squared
        y_pred = ridge.predict(X)
        ss_res = np.sum(weights * (y - y_pred)**2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'loadings': dict(zip(X.columns, ridge.coef_)),
            'alpha': ridge.intercept_,
            'r_squared': r_squared,
            'effective_samples': np.sum(weights)
        }
    
    def _extract_alpha_components(self, returns: pd.DataFrame, factors: pd.DataFrame,
                                regime_probabilities: Optional[pd.DataFrame]):
        """Extract alpha components from factor decomposition."""
        
        if self.regime_dependent and regime_probabilities is not None:
            self._extract_regime_alpha_components(returns, factors, regime_probabilities)
        else:
            self._extract_global_alpha_components(returns, factors)
    
    def _extract_global_alpha_components(self, returns: pd.DataFrame, factors: pd.DataFrame):
        """Extract global alpha components."""
        
        self.alpha_components = {}
        
        for asset in returns.columns:
            if asset not in self.factor_loadings:
                continue
            
            asset_returns = returns[asset].dropna()
            loadings = self.factor_loadings[asset]['loadings']
            alpha_intercept = self.factor_loadings[asset]['alpha']
            
            # Calculate factor-explained returns
            common_index = asset_returns.index.intersection(factors.index)
            factor_returns = pd.Series(0, index=common_index)
            
            for factor_name, loading in loadings.items():
                if factor_name in factors.columns:
                    factor_returns += loading * factors.loc[common_index, factor_name]
            
            # Alpha component is residual after factor attribution
            alpha_component = asset_returns.loc[common_index] - factor_returns
            
            self.alpha_components[asset] = {
                'alpha_series': alpha_component,
                'alpha_mean': alpha_component.mean(),
                'alpha_vol': alpha_component.std(),
                'alpha_sharpe': alpha_component.mean() / (alpha_component.std() + 1e-8),
                'factor_explained_var': factor_returns.var(),
                'alpha_var': alpha_component.var(),
                'total_var': asset_returns.loc[common_index].var()
            }
    
    def _extract_regime_alpha_components(self, returns: pd.DataFrame, factors: pd.DataFrame,
                                       regime_probabilities: pd.DataFrame):
        """Extract regime-dependent alpha components."""
        
        n_regimes = regime_probabilities.shape[1]
        self.regime_alpha_components = {}
        
        for regime in range(n_regimes):
            regime_alpha = {}
            regime_weights = regime_probabilities.iloc[:, regime]
            
            if regime not in self.regime_factor_loadings:
                continue
            
            for asset in returns.columns:
                if asset not in self.regime_factor_loadings[regime]:
                    continue
                
                asset_returns = returns[asset].dropna()
                loadings = self.regime_factor_loadings[regime][asset]['loadings']
                
                # Calculate regime-weighted factor returns
                common_index = asset_returns.index.intersection(factors.index)
                common_index = common_index.intersection(regime_weights.index)
                
                factor_returns = pd.Series(0, index=common_index)
                for factor_name, loading in loadings.items():
                    if factor_name in factors.columns:
                        factor_returns += loading * factors.loc[common_index, factor_name]
                
                # Regime-weighted alpha
                alpha_component = asset_returns.loc[common_index] - factor_returns
                weights = regime_weights.loc[common_index]
                
                # Calculate weighted statistics
                weighted_alpha_mean = np.average(alpha_component, weights=weights)
                weighted_alpha_var = np.average((alpha_component - weighted_alpha_mean)**2, weights=weights)
                
                regime_alpha[asset] = {
                    'alpha_series': alpha_component,
                    'alpha_mean': weighted_alpha_mean,
                    'alpha_vol': np.sqrt(weighted_alpha_var),
                    'alpha_sharpe': weighted_alpha_mean / (np.sqrt(weighted_alpha_var) + 1e-8),
                    'regime_weight': weights.mean()
                }
            
            self.regime_alpha_components[regime] = regime_alpha
    
    def _perform_attribution_analysis(self, returns: pd.DataFrame, factors: pd.DataFrame,
                                    regime_probabilities: Optional[pd.DataFrame]):
        """Perform detailed attribution analysis."""
        
        self.attribution_results = {}
        
        for asset in returns.columns:
            attribution = self._calculate_asset_attribution(asset, returns, factors, regime_probabilities)
            self.attribution_results[asset] = attribution
        
        # Portfolio-level attribution
        self.performance_attribution = self._calculate_portfolio_attribution(returns, factors, regime_probabilities)
    
    def _calculate_asset_attribution(self, asset: str, returns: pd.DataFrame, 
                                   factors: pd.DataFrame,
                                   regime_probabilities: Optional[pd.DataFrame]) -> Dict:
        """Calculate attribution for individual asset."""
        
        if self.regime_dependent and regime_probabilities is not None:
            return self._calculate_regime_asset_attribution(asset, returns, factors, regime_probabilities)
        else:
            return self._calculate_global_asset_attribution(asset, returns, factors)
    
    def _calculate_global_asset_attribution(self, asset: str, returns: pd.DataFrame,
                                          factors: pd.DataFrame) -> Dict:
        """Calculate global attribution for asset."""
        
        if asset not in self.factor_loadings:
            return {}
        
        loadings = self.factor_loadings[asset]['loadings']
        alpha = self.factor_loadings[asset]['alpha']
        
        # Factor contributions
        factor_contributions = {}
        total_factor_return = 0
        
        for factor_name, loading in loadings.items():
            if factor_name in factors.columns:
                factor_return = factors[factor_name].mean() * loading
                factor_contributions[factor_name] = {
                    'loading': loading,
                    'factor_return': factors[factor_name].mean(),
                    'contribution': factor_return,
                    'volatility_contribution': loading**2 * factors[factor_name].var()
                }
                total_factor_return += factor_return
        
        # Alpha contribution
        alpha_contribution = alpha
        
        return {
            'factor_contributions': factor_contributions,
            'alpha_contribution': alpha_contribution,
            'total_factor_return': total_factor_return,
            'total_attribution': total_factor_return + alpha_contribution
        }
    
    def _calculate_regime_asset_attribution(self, asset: str, returns: pd.DataFrame,
                                          factors: pd.DataFrame,
                                          regime_probabilities: pd.DataFrame) -> Dict:
        """Calculate regime-dependent attribution for asset."""
        
        n_regimes = regime_probabilities.shape[1]
        regime_attributions = {}
        
        for regime in range(n_regimes):
            if (regime in self.regime_factor_loadings and 
                asset in self.regime_factor_loadings[regime]):
                
                loadings = self.regime_factor_loadings[regime][asset]['loadings']
                regime_weight = regime_probabilities.iloc[:, regime].mean()
                
                # Factor contributions for this regime
                factor_contributions = {}
                for factor_name, loading in loadings.items():
                    if factor_name in factors.columns:
                        # Regime-weighted factor return
                        regime_weights = regime_probabilities.iloc[:, regime]
                        weighted_factor_return = np.average(factors[factor_name], weights=regime_weights)
                        
                        factor_contributions[factor_name] = {
                            'loading': loading,
                            'factor_return': weighted_factor_return,
                            'contribution': loading * weighted_factor_return * regime_weight
                        }
                
                # Alpha contribution for this regime
                if regime in self.regime_alpha_components and asset in self.regime_alpha_components[regime]:
                    alpha_contrib = self.regime_alpha_components[regime][asset]['alpha_mean'] * regime_weight
                else:
                    alpha_contrib = 0
                
                regime_attributions[regime] = {
                    'regime_weight': regime_weight,
                    'factor_contributions': factor_contributions,
                    'alpha_contribution': alpha_contrib
                }
        
        return {'regime_attributions': regime_attributions}
    
    def _calculate_portfolio_attribution(self, returns: pd.DataFrame, factors: pd.DataFrame,
                                       regime_probabilities: Optional[pd.DataFrame]) -> Dict:
        """Calculate portfolio-level attribution analysis."""
        
        # Equal-weighted portfolio
        portfolio_returns = returns.mean(axis=1)
        
        # Portfolio factor loadings (average of individual loadings)
        portfolio_loadings = {}
        
        if self.regime_dependent and regime_probabilities is not None:
            # Regime-dependent portfolio attribution
            n_regimes = regime_probabilities.shape[1]
            regime_portfolio_attribution = {}
            
            for regime in range(n_regimes):
                if regime in self.regime_factor_loadings:
                    regime_loadings = {}
                    
                    # Average loadings across assets for this regime
                    for factor_name in factors.columns:
                        loadings = []
                        for asset, asset_loadings in self.regime_factor_loadings[regime].items():
                            if factor_name in asset_loadings['loadings']:
                                loadings.append(asset_loadings['loadings'][factor_name])
                        
                        if loadings:
                            regime_loadings[factor_name] = np.mean(loadings)
                    
                    # Calculate regime portfolio factor returns
                    regime_weights = regime_probabilities.iloc[:, regime]
                    regime_factor_returns = {}
                    
                    for factor_name, loading in regime_loadings.items():
                        weighted_factor_return = np.average(factors[factor_name], weights=regime_weights)
                        regime_factor_returns[factor_name] = loading * weighted_factor_return
                    
                    regime_portfolio_attribution[regime] = {
                        'factor_loadings': regime_loadings,
                        'factor_returns': regime_factor_returns,
                        'regime_weight': regime_weights.mean()
                    }
            
            return {'regime_portfolio_attribution': regime_portfolio_attribution}
        
        else:
            # Global portfolio attribution
            for factor_name in factors.columns:
                loadings = []
                for asset in self.factor_loadings:
                    if factor_name in self.factor_loadings[asset]['loadings']:
                        loadings.append(self.factor_loadings[asset]['loadings'][factor_name])
                
                if loadings:
                    portfolio_loadings[factor_name] = np.mean(loadings)
            
            # Portfolio factor returns
            portfolio_factor_returns = {}
            for factor_name, loading in portfolio_loadings.items():
                portfolio_factor_returns[factor_name] = loading * factors[factor_name].mean()
            
            return {
                'portfolio_loadings': portfolio_loadings,
                'portfolio_factor_returns': portfolio_factor_returns,
                'total_factor_return': sum(portfolio_factor_returns.values())
            }
    
    def get_alpha_summary(self) -> pd.DataFrame:
        """
        Get summary of alpha components across assets.
        
        Returns:
            DataFrame with alpha summary statistics
        """
        if self.regime_dependent:
            return self._get_regime_alpha_summary()
        else:
            return self._get_global_alpha_summary()
    
    def _get_global_alpha_summary(self) -> pd.DataFrame:
        """Get global alpha summary."""
        
        summary_data = []
        
        for asset, alpha_data in self.alpha_components.items():
            summary_data.append({
                'asset': asset,
                'alpha_mean': alpha_data['alpha_mean'],
                'alpha_vol': alpha_data['alpha_vol'],
                'alpha_sharpe': alpha_data['alpha_sharpe'],
                'alpha_var_ratio': alpha_data['alpha_var'] / alpha_data['total_var'],
                'factor_var_ratio': alpha_data['factor_explained_var'] / alpha_data['total_var']
            })
        
        return pd.DataFrame(summary_data).sort_values('alpha_sharpe', ascending=False)
    
    def _get_regime_alpha_summary(self) -> pd.DataFrame:
        """Get regime-dependent alpha summary."""
        
        summary_data = []
        
        for regime, regime_data in self.regime_alpha_components.items():
            for asset, alpha_data in regime_data.items():
                summary_data.append({
                    'regime': regime,
                    'asset': asset,
                    'alpha_mean': alpha_data['alpha_mean'],
                    'alpha_vol': alpha_data['alpha_vol'],
                    'alpha_sharpe': alpha_data['alpha_sharpe'],
                    'regime_weight': alpha_data['regime_weight']
                })
        
        return pd.DataFrame(summary_data)
    
    def get_factor_loadings_summary(self) -> pd.DataFrame:
        """
        Get summary of factor loadings across assets.
        
        Returns:
            DataFrame with factor loading statistics
        """
        if self.regime_dependent:
            return self._get_regime_factor_summary()
        else:
            return self._get_global_factor_summary()
    
    def _get_global_factor_summary(self) -> pd.DataFrame:
        """Get global factor loadings summary."""
        
        summary_data = []
        factor_names = list(self.factor_data.keys())
        
        for asset, loading_data in self.factor_loadings.items():
            row = {'asset': asset, 'r_squared': loading_data['r_squared']}
            
            for factor_name in factor_names:
                row[f'{factor_name}_loading'] = loading_data['loadings'].get(factor_name, 0)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def _get_regime_factor_summary(self) -> pd.DataFrame:
        """Get regime-dependent factor loadings summary."""
        
        summary_data = []
        factor_names = list(self.factor_data.keys())
        
        for regime, regime_data in self.regime_factor_loadings.items():
            for asset, loading_data in regime_data.items():
                row = {
                    'regime': regime,
                    'asset': asset,
                    'r_squared': loading_data['r_squared'],
                    'effective_samples': loading_data['effective_samples']
                }
                
                for factor_name in factor_names:
                    row[f'{factor_name}_loading'] = loading_data['loadings'].get(factor_name, 0)
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def predict_alpha(self, new_factor_data: pd.DataFrame,
                     regime_probabilities: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict alpha for new factor data.
        
        Args:
            new_factor_data: New factor realizations
            regime_probabilities: Optional regime probabilities for prediction period
        
        Returns:
            DataFrame with predicted alpha for each asset
        """
        
        if self.regime_dependent and regime_probabilities is not None:
            return self._predict_regime_alpha(new_factor_data, regime_probabilities)
        else:
            return self._predict_global_alpha(new_factor_data)
    
    def _predict_global_alpha(self, new_factor_data: pd.DataFrame) -> pd.DataFrame:
        """Predict alpha using global factor model."""
        
        predictions = {}
        
        for asset, loading_data in self.factor_loadings.items():
            loadings = loading_data['loadings']
            alpha = loading_data['alpha']
            
            # Calculate factor-explained return
            factor_return = alpha  # Start with alpha intercept
            
            for factor_name, loading in loadings.items():
                if factor_name in new_factor_data.columns:
                    factor_return += loading * new_factor_data[factor_name]
            
            predictions[asset] = factor_return
        
        return pd.DataFrame(predictions, index=new_factor_data.index)
    
    def _predict_regime_alpha(self, new_factor_data: pd.DataFrame,
                            regime_probabilities: pd.DataFrame) -> pd.DataFrame:
        """Predict alpha using regime-dependent factor model."""
        
        n_regimes = regime_probabilities.shape[1]
        predictions = {}
        
        # Align data
        common_index = new_factor_data.index.intersection(regime_probabilities.index)
        
        for asset in set().union(*[list(regime_data.keys()) 
                                 for regime_data in self.regime_factor_loadings.values()]):
            
            asset_predictions = pd.Series(0, index=common_index)
            
            for regime in range(n_regimes):
                if (regime in self.regime_factor_loadings and 
                    asset in self.regime_factor_loadings[regime]):
                    
                    loadings = self.regime_factor_loadings[regime][asset]['loadings']
                    regime_weights = regime_probabilities.iloc[:, regime].loc[common_index]
                    
                    # Calculate regime-specific factor return
                    regime_factor_return = pd.Series(0, index=common_index)
                    
                    for factor_name, loading in loadings.items():
                        if factor_name in new_factor_data.columns:
                            regime_factor_return += loading * new_factor_data.loc[common_index, factor_name]
                    
                    # Weight by regime probability
                    asset_predictions += regime_weights * regime_factor_return
            
            predictions[asset] = asset_predictions
        
        return pd.DataFrame(predictions)
    
    def get_decomposition_summary(self) -> Dict:
        """
        Get comprehensive summary of the decomposition analysis.
        
        Returns:
            Dictionary with decomposition summary
        """
        summary = {
            'factor_models': self.factor_models,
            'regime_dependent': self.regime_dependent,
            'orthogonalize_factors': self.orthogonalize_factors,
            'n_factors': len(self.factor_data),
            'factor_names': list(self.factor_data.keys())
        }
        
        if self.regime_dependent:
            summary['n_regimes'] = len(self.regime_factor_loadings)
            summary['regime_factor_loadings'] = self.regime_factor_loadings
            summary['regime_alpha_components'] = len(self.regime_alpha_components)
        else:
            summary['n_assets'] = len(self.factor_loadings)
            summary['avg_r_squared'] = np.mean([data['r_squared'] for data in self.factor_loadings.values()])
        
        # Factor correlation analysis
        factor_df = pd.DataFrame(self.factor_data)
        factor_corr = factor_df.corr()
        summary['factor_correlations'] = factor_corr.to_dict()
        summary['max_factor_correlation'] = factor_corr.abs().where(~np.eye(len(factor_corr), dtype=bool)).max().max()
        
        return summary