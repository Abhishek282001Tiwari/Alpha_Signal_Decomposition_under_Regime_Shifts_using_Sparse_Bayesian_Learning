import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import logging

class RegimeAnalyzer:
    """
    Comprehensive regime characterization and analysis toolkit.
    Provides statistical analysis, visualization, and factor identification for market regimes.
    """
    
    def __init__(self):
        """Initialize regime analyzer."""
        self.logger = logging.getLogger(__name__)
        self.regime_stats = {}
        self.regime_characteristics = {}
        
    def characterize_regimes(self, 
                           data: pd.DataFrame,
                           regime_probs: pd.DataFrame,
                           price_data: Dict[str, pd.DataFrame],
                           macro_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Comprehensive regime characterization analysis.
        
        Args:
            data: Market data DataFrame
            regime_probs: Regime probability DataFrame
            price_data: Dictionary of individual asset price data
            macro_data: Macroeconomic indicators DataFrame
        
        Returns:
            Dictionary with regime characteristics
        """
        self.logger.info("Starting comprehensive regime characterization")
        
        results = {}
        n_regimes = len([col for col in regime_probs.columns if col.startswith('regime_')])
        
        # Get discrete regime assignments
        regime_states = regime_probs.iloc[:, :n_regimes].idxmax(axis=1)
        regime_states = regime_states.map(lambda x: int(x.split('_')[1]))
        
        for regime in range(n_regimes):
            regime_mask = regime_states == regime
            regime_periods = data[regime_mask]
            
            if len(regime_periods) > 5:  # Minimum periods for analysis
                characteristics = self._analyze_single_regime(
                    regime, regime_periods, price_data, macro_data, regime_mask
                )
                results[f'regime_{regime}'] = characteristics
            else:
                self.logger.warning(f"Insufficient data for regime {regime}")
                results[f'regime_{regime}'] = {}
        
        # Cross-regime analysis
        results['cross_regime_analysis'] = self._cross_regime_analysis(
            data, regime_states, price_data
        )
        
        # Regime transition analysis
        results['transition_analysis'] = self._analyze_regime_transitions(
            regime_states, data
        )
        
        self.regime_characteristics = results
        return results
    
    def _analyze_single_regime(self, 
                             regime_id: int,
                             regime_data: pd.DataFrame,
                             price_data: Dict[str, pd.DataFrame],
                             macro_data: Optional[pd.DataFrame],
                             regime_mask: pd.Series) -> Dict:
        """Analyze characteristics of a single regime."""
        characteristics = {
            'regime_id': regime_id,
            'basic_stats': {},
            'market_characteristics': {},
            'volatility_analysis': {},
            'correlation_analysis': {},
            'factor_exposure': {},
            'macro_environment': {}
        }
        
        # Basic statistics
        characteristics['basic_stats'] = {
            'duration_days': len(regime_data),
            'frequency': len(regime_data) / len(regime_mask),
            'start_dates': regime_data.index.min(),
            'end_dates': regime_data.index.max(),
            'avg_episode_length': self._calculate_avg_episode_length(regime_mask)
        }
        
        # Market characteristics
        if 'returns' in regime_data.columns:
            returns = regime_data['returns'].dropna()
            characteristics['market_characteristics'] = {
                'mean_return': returns.mean(),
                'median_return': returns.median(),
                'std_return': returns.std(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'positive_return_frequency': (returns > 0).mean(),
                'var_95': returns.quantile(0.05),
                'var_99': returns.quantile(0.01)
            }
        
        # Volatility analysis
        characteristics['volatility_analysis'] = self._analyze_regime_volatility(
            regime_data, price_data, regime_mask
        )
        
        # Correlation analysis
        characteristics['correlation_analysis'] = self._analyze_regime_correlations(
            price_data, regime_mask
        )
        
        # Factor exposure analysis
        characteristics['factor_exposure'] = self._analyze_factor_exposure(
            regime_data, price_data, regime_mask
        )
        
        # Macroeconomic environment
        if macro_data is not None:
            characteristics['macro_environment'] = self._analyze_macro_environment(
                macro_data, regime_mask
            )
        
        return characteristics
    
    def _calculate_avg_episode_length(self, regime_mask: pd.Series) -> float:
        """Calculate average length of regime episodes."""
        episodes = []
        current_episode = 0
        
        for i, is_regime in enumerate(regime_mask):
            if is_regime:
                current_episode += 1
            else:
                if current_episode > 0:
                    episodes.append(current_episode)
                    current_episode = 0
        
        # Add final episode if series ends in regime
        if current_episode > 0:
            episodes.append(current_episode)
        
        return np.mean(episodes) if episodes else 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _analyze_regime_volatility(self, 
                                 regime_data: pd.DataFrame,
                                 price_data: Dict[str, pd.DataFrame],
                                 regime_mask: pd.Series) -> Dict:
        """Analyze volatility characteristics during regime."""
        vol_analysis = {}
        
        # Overall market volatility
        if 'returns' in regime_data.columns:
            returns = regime_data['returns'].dropna()
            vol_analysis['realized_volatility'] = returns.std() * np.sqrt(252)
            
            # GARCH-like volatility clustering
            vol_analysis['volatility_clustering'] = self._test_volatility_clustering(returns)
        
        # Cross-sectional volatility
        asset_vols = {}
        for symbol, df in price_data.items():
            if len(df) > len(regime_mask):
                df = df.iloc[:len(regime_mask)]
            elif len(df) < len(regime_mask):
                continue
                
            regime_returns = df['Close'].pct_change()[regime_mask].dropna()
            if len(regime_returns) > 5:
                asset_vols[symbol] = regime_returns.std() * np.sqrt(252)
        
        if asset_vols:
            vol_analysis['cross_sectional_vol'] = {
                'mean_asset_vol': np.mean(list(asset_vols.values())),
                'median_asset_vol': np.median(list(asset_vols.values())),
                'vol_dispersion': np.std(list(asset_vols.values())),
                'min_vol': min(asset_vols.values()),
                'max_vol': max(asset_vols.values())
            }
        
        return vol_analysis
    
    def _test_volatility_clustering(self, returns: pd.Series) -> Dict:
        """Test for volatility clustering using ARCH effects."""
        try:
            # Simple ARCH test
            squared_returns = returns**2
            lagged_squared = squared_returns.shift(1).dropna()
            current_squared = squared_returns.iloc[1:].dropna()
            
            # Regression test
            correlation = current_squared.corr(lagged_squared)
            
            # Ljung-Box test on squared returns
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_stat, lb_pvalue = acorr_ljungbox(squared_returns.dropna(), lags=10, return_df=False)
            
            return {
                'arch_correlation': correlation,
                'ljung_box_statistic': lb_stat[-1],
                'ljung_box_pvalue': lb_pvalue[-1],
                'volatility_clustering_present': lb_pvalue[-1] < 0.05
            }
        except:
            return {
                'arch_correlation': np.nan,
                'ljung_box_statistic': np.nan,
                'ljung_box_pvalue': np.nan,
                'volatility_clustering_present': False
            }
    
    def _analyze_regime_correlations(self, 
                                   price_data: Dict[str, pd.DataFrame],
                                   regime_mask: pd.Series) -> Dict:
        """Analyze asset correlations during regime."""
        # Create returns matrix
        returns_dict = {}
        for symbol, df in price_data.items():
            if len(df) >= len(regime_mask):
                returns = df['Close'].pct_change().iloc[:len(regime_mask)]
                regime_returns = returns[regime_mask].dropna()
                if len(regime_returns) > 10:
                    returns_dict[symbol] = regime_returns
        
        if len(returns_dict) < 2:
            return {}
        
        # Align all return series
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        correlations = corr_matrix.values[mask]
        
        return {
            'mean_correlation': np.mean(correlations),
            'median_correlation': np.median(correlations),
            'correlation_std': np.std(correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations),
            'correlation_matrix': corr_matrix.to_dict(),
            'n_assets': len(returns_dict)
        }
    
    def _analyze_factor_exposure(self, 
                               regime_data: pd.DataFrame,
                               price_data: Dict[str, pd.DataFrame],
                               regime_mask: pd.Series) -> Dict:
        """Analyze factor exposures during regime."""
        factor_analysis = {}
        
        # Market factor analysis
        if 'returns' in regime_data.columns:
            market_returns = regime_data['returns'].dropna()
            
            # Beta analysis for individual assets
            betas = {}
            for symbol, df in price_data.items():
                if len(df) >= len(regime_mask):
                    asset_returns = df['Close'].pct_change().iloc[:len(regime_mask)]
                    regime_asset_returns = asset_returns[regime_mask].dropna()
                    
                    # Align with market returns
                    common_index = market_returns.index.intersection(regime_asset_returns.index)
                    if len(common_index) > 10:
                        market_aligned = market_returns[common_index]
                        asset_aligned = regime_asset_returns[common_index]
                        
                        # Calculate beta
                        covariance = np.cov(asset_aligned, market_aligned)[0, 1]
                        market_variance = np.var(market_aligned)
                        beta = covariance / market_variance if market_variance > 0 else np.nan
                        betas[symbol] = beta
            
            if betas:
                factor_analysis['beta_analysis'] = {
                    'mean_beta': np.mean(list(betas.values())),
                    'median_beta': np.median(list(betas.values())),
                    'beta_dispersion': np.std(list(betas.values())),
                    'high_beta_assets': [k for k, v in betas.items() if v > 1.2],
                    'low_beta_assets': [k for k, v in betas.items() if v < 0.8]
                }
        
        # Momentum factor analysis
        factor_analysis['momentum_analysis'] = self._analyze_momentum_factor(
            price_data, regime_mask
        )
        
        # Mean reversion analysis
        factor_analysis['mean_reversion_analysis'] = self._analyze_mean_reversion(
            price_data, regime_mask
        )
        
        return factor_analysis
    
    def _analyze_momentum_factor(self, 
                               price_data: Dict[str, pd.DataFrame],
                               regime_mask: pd.Series) -> Dict:
        """Analyze momentum factor during regime."""
        momentum_scores = {}
        
        for symbol, df in price_data.items():
            if len(df) >= len(regime_mask):
                prices = df['Close'].iloc[:len(regime_mask)]
                regime_prices = prices[regime_mask]
                
                if len(regime_prices) > 20:
                    # Calculate momentum (20-day return)
                    momentum = (regime_prices.iloc[-1] / regime_prices.iloc[0] - 1) * 100
                    momentum_scores[symbol] = momentum
        
        if momentum_scores:
            return {
                'mean_momentum': np.mean(list(momentum_scores.values())),
                'median_momentum': np.median(list(momentum_scores.values())),
                'momentum_dispersion': np.std(list(momentum_scores.values())),
                'positive_momentum_pct': np.mean([m > 0 for m in momentum_scores.values()]),
                'strong_momentum_assets': [k for k, v in momentum_scores.items() if abs(v) > 10]
            }
        return {}
    
    def _analyze_mean_reversion(self, 
                              price_data: Dict[str, pd.DataFrame],
                              regime_mask: pd.Series) -> Dict:
        """Analyze mean reversion characteristics during regime."""
        reversion_stats = {}
        
        for symbol, df in price_data.items():
            if len(df) >= len(regime_mask):
                returns = df['Close'].pct_change().iloc[:len(regime_mask)]
                regime_returns = returns[regime_mask].dropna()
                
                if len(regime_returns) > 10:
                    # Test for mean reversion using autocorrelation
                    autocorr_1 = regime_returns.autocorr(lag=1)
                    reversion_stats[symbol] = autocorr_1
        
        if reversion_stats:
            autocorrs = list(reversion_stats.values())
            return {
                'mean_autocorrelation': np.mean(autocorrs),
                'median_autocorrelation': np.median(autocorrs),
                'negative_autocorr_pct': np.mean([a < 0 for a in autocorrs]),
                'strong_mean_reversion_assets': [k for k, v in reversion_stats.items() if v < -0.1]
            }
        return {}
    
    def _analyze_macro_environment(self, 
                                 macro_data: pd.DataFrame,
                                 regime_mask: pd.Series) -> Dict:
        """Analyze macroeconomic environment during regime."""
        macro_analysis = {}
        
        # Align macro data with regime mask
        aligned_macro = macro_data.iloc[:len(regime_mask)]
        regime_macro = aligned_macro[regime_mask]
        
        for column in regime_macro.columns:
            if regime_macro[column].dtype in ['float64', 'int64']:
                values = regime_macro[column].dropna()
                if len(values) > 0:
                    macro_analysis[column] = {
                        'mean': values.mean(),
                        'median': values.median(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'trend': 'increasing' if values.iloc[-1] > values.iloc[0] else 'decreasing'
                    }
        
        return macro_analysis
    
    def _cross_regime_analysis(self, 
                             data: pd.DataFrame,
                             regime_states: pd.Series,
                             price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Compare characteristics across different regimes."""
        cross_analysis = {}
        
        unique_regimes = regime_states.unique()
        
        # Return comparison
        if 'returns' in data.columns:
            regime_returns = {}
            for regime in unique_regimes:
                regime_mask = regime_states == regime
                regime_returns[f'regime_{regime}'] = data['returns'][regime_mask].dropna()
            
            # Statistical tests
            if len(regime_returns) >= 2:
                regime_values = list(regime_returns.values())
                # ANOVA test
                try:
                    f_stat, p_value = stats.f_oneway(*regime_values)
                    cross_analysis['returns_anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant_difference': p_value < 0.05
                    }
                except:
                    cross_analysis['returns_anova'] = {'significant_difference': False}
        
        # Volatility comparison
        regime_volatilities = {}
        for regime in unique_regimes:
            regime_mask = regime_states == regime
            if 'returns' in data.columns:
                regime_vol = data['returns'][regime_mask].std() * np.sqrt(252)
                regime_volatilities[f'regime_{regime}'] = regime_vol
        
        cross_analysis['volatility_comparison'] = regime_volatilities
        
        # Correlation comparison
        regime_correlations = {}
        for regime in unique_regimes:
            regime_mask = regime_states == regime
            corr_analysis = self._analyze_regime_correlations(price_data, regime_mask)
            if 'mean_correlation' in corr_analysis:
                regime_correlations[f'regime_{regime}'] = corr_analysis['mean_correlation']
        
        cross_analysis['correlation_comparison'] = regime_correlations
        
        return cross_analysis
    
    def _analyze_regime_transitions(self, 
                                  regime_states: pd.Series,
                                  data: pd.DataFrame) -> Dict:
        """Analyze regime transition patterns and characteristics."""
        transition_analysis = {}
        
        # Calculate transition matrix
        unique_regimes = sorted(regime_states.unique())
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_states) - 1):
            current_regime = regime_states.iloc[i]
            next_regime = regime_states.iloc[i + 1]
            
            current_idx = unique_regimes.index(current_regime)
            next_idx = unique_regimes.index(next_regime)
            
            transition_matrix[current_idx, next_idx] += 1
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, row_sums, 
                                   out=np.zeros_like(transition_matrix), 
                                   where=row_sums!=0)
        
        transition_analysis['transition_matrix'] = transition_matrix
        transition_analysis['transition_probabilities'] = transition_probs
        
        # Persistence analysis
        persistence = {}
        for i, regime in enumerate(unique_regimes):
            persistence[f'regime_{regime}'] = transition_probs[i, i]
        
        transition_analysis['regime_persistence'] = persistence
        
        # Transition triggers analysis
        if 'returns' in data.columns:
            transition_returns = []
            for i in range(len(regime_states) - 1):
                if regime_states.iloc[i] != regime_states.iloc[i + 1]:
                    # Regime transition occurred
                    transition_returns.append(data['returns'].iloc[i])
            
            if transition_returns:
                transition_analysis['transition_triggers'] = {
                    'mean_transition_return': np.mean(transition_returns),
                    'transition_return_std': np.std(transition_returns),
                    'extreme_transitions': len([r for r in transition_returns if abs(r) > 0.02])
                }
        
        return transition_analysis
    
    def calculate_regime_statistics(self, 
                                  regime_probs: pd.DataFrame,
                                  data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive regime statistics.
        
        Args:
            regime_probs: Regime probability DataFrame
            data: Market data DataFrame
        
        Returns:
            DataFrame with regime statistics
        """
        stats_list = []
        n_regimes = len([col for col in regime_probs.columns if col.startswith('regime_')])
        
        # Get discrete regime assignments
        regime_states = regime_probs.iloc[:, :n_regimes].idxmax(axis=1)
        regime_states = regime_states.map(lambda x: int(x.split('_')[1]))
        
        for regime in range(n_regimes):
            regime_mask = regime_states == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) > 0:
                stats_dict = {
                    'regime': regime,
                    'frequency': len(regime_data) / len(data),
                    'avg_duration': self._calculate_avg_episode_length(regime_mask),
                    'total_days': len(regime_data)
                }
                
                if 'returns' in regime_data.columns:
                    returns = regime_data['returns'].dropna()
                    if len(returns) > 0:
                        stats_dict.update({
                            'mean_return': returns.mean(),
                            'volatility': returns.std() * np.sqrt(252),
                            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                            'skewness': returns.skew(),
                            'kurtosis': returns.kurtosis(),
                            'max_drawdown': self._calculate_max_drawdown(returns)
                        })
                
                stats_list.append(stats_dict)
        
        return pd.DataFrame(stats_list)
    
    def create_regime_dashboard(self, 
                              data: pd.DataFrame,
                              regime_probs: pd.DataFrame,
                              price_data: Optional[Dict[str, pd.DataFrame]] = None) -> go.Figure:
        """
        Create interactive regime visualization dashboard.
        
        Args:
            data: Market data DataFrame
            regime_probs: Regime probability DataFrame
            price_data: Optional individual asset price data
        
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Regime Probabilities Over Time',
                'Market Returns by Regime',
                'Regime Transition Heatmap',
                'Volatility by Regime',
                'Regime Duration Distribution',
                'Cumulative Returns by Regime',
                'Cross-Asset Correlations',
                'Regime Statistics Summary'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "box"}],
                [{"type": "histogram"}, {"secondary_y": True}],
                [{"type": "heatmap"}, {"type": "table"}]
            ]
        )
        
        # 1. Regime probabilities over time
        n_regimes = len([col for col in regime_probs.columns if col.startswith('regime_')])
        colors = px.colors.qualitative.Set1[:n_regimes]
        
        for i in range(n_regimes):
            fig.add_trace(
                go.Scatter(
                    x=regime_probs.index,
                    y=regime_probs[f'regime_{i}'],
                    name=f'Regime {i}',
                    line=dict(color=colors[i]),
                    fill='tonexty' if i > 0 else 'tozeroy'
                ),
                row=1, col=1
            )
        
        # Add market returns on secondary y-axis
        if 'returns' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['returns'],
                    name='Returns',
                    line=dict(color='black', width=1),
                    yaxis='y2'
                ),
                row=1, col=1, secondary_y=True
            )
        
        # 2. Market returns by regime (scatter plot)
        if 'returns' in data.columns:
            regime_states = regime_probs.iloc[:, :n_regimes].idxmax(axis=1)
            regime_states = regime_states.map(lambda x: int(x.split('_')[1]))
            
            for regime in range(n_regimes):
                regime_mask = regime_states == regime
                regime_returns = data['returns'][regime_mask]
                
                fig.add_trace(
                    go.Scatter(
                        x=regime_returns.index,
                        y=regime_returns.values,
                        mode='markers',
                        name=f'Regime {regime} Returns',
                        marker=dict(color=colors[regime], size=4)
                    ),
                    row=1, col=2
                )
        
        # 3. Regime transition heatmap
        regime_states = regime_probs.iloc[:, :n_regimes].idxmax(axis=1)
        regime_states = regime_states.map(lambda x: int(x.split('_')[1]))
        
        transition_matrix = self._calculate_transition_matrix(regime_states)
        
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix,
                x=[f'To Regime {i}' for i in range(n_regimes)],
                y=[f'From Regime {i}' for i in range(n_regimes)],
                colorscale='Blues',
                showscale=True
            ),
            row=2, col=1
        )
        
        # 4. Volatility by regime (box plot)
        if 'returns' in data.columns:
            regime_vols = []
            regime_labels = []
            
            for regime in range(n_regimes):
                regime_mask = regime_states == regime
                regime_returns = data['returns'][regime_mask].dropna()
                
                # Calculate rolling volatility
                rolling_vol = regime_returns.rolling(window=20).std() * np.sqrt(252)
                regime_vols.extend(rolling_vol.dropna().tolist())
                regime_labels.extend([f'Regime {regime}'] * len(rolling_vol.dropna()))
            
            for regime in range(n_regimes):
                regime_vol_data = [vol for vol, label in zip(regime_vols, regime_labels) if label == f'Regime {regime}']
                
                fig.add_trace(
                    go.Box(
                        y=regime_vol_data,
                        name=f'Regime {regime}',
                        marker_color=colors[regime]
                    ),
                    row=2, col=2
                )
        
        # 5. Regime duration distribution
        durations = self._calculate_regime_durations(regime_states)
        
        for regime in range(n_regimes):
            regime_durations = durations.get(regime, [])
            
            fig.add_trace(
                go.Histogram(
                    x=regime_durations,
                    name=f'Regime {regime} Duration',
                    marker_color=colors[regime],
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # 6. Cumulative returns by regime
        if 'returns' in data.columns:
            for regime in range(n_regimes):
                regime_mask = regime_states == regime
                regime_returns = data['returns'][regime_mask]
                cumulative_returns = (1 + regime_returns).cumprod() - 1
                
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns.values,
                        name=f'Regime {regime} Cumulative',
                        line=dict(color=colors[regime])
                    ),
                    row=3, col=2
                )
        
        # 7. Cross-asset correlations (if price_data available)
        if price_data:
            corr_matrix = self._calculate_cross_asset_correlations(price_data, regime_states)
            
            if corr_matrix is not None:
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        showscale=True
                    ),
                    row=4, col=1
                )
        
        # 8. Regime statistics table
        stats_df = self.calculate_regime_statistics(regime_probs, data)
        
        if not stats_df.empty:
            fig.add_trace(
                go.Table(
                    header=dict(values=list(stats_df.columns)),
                    cells=dict(values=[stats_df[col].round(4) for col in stats_df.columns])
                ),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title="Comprehensive Regime Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def _calculate_transition_matrix(self, regime_states: pd.Series) -> np.ndarray:
        """Calculate regime transition matrix."""
        unique_regimes = sorted(regime_states.unique())
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_states) - 1):
            current_regime = regime_states.iloc[i]
            next_regime = regime_states.iloc[i + 1]
            
            current_idx = unique_regimes.index(current_regime)
            next_idx = unique_regimes.index(next_regime)
            
            transition_matrix[current_idx, next_idx] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        return np.divide(transition_matrix, row_sums, 
                        out=np.zeros_like(transition_matrix), 
                        where=row_sums!=0)
    
    def _calculate_regime_durations(self, regime_states: pd.Series) -> Dict[int, List[int]]:
        """Calculate duration of each regime episode."""
        durations = {}
        current_regime = None
        current_duration = 0
        
        for regime in regime_states:
            if regime != current_regime:
                if current_regime is not None and current_duration > 0:
                    if current_regime not in durations:
                        durations[current_regime] = []
                    durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1
            else:
                current_duration += 1
        
        # Add final regime
        if current_regime is not None and current_duration > 0:
            if current_regime not in durations:
                durations[current_regime] = []
            durations[current_regime].append(current_duration)
        
        return durations
    
    def _calculate_cross_asset_correlations(self, 
                                          price_data: Dict[str, pd.DataFrame],
                                          regime_states: pd.Series) -> Optional[pd.DataFrame]:
        """Calculate cross-asset correlations."""
        if len(price_data) < 2:
            return None
        
        # Create returns matrix
        returns_dict = {}
        min_length = min(len(df) for df in price_data.values())
        min_length = min(min_length, len(regime_states))
        
        for symbol, df in price_data.items():
            returns = df['Close'].pct_change().iloc[:min_length]
            returns_dict[symbol] = returns
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return None
        
        return returns_df.corr()
    
    def identify_regime_triggers(self, 
                               data: pd.DataFrame,
                               regime_probs: pd.DataFrame,
                               macro_data: Optional[pd.DataFrame] = None,
                               lookback_window: int = 20) -> Dict:
        """
        Identify factors that trigger regime transitions.
        
        Args:
            data: Market data DataFrame
            regime_probs: Regime probability DataFrame
            macro_data: Macroeconomic data DataFrame
            lookback_window: Window for analyzing pre-transition period
        
        Returns:
            Dictionary with regime trigger analysis
        """
        self.logger.info("Identifying regime transition triggers")
        
        # Detect regime transitions
        n_regimes = len([col for col in regime_probs.columns if col.startswith('regime_')])
        regime_states = regime_probs.iloc[:, :n_regimes].idxmax(axis=1)
        regime_states = regime_states.map(lambda x: int(x.split('_')[1]))
        
        transition_points = []
        for i in range(1, len(regime_states)):
            if regime_states.iloc[i] != regime_states.iloc[i-1]:
                transition_points.append({
                    'date': regime_states.index[i],
                    'from_regime': regime_states.iloc[i-1],
                    'to_regime': regime_states.iloc[i],
                    'index': i
                })
        
        trigger_analysis = {}
        
        # Analyze market conditions before transitions
        if 'returns' in data.columns:
            pre_transition_returns = []
            pre_transition_volatility = []
            
            for transition in transition_points:
                idx = transition['index']
                start_idx = max(0, idx - lookback_window)
                
                pre_returns = data['returns'].iloc[start_idx:idx]
                pre_vol = pre_returns.std() * np.sqrt(252)
                
                pre_transition_returns.append(pre_returns.mean())
                pre_transition_volatility.append(pre_vol)
            
            trigger_analysis['market_conditions'] = {
                'avg_pre_transition_return': np.mean(pre_transition_returns),
                'avg_pre_transition_volatility': np.mean(pre_transition_volatility),
                'return_threshold_95': np.percentile(pre_transition_returns, 95),
                'volatility_threshold_95': np.percentile(pre_transition_volatility, 95)
            }
        
        # Analyze macro triggers if available
        if macro_data is not None:
            macro_triggers = {}
            
            for col in macro_data.columns:
                if macro_data[col].dtype in ['float64', 'int64']:
                    pre_transition_values = []
                    
                    for transition in transition_points:
                        idx = transition['index']
                        if idx < len(macro_data):
                            start_idx = max(0, idx - lookback_window)
                            pre_values = macro_data[col].iloc[start_idx:idx].dropna()
                            
                            if len(pre_values) > 0:
                                # Calculate trend and level
                                trend = (pre_values.iloc[-1] - pre_values.iloc[0]) / len(pre_values)
                                level = pre_values.mean()
                                pre_transition_values.append({'trend': trend, 'level': level})
                    
                    if pre_transition_values:
                        trends = [v['trend'] for v in pre_transition_values]
                        levels = [v['level'] for v in pre_transition_values]
                        
                        macro_triggers[col] = {
                            'avg_trend': np.mean(trends),
                            'avg_level': np.mean(levels),
                            'trend_std': np.std(trends),
                            'level_std': np.std(levels)
                        }
            
            trigger_analysis['macro_triggers'] = macro_triggers
        
        # Transition timing analysis
        transition_analysis = {
            'total_transitions': len(transition_points),
            'transition_frequency': len(transition_points) / len(regime_states),
            'avg_time_between_transitions': np.mean([
                (transition_points[i]['date'] - transition_points[i-1]['date']).days
                for i in range(1, len(transition_points))
            ]) if len(transition_points) > 1 else None
        }
        
        trigger_analysis['transition_timing'] = transition_analysis
        trigger_analysis['transition_points'] = transition_points
        
        return trigger_analysis