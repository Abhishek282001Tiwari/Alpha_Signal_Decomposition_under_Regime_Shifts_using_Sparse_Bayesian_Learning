import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import logging
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class RiskManager:
    """
    Comprehensive risk management system with regime-aware factor models,
    VaR/CVaR calculations, stress testing, and dynamic hedging strategies.
    """
    
    def __init__(self, 
                 confidence_levels: List[float] = [0.95, 0.99],
                 lookback_window: int = 252,
                 factor_model_type: str = 'pca',
                 regime_aware: bool = True):
        """
        Initialize Risk Manager.
        
        Args:
            confidence_levels: VaR confidence levels
            lookback_window: Rolling window for risk calculations
            factor_model_type: Factor model type ('pca', 'statistical', 'fundamental')
            regime_aware: Whether to use regime-dependent risk models
        """
        self.confidence_levels = confidence_levels
        self.lookback_window = lookback_window
        self.factor_model_type = factor_model_type
        self.regime_aware = regime_aware
        
        # Risk model components
        self.factor_exposures = None
        self.factor_returns = None
        self.factor_covariance = None
        self.specific_risks = None
        self.regime_risk_models = {}
        
        # Portfolio risk metrics
        self.portfolio_var = {}
        self.portfolio_cvar = {}
        self.component_var = {}
        self.marginal_var = {}
        
        # Stress testing
        self.stress_scenarios = {}
        self.historical_stress_tests = {}
        
        # Dynamic hedging
        self.hedge_ratios = {}
        self.hedge_effectiveness = {}
        
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def fit_factor_model(self, 
                        returns: pd.DataFrame,
                        fundamental_data: Optional[pd.DataFrame] = None,
                        regime_probabilities: Optional[pd.DataFrame] = None):
        """
        Fit factor model for risk decomposition.
        
        Args:
            returns: Asset returns DataFrame
            fundamental_data: Optional fundamental data for factor construction
            regime_probabilities: Optional regime probabilities
        """
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        
        if self.regime_aware and regime_probabilities is not None:
            self._fit_regime_factor_models(returns, fundamental_data, regime_probabilities)
        else:
            self._fit_global_factor_model(returns, fundamental_data)
    
    def _fit_global_factor_model(self, 
                                returns: pd.DataFrame,
                                fundamental_data: Optional[pd.DataFrame]):
        """Fit global factor model."""
        
        if self.factor_model_type == 'pca':
            self._fit_pca_factor_model(returns)
        elif self.factor_model_type == 'statistical':
            self._fit_statistical_factor_model(returns)
        elif self.factor_model_type == 'fundamental':
            self._fit_fundamental_factor_model(returns, fundamental_data)
        else:
            raise ValueError(f"Unknown factor model type: {self.factor_model_type}")
    
    def _fit_pca_factor_model(self, returns: pd.DataFrame, n_factors: int = 10):
        """Fit PCA-based factor model."""
        
        # Standardize returns
        returns_scaled = self.scaler.fit_transform(returns.dropna())
        
        # Principal Component Analysis
        pca = PCA(n_components=min(n_factors, self.n_assets))
        factor_returns = pca.fit_transform(returns_scaled)
        
        # Factor exposures (loadings)
        factor_loadings = pca.components_.T
        
        # Explained variance ratios
        explained_variance = pca.explained_variance_ratio_
        
        # Create DataFrames
        factor_names = [f'Factor_{i+1}' for i in range(factor_returns.shape[1])]
        
        self.factor_returns = pd.DataFrame(
            factor_returns,
            index=returns.dropna().index,
            columns=factor_names
        )
        
        self.factor_exposures = pd.DataFrame(
            factor_loadings,
            index=self.assets,
            columns=factor_names
        )
        
        # Factor covariance matrix
        self.factor_covariance = pd.DataFrame(
            np.cov(factor_returns.T),
            index=factor_names,
            columns=factor_names
        )
        
        # Specific risks (residual volatilities)
        # Reconstruct returns using factors
        reconstructed_returns = factor_returns @ factor_loadings.T
        residuals = returns_scaled - reconstructed_returns
        specific_variances = np.var(residuals, axis=0)
        
        self.specific_risks = pd.Series(
            np.sqrt(specific_variances),
            index=self.assets
        )
        
        self.logger.info(f"PCA factor model fitted with {len(factor_names)} factors, "
                        f"explaining {explained_variance.sum():.2%} of variance")
    
    def _fit_statistical_factor_model(self, returns: pd.DataFrame):
        """Fit statistical factor model using market and style factors."""
        
        # Market factor (equal-weighted portfolio)
        market_factor = returns.mean(axis=1)
        
        # Size factor (SMB proxy using cross-sectional volatility)
        rolling_vol = returns.rolling(window=63).std()
        size_factor = -rolling_vol.mean(axis=1)  # Negative vol as size proxy
        
        # Value factor (HML proxy using momentum reversal)
        short_momentum = returns.rolling(window=21).mean()
        long_momentum = returns.rolling(window=252).mean()
        value_factor = -(short_momentum.mean(axis=1) - long_momentum.mean(axis=1))
        
        # Momentum factor
        momentum_factor = long_momentum.mean(axis=1) - short_momentum.mean(axis=1)
        
        # Volatility factor
        volatility_factor = rolling_vol.mean(axis=1)
        
        # Create factor returns DataFrame
        factor_data = {
            'Market': market_factor,
            'Size': size_factor,
            'Value': value_factor,
            'Momentum': momentum_factor,
            'Volatility': volatility_factor
        }
        
        self.factor_returns = pd.DataFrame(factor_data).dropna()
        
        # Estimate factor exposures using regression
        factor_exposures = []
        specific_risks = []
        
        for asset in self.assets:
            asset_returns = returns[asset].dropna()
            
            # Align with factor returns
            common_index = asset_returns.index.intersection(self.factor_returns.index)
            if len(common_index) < 50:
                continue
            
            y = asset_returns.loc[common_index]
            X = self.factor_returns.loc[common_index]
            
            # Multiple regression
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(X, y)
            
            factor_exposures.append(reg.coef_)
            
            # Specific risk from residuals
            y_pred = reg.predict(X)
            residuals = y - y_pred
            specific_risks.append(residuals.std())
        
        self.factor_exposures = pd.DataFrame(
            factor_exposures,
            index=self.assets,
            columns=self.factor_returns.columns
        )
        
        self.specific_risks = pd.Series(specific_risks, index=self.assets)
        
        # Factor covariance matrix
        self.factor_covariance = self.factor_returns.cov()
    
    def _fit_fundamental_factor_model(self, 
                                    returns: pd.DataFrame,
                                    fundamental_data: Optional[pd.DataFrame]):
        """Fit fundamental factor model using company characteristics."""
        
        if fundamental_data is None:
            self.logger.warning("No fundamental data provided, falling back to statistical model")
            return self._fit_statistical_factor_model(returns)
        
        # Extract fundamental factors
        fundamental_factors = fundamental_data.copy()
        
        # Standardize fundamental data
        fundamental_scaled = self.scaler.fit_transform(fundamental_factors.dropna())
        
        # Use fundamental data as factor exposures
        self.factor_exposures = pd.DataFrame(
            fundamental_scaled,
            index=fundamental_factors.dropna().index,
            columns=fundamental_factors.columns
        )
        
        # Estimate factor returns using cross-sectional regression
        factor_returns_list = []
        
        for date in returns.index:
            if date in self.factor_exposures.index:
                # Cross-sectional regression: returns ~ exposures
                asset_returns = returns.loc[date].dropna()
                exposures = self.factor_exposures.loc[date]
                
                # Align assets
                common_assets = asset_returns.index.intersection(exposures.index)
                if len(common_assets) > 5:
                    y = asset_returns[common_assets]
                    X = exposures[common_assets]
                    
                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression()
                    reg.fit(X.values.reshape(-1, len(X)), y)
                    
                    factor_returns_list.append(reg.coef_)
        
        if factor_returns_list:
            self.factor_returns = pd.DataFrame(
                factor_returns_list,
                index=returns.index[:len(factor_returns_list)],
                columns=fundamental_factors.columns
            )
            
            self.factor_covariance = self.factor_returns.cov()
            
            # Calculate specific risks
            specific_risks = []
            for asset in self.assets:
                if asset in self.factor_exposures.index:
                    asset_returns = returns[asset]
                    exposures = self.factor_exposures.loc[asset]
                    
                    # Estimated returns from factor model
                    factor_contrib = self.factor_returns @ exposures
                    residuals = asset_returns - factor_contrib
                    specific_risks.append(residuals.std())
                else:
                    specific_risks.append(returns[asset].std())
            
            self.specific_risks = pd.Series(specific_risks, index=self.assets)
    
    def _fit_regime_factor_models(self, 
                                returns: pd.DataFrame,
                                fundamental_data: Optional[pd.DataFrame],
                                regime_probabilities: pd.DataFrame):
        """Fit regime-dependent factor models."""
        
        n_regimes = regime_probabilities.shape[1]
        
        for regime in range(n_regimes):
            regime_weights = regime_probabilities.iloc[:, regime]
            
            # Weighted returns for this regime
            weighted_returns = returns.multiply(np.sqrt(regime_weights), axis=0)
            
            # Fit factor model for this regime
            temp_risk_manager = RiskManager(
                factor_model_type=self.factor_model_type,
                regime_aware=False
            )
            
            temp_risk_manager._fit_global_factor_model(weighted_returns, fundamental_data)
            
            self.regime_risk_models[regime] = {
                'factor_exposures': temp_risk_manager.factor_exposures,
                'factor_returns': temp_risk_manager.factor_returns,
                'factor_covariance': temp_risk_manager.factor_covariance,
                'specific_risks': temp_risk_manager.specific_risks,
                'regime_weight': regime_weights.mean()
            }
        
        # Create aggregate model weighted by regime probabilities
        self._create_aggregate_risk_model(regime_probabilities)
    
    def _create_aggregate_risk_model(self, regime_probabilities: pd.DataFrame):
        """Create aggregate risk model from regime models."""
        
        current_regime_probs = regime_probabilities.iloc[-1].values
        
        # Weighted factor exposures
        self.factor_exposures = None
        for regime, prob in enumerate(current_regime_probs):
            if regime in self.regime_risk_models:
                exposures = self.regime_risk_models[regime]['factor_exposures']
                if self.factor_exposures is None:
                    self.factor_exposures = prob * exposures
                else:
                    self.factor_exposures += prob * exposures
        
        # Weighted factor covariance
        self.factor_covariance = None
        for regime, prob in enumerate(current_regime_probs):
            if regime in self.regime_risk_models:
                cov = self.regime_risk_models[regime]['factor_covariance']
                if self.factor_covariance is None:
                    self.factor_covariance = prob * cov
                else:
                    self.factor_covariance += prob * cov
        
        # Weighted specific risks
        self.specific_risks = None
        for regime, prob in enumerate(current_regime_probs):
            if regime in self.regime_risk_models:
                risks = self.regime_risk_models[regime]['specific_risks']
                if self.specific_risks is None:
                    self.specific_risks = prob * risks
                else:
                    self.specific_risks += prob * risks
    
    def calculate_portfolio_risk(self, weights: np.ndarray) -> Dict:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Dictionary with risk metrics
        """
        # Portfolio variance decomposition
        if self.factor_exposures is not None:
            factor_contrib, specific_contrib, total_var = self._decompose_portfolio_variance(weights)
        else:
            # Fallback to sample covariance
            cov_matrix = self.returns.cov()
            total_var = weights.T @ cov_matrix.values @ weights
            factor_contrib = 0
            specific_contrib = total_var
        
        portfolio_vol = np.sqrt(total_var)
        
        # VaR and CVaR calculations
        var_estimates = {}
        cvar_estimates = {}
        
        for confidence_level in self.confidence_levels:
            var, cvar = self._calculate_var_cvar(weights, confidence_level)
            var_estimates[confidence_level] = var
            cvar_estimates[confidence_level] = cvar
        
        # Component and marginal VaR
        component_var = self._calculate_component_var(weights)
        marginal_var = self._calculate_marginal_var(weights)
        
        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_variance': total_var,
            'factor_contribution': factor_contrib,
            'specific_contribution': specific_contrib,
            'var_estimates': var_estimates,
            'cvar_estimates': cvar_estimates,
            'component_var': component_var,
            'marginal_var': marginal_var,
            'risk_concentration': self._calculate_risk_concentration(weights)
        }
    
    def _decompose_portfolio_variance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Decompose portfolio variance into factor and specific components."""
        
        # Factor contribution: w' * B * F * B' * w
        B = self.factor_exposures.values
        F = self.factor_covariance.values
        factor_contrib = weights.T @ B @ F @ B.T @ weights
        
        # Specific contribution: w' * D * w
        D = np.diag(self.specific_risks.values**2)
        specific_contrib = weights.T @ D @ weights
        
        total_var = factor_contrib + specific_contrib
        
        return factor_contrib, specific_contrib, total_var
    
    def _calculate_var_cvar(self, weights: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        
        # Portfolio returns
        portfolio_returns = self.returns @ weights
        
        # Historical simulation VaR
        var_hist = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # CVaR (Expected Shortfall)
        cvar_hist = portfolio_returns[portfolio_returns <= var_hist].mean()
        
        # Parametric VaR (assuming normality)
        portfolio_mean = portfolio_returns.mean()
        portfolio_std = portfolio_returns.std()
        
        z_score = stats.norm.ppf(1 - confidence_level)
        var_parametric = portfolio_mean + z_score * portfolio_std
        
        # Use historical VaR as it's more robust
        return var_hist, cvar_hist
    
    def _calculate_component_var(self, weights: np.ndarray) -> pd.Series:
        """Calculate component VaR for each asset."""
        
        portfolio_returns = self.returns @ weights
        portfolio_var_95 = np.percentile(portfolio_returns, 5)
        
        component_vars = []
        
        for i, asset in enumerate(self.assets):
            # Marginal contribution to VaR
            asset_returns = self.returns[asset]
            
            # Calculate correlation with portfolio at VaR level
            portfolio_stress = portfolio_returns <= portfolio_var_95
            if portfolio_stress.sum() > 5:
                asset_stress_returns = asset_returns[portfolio_stress]
                marginal_contrib = asset_stress_returns.mean()
                component_var = weights[i] * marginal_contrib
            else:
                component_var = 0
            
            component_vars.append(component_var)
        
        return pd.Series(component_vars, index=self.assets)
    
    def _calculate_marginal_var(self, weights: np.ndarray) -> pd.Series:
        """Calculate marginal VaR for each asset."""
        
        # Small perturbation for numerical derivative
        epsilon = 1e-6
        base_var = self._calculate_var_cvar(weights, 0.95)[0]
        
        marginal_vars = []
        
        for i in range(self.n_assets):
            # Perturb weight
            perturbed_weights = weights.copy()
            perturbed_weights[i] += epsilon
            
            # Renormalize
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            perturbed_var = self._calculate_var_cvar(perturbed_weights, 0.95)[0]
            marginal_var = (perturbed_var - base_var) / epsilon
            marginal_vars.append(marginal_var)
        
        return pd.Series(marginal_vars, index=self.assets)
    
    def _calculate_risk_concentration(self, weights: np.ndarray) -> float:
        """Calculate risk concentration using Herfindahl index."""
        
        if self.factor_exposures is not None:
            # Risk contributions from factor model
            B = self.factor_exposures.values
            F = self.factor_covariance.values
            D = np.diag(self.specific_risks.values**2)
            
            # Total portfolio variance
            total_var = weights.T @ (B @ F @ B.T + D) @ weights
            
            # Individual risk contributions
            risk_contributions = (B @ F @ B.T + D) @ weights * weights / total_var
        else:
            # Fallback to sample covariance
            cov_matrix = self.returns.cov().values
            total_var = weights.T @ cov_matrix @ weights
            risk_contributions = cov_matrix @ weights * weights / total_var
        
        # Herfindahl index of risk contributions
        return np.sum(risk_contributions**2)
    
    def stress_test_portfolio(self, 
                            weights: np.ndarray,
                            stress_scenarios: Optional[Dict] = None) -> Dict:
        """
        Perform stress testing on portfolio.
        
        Args:
            weights: Portfolio weights
            stress_scenarios: Custom stress scenarios
        
        Returns:
            Dictionary with stress test results
        """
        if stress_scenarios is None:
            stress_scenarios = self._create_default_stress_scenarios()
        
        stress_results = {}
        
        for scenario_name, scenario in stress_scenarios.items():
            if scenario['type'] == 'factor_shock':
                result = self._factor_shock_stress_test(weights, scenario)
            elif scenario['type'] == 'historical_scenario':
                result = self._historical_scenario_stress_test(weights, scenario)
            elif scenario['type'] == 'monte_carlo':
                result = self._monte_carlo_stress_test(weights, scenario)
            else:
                continue
            
            stress_results[scenario_name] = result
        
        return stress_results
    
    def _create_default_stress_scenarios(self) -> Dict:
        """Create default stress testing scenarios."""
        
        scenarios = {
            'market_crash': {
                'type': 'factor_shock',
                'factor_shocks': {'Market': -0.20},  # 20% market decline
                'description': '20% market crash scenario'
            },
            'volatility_spike': {
                'type': 'factor_shock',
                'factor_shocks': {'Volatility': 0.50},  # 50% increase in volatility
                'description': 'Volatility spike scenario'
            },
            'interest_rate_shock': {
                'type': 'factor_shock',
                'factor_shocks': {'Value': 0.30},  # 30% value factor shock
                'description': 'Interest rate shock scenario'
            }
        }
        
        # Add historical stress scenarios if available
        if hasattr(self, 'returns') and len(self.returns) > 252:
            # Worst performing periods
            portfolio_returns = self.returns @ np.ones(self.n_assets) / self.n_assets
            worst_periods = portfolio_returns.nsmallest(10).index
            
            for i, date in enumerate(worst_periods[:3]):
                scenarios[f'historical_stress_{i+1}'] = {
                    'type': 'historical_scenario',
                    'date': date,
                    'description': f'Historical stress period {date.strftime("%Y-%m-%d")}'
                }
        
        return scenarios
    
    def _factor_shock_stress_test(self, weights: np.ndarray, scenario: Dict) -> Dict:
        """Perform factor shock stress test."""
        
        if self.factor_exposures is None:
            return {'error': 'No factor model available'}
        
        factor_shocks = scenario['factor_shocks']
        
        # Calculate portfolio impact
        portfolio_impact = 0
        factor_impacts = {}
        
        for factor_name, shock_size in factor_shocks.items():
            if factor_name in self.factor_exposures.columns:
                # Portfolio exposure to this factor
                factor_exposure = weights.T @ self.factor_exposures[factor_name].values
                
                # Impact of shock
                impact = factor_exposure * shock_size
                portfolio_impact += impact
                factor_impacts[factor_name] = impact
        
        return {
            'portfolio_impact': portfolio_impact,
            'factor_impacts': factor_impacts,
            'portfolio_value_change': portfolio_impact,  # Assuming $1 portfolio
            'scenario_description': scenario.get('description', 'Factor shock')
        }
    
    def _historical_scenario_stress_test(self, weights: np.ndarray, scenario: Dict) -> Dict:
        """Perform historical scenario stress test."""
        
        stress_date = scenario['date']
        
        if stress_date not in self.returns.index:
            return {'error': f'Date {stress_date} not found in returns data'}
        
        # Returns on stress date
        stress_returns = self.returns.loc[stress_date]
        
        # Portfolio impact
        portfolio_impact = weights.T @ stress_returns.values
        
        # Individual asset impacts
        asset_impacts = weights * stress_returns.values
        
        return {
            'portfolio_impact': portfolio_impact,
            'asset_impacts': dict(zip(self.assets, asset_impacts)),
            'stress_date': stress_date,
            'scenario_description': scenario.get('description', f'Historical scenario {stress_date}')
        }
    
    def _monte_carlo_stress_test(self, 
                               weights: np.ndarray, 
                               scenario: Dict,
                               n_simulations: int = 10000) -> Dict:
        """Perform Monte Carlo stress test."""
        
        # Generate random scenarios
        if self.factor_exposures is not None:
            # Use factor model for simulation
            factor_simulations = np.random.multivariate_normal(
                mean=np.zeros(len(self.factor_covariance)),
                cov=self.factor_covariance.values,
                size=n_simulations
            )
            
            # Convert to asset returns
            asset_simulations = factor_simulations @ self.factor_exposures.values.T
        else:
            # Use sample covariance
            mean_returns = self.returns.mean().values
            cov_matrix = self.returns.cov().values
            
            asset_simulations = np.random.multivariate_normal(
                mean=mean_returns,
                cov=cov_matrix,
                size=n_simulations
            )
        
        # Portfolio returns for each simulation
        portfolio_simulations = asset_simulations @ weights
        
        # Calculate stress metrics
        var_95 = np.percentile(portfolio_simulations, 5)
        var_99 = np.percentile(portfolio_simulations, 1)
        expected_shortfall_95 = portfolio_simulations[portfolio_simulations <= var_95].mean()
        expected_shortfall_99 = portfolio_simulations[portfolio_simulations <= var_99].mean()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': expected_shortfall_95,
            'expected_shortfall_99': expected_shortfall_99,
            'worst_case': portfolio_simulations.min(),
            'best_case': portfolio_simulations.max(),
            'n_simulations': n_simulations,
            'scenario_description': scenario.get('description', 'Monte Carlo simulation')
        }
    
    def calculate_hedge_ratios(self, 
                             portfolio_weights: np.ndarray,
                             hedge_instruments: List[str]) -> Dict:
        """
        Calculate optimal hedge ratios for portfolio.
        
        Args:
            portfolio_weights: Current portfolio weights
            hedge_instruments: List of hedging instruments
        
        Returns:
            Dictionary with hedge ratios and effectiveness
        """
        portfolio_returns = self.returns @ portfolio_weights
        
        hedge_results = {}
        
        for instrument in hedge_instruments:
            if instrument in self.returns.columns:
                hedge_returns = self.returns[instrument]
                
                # Calculate optimal hedge ratio using regression
                common_index = portfolio_returns.index.intersection(hedge_returns.index)
                if len(common_index) > 50:
                    y = portfolio_returns.loc[common_index]
                    x = hedge_returns.loc[common_index]
                    
                    # OLS regression: portfolio_returns = alpha + beta * hedge_returns
                    cov_xy = np.cov(x, y)[0, 1]
                    var_x = np.var(x)
                    hedge_ratio = -cov_xy / var_x  # Negative for hedging
                    
                    # Calculate hedge effectiveness (R-squared)
                    correlation = np.corrcoef(x, y)[0, 1]
                    hedge_effectiveness = correlation**2
                    
                    # Hedged portfolio variance
                    var_portfolio = np.var(y)
                    var_hedge = np.var(x)
                    hedged_variance = var_portfolio + hedge_ratio**2 * var_hedge + 2 * hedge_ratio * cov_xy
                    variance_reduction = (var_portfolio - hedged_variance) / var_portfolio
                    
                    hedge_results[instrument] = {
                        'hedge_ratio': hedge_ratio,
                        'hedge_effectiveness': hedge_effectiveness,
                        'variance_reduction': variance_reduction,
                        'correlation': correlation
                    }
        
        return hedge_results
    
    def regime_risk_attribution(self, 
                              weights: np.ndarray,
                              regime_probabilities: pd.DataFrame) -> Dict:
        """
        Perform risk attribution across regimes.
        
        Args:
            weights: Portfolio weights
            regime_probabilities: Regime probabilities
        
        Returns:
            Risk attribution by regime
        """
        if not self.regime_aware or not self.regime_risk_models:
            return {}
        
        n_regimes = regime_probabilities.shape[1]
        current_regime_probs = regime_probabilities.iloc[-1].values
        
        regime_attribution = {}
        
        for regime in range(n_regimes):
            if regime in self.regime_risk_models:
                regime_model = self.regime_risk_models[regime]
                regime_prob = current_regime_probs[regime]
                
                # Calculate regime-specific risk
                B = regime_model['factor_exposures'].values
                F = regime_model['factor_covariance'].values
                D = np.diag(regime_model['specific_risks'].values**2)
                
                regime_variance = weights.T @ (B @ F @ B.T + D) @ weights
                regime_vol = np.sqrt(regime_variance)
                
                # Risk contribution from this regime
                risk_contribution = regime_prob * regime_variance
                
                regime_attribution[regime] = {
                    'regime_probability': regime_prob,
                    'regime_volatility': regime_vol,
                    'risk_contribution': risk_contribution,
                    'weighted_risk_contribution': regime_prob * regime_vol
                }
        
        return regime_attribution
    
    def get_risk_summary(self, weights: np.ndarray) -> Dict:
        """Get comprehensive risk summary for portfolio."""
        
        risk_metrics = self.calculate_portfolio_risk(weights)
        
        summary = {
            'portfolio_metrics': risk_metrics,
            'risk_model_type': self.factor_model_type,
            'regime_aware': self.regime_aware,
            'n_factors': len(self.factor_covariance.columns) if self.factor_covariance is not None else 0,
            'lookback_window': self.lookback_window,
            'confidence_levels': self.confidence_levels
        }
        
        # Add factor model summary
        if self.factor_exposures is not None:
            summary['factor_model'] = {
                'n_factors': len(self.factor_exposures.columns),
                'factor_names': list(self.factor_exposures.columns),
                'average_specific_risk': self.specific_risks.mean(),
                'factor_concentration': np.sum((weights.T @ self.factor_exposures.values)**2)
            }
        
        # Add regime information
        if self.regime_aware and self.regime_risk_models:
            summary['regime_models'] = {
                'n_regimes': len(self.regime_risk_models),
                'regime_weights': {regime: model['regime_weight'] 
                                 for regime, model in self.regime_risk_models.items()}
            }
        
        return summary