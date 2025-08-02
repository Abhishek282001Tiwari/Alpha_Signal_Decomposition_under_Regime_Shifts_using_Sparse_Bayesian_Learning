import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("cvxpy not available. Install with: pip install cvxpy")

from scipy import optimize, linalg
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    method: str = 'mean_variance'  # 'mean_variance', 'black_litterman', 'risk_parity', 'hierarchical_risk_parity'
    regime_aware: bool = True
    risk_aversion: float = 1.0
    max_leverage: float = 1.0
    max_position_size: float = 0.1
    min_position_size: float = 0.0
    turnover_penalty: float = 0.01
    transaction_cost_bps: float = 5.0  # basis points
    confidence_level: float = 0.05
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly'
    use_robust_covariance: bool = True

class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization engine with regime-aware expected returns,
    multiple optimization methods, comprehensive risk management, and transaction costs.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize Advanced Portfolio Optimizer.
        
        Args:
            config: Configuration object for optimization parameters
        """
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.regime_expected_returns = {}
        self.regime_covariances = {}
        self.regime_probabilities = None
        self.current_weights = None
        self.universe = []
        
        # Risk models
        self.risk_model = None
        self.factor_exposures = None
        self.factor_returns = None
        
        # Optimization results
        self.optimization_results = {}
        self.performance_attribution = {}
        
        # Transaction cost model
        self.transaction_costs = {}
        
    def optimize_portfolio(self,
                          alpha_signals: pd.DataFrame,
                          regime_probabilities: pd.DataFrame,
                          returns_data: pd.DataFrame,
                          current_positions: Optional[pd.Series] = None,
                          market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize portfolio given alpha signals and regime information.
        
        Args:
            alpha_signals: DataFrame with alpha signals by asset and date
            regime_probabilities: DataFrame with regime probabilities by date
            returns_data: Historical returns data for risk modeling
            current_positions: Current portfolio positions
            market_data: Additional market data for risk modeling
        
        Returns:
            Dictionary containing optimal weights and optimization details
        """
        self.logger.info(f"Optimizing portfolio using {self.config.method} method")
        
        try:
            # Prepare optimization inputs
            inputs = self._prepare_optimization_inputs(
                alpha_signals, regime_probabilities, returns_data, current_positions
            )
            
            if not inputs:
                self.logger.error("Failed to prepare optimization inputs")
                return {}
            
            # Apply optimization method
            if self.config.method == 'mean_variance':
                result = self._optimize_mean_variance(inputs)
            elif self.config.method == 'black_litterman':
                result = self._optimize_black_litterman(inputs)
            elif self.config.method == 'risk_parity':
                result = self._optimize_risk_parity(inputs)
            elif self.config.method == 'hierarchical_risk_parity':
                result = self._optimize_hierarchical_risk_parity(inputs)
            elif self.config.method == 'robust_optimization':
                result = self._optimize_robust(inputs)
            else:
                self.logger.error(f"Unknown optimization method: {self.config.method}")
                return {}
            
            # Post-process results
            if result:
                result = self._postprocess_optimization_result(result, inputs)
                self.optimization_results = result
                self.current_weights = result.get('weights', pd.Series())
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            return {}
    
    def _prepare_optimization_inputs(self,
                                   alpha_signals: pd.DataFrame,
                                   regime_probabilities: pd.DataFrame,
                                   returns_data: pd.DataFrame,
                                   current_positions: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Prepare inputs for portfolio optimization."""
        try:
            # Get latest signals and regime probabilities
            latest_date = alpha_signals.index[-1]
            latest_signals = alpha_signals.loc[latest_date]
            latest_regime_probs = regime_probabilities.loc[latest_date]
            
            # Get universe of assets
            universe = latest_signals.dropna().index.tolist()
            if not universe:
                self.logger.error("No valid assets in universe")
                return {}
            
            self.universe = universe
            n_assets = len(universe)
            
            # Prepare expected returns
            if self.config.regime_aware:
                expected_returns = self._calculate_regime_aware_returns(
                    latest_signals, latest_regime_probs, returns_data
                )
            else:
                expected_returns = latest_signals[universe]
            
            # Prepare covariance matrix
            covariance_matrix = self._estimate_covariance_matrix(returns_data, universe)
            
            # Current positions
            if current_positions is None:
                current_positions = pd.Series(0.0, index=universe)
            else:
                current_positions = current_positions.reindex(universe, fill_value=0.0)
            
            # Risk model (factor-based)
            risk_model = self._build_risk_model(returns_data, universe)
            
            inputs = {
                'expected_returns': expected_returns,
                'covariance_matrix': covariance_matrix,
                'current_positions': current_positions,
                'universe': universe,
                'n_assets': n_assets,
                'regime_probabilities': latest_regime_probs,
                'risk_model': risk_model,
                'alpha_signals': latest_signals[universe]
            }
            
            return inputs
            
        except Exception as e:
            self.logger.error(f"Error preparing optimization inputs: {str(e)}")
            return {}
    
    def _calculate_regime_aware_returns(self,
                                      signals: pd.Series,
                                      regime_probs: pd.Series,
                                      returns_data: pd.DataFrame) -> pd.Series:
        """Calculate regime-aware expected returns."""
        try:
            # For each asset, weight expected returns by regime probabilities
            universe = signals.dropna().index.tolist()
            regime_aware_returns = pd.Series(0.0, index=universe)
            
            n_regimes = len(regime_probs)
            
            for asset in universe:
                if asset in returns_data.columns:
                    asset_returns = returns_data[asset].dropna()
                    
                    # Estimate regime-specific expected returns
                    # This is simplified - in practice, use regime model outputs
                    historical_mean = asset_returns.mean()
                    historical_vol = asset_returns.std()
                    
                    # Regime-specific adjustments based on signals
                    signal_strength = signals[asset]
                    
                    # Weight by regime probabilities
                    for regime in range(n_regimes):
                        regime_prob = regime_probs.iloc[regime]
                        # Simplified regime-specific return adjustment
                        regime_adjustment = signal_strength * (1 + regime * 0.1)  # Higher regimes = higher sensitivity
                        regime_aware_returns[asset] += regime_prob * regime_adjustment
                
            return regime_aware_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating regime-aware returns: {str(e)}")
            return signals.fillna(0.0)
    
    def _estimate_covariance_matrix(self, returns_data: pd.DataFrame, universe: List[str]) -> np.ndarray:
        """Estimate covariance matrix with robust methods."""
        try:
            # Filter returns data to universe
            returns_subset = returns_data[universe].dropna()
            
            if len(returns_subset) < 50:  # Minimum observations
                self.logger.warning("Insufficient data for covariance estimation")
                return np.eye(len(universe)) * 0.01  # Identity matrix fallback
            
            if self.config.use_robust_covariance:
                # Ledoit-Wolf shrinkage estimator
                cov_estimator = LedoitWolf()
                covariance_matrix = cov_estimator.fit(returns_subset).covariance_
            else:
                # Sample covariance
                covariance_matrix = returns_subset.cov().values
            
            # Ensure positive definite
            try:
                # Check if matrix is positive definite
                np.linalg.cholesky(covariance_matrix)
            except np.linalg.LinAlgError:
                # If not, use nearest positive definite matrix
                eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
                eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
                covariance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            return covariance_matrix
            
        except Exception as e:
            self.logger.error(f"Error estimating covariance matrix: {str(e)}")
            return np.eye(len(universe)) * 0.01
    
    def _build_risk_model(self, returns_data: pd.DataFrame, universe: List[str]) -> Dict[str, Any]:
        """Build factor-based risk model."""
        try:
            # Simple factor model using PCA
            from sklearn.decomposition import PCA
            
            returns_subset = returns_data[universe].dropna()
            
            if len(returns_subset) < 100:
                return {'type': 'sample_covariance'}
            
            # Fit PCA to extract common factors
            n_factors = min(10, len(universe) // 2)  # Use up to 10 factors
            pca = PCA(n_components=n_factors)
            factor_returns = pca.fit_transform(returns_subset)
            factor_loadings = pca.components_.T  # (n_assets, n_factors)
            
            # Specific risk (idiosyncratic risk)
            reconstructed_returns = factor_returns @ pca.components_
            specific_risk = np.var(returns_subset.values - reconstructed_returns, axis=0)
            
            risk_model = {
                'type': 'factor_model',
                'factor_loadings': factor_loadings,
                'factor_covariance': np.cov(factor_returns.T),
                'specific_risk': specific_risk,
                'explained_variance_ratio': pca.explained_variance_ratio_
            }
            
            return risk_model
            
        except Exception as e:
            self.logger.error(f"Error building risk model: {str(e)}")
            return {'type': 'sample_covariance'}
    
    def _optimize_mean_variance(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mean-variance optimization."""
        if not CVXPY_AVAILABLE:
            return self._optimize_mean_variance_scipy(inputs)
        
        try:
            expected_returns = inputs['expected_returns'].values
            covariance_matrix = inputs['covariance_matrix']
            current_positions = inputs['current_positions'].values
            n_assets = inputs['n_assets']
            
            # Decision variables
            weights = cp.Variable(n_assets)
            
            # Objective: maximize utility (return - risk penalty)
            portfolio_return = expected_returns.T @ weights
            portfolio_risk = cp.quad_form(weights, covariance_matrix)
            
            # Transaction costs
            turnover = cp.norm(weights - current_positions, 1)
            transaction_costs = self.config.transaction_cost_bps / 10000 * turnover
            
            # Objective function
            utility = portfolio_return - 0.5 * self.config.risk_aversion * portfolio_risk - transaction_costs
            objective = cp.Maximize(utility)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Fully invested
                weights >= -self.config.max_leverage,  # No excessive shorting
                weights <= self.config.max_position_size,  # Position size limits
                weights >= -self.config.max_position_size,
                cp.norm(weights, 1) <= self.config.max_leverage  # Leverage constraint
            ]
            
            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = pd.Series(weights.value, index=inputs['universe'])
                
                # Calculate expected portfolio metrics
                portfolio_return_val = np.dot(expected_returns, weights.value)
                portfolio_vol = np.sqrt(np.dot(weights.value, np.dot(covariance_matrix, weights.value)))
                
                result = {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return_val,
                    'expected_volatility': portfolio_vol,
                    'sharpe_ratio': portfolio_return_val / portfolio_vol if portfolio_vol > 0 else 0,
                    'turnover': np.sum(np.abs(weights.value - current_positions)),
                    'optimization_status': problem.status,
                    'objective_value': problem.value
                }
                
                return result
            else:
                self.logger.error(f"Optimization failed with status: {problem.status}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in mean-variance optimization: {str(e)}")
            return {}
    
    def _optimize_mean_variance_scipy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mean-variance optimization using scipy (fallback when cvxpy not available)."""
        try:
            expected_returns = inputs['expected_returns'].values
            covariance_matrix = inputs['covariance_matrix']
            current_positions = inputs['current_positions'].values
            n_assets = inputs['n_assets']
            
            # Objective function (negative because scipy minimizes)
            def objective(weights):
                portfolio_return = np.dot(expected_returns, weights)
                portfolio_risk = np.dot(weights, np.dot(covariance_matrix, weights))
                turnover = np.sum(np.abs(weights - current_positions))
                transaction_costs = self.config.transaction_cost_bps / 10000 * turnover
                
                return -(portfolio_return - 0.5 * self.config.risk_aversion * portfolio_risk - transaction_costs)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Fully invested
            ]
            
            # Bounds
            bounds = [(-self.config.max_position_size, self.config.max_position_size) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = optimize.minimize(
                objective, x0, method='SLSQP', 
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=inputs['universe'])
                
                # Calculate portfolio metrics
                portfolio_return_val = np.dot(expected_returns, result.x)
                portfolio_vol = np.sqrt(np.dot(result.x, np.dot(covariance_matrix, result.x)))
                
                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return_val,
                    'expected_volatility': portfolio_vol,
                    'sharpe_ratio': portfolio_return_val / portfolio_vol if portfolio_vol > 0 else 0,
                    'turnover': np.sum(np.abs(result.x - current_positions)),
                    'optimization_status': 'optimal',
                    'objective_value': -result.fun
                }
            else:
                self.logger.error(f"Scipy optimization failed: {result.message}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in scipy mean-variance optimization: {str(e)}")
            return {}
    
    def _optimize_black_litterman(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Black-Litterman optimization with regime-aware views."""
        try:
            covariance_matrix = inputs['covariance_matrix']
            alpha_signals = inputs['alpha_signals']
            n_assets = inputs['n_assets']
            universe = inputs['universe']
            
            # Market capitalization weights (proxy using equal weights)
            market_weights = np.ones(n_assets) / n_assets
            
            # Risk aversion parameter
            risk_aversion = self.config.risk_aversion
            
            # Implied equilibrium returns
            pi = risk_aversion * np.dot(covariance_matrix, market_weights)
            
            # Views matrix (P) and view returns (Q)
            # Use alpha signals as views
            P = np.eye(n_assets)  # Each asset has a view
            Q = alpha_signals.values  # Alpha signals as view returns
            
            # Uncertainty in views (Omega)
            # Higher uncertainty for weaker signals
            signal_strength = np.abs(alpha_signals.values)
            normalized_strength = signal_strength / (signal_strength.max() + 1e-8)
            view_uncertainty = np.diag(1.0 - normalized_strength) * np.mean(np.diag(covariance_matrix))
            Omega = view_uncertainty
            
            # Confidence in views (tau)
            tau = 1.0 / len(alpha_signals)  # Typical value
            
            # Black-Litterman formula
            M1 = linalg.inv(tau * covariance_matrix)
            M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
            M3 = np.dot(linalg.inv(tau * covariance_matrix), pi)
            M4 = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
            
            # New expected returns
            mu_bl = np.dot(linalg.inv(M1 + M2), M3 + M4)
            
            # New covariance matrix
            cov_bl = linalg.inv(M1 + M2)
            
            # Optimize with Black-Litterman inputs
            bl_inputs = inputs.copy()
            bl_inputs['expected_returns'] = pd.Series(mu_bl, index=universe)
            bl_inputs['covariance_matrix'] = cov_bl
            
            # Use mean-variance optimization with BL inputs
            result = self._optimize_mean_variance(bl_inputs)
            
            if result:
                result['method'] = 'black_litterman'
                result['implied_returns'] = pd.Series(pi, index=universe)
                result['bl_returns'] = pd.Series(mu_bl, index=universe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {}
    
    def _optimize_risk_parity(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Risk parity optimization."""
        try:
            covariance_matrix = inputs['covariance_matrix']
            n_assets = inputs['n_assets']
            universe = inputs['universe']
            
            # Risk budgeting - equal risk contribution
            target_risk_budget = np.ones(n_assets) / n_assets
            
            def risk_budget_objective(weights):
                # Portfolio volatility
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                
                # Marginal contributions to risk
                marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
                
                # Risk contributions
                risk_contrib = weights * marginal_contrib / portfolio_vol
                
                # Minimize sum of squared deviations from target
                return np.sum((risk_contrib - target_risk_budget)**2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Fully invested
                {'type': 'ineq', 'fun': lambda x: x}  # Long only
            ]
            
            # Bounds
            bounds = [(0.001, self.config.max_position_size) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = optimize.minimize(
                risk_budget_objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=universe)
                
                # Calculate portfolio metrics
                portfolio_vol = np.sqrt(np.dot(result.x, np.dot(covariance_matrix, result.x)))
                
                return {
                    'weights': optimal_weights,
                    'expected_volatility': portfolio_vol,
                    'optimization_status': 'optimal',
                    'method': 'risk_parity'
                }
            else:
                self.logger.error(f"Risk parity optimization failed: {result.message}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {str(e)}")
            return {}
    
    def _optimize_hierarchical_risk_parity(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Hierarchical Risk Parity optimization."""
        try:
            covariance_matrix = inputs['covariance_matrix']
            universe = inputs['universe']
            
            # Calculate correlation matrix
            std_devs = np.sqrt(np.diag(covariance_matrix))
            correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
            
            # Hierarchical clustering
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Distance matrix from correlation
            distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
            
            # Hierarchical clustering
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')
            
            # Recursive bisection for weight allocation
            weights = self._hrp_recursive_bisection(
                linkage_matrix, correlation_matrix, std_devs, list(range(len(universe)))
            )
            
            optimal_weights = pd.Series(weights, index=universe)
            
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            return {
                'weights': optimal_weights,
                'expected_volatility': portfolio_vol,
                'method': 'hierarchical_risk_parity',
                'optimization_status': 'optimal'
            }
            
        except Exception as e:
            self.logger.error(f"Error in HRP optimization: {str(e)}")
            return {}
    
    def _hrp_recursive_bisection(self, linkage_matrix, correlation_matrix, std_devs, assets):
        """Recursive bisection for HRP weight allocation."""
        # Base case: single asset
        if len(assets) == 1:
            return {assets[0]: 1.0}
        
        # Find the split point in the dendrogram
        # This is a simplified implementation
        n_assets = len(assets)
        
        if n_assets == 2:
            # Calculate inverse volatility weights for two assets
            vol1, vol2 = std_devs[assets[0]], std_devs[assets[1]]
            weight1 = (1/vol2) / (1/vol1 + 1/vol2)
            weight2 = 1 - weight1
            return {assets[0]: weight1, assets[1]: weight2}
        
        # Split into two clusters (simplified)
        mid_point = n_assets // 2
        cluster1 = assets[:mid_point]
        cluster2 = assets[mid_point:]
        
        # Recursive calls
        weights1 = self._hrp_recursive_bisection(linkage_matrix, correlation_matrix, std_devs, cluster1)
        weights2 = self._hrp_recursive_bisection(linkage_matrix, correlation_matrix, std_devs, cluster2)
        
        # Calculate cluster volatilities
        cluster1_vol = self._calculate_cluster_volatility(cluster1, correlation_matrix, std_devs, weights1)
        cluster2_vol = self._calculate_cluster_volatility(cluster2, correlation_matrix, std_devs, weights2)
        
        # Allocate between clusters using inverse volatility
        total_inv_vol = 1/cluster1_vol + 1/cluster2_vol
        cluster1_weight = (1/cluster1_vol) / total_inv_vol
        cluster2_weight = 1 - cluster1_weight
        
        # Scale individual weights
        final_weights = {}
        for asset, weight in weights1.items():
            final_weights[asset] = weight * cluster1_weight
        for asset, weight in weights2.items():
            final_weights[asset] = weight * cluster2_weight
        
        return final_weights
    
    def _calculate_cluster_volatility(self, cluster, correlation_matrix, std_devs, weights):
        """Calculate volatility of a cluster."""
        cluster_weights = np.array([weights[asset] for asset in cluster])
        cluster_cov = correlation_matrix[np.ix_(cluster, cluster)] * np.outer(
            std_devs[cluster], std_devs[cluster]
        )
        return np.sqrt(np.dot(cluster_weights, np.dot(cluster_cov, cluster_weights)))
    
    def _optimize_robust(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Robust optimization accounting for parameter uncertainty."""
        try:
            # Implement robust optimization using worst-case scenarios
            # This is a simplified version - full implementation would use uncertainty sets
            
            expected_returns = inputs['expected_returns'].values
            covariance_matrix = inputs['covariance_matrix']
            
            # Add uncertainty to expected returns (±20% of signal strength)
            uncertainty_factor = 0.2
            return_uncertainty = np.abs(expected_returns) * uncertainty_factor
            
            # Robust expected returns (conservative estimate)
            robust_returns = expected_returns - return_uncertainty
            
            # Inflate covariance matrix to account for estimation error
            covariance_inflation = 1.2
            robust_covariance = covariance_matrix * covariance_inflation
            
            # Update inputs with robust estimates
            robust_inputs = inputs.copy()
            robust_inputs['expected_returns'] = pd.Series(robust_returns, index=inputs['universe'])
            robust_inputs['covariance_matrix'] = robust_covariance
            
            # Use mean-variance optimization with robust inputs
            result = self._optimize_mean_variance(robust_inputs)
            
            if result:
                result['method'] = 'robust_optimization'
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in robust optimization: {str(e)}")
            return {}
    
    def _postprocess_optimization_result(self, result: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process optimization results."""
        try:
            if 'weights' not in result:
                return result
            
            weights = result['weights']
            
            # Calculate additional portfolio metrics
            result['portfolio_metrics'] = self._calculate_portfolio_metrics(weights, inputs)
            
            # Risk attribution
            result['risk_attribution'] = self._calculate_risk_attribution(weights, inputs)
            
            # Transaction costs
            if 'current_positions' in inputs:
                result['transaction_analysis'] = self._analyze_transactions(
                    weights, inputs['current_positions']
                )
            
            # Factor exposures (if risk model available)
            if inputs.get('risk_model', {}).get('type') == 'factor_model':
                result['factor_exposures'] = self._calculate_factor_exposures(weights, inputs)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error post-processing results: {str(e)}")
            return result
    
    def _calculate_portfolio_metrics(self, weights: pd.Series, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['gross_exposure'] = weights.abs().sum()
            metrics['net_exposure'] = weights.sum()
            metrics['long_exposure'] = weights[weights > 0].sum()
            metrics['short_exposure'] = weights[weights < 0].sum()
            metrics['n_positions'] = (weights.abs() > 1e-6).sum()
            
            # Concentration metrics
            metrics['max_position'] = weights.abs().max()
            metrics['top_5_concentration'] = weights.abs().nlargest(5).sum()
            metrics['herfindahl_index'] = (weights ** 2).sum()
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(inputs['covariance_matrix']))
            weighted_avg_vol = np.sum(weights.abs() * individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights.values, np.dot(inputs['covariance_matrix'], weights.values)))
            metrics['diversification_ratio'] = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def _calculate_risk_attribution(self, weights: pd.Series, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk attribution by asset."""
        try:
            covariance_matrix = inputs['covariance_matrix']
            weights_array = weights.values
            
            # Portfolio variance
            portfolio_var = np.dot(weights_array, np.dot(covariance_matrix, weights_array))
            
            # Marginal contribution to risk
            marginal_contrib = np.dot(covariance_matrix, weights_array)
            
            # Component contribution to risk
            component_contrib = weights_array * marginal_contrib
            
            # Percentage contributions
            pct_contrib = component_contrib / portfolio_var * 100
            
            risk_attribution = pd.Series(pct_contrib, index=weights.index)
            
            return {
                'marginal_contributions': pd.Series(marginal_contrib, index=weights.index),
                'component_contributions': pd.Series(component_contrib, index=weights.index),
                'percentage_contributions': risk_attribution
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk attribution: {str(e)}")
            return {}
    
    def _analyze_transactions(self, new_weights: pd.Series, current_weights: pd.Series) -> Dict[str, Any]:
        """Analyze transactions and costs."""
        try:
            # Calculate trades
            trades = new_weights - current_weights.reindex(new_weights.index, fill_value=0.0)
            
            # Transaction costs
            transaction_costs = trades.abs() * self.config.transaction_cost_bps / 10000
            
            return {
                'trades': trades,
                'total_turnover': trades.abs().sum(),
                'transaction_costs': transaction_costs,
                'total_transaction_cost': transaction_costs.sum(),
                'buys': trades[trades > 0],
                'sells': trades[trades < 0]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing transactions: {str(e)}")
            return {}
    
    def _calculate_factor_exposures(self, weights: pd.Series, inputs: Dict[str, Any]) -> pd.Series:
        """Calculate factor exposures."""
        try:
            risk_model = inputs['risk_model']
            factor_loadings = risk_model['factor_loadings']
            
            # Portfolio factor exposures
            portfolio_exposures = np.dot(weights.values, factor_loadings)
            
            factor_names = [f'Factor_{i}' for i in range(len(portfolio_exposures))]
            
            return pd.Series(portfolio_exposures, index=factor_names)
            
        except Exception as e:
            self.logger.error(f"Error calculating factor exposures: {str(e)}")
            return pd.Series()
    
    def generate_optimization_report(self, optimization_result: Dict[str, Any]) -> str:
        """Generate a comprehensive optimization report."""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("PORTFOLIO OPTIMIZATION REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Optimization Method: {optimization_result.get('method', 'Unknown')}")
            report_lines.append("")
            
            # Portfolio weights
            if 'weights' in optimization_result:
                weights = optimization_result['weights']
                report_lines.append("PORTFOLIO WEIGHTS")
                report_lines.append("-" * 40)
                
                # Top 10 positions
                top_positions = weights.abs().nlargest(10)
                for asset, weight in top_positions.items():
                    report_lines.append(f"{asset:10s}: {weight:8.4f} ({weight*100:6.2f}%)")
                report_lines.append("")
            
            # Portfolio metrics
            if 'portfolio_metrics' in optimization_result:
                metrics = optimization_result['portfolio_metrics']
                report_lines.append("PORTFOLIO METRICS")
                report_lines.append("-" * 40)
                report_lines.append(f"Gross Exposure: {metrics.get('gross_exposure', 0):.4f}")
                report_lines.append(f"Net Exposure: {metrics.get('net_exposure', 0):.4f}")
                report_lines.append(f"Number of Positions: {metrics.get('n_positions', 0)}")
                report_lines.append(f"Max Position: {metrics.get('max_position', 0):.4f}")
                report_lines.append(f"Top 5 Concentration: {metrics.get('top_5_concentration', 0):.4f}")
                report_lines.append("")
            
            # Expected performance
            if 'expected_return' in optimization_result:
                report_lines.append("EXPECTED PERFORMANCE")
                report_lines.append("-" * 40)
                report_lines.append(f"Expected Return: {optimization_result['expected_return']:.4f}")
                report_lines.append(f"Expected Volatility: {optimization_result.get('expected_volatility', 0):.4f}")
                report_lines.append(f"Expected Sharpe Ratio: {optimization_result.get('sharpe_ratio', 0):.4f}")
                report_lines.append("")
            
            # Transaction analysis
            if 'transaction_analysis' in optimization_result:
                txn = optimization_result['transaction_analysis']
                report_lines.append("TRANSACTION ANALYSIS")
                report_lines.append("-" * 40)
                report_lines.append(f"Total Turnover: {txn.get('total_turnover', 0):.4f}")
                report_lines.append(f"Transaction Costs: {txn.get('total_transaction_cost', 0):.6f}")
                report_lines.append("")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return "Error generating optimization report"