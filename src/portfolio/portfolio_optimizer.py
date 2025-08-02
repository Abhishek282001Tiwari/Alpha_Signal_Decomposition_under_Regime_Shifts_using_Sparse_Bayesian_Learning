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
            portfolio_vol = result.get('expected_volatility', 0)
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
    
    def calculate_portfolio_performance(self, 
                                      weights: pd.Series,
                                      returns_data: pd.DataFrame,
                                      start_date: datetime,
                                      end_date: datetime) -> Dict[str, Any]:
        """Calculate historical portfolio performance."""
        try:
            # Filter returns data
            period_returns = returns_data.loc[start_date:end_date]
            
            # Calculate portfolio returns
            portfolio_returns = (period_returns * weights).sum(axis=1)
            
            # Performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns / rolling_max) - 1
            max_drawdown = drawdowns.min()
            
            # Additional metrics
            sortino_ratio = annualized_return / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252))
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            performance = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': (portfolio_returns > 0).mean(),
                'portfolio_returns': portfolio_returns,
                'cumulative_returns': cumulative_returns,
                'drawdowns': drawdowns
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio performance: {str(e)}")
            return {}
    
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
                       returns: pd.DataFrame,
                       market_caps: Optional[pd.Series] = None,
                       regime_probabilities: Optional[pd.DataFrame] = None):
        """
        Set market data for optimization.
        
        Args:
            returns: Historical returns DataFrame
            market_caps: Market capitalizations for assets
            regime_probabilities: Regime probability DataFrame
        """
        self.returns = returns
        self.market_caps = market_caps
        self.regime_probabilities = regime_probabilities
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        
        # Calculate regime-specific parameters if regime-aware
        if self.regime_aware and regime_probabilities is not None:
            self._calculate_regime_parameters()
        else:
            self._calculate_global_parameters()
    
    def _calculate_regime_parameters(self):
        """Calculate regime-specific expected returns and covariances."""
        n_regimes = self.regime_probabilities.shape[1]
        
        for regime in range(n_regimes):
            regime_weights = self.regime_probabilities.iloc[:, regime]
            
            # Weighted expected returns
            weighted_returns = self.returns.multiply(regime_weights, axis=0)
            regime_mean = weighted_returns.sum() / regime_weights.sum()
            
            # Weighted covariance matrix
            demeaned_returns = self.returns.subtract(regime_mean, axis=1)
            weighted_demeaned = demeaned_returns.multiply(np.sqrt(regime_weights), axis=0)
            regime_cov = weighted_demeaned.T @ weighted_demeaned / regime_weights.sum()
            
            # Regularize covariance matrix
            regime_cov = self._regularize_covariance(regime_cov)
            
            self.regime_expected_returns[regime] = regime_mean
            self.regime_covariances[regime] = regime_cov
        
        # Calculate aggregate expected returns and covariance
        self._calculate_aggregate_parameters()
    
    def _calculate_global_parameters(self):
        """Calculate global expected returns and covariance."""
        self.expected_returns = self.returns.mean()
        self.covariance_matrix = self._regularize_covariance(self.returns.cov())
    
    def _calculate_aggregate_parameters(self):
        """Calculate regime-weighted aggregate parameters."""
        if not self.regime_probabilities.empty:
            # Current regime probabilities (last observation)
            current_regime_probs = self.regime_probabilities.iloc[-1].values
            
            # Weighted expected returns
            self.expected_returns = pd.Series(0, index=self.assets)
            for regime, prob in enumerate(current_regime_probs):
                if regime in self.regime_expected_returns:
                    self.expected_returns += prob * self.regime_expected_returns[regime]
            
            # Weighted covariance (accounting for regime uncertainty)
            self.covariance_matrix = pd.DataFrame(0, index=self.assets, columns=self.assets)
            for regime, prob in enumerate(current_regime_probs):
                if regime in self.regime_covariances:
                    regime_cov = self.regime_covariances[regime]
                    regime_mean = self.regime_expected_returns[regime]
                    
                    # Covariance decomposition: E[Var] + Var[E]
                    self.covariance_matrix += prob * regime_cov
                    
                    # Add uncertainty from regime switching
                    mean_diff = (regime_mean - self.expected_returns).values
                    self.covariance_matrix += prob * np.outer(mean_diff, mean_diff)
    
    def _regularize_covariance(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Regularize covariance matrix using Ledoit-Wolf shrinkage."""
        # Convert to numpy for regularization
        cov_np = cov_matrix.values
        
        # Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        cov_regularized = lw.fit(self.returns.values).covariance_
        
        # Convert back to DataFrame
        return pd.DataFrame(cov_regularized, index=cov_matrix.index, columns=cov_matrix.columns)
    
    def set_views(self, 
                 views_matrix: np.ndarray,
                 views_returns: np.ndarray,
                 views_uncertainty: Optional[np.ndarray] = None):
        """
        Set Black-Litterman views.
        
        Args:
            views_matrix: Matrix mapping assets to views (n_views x n_assets)
            views_returns: Expected returns for each view
            views_uncertainty: Uncertainty matrix for views
        """
        self.views_matrix = views_matrix
        self.views_returns = views_returns
        
        if views_uncertainty is None:
            # Default uncertainty based on view confidence
            self.views_uncertainty = np.eye(len(views_returns)) * 0.01
        else:
            self.views_uncertainty = views_uncertainty
    
    def optimize_portfolio(self, 
                          current_weights: Optional[np.ndarray] = None,
                          constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio weights based on selected method.
        
        Args:
            current_weights: Current portfolio weights for turnover calculation
            constraints: Additional portfolio constraints
        
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Optimizing portfolio using {self.optimization_method}")
        
        if current_weights is None:
            current_weights = np.zeros(self.n_assets)
        
        if self.optimization_method == 'mean_variance':
            return self._optimize_mean_variance(current_weights, constraints)
        elif self.optimization_method == 'black_litterman':
            return self._optimize_black_litterman(current_weights, constraints)
        elif self.optimization_method == 'risk_parity':
            return self._optimize_risk_parity(current_weights, constraints)
        elif self.optimization_method == 'hierarchical_risk_parity':
            return self._optimize_hierarchical_risk_parity(current_weights, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _optimize_mean_variance(self, 
                              current_weights: np.ndarray,
                              constraints: Optional[Dict]) -> Dict:
        """Mean-variance optimization with regime awareness."""
        
        # Decision variable: portfolio weights
        w = cp.Variable(self.n_assets)
        
        # Expected return
        mu = self.expected_returns.values
        portfolio_return = mu.T @ w
        
        # Portfolio variance
        Sigma = self.covariance_matrix.values
        portfolio_variance = cp.quad_form(w, Sigma)
        
        # Transaction costs
        if current_weights is not None:
            turnover = cp.norm(w - current_weights, 1)
            transaction_costs = self.turnover_penalty * turnover
        else:
            transaction_costs = 0
        
        # Objective: maximize return - risk penalty - transaction costs
        objective = cp.Maximize(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance - transaction_costs)
        
        # Constraints
        constraints_list = [cp.sum(w) == 1]  # Fully invested
        
        # Leverage constraint
        if self.max_leverage < np.inf:
            constraints_list.append(cp.norm(w, 1) <= self.max_leverage)
        
        # Position size constraints
        if self.max_position_size < 1.0:
            constraints_list.append(w <= self.max_position_size)
            constraints_list.append(w >= -self.max_position_size)
        
        # Additional constraints
        if constraints:
            if 'long_only' in constraints and constraints['long_only']:
                constraints_list.append(w >= 0)
            
            if 'sector_constraints' in constraints:
                sector_constraints = constraints['sector_constraints']
                for sector, (min_weight, max_weight) in sector_constraints.items():
                    sector_mask = constraints.get('sector_mappings', {}).get(sector, [])
                    if sector_mask:
                        sector_weight = cp.sum(w[sector_mask])
                        constraints_list.append(sector_weight >= min_weight)
                        constraints_list.append(sector_weight <= max_weight)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = w.value
                
                # Calculate portfolio statistics
                portfolio_stats = self._calculate_portfolio_statistics(optimal_weights)
                
                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_stats['expected_return'],
                    'volatility': portfolio_stats['volatility'],
                    'sharpe_ratio': portfolio_stats['sharpe_ratio'],
                    'turnover': np.sum(np.abs(optimal_weights - current_weights)),
                    'max_weight': np.max(np.abs(optimal_weights)),
                    'optimization_status': 'optimal',
                    'solver_status': problem.status
                }
            else:
                self.logger.error(f"Optimization failed with status: {problem.status}")
                return {'optimization_status': 'failed', 'solver_status': problem.status}
                
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def _optimize_black_litterman(self, 
                                current_weights: np.ndarray,
                                constraints: Optional[Dict]) -> Dict:
        """Black-Litterman optimization with sparse Bayesian views."""
        
        # Market equilibrium returns (reverse optimization)
        if self.market_caps is not None:
            market_weights = self.market_caps / self.market_caps.sum()
            market_implied_returns = self.risk_aversion * self.covariance_matrix.values @ market_weights.values
        else:
            # Equal-weighted market portfolio
            market_weights = np.ones(self.n_assets) / self.n_assets
            market_implied_returns = self.risk_aversion * self.covariance_matrix.values @ market_weights
        
        # Prior covariance of expected returns
        prior_covariance = self.tau * self.covariance_matrix.values
        
        # Black-Litterman posterior
        if self.views_matrix is not None and self.views_returns is not None:
            # Incorporate views
            P = self.views_matrix
            Q = self.views_returns
            Omega = self.views_uncertainty
            
            # Posterior mean and covariance
            M1 = linalg.inv(prior_covariance)
            M2 = P.T @ linalg.inv(Omega) @ P
            M3 = linalg.inv(M1 + M2)
            
            posterior_mean = M3 @ (M1 @ market_implied_returns + P.T @ linalg.inv(Omega) @ Q)
            posterior_covariance = M3
        else:
            # No views - use market equilibrium
            posterior_mean = market_implied_returns
            posterior_covariance = prior_covariance
        
        # Update covariance matrix for optimization
        bl_covariance = self.covariance_matrix.values + posterior_covariance
        
        # Optimize using Black-Litterman parameters
        w = cp.Variable(self.n_assets)
        
        portfolio_return = posterior_mean.T @ w
        portfolio_variance = cp.quad_form(w, bl_covariance)
        
        # Transaction costs
        if current_weights is not None:
            turnover = cp.norm(w - current_weights, 1)
            transaction_costs = self.turnover_penalty * turnover
        else:
            transaction_costs = 0
        
        objective = cp.Maximize(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance - transaction_costs)
        
        # Standard constraints
        constraints_list = [cp.sum(w) == 1]
        
        if self.max_leverage < np.inf:
            constraints_list.append(cp.norm(w, 1) <= self.max_leverage)
        
        if self.max_position_size < 1.0:
            constraints_list.append(w <= self.max_position_size)
            constraints_list.append(w >= -self.max_position_size)
        
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = w.value
                portfolio_stats = self._calculate_portfolio_statistics(optimal_weights)
                
                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_stats['expected_return'],
                    'volatility': portfolio_stats['volatility'],
                    'sharpe_ratio': portfolio_stats['sharpe_ratio'],
                    'turnover': np.sum(np.abs(optimal_weights - current_weights)),
                    'bl_posterior_return': posterior_mean,
                    'bl_posterior_covariance': posterior_covariance,
                    'optimization_status': 'optimal'
                }
            else:
                return {'optimization_status': 'failed', 'solver_status': problem.status}
                
        except Exception as e:
            self.logger.error(f"Black-Litterman optimization error: {str(e)}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def _optimize_risk_parity(self, 
                            current_weights: np.ndarray,
                            constraints: Optional[Dict]) -> Dict:
        """Risk parity optimization."""
        
        def risk_parity_objective(weights):
            """Objective function for risk parity."""
            weights = np.array(weights)
            
            # Portfolio variance
            portfolio_var = weights.T @ self.covariance_matrix.values @ weights
            
            # Marginal risk contributions
            marginal_risk = self.covariance_matrix.values @ weights
            
            # Risk contributions
            risk_contributions = weights * marginal_risk / portfolio_var
            
            # Target equal risk contributions
            target_risk = np.ones(self.n_assets) / self.n_assets
            
            # Sum of squared deviations from target
            return np.sum((risk_contributions - target_risk)**2)
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Fully invested
        ]
        
        # Bounds
        bounds = [(0, self.max_position_size) for _ in range(self.n_assets)]
        
        # Optimize
        try:
            result = optimize.minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_stats = self._calculate_portfolio_statistics(optimal_weights)
                
                # Calculate risk contributions
                portfolio_var = optimal_weights.T @ self.covariance_matrix.values @ optimal_weights
                marginal_risk = self.covariance_matrix.values @ optimal_weights
                risk_contributions = optimal_weights * marginal_risk / portfolio_var
                
                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_stats['expected_return'],
                    'volatility': portfolio_stats['volatility'],
                    'sharpe_ratio': portfolio_stats['sharpe_ratio'],
                    'risk_contributions': risk_contributions,
                    'risk_concentration': np.sum(risk_contributions**2),
                    'turnover': np.sum(np.abs(optimal_weights - current_weights)),
                    'optimization_status': 'optimal'
                }
            else:
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization error: {str(e)}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def _optimize_hierarchical_risk_parity(self, 
                                         current_weights: np.ndarray,
                                         constraints: Optional[Dict]) -> Dict:
        """Hierarchical Risk Parity (HRP) optimization."""
        
        # Calculate distance matrix from correlation
        corr_matrix = self.returns.corr()
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
        from scipy.spatial.distance import squareform
        
        # Convert to condensed distance matrix
        condensed_distances = squareform(distance_matrix.values, checks=False)
        
        # Perform clustering
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # Get cluster tree
        cluster_tree = to_tree(linkage_matrix, rd_return_dict=False)
        
        # Recursive bisection for weight allocation
        def _recursive_bisection(cluster_node, cov_matrix):
            """Recursively allocate weights using bisection."""
            if cluster_node.is_leaf():
                return {cluster_node.id: 1.0}
            
            # Get left and right clusters
            left_cluster = cluster_node.left
            right_cluster = cluster_node.right
            
            # Get cluster members
            left_members = self._get_cluster_members(left_cluster)
            right_members = self._get_cluster_members(right_cluster)
            
            # Calculate cluster variances
            left_cov = cov_matrix.loc[left_members, left_members]
            right_cov = cov_matrix.loc[right_members, right_members]
            
            # Inverse variance allocation
            left_var = self._calculate_cluster_variance(left_cov)
            right_var = self._calculate_cluster_variance(right_cov)
            
            # Allocation weights
            total_inv_var = 1/left_var + 1/right_var
            left_weight = (1/left_var) / total_inv_var
            right_weight = (1/right_var) / total_inv_var
            
            # Recursive allocation
            left_weights = _recursive_bisection(left_cluster, cov_matrix)
            right_weights = _recursive_bisection(right_cluster, cov_matrix)
            
            # Scale weights
            weights = {}
            for asset, weight in left_weights.items():
                weights[asset] = weight * left_weight
            for asset, weight in right_weights.items():
                weights[asset] = weight * right_weight
            
            return weights
        
        # Calculate HRP weights
        try:
            weight_dict = _recursive_bisection(cluster_tree, self.covariance_matrix)
            
            # Convert to array
            optimal_weights = np.array([weight_dict.get(i, 0.0) for i in range(self.n_assets)])
            
            # Normalize weights
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            portfolio_stats = self._calculate_portfolio_statistics(optimal_weights)
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_stats['expected_return'],
                'volatility': portfolio_stats['volatility'],
                'sharpe_ratio': portfolio_stats['sharpe_ratio'],
                'turnover': np.sum(np.abs(optimal_weights - current_weights)),
                'clustering_info': {
                    'linkage_matrix': linkage_matrix,
                    'distance_matrix': distance_matrix.values
                },
                'optimization_status': 'optimal'
            }
            
        except Exception as e:
            self.logger.error(f"HRP optimization error: {str(e)}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def _get_cluster_members(self, cluster_node):
        """Get member indices of a cluster node."""
        if cluster_node.is_leaf():
            return [cluster_node.id]
        else:
            left_members = self._get_cluster_members(cluster_node.left)
            right_members = self._get_cluster_members(cluster_node.right)
            return left_members + right_members
    
    def _calculate_cluster_variance(self, cluster_cov):
        """Calculate variance of equal-weighted cluster."""
        n = len(cluster_cov)
        equal_weights = np.ones(n) / n
        return equal_weights.T @ cluster_cov.values @ equal_weights
    
    def _calculate_portfolio_statistics(self, weights: np.ndarray) -> Dict:
        """Calculate portfolio statistics for given weights."""
        expected_return = np.dot(weights, self.expected_returns.values)
        portfolio_variance = weights.T @ self.covariance_matrix.values @ weights
        volatility = np.sqrt(portfolio_variance)
        
        # Assume risk-free rate of 2% for Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_variance': portfolio_variance
        }
    
    def calculate_efficient_frontier(self, 
                                   n_points: int = 50,
                                   return_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Calculate efficient frontier for mean-variance optimization.
        
        Args:
            n_points: Number of points on the frontier
            return_range: Range of returns to consider
        
        Returns:
            Dictionary with frontier data
        """
        if return_range is None:
            # Use range based on individual asset returns
            min_return = self.expected_returns.min()
            max_return = self.expected_returns.max()
            return_range = (min_return, max_return)
        
        target_returns = np.linspace(return_range[0], return_range[1], n_points)
        
        frontier_weights = []
        frontier_volatilities = []
        frontier_returns = []
        
        for target_return in target_returns:
            # Optimize for minimum variance given target return
            w = cp.Variable(self.n_assets)
            
            portfolio_variance = cp.quad_form(w, self.covariance_matrix.values)
            
            constraints = [
                cp.sum(w) == 1,  # Fully invested
                self.expected_returns.values.T @ w == target_return  # Target return
            ]
            
            # Position size constraints
            if self.max_position_size < 1.0:
                constraints.extend([
                    w <= self.max_position_size,
                    w >= -self.max_position_size
                ])
            
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
            
            try:
                problem.solve(solver=cp.ECOS)
                
                if problem.status == cp.OPTIMAL:
                    weights = w.value
                    volatility = np.sqrt(portfolio_variance.value)
                    
                    frontier_weights.append(weights)
                    frontier_volatilities.append(volatility)
                    frontier_returns.append(target_return)
                    
            except:
                continue
        
        return {
            'returns': frontier_returns,
            'volatilities': frontier_volatilities,
            'weights': frontier_weights,
            'sharpe_ratios': [(r - 0.02) / v for r, v in zip(frontier_returns, frontier_volatilities)]
        }
    
    def robust_optimization(self, 
                          uncertainty_sets: Dict,
                          current_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Robust optimization accounting for parameter uncertainty.
        
        Args:
            uncertainty_sets: Dictionary defining uncertainty in parameters
            current_weights: Current portfolio weights
        
        Returns:
            Robust optimization results
        """
        if current_weights is None:
            current_weights = np.zeros(self.n_assets)
        
        # Decision variables
        w = cp.Variable(self.n_assets)
        
        # Worst-case return uncertainty
        mu_nominal = self.expected_returns.values
        if 'return_uncertainty' in uncertainty_sets:
            kappa_mu = uncertainty_sets['return_uncertainty']
            # Worst-case expected return
            worst_case_return = mu_nominal.T @ w - kappa_mu * cp.norm(w, 2)
        else:
            worst_case_return = mu_nominal.T @ w
        
        # Worst-case variance uncertainty
        Sigma_nominal = self.covariance_matrix.values
        if 'covariance_uncertainty' in uncertainty_sets:
            kappa_sigma = uncertainty_sets['covariance_uncertainty']
            # Robust covariance matrix
            robust_variance = cp.quad_form(w, Sigma_nominal) + kappa_sigma * cp.norm(w, 2)**2
        else:
            robust_variance = cp.quad_form(w, Sigma_nominal)
        
        # Transaction costs
        turnover = cp.norm(w - current_weights, 1)
        transaction_costs = self.turnover_penalty * turnover
        
        # Robust objective: maximize worst-case utility
        objective = cp.Maximize(worst_case_return - 0.5 * self.risk_aversion * robust_variance - transaction_costs)
        
        # Constraints
        constraints = [cp.sum(w) == 1]
        
        if self.max_leverage < np.inf:
            constraints.append(cp.norm(w, 1) <= self.max_leverage)
        
        if self.max_position_size < 1.0:
            constraints.extend([
                w <= self.max_position_size,
                w >= -self.max_position_size
            ])
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = w.value
                portfolio_stats = self._calculate_portfolio_statistics(optimal_weights)
                
                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_stats['expected_return'],
                    'volatility': portfolio_stats['volatility'],
                    'sharpe_ratio': portfolio_stats['sharpe_ratio'],
                    'turnover': np.sum(np.abs(optimal_weights - current_weights)),
                    'robust_return': worst_case_return.value,
                    'robust_variance': robust_variance.value,
                    'optimization_status': 'optimal'
                }
            else:
                return {'optimization_status': 'failed', 'solver_status': problem.status}
                
        except Exception as e:
            self.logger.error(f"Robust optimization error: {str(e)}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def set_factor_model(self, 
                        factor_exposures: pd.DataFrame,
                        factor_covariance: pd.DataFrame,
                        specific_risks: pd.Series):
        """
        Set factor model for risk decomposition.
        
        Args:
            factor_exposures: Asset exposures to factors (assets x factors)
            factor_covariance: Factor covariance matrix
            specific_risks: Asset-specific risk (idiosyncratic)
        """
        self.factor_exposures = factor_exposures
        self.factor_covariance = factor_covariance
        self.specific_risks = specific_risks
        
        # Update covariance matrix using factor model
        B = factor_exposures.values
        F = factor_covariance.values
        D = np.diag(specific_risks.values**2)
        
        # Covariance = B * F * B' + D
        self.covariance_matrix = pd.DataFrame(
            B @ F @ B.T + D,
            index=factor_exposures.index,
            columns=factor_exposures.index
        )
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization setup and parameters."""
        return {
            'optimization_method': self.optimization_method,
            'regime_aware': self.regime_aware,
            'transaction_cost_model': self.transaction_cost_model,
            'max_leverage': self.max_leverage,
            'max_position_size': self.max_position_size,
            'risk_aversion': self.risk_aversion,
            'turnover_penalty': self.turnover_penalty,
            'n_assets': self.n_assets,
            'assets': self.assets,
            'has_views': self.views_matrix is not None,
            'has_factor_model': self.factor_exposures is not None,
            'n_regimes': len(self.regime_expected_returns) if self.regime_aware else None
        }