import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available. Install with: pip install hmmlearn")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.stats.diagnostic import breaks_cusumolsresid, het_white
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.tsa.regime_switching import markov_regression, markov_autoregression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. Install with: pip install statsmodels")

warnings.filterwarnings('ignore')

@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection methods."""
    n_regimes: int = 3
    methods: List[str] = None  # ['hmm', 'ms_var', 'structural_breaks', 'gaussian_mixture']
    lookback_window: int = 252  # Trading days
    min_regime_duration: int = 20  # Minimum days in regime
    confidence_threshold: float = 0.7  # Minimum probability for regime assignment
    features_to_use: List[str] = None  # If None, use all available

class AdvancedRegimeDetector:
    """
    Advanced multi-regime identification using various econometric and machine learning approaches.
    Supports HMM, MS-VAR, Structural Breaks, Gaussian Mixture Models, and ensemble methods.
    """
    
    def __init__(self, config: RegimeDetectionConfig = None):
        """
        Initialize advanced regime detector.
        
        Args:
            config: Configuration object for regime detection
        """
        self.config = config or RegimeDetectionConfig()
        if self.config.methods is None:
            self.config.methods = ['hmm', 'gaussian_mixture', 'structural_breaks']
        
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
        # Model storage
        self.models = {}
        self.regime_probabilities = {}
        self.regime_states = {}
        self.ensemble_weights = {}
        self.feature_importance = {}
        
        # Performance tracking
        self.model_performance = {}
        self.regime_statistics = {}
        
    def detect_regimes(self, 
                      data: pd.DataFrame,
                      target_variable: str = 'returns',
                      feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive regime detection using multiple methods.
        
        Args:
            data: Input DataFrame with market data and features
            target_variable: Target variable for regime detection (e.g., 'returns')
            feature_columns: Specific features to use for regime detection
        
        Returns:
            Dictionary containing regime probabilities and states from all methods
        """
        self.logger.info("Starting comprehensive regime detection")
        
        # Prepare data
        processed_data = self._prepare_data(data, target_variable, feature_columns)
        
        if processed_data.empty:
            self.logger.error("No valid data for regime detection")
            return {}
        
        results = {}
        
        # Apply each detection method
        for method in self.config.methods:
            try:
                self.logger.info(f"Applying {method} regime detection")
                
                if method == 'hmm' and HMM_AVAILABLE:
                    method_result = self._fit_hmm_regime(processed_data)
                elif method == 'ms_var' and STATSMODELS_AVAILABLE:
                    method_result = self._fit_ms_var_regime(processed_data, target_variable)
                elif method == 'gaussian_mixture':
                    method_result = self._fit_gaussian_mixture_regime(processed_data)
                elif method == 'structural_breaks' and STATSMODELS_AVAILABLE:
                    method_result = self._detect_structural_breaks(processed_data, target_variable)
                elif method == 'threshold_var':
                    method_result = self._fit_threshold_var_regime(processed_data, target_variable)
                else:
                    self.logger.warning(f"Method {method} not available or not implemented")
                    continue
                
                if method_result:
                    results[method] = method_result
                    self.models[method] = method_result.get('model')
                    self.regime_probabilities[method] = method_result.get('probabilities')
                    self.regime_states[method] = method_result.get('states')
                
            except Exception as e:
                self.logger.error(f"Error in {method} regime detection: {str(e)}")
                continue
        
        if not results:
            self.logger.error("No regime detection methods succeeded")
            return {}
        
        # Ensemble combination of methods
        ensemble_result = self._create_ensemble_regimes(results, processed_data.index)
        results['ensemble'] = ensemble_result
        
        # Calculate regime statistics
        self._calculate_regime_statistics(results, processed_data)
        
        self.logger.info(f"Regime detection completed using {len(results)} methods")
        return results
    
    def _prepare_data(self, 
                     data: pd.DataFrame, 
                     target_variable: str,
                     feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare and validate data for regime detection."""
        try:
            # Select features
            if feature_columns is None:
                if self.config.features_to_use:
                    feature_columns = self.config.features_to_use
                else:
                    # Use numeric columns excluding the target
                    feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                                     if col != target_variable]
            
            # Ensure target variable is included
            all_columns = list(set([target_variable] + feature_columns))
            available_columns = [col for col in all_columns if col in data.columns]
            
            if not available_columns:
                self.logger.error("No valid columns found in data")
                return pd.DataFrame()
            
            # Extract relevant data
            regime_data = data[available_columns].copy()
            
            # Handle missing values
            regime_data = regime_data.dropna()
            
            if len(regime_data) < self.config.lookback_window:
                self.logger.warning(f"Insufficient data: {len(regime_data)} < {self.config.lookback_window}")
            
            # Normalize features (excluding target if it's returns)
            feature_cols_to_normalize = [col for col in regime_data.columns if col != target_variable]
            if feature_cols_to_normalize:
                regime_data[feature_cols_to_normalize] = self.scaler.fit_transform(
                    regime_data[feature_cols_to_normalize]
                )
            
            return regime_data
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return pd.DataFrame()
    
    def _fit_hmm_regime(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Fit Hidden Markov Model for regime detection."""
        if not HMM_AVAILABLE:
            return None
        
        try:
            # Use multiple features for HMM
            X = data.values
            
            # Try different covariance types
            best_model = None
            best_score = -np.inf
            
            for covariance_type in ['diag', 'full', 'tied']:
                try:
                    # Gaussian HMM
                    model = hmm.GaussianHMM(
                        n_components=self.config.n_regimes,
                        covariance_type=covariance_type,
                        n_iter=1000,
                        random_state=42
                    )
                    
                    model.fit(X)
                    score = model.score(X)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        
                except Exception as e:
                    self.logger.warning(f"HMM with {covariance_type} covariance failed: {str(e)}")
                    continue
            
            if best_model is None:
                return None
            
            # Get regime states and probabilities
            states = best_model.predict(X)
            probabilities = best_model.predict_proba(X)
            
            # Create probability DataFrame
            prob_df = pd.DataFrame(
                probabilities,
                index=data.index,
                columns=[f'Regime_{i}' for i in range(self.config.n_regimes)]
            )
            
            # Calculate regime statistics
            regime_stats = self._calculate_hmm_statistics(best_model, X, states)
            
            return {
                'model': best_model,
                'states': pd.Series(states, index=data.index),
                'probabilities': prob_df,
                'log_likelihood': best_score,
                'aic': -2 * best_score + 2 * self._count_hmm_parameters(best_model),
                'bic': -2 * best_score + np.log(len(X)) * self._count_hmm_parameters(best_model),
                'regime_statistics': regime_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error in HMM regime detection: {str(e)}")
            return None
    
    def _fit_ms_var_regime(self, data: pd.DataFrame, target_variable: str) -> Optional[Dict[str, Any]]:
        """Fit Markov-Switching Vector Autoregression model."""
        if not STATSMODELS_AVAILABLE:
            return None
        
        try:
            # Use target variable and a few key features
            key_features = [col for col in data.columns if col != target_variable][:3]  # Limit features
            model_data = data[[target_variable] + key_features]
            
            # Fit MS-VAR model
            try:
                # Use Markov Autoregression as a simpler alternative
                model = markov_autoregression.MarkovAutoregression(
                    model_data[target_variable].values,
                    k_regimes=self.config.n_regimes,
                    order=1,
                    switching_ar=True,
                    switching_variance=True
                )
                
                fitted_model = model.fit(maxiter=1000, em_iter=20)
                
                # Get regime probabilities
                probabilities = fitted_model.smoothed_marginal_probabilities
                states = np.argmax(probabilities, axis=1)
                
                # Create probability DataFrame
                prob_df = pd.DataFrame(
                    probabilities,
                    index=data.index,
                    columns=[f'Regime_{i}' for i in range(self.config.n_regimes)]
                )
                
                return {
                    'model': fitted_model,
                    'states': pd.Series(states, index=data.index),
                    'probabilities': prob_df,
                    'log_likelihood': fitted_model.llf,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'regime_statistics': self._extract_ms_var_statistics(fitted_model)
                }
                
            except Exception as e:
                self.logger.warning(f"MS-VAR model failed, trying simpler MS regression: {str(e)}")
                
                # Fallback to Markov Regression
                model = markov_regression.MarkovRegression(
                    model_data[target_variable].values,
                    k_regimes=self.config.n_regimes,
                    switching_variance=True
                )
                
                fitted_model = model.fit()
                
                probabilities = fitted_model.smoothed_marginal_probabilities
                states = np.argmax(probabilities, axis=1)
                
                prob_df = pd.DataFrame(
                    probabilities,
                    index=data.index,
                    columns=[f'Regime_{i}' for i in range(self.config.n_regimes)]
                )
                
                return {
                    'model': fitted_model,
                    'states': pd.Series(states, index=data.index),
                    'probabilities': prob_df,
                    'log_likelihood': fitted_model.llf,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'regime_statistics': {}
                }
            
        except Exception as e:
            self.logger.error(f"Error in MS-VAR regime detection: {str(e)}")
            return None
    
    def _fit_gaussian_mixture_regime(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Fit Gaussian Mixture Model for regime detection."""
        try:
            X = data.values
            
            # Try different covariance types
            best_model = None
            best_score = -np.inf
            
            for covariance_type in ['full', 'tied', 'diag', 'spherical']:
                try:
                    model = GaussianMixture(
                        n_components=self.config.n_regimes,
                        covariance_type=covariance_type,
                        max_iter=1000,
                        random_state=42
                    )
                    
                    model.fit(X)
                    score = model.score(X)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        
                except Exception as e:
                    self.logger.warning(f"GMM with {covariance_type} covariance failed: {str(e)}")
                    continue
            
            if best_model is None:
                return None
            
            # Get regime assignments and probabilities
            states = best_model.predict(X)
            probabilities = best_model.predict_proba(X)
            
            # Apply smoothing to reduce rapid regime switching
            smoothed_probs = self._smooth_probabilities(probabilities)
            smoothed_states = np.argmax(smoothed_probs, axis=1)
            
            # Create probability DataFrame
            prob_df = pd.DataFrame(
                smoothed_probs,
                index=data.index,
                columns=[f'Regime_{i}' for i in range(self.config.n_regimes)]
            )
            
            return {
                'model': best_model,
                'states': pd.Series(smoothed_states, index=data.index),
                'probabilities': prob_df,
                'log_likelihood': best_score * len(X),
                'aic': -2 * best_score * len(X) + 2 * self._count_gmm_parameters(best_model),
                'bic': -2 * best_score * len(X) + np.log(len(X)) * self._count_gmm_parameters(best_model),
                'regime_statistics': self._calculate_gmm_statistics(best_model, X)
            }
            
        except Exception as e:
            self.logger.error(f"Error in Gaussian Mixture regime detection: {str(e)}")
            return None
    
    def _detect_structural_breaks(self, data: pd.DataFrame, target_variable: str) -> Optional[Dict[str, Any]]:
        """Detect structural breaks using statistical tests."""
        if not STATSMODELS_AVAILABLE:
            return None
        
        try:
            y = data[target_variable].values
            
            # Use CUSUM test for structural breaks
            break_points = []
            
            # Sliding window CUSUM test
            window_size = max(50, len(y) // 10)
            
            for i in range(window_size, len(y) - window_size):
                y_segment = y[i-window_size:i+window_size]
                
                try:
                    # Simple regression for CUSUM test
                    X = np.arange(len(y_segment)).reshape(-1, 1)
                    X = sm.add_constant(X)
                    
                    # OLS regression
                    model = sm.OLS(y_segment, X).fit()
                    
                    # CUSUM test
                    cusum_stat, cusum_pvalue = breaks_cusumolsresid(model.resid)
                    
                    if cusum_pvalue < 0.05:  # Significant break
                        break_points.append(i)
                        
                except Exception:
                    continue
            
            # Remove close break points
            filtered_breaks = []
            if break_points:
                filtered_breaks = [break_points[0]]
                for bp in break_points[1:]:
                    if bp - filtered_breaks[-1] > self.config.min_regime_duration:
                        filtered_breaks.append(bp)
            
            # Create regime states based on break points
            states = np.zeros(len(y), dtype=int)
            
            if filtered_breaks:
                # Assign regimes based on break points
                regime_id = 0
                start_idx = 0
                
                for break_point in filtered_breaks:
                    if regime_id < self.config.n_regimes - 1:
                        states[start_idx:break_point] = regime_id
                        regime_id += 1
                        start_idx = break_point
                
                # Assign remaining observations to last regime
                states[start_idx:] = min(regime_id, self.config.n_regimes - 1)
            
            # Convert states to probabilities
            probabilities = np.zeros((len(y), self.config.n_regimes))
            for i, state in enumerate(states):
                probabilities[i, state] = 1.0
            
            # Apply some smoothing around break points
            smoothed_probs = self._smooth_probabilities(probabilities, window=5)
            
            prob_df = pd.DataFrame(
                smoothed_probs,
                index=data.index,
                columns=[f'Regime_{i}' for i in range(self.config.n_regimes)]
            )
            
            return {
                'model': {'break_points': filtered_breaks, 'method': 'cusum'},
                'states': pd.Series(np.argmax(smoothed_probs, axis=1), index=data.index),
                'probabilities': prob_df,
                'break_points': filtered_breaks,
                'regime_statistics': self._calculate_break_statistics(data, filtered_breaks)
            }
            
        except Exception as e:
            self.logger.error(f"Error in structural break detection: {str(e)}")
            return None
    
    def _fit_threshold_var_regime(self, data: pd.DataFrame, target_variable: str) -> Optional[Dict[str, Any]]:
        """Fit Threshold Vector Autoregression model."""
        try:
            # Simple threshold model based on volatility regimes
            returns = data[target_variable].values
            
            # Calculate rolling volatility as threshold variable
            window = 20
            volatility = pd.Series(returns).rolling(window=window).std().values
            volatility = volatility[window-1:]  # Remove NaN values
            returns_trimmed = returns[window-1:]
            
            if len(volatility) < 100:  # Insufficient data
                return None
            
            # Define thresholds based on volatility quantiles
            thresholds = np.quantile(volatility, np.linspace(0, 1, self.config.n_regimes + 1)[1:-1])
            
            # Assign regimes based on thresholds
            states = np.zeros(len(volatility), dtype=int)
            
            for i, vol in enumerate(volatility):
                regime = 0
                for j, threshold in enumerate(thresholds):
                    if vol > threshold:
                        regime = j + 1
                states[i] = min(regime, self.config.n_regimes - 1)
            
            # Create smooth probabilities
            probabilities = np.zeros((len(states), self.config.n_regimes))
            for i, state in enumerate(states):
                probabilities[i, state] = 1.0
            
            # Apply smoothing
            smoothed_probs = self._smooth_probabilities(probabilities)
            
            # Extend to original data length
            full_probs = np.zeros((len(data), self.config.n_regimes))
            full_probs[:window-1, 0] = 1.0  # Assign first regime to initial period
            full_probs[window-1:] = smoothed_probs
            
            prob_df = pd.DataFrame(
                full_probs,
                index=data.index,
                columns=[f'Regime_{i}' for i in range(self.config.n_regimes)]
            )
            
            full_states = np.argmax(full_probs, axis=1)
            
            return {
                'model': {'thresholds': thresholds, 'method': 'volatility_threshold'},
                'states': pd.Series(full_states, index=data.index),
                'probabilities': prob_df,
                'thresholds': thresholds,
                'regime_statistics': self._calculate_threshold_statistics(data, full_states, thresholds)
            }
            
        except Exception as e:
            self.logger.error(f"Error in threshold VAR regime detection: {str(e)}")
            return None
    
    def _smooth_probabilities(self, probabilities: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply smoothing to regime probabilities."""
        smoothed = probabilities.copy()
        
        for i in range(self.config.n_regimes):
            smoothed[:, i] = pd.Series(probabilities[:, i]).rolling(
                window=window, center=True, min_periods=1
            ).mean().values
        
        # Renormalize
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        smoothed = smoothed / row_sums
        
        return smoothed
    
    def _create_ensemble_regimes(self, results: Dict[str, Any], index: pd.Index) -> Dict[str, Any]:
        """Create ensemble regime probabilities from multiple methods."""
        try:
            method_names = list(results.keys())
            if len(method_names) == 1:
                # Only one method, return as is
                return results[method_names[0]]
            
            # Collect probabilities from all methods
            all_probabilities = []
            method_weights = []
            
            for method_name, method_result in results.items():
                if 'probabilities' in method_result:
                    probs = method_result['probabilities'].values
                    all_probabilities.append(probs)
                    
                    # Calculate weight based on information criteria if available
                    weight = 1.0
                    if 'bic' in method_result:
                        # Lower BIC is better, so use negative BIC for weight
                        weight = np.exp(-method_result['bic'] / 1000)  # Scale for numerical stability
                    
                    method_weights.append(weight)
            
            if not all_probabilities:
                return {}
            
            # Normalize weights
            method_weights = np.array(method_weights)
            method_weights = method_weights / method_weights.sum()
            
            # Weighted average of probabilities
            ensemble_probs = np.zeros_like(all_probabilities[0])
            for i, (probs, weight) in enumerate(zip(all_probabilities, method_weights)):
                ensemble_probs += weight * probs
                self.ensemble_weights[method_names[i]] = weight
            
            # Create probability DataFrame
            prob_df = pd.DataFrame(
                ensemble_probs,
                index=index,
                columns=[f'Regime_{i}' for i in range(self.config.n_regimes)]
            )
            
            # Get most likely states
            states = np.argmax(ensemble_probs, axis=1)
            
            return {
                'method': 'ensemble',
                'states': pd.Series(states, index=index),
                'probabilities': prob_df,
                'method_weights': dict(zip(method_names, method_weights)),
                'regime_statistics': self._calculate_ensemble_statistics(ensemble_probs, index)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble regimes: {str(e)}")
            return {}
    
    # Helper methods for model parameter counting and statistics
    def _count_hmm_parameters(self, model) -> int:
        """Count parameters in HMM model."""
        n_states = model.n_components
        n_features = model.n_features
        
        # Transition matrix parameters
        transition_params = n_states * (n_states - 1)
        
        # Emission parameters (means and covariances)
        if model.covariance_type == 'full':
            emission_params = n_states * (n_features + n_features * (n_features + 1) // 2)
        elif model.covariance_type == 'diag':
            emission_params = n_states * (n_features + n_features)
        else:  # tied or spherical
            emission_params = n_states * n_features + n_features * (n_features + 1) // 2
        
        return transition_params + emission_params
    
    def _count_gmm_parameters(self, model) -> int:
        """Count parameters in GMM model."""
        n_components = model.n_components
        n_features = model.n_features_in_
        
        # Mixing weights
        weight_params = n_components - 1
        
        # Means
        mean_params = n_components * n_features
        
        # Covariances
        if model.covariance_type == 'full':
            cov_params = n_components * n_features * (n_features + 1) // 2
        elif model.covariance_type == 'diag':
            cov_params = n_components * n_features
        elif model.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) // 2
        else:  # spherical
            cov_params = n_components
        
        return weight_params + mean_params + cov_params
    
    def _calculate_hmm_statistics(self, model, X: np.ndarray, states: np.ndarray) -> Dict:
        """Calculate statistics for HMM model."""
        try:
            stats = {}
            
            # Regime duration statistics
            regime_durations = self._calculate_regime_durations(states)
            stats['regime_durations'] = regime_durations
            
            # Transition probabilities
            stats['transition_matrix'] = model.transmat_.tolist()
            
            # Regime means and covariances
            stats['regime_means'] = model.means_.tolist()
            if hasattr(model, 'covars_'):
                stats['regime_covariances'] = model.covars_.tolist()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating HMM statistics: {str(e)}")
            return {}
    
    def _calculate_regime_durations(self, states: np.ndarray) -> Dict:
        """Calculate regime duration statistics."""
        durations = {i: [] for i in range(self.config.n_regimes)}
        
        current_regime = states[0]
        current_duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = states[i]
                current_duration = 1
        
        # Add the last duration
        durations[current_regime].append(current_duration)
        
        # Calculate statistics
        duration_stats = {}
        for regime, regime_durations in durations.items():
            if regime_durations:
                duration_stats[f'regime_{regime}'] = {
                    'mean_duration': np.mean(regime_durations),
                    'median_duration': np.median(regime_durations),
                    'max_duration': np.max(regime_durations),
                    'min_duration': np.min(regime_durations),
                    'count': len(regime_durations)
                }
        
        return duration_stats
    
    def _calculate_regime_statistics(self, results: Dict, data: pd.DataFrame):
        """Calculate comprehensive regime statistics."""
        for method_name, method_result in results.items():
            if 'states' in method_result:
                states = method_result['states'].values
                probabilities = method_result['probabilities'].values
                
                regime_stats = {
                    'regime_frequencies': {},
                    'regime_persistence': {},
                    'regime_characteristics': {}
                }
                
                # Calculate regime frequencies
                for regime in range(self.config.n_regimes):
                    regime_freq = np.mean(states == regime)
                    regime_stats['regime_frequencies'][f'regime_{regime}'] = regime_freq
                
                # Calculate persistence (probability of staying in same regime)
                for regime in range(self.config.n_regimes):
                    regime_mask = states == regime
                    if np.sum(regime_mask) > 1:
                        regime_indices = np.where(regime_mask)[0]
                        consecutive_stays = 0
                        total_transitions = 0
                        
                        for i in range(len(regime_indices) - 1):
                            if regime_indices[i+1] == regime_indices[i] + 1:
                                consecutive_stays += 1
                            total_transitions += 1
                        
                        persistence = consecutive_stays / total_transitions if total_transitions > 0 else 0
                        regime_stats['regime_persistence'][f'regime_{regime}'] = persistence
                
                # Calculate regime characteristics using available data
                if 'returns' in data.columns:
                    for regime in range(self.config.n_regimes):
                        regime_mask = states == regime
                        if np.sum(regime_mask) > 0:
                            regime_returns = data.loc[regime_mask, 'returns']
                            regime_stats['regime_characteristics'][f'regime_{regime}'] = {
                                'mean_return': regime_returns.mean(),
                                'volatility': regime_returns.std(),
                                'skewness': regime_returns.skew(),
                                'kurtosis': regime_returns.kurt()
                            }
                
                self.regime_statistics[method_name] = regime_stats
    
    def get_current_regime(self, method: str = 'ensemble') -> Tuple[int, float]:
        """Get the most recent regime and its probability."""
        if method not in self.regime_probabilities:
            available_methods = list(self.regime_probabilities.keys())
            if available_methods:
                method = available_methods[0]
            else:
                return 0, 0.0
        
        latest_probs = self.regime_probabilities[method].iloc[-1]
        regime = np.argmax(latest_probs.values)
        probability = latest_probs.values[regime]
        
        return regime, probability
    
    def predict_regime_transition(self, 
                                 method: str = 'ensemble',
                                 horizon: int = 5) -> pd.DataFrame:
        """Predict regime transitions over a specified horizon."""
        if method not in self.models or self.models[method] is None:
            return pd.DataFrame()
        
        try:
            model = self.models[method]
            current_regime, _ = self.get_current_regime(method)
            
            # For HMM models, use transition matrix
            if hasattr(model, 'transmat_'):
                transition_matrix = model.transmat_
                
                predictions = []
                current_state_probs = np.zeros(self.config.n_regimes)
                current_state_probs[current_regime] = 1.0
                
                for step in range(1, horizon + 1):
                    # Multiply by transition matrix
                    current_state_probs = current_state_probs @ transition_matrix
                    predictions.append(current_state_probs.copy())
                
                pred_df = pd.DataFrame(
                    predictions,
                    columns=[f'Regime_{i}' for i in range(self.config.n_regimes)],
                    index=range(1, horizon + 1)
                )
                
                return pred_df
            
            # For other models, return current probabilities
            current_probs = self.regime_probabilities[method].iloc[-1]
            pred_df = pd.DataFrame(
                [current_probs.values] * horizon,
                columns=current_probs.index,
                index=range(1, horizon + 1)
            )
            
            return pred_df
            
        except Exception as e:
            self.logger.error(f"Error predicting regime transitions: {str(e)}")
            return pd.DataFrame()
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of regime detection results."""
        summary = {
            'methods_used': list(self.models.keys()),
            'n_regimes': self.config.n_regimes,
            'current_regimes': {},
            'regime_statistics': self.regime_statistics,
            'ensemble_weights': self.ensemble_weights,
            'model_performance': {}
        }
        
        # Current regime for each method
        for method in self.regime_probabilities:
            regime, prob = self.get_current_regime(method)
            summary['current_regimes'][method] = {
                'regime': regime,
                'probability': prob
            }
        
        # Model performance comparison
        for method, result in self.models.items():
            if isinstance(result, dict):
                perf = {}
                if 'aic' in result:
                    perf['aic'] = result['aic']
                if 'bic' in result:
                    perf['bic'] = result['bic']
                if 'log_likelihood' in result:
                    perf['log_likelihood'] = result['log_likelihood']
                
                summary['model_performance'][method] = perf
        
        return summary
    
    # Additional helper methods for specific model statistics
    def _extract_ms_var_statistics(self, fitted_model) -> Dict:
        """Extract statistics from MS-VAR model."""
        try:
            return {
                'parameters': fitted_model.params.tolist() if hasattr(fitted_model, 'params') else [],
                'regime_means': fitted_model.regime_means if hasattr(fitted_model, 'regime_means') else [],
                'regime_variances': fitted_model.regime_variances if hasattr(fitted_model, 'regime_variances') else []
            }
        except:
            return {}
    
    def _calculate_gmm_statistics(self, model, X: np.ndarray) -> Dict:
        """Calculate statistics for GMM model."""
        try:
            return {
                'mixture_weights': model.weights_.tolist(),
                'means': model.means_.tolist(),
                'covariances': model.covariances_.tolist() if hasattr(model, 'covariances_') else []
            }
        except:
            return {}
    
    def _calculate_break_statistics(self, data: pd.DataFrame, break_points: List[int]) -> Dict:
        """Calculate statistics for structural break detection."""
        try:
            return {
                'n_breaks': len(break_points),
                'break_dates': [data.index[bp] for bp in break_points] if break_points else [],
                'regime_lengths': np.diff([0] + break_points + [len(data)]).tolist()
            }
        except:
            return {}
    
    def _calculate_threshold_statistics(self, data: pd.DataFrame, states: np.ndarray, thresholds: np.ndarray) -> Dict:
        """Calculate statistics for threshold model."""
        try:
            return {
                'thresholds': thresholds.tolist(),
                'regime_frequencies': [np.mean(states == i) for i in range(self.config.n_regimes)]
            }
        except:
            return {}
    
    def _calculate_ensemble_statistics(self, probabilities: np.ndarray, index: pd.Index) -> Dict:
        """Calculate statistics for ensemble method."""
        try:
            return {
                'ensemble_entropy': -np.mean(np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)),
                'regime_uncertainty': np.mean(1 - np.max(probabilities, axis=1))
            }
        except:
            return {}
            features: List of feature column names
            model_type: HMM type ('gaussian', 'gmm')
        
        Returns:
            Dictionary with model results
        """
        self.logger.info("Fitting HMM regime detection model")
        
        # Prepare data
        X = data[features].dropna().values
        X_scaled = self.scaler.fit_transform(X)
        
        try:
            if model_type == 'gaussian':
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    random_state=self.random_state,
                    n_iter=1000
                )
            else:
                model = hmm.GMMHMM(
                    n_components=self.n_regimes,
                    n_mix=2,
                    random_state=self.random_state,
                    n_iter=1000
                )
            
            # Fit model
            model.fit(X_scaled)
            
            # Predict regime states and probabilities
            regime_states = model.predict(X_scaled)
            regime_probs = model.predict_proba(X_scaled)
            
            # Calculate information criteria
            log_likelihood = model.score(X_scaled)
            n_params = self._count_hmm_parameters(model)
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(len(X_scaled)) * n_params
            
            results = {
                'model': model,
                'regime_states': regime_states,
                'regime_probabilities': regime_probs,
                'transition_matrix': model.transmat_,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'steady_state_probs': self._calculate_steady_state(model.transmat_)
            }
            
            # Store results
            self.models['hmm'] = results
            self.regime_probabilities['hmm'] = pd.DataFrame(
                regime_probs, 
                index=data[features].dropna().index,
                columns=[f'regime_{i}' for i in range(self.n_regimes)]
            )
            
            self.logger.info(f"HMM model fitted. AIC: {aic:.2f}, BIC: {bic:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error fitting HMM model: {str(e)}")
            raise
    
    def fit_ms_var_regime(self, 
                         data: pd.DataFrame,
                         features: List[str],
                         lags: int = 2) -> Dict:
        """
        Fit Markov Switching Vector Autoregression model.
        
        Args:
            data: Input DataFrame
            features: List of feature columns
            lags: Number of VAR lags
        
        Returns:
            Dictionary with model results
        """
        self.logger.info("Fitting MS-VAR regime detection model")
        
        try:
            # Prepare data
            df = data[features].dropna()
            
            # Custom MS-VAR implementation (simplified)
            results = self._fit_ms_var_custom(df, lags)
            
            self.models['ms_var'] = results
            
            self.logger.info("MS-VAR model fitted successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error fitting MS-VAR model: {str(e)}")
            raise
    
    def _fit_ms_var_custom(self, data: pd.DataFrame, lags: int) -> Dict:
        """
        Custom implementation of MS-VAR using EM algorithm.
        """
        n_obs, n_vars = data.shape
        
        # Initialize parameters
        regime_probs = np.random.dirichlet([1] * self.n_regimes, n_obs)
        transition_matrix = np.random.dirichlet([1] * self.n_regimes, self.n_regimes)
        
        # VAR coefficients for each regime
        var_coeffs = {}
        var_residuals = {}
        
        for regime in range(self.n_regimes):
            # Fit VAR model for each regime
            var_model = VAR(data)
            var_results = var_model.fit(lags, verbose=False)
            var_coeffs[regime] = var_results.params
            var_residuals[regime] = var_results.resid
        
        # EM algorithm iterations
        max_iterations = 100
        tolerance = 1e-6
        log_likelihood_prev = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: Update regime probabilities
            regime_probs = self._ms_var_e_step(data, var_coeffs, transition_matrix, lags)
            
            # M-step: Update parameters
            var_coeffs, transition_matrix = self._ms_var_m_step(data, regime_probs, lags)
            
            # Calculate log-likelihood
            log_likelihood = self._ms_var_log_likelihood(data, var_coeffs, regime_probs, lags)
            
            # Check convergence
            if abs(log_likelihood - log_likelihood_prev) < tolerance:
                break
            log_likelihood_prev = log_likelihood
        
        # Final regime classification
        regime_states = np.argmax(regime_probs, axis=1)
        
        # Calculate information criteria
        n_params = self._count_ms_var_parameters(n_vars, lags)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_obs) * n_params
        
        return {
            'regime_states': regime_states,
            'regime_probabilities': regime_probs,
            'var_coefficients': var_coeffs,
            'transition_matrix': transition_matrix,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'convergence_iteration': iteration
        }
    
    def _ms_var_e_step(self, data: pd.DataFrame, var_coeffs: Dict, 
                      transition_matrix: np.ndarray, lags: int) -> np.ndarray:
        """E-step of MS-VAR EM algorithm."""
        n_obs = len(data)
        regime_probs = np.zeros((n_obs, self.n_regimes))
        
        # Forward-backward algorithm for regime probabilities
        for t in range(lags, n_obs):
            for regime in range(self.n_regimes):
                # Calculate likelihood for current observation
                y_t = data.iloc[t].values
                X_t = self._create_var_regressors(data.iloc[t-lags:t], lags)
                
                # Prediction from VAR model
                y_pred = X_t @ var_coeffs[regime].T
                residual = y_t - y_pred
                
                # Multivariate normal likelihood
                cov_matrix = np.cov(residual.T) + 1e-6 * np.eye(len(residual))
                likelihood = stats.multivariate_normal.pdf(residual, cov=cov_matrix)
                
                # Update probability
                regime_probs[t, regime] = likelihood
        
        # Normalize probabilities
        regime_probs = regime_probs / (regime_probs.sum(axis=1, keepdims=True) + 1e-8)
        
        return regime_probs
    
    def _ms_var_m_step(self, data: pd.DataFrame, regime_probs: np.ndarray, 
                      lags: int) -> Tuple[Dict, np.ndarray]:
        """M-step of MS-VAR EM algorithm."""
        n_obs = len(data)
        var_coeffs = {}
        
        # Update VAR coefficients for each regime
        for regime in range(self.n_regimes):
            # Weighted least squares with regime probabilities as weights
            weights = regime_probs[:, regime]
            
            # Prepare regression data
            Y = []
            X = []
            W = []
            
            for t in range(lags, n_obs):
                Y.append(data.iloc[t].values)
                X.append(self._create_var_regressors(data.iloc[t-lags:t], lags))
                W.append(weights[t])
            
            Y = np.array(Y)
            X = np.array(X)
            W = np.array(W)
            
            # Weighted regression
            XTW = X.T * W
            var_coeffs[regime] = np.linalg.solve(XTW @ X + 1e-6 * np.eye(X.shape[1]), 
                                               XTW @ Y)
        
        # Update transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = np.sum(regime_probs[:-1, i] * regime_probs[1:, j])
                denominator = np.sum(regime_probs[:-1, i]) + 1e-8
                transition_matrix[i, j] = numerator / denominator
        
        # Normalize transition matrix
        transition_matrix = transition_matrix / (transition_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        return var_coeffs, transition_matrix
    
    def _create_var_regressors(self, data: pd.DataFrame, lags: int) -> np.ndarray:
        """Create VAR regressor matrix."""
        regressors = []
        for lag in range(1, lags + 1):
            regressors.append(data.iloc[-lag].values)
        regressors.append(1)  # Constant term
        return np.concatenate(regressors)
    
    def fit_tvar_regime(self, 
                       data: pd.DataFrame,
                       features: List[str],
                       threshold_variable: str,
                       lags: int = 2) -> Dict:
        """
        Fit Threshold Vector Autoregression model.
        
        Args:
            data: Input DataFrame
            features: List of feature columns
            threshold_variable: Variable to use for threshold
            lags: Number of VAR lags
        
        Returns:
            Dictionary with model results
        """
        self.logger.info("Fitting TVAR regime detection model")
        
        try:
            df = data[features + [threshold_variable]].dropna()
            threshold_values = df[threshold_variable].values
            
            # Grid search for optimal threshold
            threshold_grid = np.percentile(threshold_values, np.linspace(10, 90, 20))
            best_threshold = None
            best_aic = np.inf
            best_results = None
            
            for threshold in threshold_grid:
                results = self._fit_tvar_single_threshold(df[features], threshold_values, 
                                                        threshold, lags)
                if results['aic'] < best_aic:
                    best_aic = results['aic']
                    best_threshold = threshold
                    best_results = results
            
            best_results['threshold'] = best_threshold
            self.models['tvar'] = best_results
            
            self.logger.info(f"TVAR model fitted. Threshold: {best_threshold:.4f}, AIC: {best_aic:.2f}")
            return best_results
            
        except Exception as e:
            self.logger.error(f"Error fitting TVAR model: {str(e)}")
            raise
    
    def _fit_tvar_single_threshold(self, data: pd.DataFrame, threshold_var: np.ndarray,
                                 threshold: float, lags: int) -> Dict:
        """Fit TVAR model for a single threshold value."""
        n_obs = len(data)
        
        # Split data based on threshold
        low_regime_mask = threshold_var <= threshold
        high_regime_mask = threshold_var > threshold
        
        # Fit VAR models for each regime
        var_results = {}
        regime_states = np.zeros(n_obs)
        regime_states[high_regime_mask] = 1
        
        for regime, mask in enumerate([low_regime_mask, high_regime_mask]):
            if np.sum(mask) > lags * len(data.columns) + 10:  # Minimum observations
                regime_data = data[mask]
                var_model = VAR(regime_data)
                try:
                    var_fit = var_model.fit(lags, verbose=False)
                    var_results[regime] = {
                        'coefficients': var_fit.params,
                        'residuals': var_fit.resid,
                        'fitted_values': var_fit.fittedvalues,
                        'aic': var_fit.aic,
                        'bic': var_fit.bic
                    }
                except:
                    # Fallback to simple AR model if VAR fails
                    var_results[regime] = None
            else:
                var_results[regime] = None
        
        # Calculate overall AIC
        total_aic = 0
        valid_regimes = 0
        for regime in var_results:
            if var_results[regime] is not None:
                total_aic += var_results[regime]['aic']
                valid_regimes += 1
        
        if valid_regimes > 0:
            total_aic /= valid_regimes
        else:
            total_aic = np.inf
        
        # Create regime probabilities (deterministic for TVAR)
        regime_probs = np.zeros((n_obs, 2))
        regime_probs[low_regime_mask, 0] = 1.0
        regime_probs[high_regime_mask, 1] = 1.0
        
        return {
            'regime_states': regime_states,
            'regime_probabilities': regime_probs,
            'var_results': var_results,
            'threshold_variable': threshold_var,
            'aic': total_aic,
            'n_regime_0': np.sum(low_regime_mask),
            'n_regime_1': np.sum(high_regime_mask)
        }
    
    def detect_structural_breaks(self, 
                               data: pd.DataFrame,
                               feature: str,
                               method: str = 'cusum') -> Dict:
        """
        Detect structural breaks using statistical tests.
        
        Args:
            data: Input DataFrame
            feature: Feature column to test
            method: Test method ('cusum', 'chow', 'recursive_residuals')
        
        Returns:
            Dictionary with break detection results
        """
        self.logger.info(f"Detecting structural breaks using {method} method")
        
        try:
            series = data[feature].dropna()
            
            if method == 'cusum':
                results = self._cusum_test(series)
            elif method == 'chow':
                results = self._chow_test(series)
            elif method == 'recursive_residuals':
                results = self._recursive_residuals_test(series)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            self.models[f'breaks_{method}'] = results
            return results
            
        except Exception as e:
            self.logger.error(f"Error detecting structural breaks: {str(e)}")
            raise
    
    def _cusum_test(self, series: pd.Series) -> Dict:
        """CUSUM test for structural breaks."""
        # Fit AR model
        y = series.values
        n = len(y)
        
        # Create lagged variables
        y_lag = np.roll(y, 1)[1:]
        y_current = y[1:]
        
        # Fit regression
        X = sm.add_constant(y_lag)
        model = sm.OLS(y_current, X).fit()
        
        # CUSUM test
        try:
            cusum_stat, cusum_pvalue = breaks_cusumolsresid(model.resid)
            
            # Calculate CUSUM statistics
            residuals = model.resid
            cusum_stats = np.cumsum(residuals) / np.sqrt(np.sum(residuals**2))
            
            # Find break points (simplified)
            break_candidates = np.where(np.abs(cusum_stats) > 0.5)[0]
            
            return {
                'cusum_statistic': cusum_stat,
                'p_value': cusum_pvalue,
                'cusum_path': cusum_stats,
                'break_candidates': break_candidates,
                'significant_breaks': len(break_candidates) > 0 and cusum_pvalue < 0.05
            }
        except:
            return {
                'cusum_statistic': np.nan,
                'p_value': np.nan,
                'cusum_path': np.array([]),
                'break_candidates': np.array([]),
                'significant_breaks': False
            }
    
    def _chow_test(self, series: pd.Series) -> Dict:
        """Chow test for structural breaks."""
        y = series.values
        n = len(y)
        
        # Test multiple break points
        break_points = []
        f_statistics = []
        p_values = []
        
        # Test break points from 20% to 80% of sample
        for break_point in range(int(0.2 * n), int(0.8 * n)):
            try:
                # Split sample
                y1 = y[:break_point]
                y2 = y[break_point:]
                
                if len(y1) > 5 and len(y2) > 5:
                    # Fit models
                    X1 = sm.add_constant(np.arange(len(y1)))
                    X2 = sm.add_constant(np.arange(len(y2)))
                    
                    model1 = sm.OLS(y1, X1).fit()
                    model2 = sm.OLS(y2, X2).fit()
                    
                    # Full model
                    X_full = sm.add_constant(np.arange(len(y)))
                    model_full = sm.OLS(y, X_full).fit()
                    
                    # Chow test statistic
                    rss_restricted = model_full.ssr
                    rss_unrestricted = model1.ssr + model2.ssr
                    
                    f_stat = ((rss_restricted - rss_unrestricted) / 2) / (rss_unrestricted / (n - 4))
                    p_val = 1 - stats.f.cdf(f_stat, 2, n - 4)
                    
                    break_points.append(break_point)
                    f_statistics.append(f_stat)
                    p_values.append(p_val)
            except:
                continue
        
        if f_statistics:
            max_f_idx = np.argmax(f_statistics)
            most_likely_break = break_points[max_f_idx]
            max_f_stat = f_statistics[max_f_idx]
            min_p_value = p_values[max_f_idx]
        else:
            most_likely_break = None
            max_f_stat = np.nan
            min_p_value = np.nan
        
        return {
            'break_points': break_points,
            'f_statistics': f_statistics,
            'p_values': p_values,
            'most_likely_break': most_likely_break,
            'max_f_statistic': max_f_stat,
            'min_p_value': min_p_value,
            'significant_break': min_p_value < 0.05 if not np.isnan(min_p_value) else False
        }
    
    def _recursive_residuals_test(self, series: pd.Series) -> Dict:
        """Recursive residuals test for structural breaks."""
        y = series.values
        n = len(y)
        
        # Calculate recursive residuals
        recursive_residuals = []
        for t in range(20, n):  # Start after minimum window
            # Fit model on data up to time t
            y_subset = y[:t]
            X_subset = sm.add_constant(np.arange(len(y_subset)))
            
            try:
                model = sm.OLS(y_subset, X_subset).fit()
                
                # Predict next observation
                X_next = sm.add_constant([t])
                y_pred = model.predict(X_next)[0]
                
                # Calculate recursive residual
                recursive_resid = y[t] - y_pred
                recursive_residuals.append(recursive_resid)
            except:
                recursive_residuals.append(np.nan)
        
        recursive_residuals = np.array(recursive_residuals)
        
        # Calculate test statistics
        valid_residuals = recursive_residuals[~np.isnan(recursive_residuals)]
        if len(valid_residuals) > 0:
            # Standardize residuals
            std_residuals = valid_residuals / np.std(valid_residuals)
            
            # Test for breaks (simplified)
            break_threshold = 2.0
            break_candidates = np.where(np.abs(std_residuals) > break_threshold)[0]
        else:
            std_residuals = np.array([])
            break_candidates = np.array([])
        
        return {
            'recursive_residuals': recursive_residuals,
            'standardized_residuals': std_residuals,
            'break_candidates': break_candidates,
            'significant_breaks': len(break_candidates) > 0
        }
    
    def ensemble_regime_classification(self, 
                                     data: pd.DataFrame,
                                     features: List[str]) -> pd.DataFrame:
        """
        Combine multiple regime detection methods for robust classification.
        
        Args:
            data: Input DataFrame
            features: List of features for regime detection
        
        Returns:
            DataFrame with ensemble regime probabilities
        """
        self.logger.info("Creating ensemble regime classification")
        
        # Fit all models
        self.fit_hmm_regime(data, features)
        # Note: MS-VAR and TVAR might need different feature sets
        
        # Collect regime probabilities from all models
        ensemble_probs = []
        weights = []
        
        for model_name, model_results in self.models.items():
            if 'regime_probabilities' in model_results:
                probs = model_results['regime_probabilities']
                
                # Weight by model quality (inverse AIC)
                if 'aic' in model_results:
                    weight = 1.0 / (1.0 + model_results['aic'])
                else:
                    weight = 1.0
                
                ensemble_probs.append(probs * weight)
                weights.append(weight)
        
        # Combine probabilities
        if ensemble_probs:
            total_weight = sum(weights)
            combined_probs = sum(ensemble_probs) / total_weight
            
            # Normalize to ensure probabilities sum to 1
            combined_probs = combined_probs / combined_probs.sum(axis=1, keepdims=True)
            
            # Create ensemble regime states
            ensemble_states = np.argmax(combined_probs, axis=1)
            
            result_df = pd.DataFrame(
                combined_probs,
                index=combined_probs.index if hasattr(combined_probs, 'index') else data.index,
                columns=[f'regime_{i}' for i in range(self.n_regimes)]
            )
            result_df['ensemble_regime'] = ensemble_states
            
            return result_df
        else:
            self.logger.warning("No valid regime detection models found")
            return pd.DataFrame()
    
    def regime_probability_smoothing(self, 
                                   regime_probs: pd.DataFrame,
                                   smoothing_factor: float = 0.1) -> pd.DataFrame:
        """
        Apply smoothing to regime probabilities to reduce noise.
        
        Args:
            regime_probs: DataFrame with regime probabilities
            smoothing_factor: Exponential smoothing factor
        
        Returns:
            Smoothed regime probabilities
        """
        smoothed_probs = regime_probs.copy()
        
        for col in regime_probs.columns:
            if col.startswith('regime_'):
                smoothed_probs[col] = regime_probs[col].ewm(alpha=smoothing_factor).mean()
        
        # Renormalize
        regime_cols = [col for col in smoothed_probs.columns if col.startswith('regime_')]
        smoothed_probs[regime_cols] = smoothed_probs[regime_cols].div(
            smoothed_probs[regime_cols].sum(axis=1), axis=0
        )
        
        return smoothed_probs
    
    def _calculate_steady_state(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Calculate steady-state probabilities from transition matrix."""
        eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmin(np.abs(eigenvals - 1.0))
        steady_state = np.real(eigenvecs[:, stationary_idx])
        return steady_state / steady_state.sum()
    
    def _count_hmm_parameters(self, model) -> int:
        """Count number of parameters in HMM model."""
        n_states = model.n_components
        n_features = model.n_features
        
        # Transition matrix parameters
        transition_params = n_states * (n_states - 1)
        
        # Emission parameters (means and covariances)
        emission_params = n_states * n_features  # means
        emission_params += n_states * n_features * (n_features + 1) // 2  # covariances
        
        return transition_params + emission_params
    
    def _count_ms_var_parameters(self, n_vars: int, lags: int) -> int:
        """Count number of parameters in MS-VAR model."""
        # VAR coefficients for each regime
        var_params_per_regime = n_vars * (n_vars * lags + 1)  # +1 for constant
        var_params = self.n_regimes * var_params_per_regime
        
        # Transition matrix parameters
        transition_params = self.n_regimes * (self.n_regimes - 1)
        
        return var_params + transition_params
    
    def _ms_var_log_likelihood(self, data: pd.DataFrame, var_coeffs: Dict,
                              regime_probs: np.ndarray, lags: int) -> float:
        """Calculate log-likelihood for MS-VAR model."""
        log_likelihood = 0.0
        n_obs = len(data)
        
        for t in range(lags, n_obs):
            for regime in range(self.n_regimes):
                y_t = data.iloc[t].values
                X_t = self._create_var_regressors(data.iloc[t-lags:t], lags)
                
                # Prediction
                y_pred = X_t @ var_coeffs[regime].T
                residual = y_t - y_pred
                
                # Log-likelihood contribution
                try:
                    cov_matrix = np.cov(residual.T) + 1e-6 * np.eye(len(residual))
                    ll_contribution = stats.multivariate_normal.logpdf(residual, cov=cov_matrix)
                    log_likelihood += regime_probs[t, regime] * ll_contribution
                except:
                    continue
        
        return log_likelihood