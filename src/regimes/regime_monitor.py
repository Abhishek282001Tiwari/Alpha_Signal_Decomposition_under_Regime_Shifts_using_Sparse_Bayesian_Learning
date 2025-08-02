import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
from collections import deque
import threading
import time

warnings.filterwarnings('ignore')

class RealTimeRegimeMonitor:
    """
    Real-time regime monitoring system with online learning and early warning capabilities.
    Provides dynamic regime updates using Bayesian filtering and adaptive algorithms.
    """
    
    def __init__(self, 
                 regime_detector,
                 update_frequency: int = 1,  # days
                 memory_window: int = 252,   # trading days
                 confidence_threshold: float = 0.7,
                 alert_threshold: float = 0.8):
        """
        Initialize real-time regime monitor.
        
        Args:
            regime_detector: Trained regime detection model
            update_frequency: Frequency of regime updates (days)
            memory_window: Rolling window for regime estimation
            confidence_threshold: Minimum confidence for regime classification
            alert_threshold: Threshold for regime transition alerts
        """
        self.regime_detector = regime_detector
        self.update_frequency = update_frequency
        self.memory_window = memory_window
        self.confidence_threshold = confidence_threshold
        self.alert_threshold = alert_threshold
        
        # State variables
        self.current_regime = None
        self.current_regime_probs = None
        self.regime_history = deque(maxlen=memory_window)
        self.data_buffer = deque(maxlen=memory_window)
        self.last_update = None
        
        # Online learning components
        self.online_parameters = {}
        self.adaptation_rate = 0.01
        self.forgetting_factor = 0.95
        
        # Alert system
        self.alert_callbacks = []
        self.alert_history = []
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def add_alert_callback(self, callback: Callable):
        """Add callback function for regime transition alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start real-time monitoring in separate thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("Real-time regime monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Real-time regime monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active:
            try:
                # Check if update is needed
                if self._should_update():
                    self._perform_update()
                
                # Sleep until next check
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _should_update(self) -> bool:
        """Check if regime update is needed."""
        if self.last_update is None:
            return True
        
        time_since_update = (datetime.now() - self.last_update).days
        return time_since_update >= self.update_frequency
    
    def _perform_update(self):
        """Perform regime update using latest data."""
        if len(self.data_buffer) < 10:
            return
        
        # Convert buffer to DataFrame
        recent_data = pd.DataFrame(list(self.data_buffer))
        
        # Update regime probabilities
        new_regime_probs = self.update_regime_probabilities(recent_data)
        
        # Check for regime transition
        if self.current_regime_probs is not None:
            self._check_regime_transition(new_regime_probs)
        
        self.current_regime_probs = new_regime_probs
        self.last_update = datetime.now()
    
    def update_regime_probabilities(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Update regime probabilities with new data using Bayesian filtering.
        
        Args:
            new_data: New market data
        
        Returns:
            Updated regime probabilities
        """
        if new_data.empty:
            return self.current_regime_probs
        
        # Prepare features
        features = self._extract_features(new_data)
        if features is None:
            return self.current_regime_probs
        
        # Online Bayesian update
        if self.current_regime_probs is None:
            # Initial prediction
            updated_probs = self._initial_regime_prediction(features)
        else:
            # Bayesian filter update
            updated_probs = self._bayesian_filter_update(features, self.current_regime_probs)
        
        # Online parameter adaptation
        self._adapt_parameters(features, updated_probs)
        
        # Store in history
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime_probs': updated_probs,
            'features': features
        })
        
        return updated_probs
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for regime detection."""
        try:
            # Basic features
            features = []
            
            if 'returns' in data.columns:
                returns = data['returns'].dropna()
                if len(returns) > 0:
                    features.extend([
                        returns.mean(),
                        returns.std(),
                        returns.skew() if len(returns) > 2 else 0,
                        returns.kurtosis() if len(returns) > 3 else 0
                    ])
            
            if 'volume' in data.columns:
                volume = data['volume'].dropna()
                if len(volume) > 0:
                    features.append(volume.mean())
            
            # VIX-like volatility measure
            if 'volatility' in data.columns:
                vol = data['volatility'].dropna()
                if len(vol) > 0:
                    features.append(vol.mean())
            
            return np.array(features) if features else None
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def _initial_regime_prediction(self, features: np.ndarray) -> np.ndarray:
        """Initial regime prediction for new data."""
        try:
            # Use trained regime detector
            if hasattr(self.regime_detector, 'models') and 'hmm' in self.regime_detector.models:
                model = self.regime_detector.models['hmm']['model']
                
                # Reshape features for prediction
                features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
                regime_probs = model.predict_proba(features_scaled)[0]
                
                return regime_probs
            else:
                # Fallback: uniform distribution
                n_regimes = getattr(self.regime_detector, 'n_regimes', 3)
                return np.ones(n_regimes) / n_regimes
                
        except Exception as e:
            self.logger.error(f"Error in initial prediction: {str(e)}")
            # Return uniform distribution as fallback
            n_regimes = getattr(self.regime_detector, 'n_regimes', 3)
            return np.ones(n_regimes) / n_regimes
    
    def _bayesian_filter_update(self, 
                              features: np.ndarray,
                              prior_probs: np.ndarray) -> np.ndarray:
        """
        Bayesian filter update for regime probabilities.
        
        Args:
            features: New feature vector
            prior_probs: Prior regime probabilities
        
        Returns:
            Updated regime probabilities
        """
        try:
            n_regimes = len(prior_probs)
            
            # Calculate likelihood for each regime
            likelihoods = np.zeros(n_regimes)
            
            for regime in range(n_regimes):
                # Get regime-specific parameters
                if regime in self.online_parameters:
                    mean = self.online_parameters[regime]['mean']
                    cov = self.online_parameters[regime]['cov']
                    
                    # Multivariate normal likelihood
                    try:
                        likelihood = stats.multivariate_normal.pdf(features, mean, cov)
                    except:
                        likelihood = 1.0 / n_regimes
                else:
                    # Default likelihood if no parameters available
                    likelihood = 1.0 / n_regimes
                
                likelihoods[regime] = likelihood
            
            # Bayesian update: posterior ∝ likelihood × prior
            posterior = likelihoods * prior_probs
            
            # Normalize
            posterior_sum = posterior.sum()
            if posterior_sum > 0:
                posterior = posterior / posterior_sum
            else:
                posterior = prior_probs  # Fallback to prior
            
            # Apply forgetting factor for transition probabilities
            if hasattr(self.regime_detector, 'models') and 'hmm' in self.regime_detector.models:
                transition_matrix = self.regime_detector.models['hmm']['transition_matrix']
                # Apply transition dynamics
                posterior = self.forgetting_factor * (transition_matrix.T @ prior_probs) + \
                           (1 - self.forgetting_factor) * posterior
                
                # Renormalize
                posterior = posterior / posterior.sum()
            
            return posterior
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian filter update: {str(e)}")
            return prior_probs
    
    def _adapt_parameters(self, features: np.ndarray, regime_probs: np.ndarray):
        """
        Adapt regime-specific parameters using online learning.
        
        Args:
            features: Current feature vector
            regime_probs: Current regime probabilities
        """
        n_regimes = len(regime_probs)
        
        for regime in range(n_regimes):
            if regime not in self.online_parameters:
                # Initialize parameters
                self.online_parameters[regime] = {
                    'mean': features.copy(),
                    'cov': np.eye(len(features)) * 0.1,
                    'count': 1
                }
            else:
                # Online update using exponential moving average
                params = self.online_parameters[regime]
                weight = regime_probs[regime]  # Weight by regime probability
                
                # Update mean
                params['mean'] = (1 - self.adaptation_rate * weight) * params['mean'] + \
                                self.adaptation_rate * weight * features
                
                # Update covariance (simplified)
                diff = features - params['mean']
                outer_product = np.outer(diff, diff)
                params['cov'] = (1 - self.adaptation_rate * weight) * params['cov'] + \
                               self.adaptation_rate * weight * outer_product
                
                # Add small regularization
                params['cov'] += 1e-6 * np.eye(len(features))
                
                params['count'] += weight
    
    def _check_regime_transition(self, new_regime_probs: np.ndarray):
        """
        Check for regime transitions and trigger alerts.
        
        Args:
            new_regime_probs: New regime probabilities
        """
        if self.current_regime_probs is None:
            return
        
        # Calculate regime change intensity
        prob_change = np.abs(new_regime_probs - self.current_regime_probs)
        max_change = prob_change.max()
        
        # Check if significant regime change occurred
        if max_change > self.alert_threshold:
            old_regime = np.argmax(self.current_regime_probs)
            new_regime = np.argmax(new_regime_probs)
            
            # Create alert
            alert = {
                'timestamp': datetime.now(),
                'type': 'regime_transition',
                'from_regime': old_regime,
                'to_regime': new_regime,
                'confidence': new_regime_probs[new_regime],
                'change_intensity': max_change,
                'regime_probs': new_regime_probs.copy()
            }
            
            self.alert_history.append(alert)
            self.logger.warning(f"Regime transition detected: {old_regime} -> {new_regime} "
                              f"(confidence: {new_regime_probs[new_regime]:.3f})")
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
    
    def add_data_point(self, data_point: Dict):
        """
        Add new data point to monitoring system.
        
        Args:
            data_point: Dictionary with market data
        """
        # Add timestamp if not present
        if 'timestamp' not in data_point:
            data_point['timestamp'] = datetime.now()
        
        # Add to buffer
        self.data_buffer.append(data_point)
        
        # Trigger update if enough new data
        if len(self.data_buffer) >= 10 and self._should_update():
            self._perform_update()
    
    def get_current_regime(self) -> Dict:
        """
        Get current regime information.
        
        Returns:
            Dictionary with current regime details
        """
        if self.current_regime_probs is None:
            return {'regime': None, 'confidence': 0.0, 'probabilities': None}
        
        current_regime = np.argmax(self.current_regime_probs)
        confidence = self.current_regime_probs[current_regime]
        
        return {
            'regime': current_regime,
            'confidence': confidence,
            'probabilities': self.current_regime_probs.copy(),
            'last_update': self.last_update,
            'high_confidence': confidence >= self.confidence_threshold
        }
    
    def get_regime_forecast(self, horizon: int = 5) -> Dict:
        """
        Generate regime probability forecast.
        
        Args:
            horizon: Forecast horizon in days
        
        Returns:
            Dictionary with regime forecasts
        """
        if self.current_regime_probs is None:
            return {}
        
        # Use transition matrix for forecasting
        forecasts = {}
        current_probs = self.current_regime_probs.copy()
        
        if (hasattr(self.regime_detector, 'models') and 
            'hmm' in self.regime_detector.models):
            
            transition_matrix = self.regime_detector.models['hmm']['transition_matrix']
            
            for day in range(1, horizon + 1):
                # Apply transition matrix
                current_probs = transition_matrix.T @ current_probs
                forecasts[f'day_{day}'] = {
                    'probabilities': current_probs.copy(),
                    'most_likely_regime': np.argmax(current_probs),
                    'confidence': np.max(current_probs)
                }
        
        return forecasts
    
    def calculate_regime_stability(self, window: int = 20) -> Dict:
        """
        Calculate regime stability metrics.
        
        Args:
            window: Window for stability calculation
        
        Returns:
            Dictionary with stability metrics
        """
        if len(self.regime_history) < window:
            return {}
        
        # Get recent regime probabilities
        recent_history = list(self.regime_history)[-window:]
        prob_matrix = np.array([h['regime_probs'] for h in recent_history])
        
        # Calculate stability metrics
        stability_metrics = {}
        
        # Regime consistency (how often the most likely regime stays the same)
        most_likely_regimes = np.argmax(prob_matrix, axis=1)
        regime_changes = np.sum(most_likely_regimes[1:] != most_likely_regimes[:-1])
        stability_metrics['regime_consistency'] = 1 - (regime_changes / (len(most_likely_regimes) - 1))
        
        # Probability volatility (standard deviation of probabilities)
        prob_volatility = np.std(prob_matrix, axis=0)
        stability_metrics['probability_volatility'] = prob_volatility.tolist()
        stability_metrics['avg_probability_volatility'] = np.mean(prob_volatility)
        
        # Entropy-based stability
        entropies = [-np.sum(probs * np.log(probs + 1e-8)) for probs in prob_matrix]
        stability_metrics['avg_entropy'] = np.mean(entropies)
        stability_metrics['entropy_volatility'] = np.std(entropies)
        
        return stability_metrics
    
    def detect_regime_stress(self) -> Dict:
        """
        Detect regime stress indicators and early warning signals.
        
        Returns:
            Dictionary with stress indicators
        """
        stress_indicators = {}
        
        if len(self.regime_history) < 10:
            return stress_indicators
        
        # Get recent data
        recent_data = list(self.regime_history)[-10:]
        
        # Probability volatility stress
        prob_matrix = np.array([h['regime_probs'] for h in recent_data])
        prob_volatility = np.std(prob_matrix, axis=0)
        max_prob_vol = np.max(prob_volatility)
        
        stress_indicators['probability_volatility_stress'] = {
            'level': max_prob_vol,
            'alert': max_prob_vol > 0.3,  # Threshold for high volatility
            'description': 'High volatility in regime probabilities'
        }
        
        # Regime uncertainty stress
        avg_entropy = np.mean([-np.sum(probs * np.log(probs + 1e-8)) for probs in prob_matrix])
        max_entropy = np.log(len(prob_matrix[0]))  # Maximum possible entropy
        uncertainty_ratio = avg_entropy / max_entropy
        
        stress_indicators['regime_uncertainty_stress'] = {
            'level': uncertainty_ratio,
            'alert': uncertainty_ratio > 0.8,  # High uncertainty threshold
            'description': 'High uncertainty in regime classification'
        }
        
        # Transition frequency stress
        most_likely_regimes = np.argmax(prob_matrix, axis=1)
        transition_count = np.sum(most_likely_regimes[1:] != most_likely_regimes[:-1])
        transition_frequency = transition_count / (len(most_likely_regimes) - 1)
        
        stress_indicators['transition_frequency_stress'] = {
            'level': transition_frequency,
            'alert': transition_frequency > 0.5,  # High transition frequency
            'description': 'High frequency of regime transitions'
        }
        
        # Feature drift stress
        if len(recent_data) >= 2:
            feature_drift = self._calculate_feature_drift(recent_data)
            stress_indicators['feature_drift_stress'] = {
                'level': feature_drift,
                'alert': feature_drift > 2.0,  # Threshold for significant drift
                'description': 'Significant drift in input features'
            }
        
        return stress_indicators
    
    def _calculate_feature_drift(self, data_history: List[Dict]) -> float:
        """Calculate feature drift using KL divergence."""
        try:
            # Get feature vectors
            features = [h['features'] for h in data_history if h['features'] is not None]
            
            if len(features) < 2:
                return 0.0
            
            # Compare recent features with historical
            recent_features = np.array(features[-5:])  # Last 5 observations
            historical_features = np.array(features[:-5])  # Earlier observations
            
            if len(historical_features) == 0:
                return 0.0
            
            # Calculate simple drift measure (Euclidean distance between means)
            recent_mean = np.mean(recent_features, axis=0)
            historical_mean = np.mean(historical_features, axis=0)
            
            drift = np.linalg.norm(recent_mean - historical_mean)
            return drift
            
        except Exception as e:
            self.logger.error(f"Error calculating feature drift: {str(e)}")
            return 0.0
    
    def generate_early_warning_signals(self) -> List[Dict]:
        """
        Generate early warning signals for potential regime changes.
        
        Returns:
            List of warning signals
        """
        warnings = []
        
        # Get stress indicators
        stress_indicators = self.detect_regime_stress()
        
        # Check each stress indicator
        for indicator_name, indicator_data in stress_indicators.items():
            if indicator_data.get('alert', False):
                warning = {
                    'timestamp': datetime.now(),
                    'type': 'early_warning',
                    'indicator': indicator_name,
                    'level': indicator_data['level'],
                    'description': indicator_data['description'],
                    'severity': 'high' if indicator_data['level'] > 0.8 else 'medium'
                }
                warnings.append(warning)
        
        # Check regime probability trends
        if len(self.regime_history) >= 5:
            prob_trends = self._analyze_probability_trends()
            for trend_warning in prob_trends:
                warnings.append(trend_warning)
        
        return warnings
    
    def _analyze_probability_trends(self) -> List[Dict]:
        """Analyze trends in regime probabilities for early warnings."""
        warnings = []
        
        recent_data = list(self.regime_history)[-5:]
        prob_matrix = np.array([h['regime_probs'] for h in recent_data])
        
        # Check for consistent trends in each regime probability
        for regime in range(prob_matrix.shape[1]):
            regime_probs = prob_matrix[:, regime]
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(regime_probs))
            slope, _, r_value, p_value, _ = stats.linregress(x, regime_probs)
            
            # Strong upward trend indicates potential regime transition
            if slope > 0.1 and p_value < 0.1 and r_value > 0.7:
                warning = {
                    'timestamp': datetime.now(),
                    'type': 'trend_warning',
                    'indicator': 'regime_probability_trend',
                    'regime': regime,
                    'trend_strength': slope,
                    'r_squared': r_value**2,
                    'description': f'Strong upward trend in regime {regime} probability',
                    'severity': 'medium'
                }
                warnings.append(warning)
        
        return warnings
    
    def get_monitoring_status(self) -> Dict:
        """
        Get comprehensive monitoring system status.
        
        Returns:
            Dictionary with system status
        """
        status = {
            'monitoring_active': self.monitoring_active,
            'last_update': self.last_update,
            'data_buffer_size': len(self.data_buffer),
            'regime_history_size': len(self.regime_history),
            'current_regime_info': self.get_current_regime(),
            'alert_count': len(self.alert_history),
            'recent_alerts': self.alert_history[-5:] if self.alert_history else []
        }
        
        # Add stability metrics if available
        if len(self.regime_history) >= 10:
            status['stability_metrics'] = self.calculate_regime_stability()
        
        # Add stress indicators
        status['stress_indicators'] = self.detect_regime_stress()
        
        # Add early warning signals
        status['early_warnings'] = self.generate_early_warning_signals()
        
        return status