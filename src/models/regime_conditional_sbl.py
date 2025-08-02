import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import logging
import warnings
from .sparse_bayesian_learner import SparseBayesianLearner
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

class RegimeConditionalSBL:
    """
    Regime-conditional Sparse Bayesian Learning system that maintains separate
    sparse models for each market regime with hierarchical Bayesian structure.
    """
    
    def __init__(self, 
                 n_regimes: int = 3,
                 prior_type: str = 'laplace',
                 hierarchical_structure: bool = True,
                 regime_switching_penalty: float = 0.1,
                 cross_regime_regularization: float = 0.05):
        """
        Initialize regime-conditional SBL system.
        
        Args:
            n_regimes: Number of market regimes
            prior_type: Sparsity prior type for individual models
            hierarchical_structure: Enable hierarchical Bayesian structure
            regime_switching_penalty: Penalty for feature set differences across regimes
            cross_regime_regularization: Regularization between regime models
        """
        self.n_regimes = n_regimes
        self.prior_type = prior_type
        self.hierarchical_structure = hierarchical_structure
        self.regime_switching_penalty = regime_switching_penalty
        self.cross_regime_regularization = cross_regime_regularization
        
        # Individual regime models
        self.regime_models = {}
        self.regime_features = {}
        self.regime_performance = {}
        
        # Hierarchical parameters
        self.global_feature_prior = None
        self.regime_feature_coupling = None
        
        # Cross-validation and model selection
        self.cv_results = {}
        self.ensemble_weights = None
        
        self.logger = logging.getLogger(__name__)
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            regime_probabilities: np.ndarray,
            feature_names: Optional[List[str]] = None,
            cv_folds: int = 5) -> 'RegimeConditionalSBL':
        """
        Fit regime-conditional sparse Bayesian models.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            regime_probabilities: Regime probability matrix (n_samples, n_regimes)
            feature_names: Optional feature names
            cv_folds: Number of cross-validation folds
        
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting regime-conditional SBL with {self.n_regimes} regimes")
        
        X = self._validate_input(X)
        y = self._validate_input(y).flatten()
        regime_probabilities = self._validate_input(regime_probabilities)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        self.feature_names = feature_names or [f'feature_{i}' for i in range(n_features)]
        
        # Initialize hierarchical structure
        if self.hierarchical_structure:
            self._initialize_hierarchical_priors(n_features)
        
        # Fit individual regime models
        self._fit_regime_models(X, y, regime_probabilities)
        
        # Apply hierarchical coupling if enabled
        if self.hierarchical_structure:
            self._apply_hierarchical_coupling()
        
        # Cross-validation for regime-aware model selection
        self._perform_regime_aware_cv(X, y, regime_probabilities, cv_folds)
        
        # Calculate ensemble weights
        self._calculate_ensemble_weights()
        
        self.logger.info("Regime-conditional SBL fitting completed")
        return self
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.array(X)
    
    def _initialize_hierarchical_priors(self, n_features: int):
        """Initialize hierarchical Bayesian structure for cross-regime learning."""
        # Global feature relevance prior
        self.global_feature_prior = {
            'alpha': np.ones(n_features) * 1.0,  # Global relevance parameters
            'beta': np.ones(n_features) * 1.0,   # Global precision parameters
        }
        
        # Cross-regime feature coupling matrix
        self.regime_feature_coupling = np.eye(self.n_regimes) * 0.1 + \
                                      np.ones((self.n_regimes, self.n_regimes)) * 0.01
    
    def _fit_regime_models(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          regime_probabilities: np.ndarray):
        """Fit individual SBL models for each regime."""
        
        for regime in range(self.n_regimes):
            self.logger.info(f"Fitting model for regime {regime}")
            
            # Extract regime-specific data using soft assignment
            regime_weights = regime_probabilities[:, regime]
            
            # Skip if regime has insufficient data
            effective_samples = np.sum(regime_weights > 0.1)
            if effective_samples < 10:
                self.logger.warning(f"Insufficient data for regime {regime}")
                continue
            
            # Create weighted SBL model
            regime_model = self._create_weighted_sbl_model(regime)
            
            # Fit model with regime weights
            self._fit_weighted_model(regime_model, X, y, regime_weights)
            
            # Store model and extract features
            self.regime_models[regime] = regime_model
            self.regime_features[regime] = regime_model.selected_features
            
            # Calculate regime-specific performance
            regime_pred = regime_model.predict(X)
            weighted_mse = np.average((y - regime_pred)**2, weights=regime_weights)
            
            self.regime_performance[regime] = {
                'weighted_mse': weighted_mse,
                'effective_samples': effective_samples,
                'selected_features': len(regime_model.selected_features) if regime_model.selected_features is not None else 0,
                'sparsity': regime_model.get_model_summary()['weight_sparsity']
            }
    
    def _create_weighted_sbl_model(self, regime: int) -> SparseBayesianLearner:
        """Create SBL model with regime-specific priors."""
        # Adjust priors based on hierarchical structure
        if self.hierarchical_structure:
            alpha_prior = self.global_feature_prior['alpha'].mean()
            beta_prior = self.global_feature_prior['beta'].mean()
        else:
            alpha_prior = 1.0
            beta_prior = 1.0
        
        model = SparseBayesianLearner(
            prior_type=self.prior_type,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            automatic_relevance_determination=True
        )
        
        return model
    
    def _fit_weighted_model(self, 
                           model: SparseBayesianLearner,
                           X: np.ndarray, 
                           y: np.ndarray,
                           weights: np.ndarray):
        """Fit SBL model with sample weights (regime probabilities)."""
        
        # Create weighted dataset by resampling based on weights
        n_samples = len(weights)
        weighted_indices = np.random.choice(
            n_samples, 
            size=n_samples, 
            p=weights / weights.sum(),
            replace=True
        )
        
        X_weighted = X[weighted_indices]
        y_weighted = y[weighted_indices]
        
        # Fit the model
        model.fit(X_weighted, y_weighted, feature_names=self.feature_names)
    
    def _apply_hierarchical_coupling(self):
        """Apply hierarchical Bayesian coupling between regime models."""
        if not self.hierarchical_structure:
            return
        
        # Update global priors based on regime models
        self._update_global_priors()
        
        # Apply cross-regime regularization
        self._apply_cross_regime_regularization()
    
    def _update_global_priors(self):
        """Update global feature priors based on regime model posteriors."""
        
        # Collect feature importance from all regime models
        all_importances = []
        regime_weights = []
        
        for regime, model in self.regime_models.items():
            if model.feature_importance is not None:
                all_importances.append(model.feature_importance)
                regime_weights.append(self.regime_performance[regime]['effective_samples'])
        
        if not all_importances:
            return
        
        # Weighted average of feature importances
        regime_weights = np.array(regime_weights)
        regime_weights = regime_weights / regime_weights.sum()
        
        global_importance = np.zeros(self.n_features)
        for importance, weight in zip(all_importances, regime_weights):
            global_importance += weight * importance
        
        # Update global priors
        self.global_feature_prior['alpha'] = global_importance + 0.1
        self.global_feature_prior['beta'] = 1.0 / (global_importance + 0.1)
    
    def _apply_cross_regime_regularization(self):
        """Apply regularization to encourage similar feature selection across regimes."""
        
        # Calculate feature selection similarity matrix
        regime_list = list(self.regime_models.keys())
        n_active_regimes = len(regime_list)
        
        if n_active_regimes < 2:
            return
        
        # Penalize models with very different feature selections
        for i, regime_i in enumerate(regime_list):
            for j, regime_j in enumerate(regime_list[i+1:], i+1):
                
                features_i = set(self.regime_features.get(regime_i, []))
                features_j = set(self.regime_features.get(regime_j, []))
                
                # Calculate Jaccard similarity
                intersection = len(features_i.intersection(features_j))
                union = len(features_i.union(features_j))
                similarity = intersection / union if union > 0 else 0
                
                # Apply penalty for dissimilar feature sets
                if similarity < 0.5:  # Threshold for similarity
                    penalty = self.regime_switching_penalty * (1 - similarity)
                    
                    # Adjust model parameters (simplified implementation)
                    if regime_i in self.regime_models:
                        model_i = self.regime_models[regime_i]
                        if hasattr(model_i, 'alpha') and isinstance(model_i.alpha, np.ndarray):
                            different_features = features_i.symmetric_difference(features_j)
                            for feat_idx in different_features:
                                if feat_idx < len(model_i.alpha):
                                    model_i.alpha[feat_idx] *= (1 + penalty)
    
    def _perform_regime_aware_cv(self, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               regime_probabilities: np.ndarray,
                               cv_folds: int):
        """Perform regime-aware cross-validation."""
        
        self.logger.info("Performing regime-aware cross-validation")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {regime: [] for regime in range(self.n_regimes)}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            regime_train = regime_probabilities[train_idx]
            regime_val = regime_probabilities[val_idx]
            
            # Fit models on training fold
            fold_models = {}
            for regime in range(self.n_regimes):
                regime_weights = regime_train[:, regime]
                effective_samples = np.sum(regime_weights > 0.1)
                
                if effective_samples >= 5:  # Minimum samples for training
                    model = self._create_weighted_sbl_model(regime)
                    self._fit_weighted_model(model, X_train, y_train, regime_weights)
                    fold_models[regime] = model
            
            # Evaluate on validation fold
            for regime in fold_models:
                model = fold_models[regime]
                val_weights = regime_val[:, regime]
                
                if np.sum(val_weights > 0.1) > 0:  # Has validation samples for this regime
                    y_pred = model.predict(X_val)
                    
                    # Weighted validation score
                    weighted_mse = np.average((y_val - y_pred)**2, weights=val_weights)
                    cv_scores[regime].append(weighted_mse)
        
        # Store CV results
        for regime in cv_scores:
            if cv_scores[regime]:
                self.cv_results[regime] = {
                    'mean_cv_score': np.mean(cv_scores[regime]),
                    'std_cv_score': np.std(cv_scores[regime]),
                    'cv_scores': cv_scores[regime]
                }
            else:
                self.cv_results[regime] = {'mean_cv_score': np.inf}
    
    def _calculate_ensemble_weights(self):
        """Calculate ensemble weights based on regime performance and CV scores."""
        
        regime_scores = []
        regime_list = []
        
        for regime in self.regime_models:
            # Combine training performance and CV score
            train_score = self.regime_performance[regime]['weighted_mse']
            cv_score = self.cv_results.get(regime, {}).get('mean_cv_score', np.inf)
            
            # Combined score (lower is better)
            combined_score = 0.7 * cv_score + 0.3 * train_score
            
            regime_scores.append(combined_score)
            regime_list.append(regime)
        
        if regime_scores:
            # Convert to weights (higher weight for lower score)
            regime_scores = np.array(regime_scores)
            
            # Handle infinite scores
            finite_mask = np.isfinite(regime_scores)
            if np.any(finite_mask):
                # Inverse weighting with softmax
                finite_scores = regime_scores[finite_mask]
                weights = np.zeros(len(regime_scores))
                
                # Use negative scores for softmax (higher score = lower weight)
                exp_scores = np.exp(-finite_scores / np.std(finite_scores) if np.std(finite_scores) > 0 else -finite_scores)
                weights[finite_mask] = exp_scores / np.sum(exp_scores)
            else:
                # Uniform weights if all scores are infinite
                weights = np.ones(len(regime_scores)) / len(regime_scores)
            
            self.ensemble_weights = dict(zip(regime_list, weights))
        else:
            self.ensemble_weights = {}
    
    def predict(self, 
               X: np.ndarray,
               regime_probabilities: np.ndarray,
               method: str = 'regime_weighted') -> np.ndarray:
        """
        Make predictions using regime-conditional models.
        
        Args:
            X: Feature matrix for prediction
            regime_probabilities: Current regime probabilities
            method: Prediction method ('regime_weighted', 'best_regime', 'ensemble')
        
        Returns:
            Predictions
        """
        X = self._validate_input(X)
        regime_probabilities = self._validate_input(regime_probabilities)
        
        if method == 'regime_weighted':
            return self._predict_regime_weighted(X, regime_probabilities)
        elif method == 'best_regime':
            return self._predict_best_regime(X, regime_probabilities)
        elif method == 'ensemble':
            return self._predict_ensemble(X, regime_probabilities)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
    
    def _predict_regime_weighted(self, 
                               X: np.ndarray,
                               regime_probabilities: np.ndarray) -> np.ndarray:
        """Make regime-weighted predictions."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for regime, model in self.regime_models.items():
            regime_pred = model.predict(X)
            regime_weights = regime_probabilities[:, regime]
            predictions += regime_weights * regime_pred
        
        return predictions
    
    def _predict_best_regime(self, 
                           X: np.ndarray,
                           regime_probabilities: np.ndarray) -> np.ndarray:
        """Make predictions using the most likely regime for each sample."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        # Find most likely regime for each sample
        best_regimes = np.argmax(regime_probabilities, axis=1)
        
        for regime, model in self.regime_models.items():
            regime_mask = best_regimes == regime
            if np.any(regime_mask):
                regime_pred = model.predict(X[regime_mask])
                predictions[regime_mask] = regime_pred
        
        return predictions
    
    def _predict_ensemble(self, 
                        X: np.ndarray,
                        regime_probabilities: np.ndarray) -> np.ndarray:
        """Make ensemble predictions using learned model weights."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        total_weight = 0.0
        for regime, model in self.regime_models.items():
            regime_pred = model.predict(X)
            ensemble_weight = self.ensemble_weights.get(regime, 0.0)
            
            # Combine ensemble weight with regime probability
            regime_weights = regime_probabilities[:, regime] * ensemble_weight
            predictions += regime_weights * regime_pred
            total_weight += ensemble_weight
        
        # Normalize if needed
        if total_weight > 0:
            predictions = predictions / total_weight
        
        return predictions
    
    def predict_with_uncertainty(self, 
                               X: np.ndarray,
                               regime_probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            X: Feature matrix
            regime_probabilities: Regime probabilities
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        X = self._validate_input(X)
        regime_probabilities = self._validate_input(regime_probabilities)
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        uncertainties = np.zeros(n_samples)
        
        # Collect predictions and uncertainties from all regimes
        regime_predictions = {}
        regime_uncertainties = {}
        
        for regime, model in self.regime_models.items():
            pred, unc = model.predict(X, return_uncertainty=True)
            regime_predictions[regime] = pred
            regime_uncertainties[regime] = unc
        
        # Calculate weighted predictions and uncertainties
        for i in range(n_samples):
            sample_pred = 0.0
            sample_var = 0.0
            total_weight = 0.0
            
            for regime in self.regime_models:
                weight = regime_probabilities[i, regime]
                pred = regime_predictions[regime][i]
                unc = regime_uncertainties[regime][i]
                
                sample_pred += weight * pred
                sample_var += weight * (unc**2 + pred**2)
                total_weight += weight
            
            if total_weight > 0:
                sample_pred /= total_weight
                # Uncertainty includes both model uncertainty and regime uncertainty
                sample_var = sample_var / total_weight - sample_pred**2
                sample_var = max(0, sample_var)  # Ensure non-negative
                
                predictions[i] = sample_pred
                uncertainties[i] = np.sqrt(sample_var)
        
        return predictions, uncertainties
    
    def get_regime_feature_analysis(self) -> pd.DataFrame:
        """
        Get analysis of feature selection across regimes.
        
        Returns:
            DataFrame with feature analysis across regimes
        """
        if not self.regime_models:
            return pd.DataFrame()
        
        feature_analysis = []
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            feature_info = {
                'feature_idx': feature_idx,
                'feature_name': feature_name,
                'selected_regimes': [],
                'importance_scores': {},
                'weight_means': {},
                'weight_stds': {}
            }
            
            for regime, model in self.regime_models.items():
                if (model.selected_features is not None and 
                    feature_idx in model.selected_features):
                    feature_info['selected_regimes'].append(regime)
                
                if (model.feature_importance is not None and 
                    feature_idx < len(model.feature_importance)):
                    feature_info['importance_scores'][regime] = model.feature_importance[feature_idx]
                    feature_info['weight_means'][regime] = model.weights_mean[feature_idx]
                    feature_info['weight_stds'][regime] = np.sqrt(model.weights_cov[feature_idx, feature_idx])
            
            # Summary statistics
            feature_info['n_regimes_selected'] = len(feature_info['selected_regimes'])
            feature_info['selection_consistency'] = len(feature_info['selected_regimes']) / len(self.regime_models)
            
            if feature_info['importance_scores']:
                importances = list(feature_info['importance_scores'].values())
                feature_info['avg_importance'] = np.mean(importances)
                feature_info['importance_std'] = np.std(importances)
            else:
                feature_info['avg_importance'] = 0.0
                feature_info['importance_std'] = 0.0
            
            feature_analysis.append(feature_info)
        
        # Convert to DataFrame
        df_data = []
        for info in feature_analysis:
            row = {
                'feature_idx': info['feature_idx'],
                'feature_name': info['feature_name'],
                'n_regimes_selected': info['n_regimes_selected'],
                'selection_consistency': info['selection_consistency'],
                'avg_importance': info['avg_importance'],
                'importance_std': info['importance_std']
            }
            
            # Add regime-specific columns
            for regime in self.regime_models:
                row[f'selected_regime_{regime}'] = regime in info['selected_regimes']
                row[f'importance_regime_{regime}'] = info['importance_scores'].get(regime, 0.0)
            
            df_data.append(row)
        
        return pd.DataFrame(df_data).sort_values('avg_importance', ascending=False)
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive summary of regime-conditional models.
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'n_regimes': self.n_regimes,
            'n_fitted_regimes': len(self.regime_models),
            'hierarchical_structure': self.hierarchical_structure,
            'regime_performance': self.regime_performance.copy(),
            'cv_results': self.cv_results.copy(),
            'ensemble_weights': self.ensemble_weights.copy() if self.ensemble_weights else {}
        }
        
        # Add regime model summaries
        regime_summaries = {}
        for regime, model in self.regime_models.items():
            regime_summaries[regime] = model.get_model_summary()
        
        summary['regime_model_summaries'] = regime_summaries
        
        # Cross-regime statistics
        if len(self.regime_models) > 1:
            feature_counts = [len(features) for features in self.regime_features.values()]
            summary['feature_selection_stats'] = {
                'avg_features_per_regime': np.mean(feature_counts),
                'std_features_per_regime': np.std(feature_counts),
                'min_features': min(feature_counts),
                'max_features': max(feature_counts)
            }
            
            # Feature overlap analysis
            all_features = set()
            for features in self.regime_features.values():
                all_features.update(features)
            
            common_features = set(self.regime_features[list(self.regime_features.keys())[0]])
            for features in list(self.regime_features.values())[1:]:
                common_features = common_features.intersection(set(features))
            
            summary['feature_overlap'] = {
                'total_unique_features': len(all_features),
                'common_features': len(common_features),
                'feature_diversity': len(all_features) / self.n_features
            }
        
        return summary
    
    def dynamic_feature_selection(self, 
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 regime_probabilities: np.ndarray,
                                 time_window: int = 252,
                                 min_regime_probability: float = 0.3) -> Dict:
        """
        Perform dynamic feature selection based on regime probability evolution.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            regime_probabilities: Regime probability matrix (n_samples, n_regimes)
            time_window: Rolling window for dynamic selection
            min_regime_probability: Minimum regime probability threshold
        
        Returns:
            Dictionary with dynamic feature selection results
        """
        self.logger.info("Performing dynamic feature selection")
        
        X = self._validate_input(X)
        y = self._validate_input(y).flatten()
        regime_probabilities = self._validate_input(regime_probabilities)
        
        n_samples, n_features = X.shape
        
        # Initialize tracking structures
        dynamic_features = {regime: [] for regime in range(self.n_regimes)}
        feature_stability = {regime: [] for regime in range(self.n_regimes)}
        regime_transitions = []
        
        # Rolling window analysis
        for start_idx in range(0, n_samples - time_window + 1, time_window // 4):
            end_idx = min(start_idx + time_window, n_samples)
            
            # Extract window data
            X_window = X[start_idx:end_idx]
            y_window = y[start_idx:end_idx]
            regime_window = regime_probabilities[start_idx:end_idx]
            
            # Identify dominant regimes in this window
            avg_regime_probs = np.mean(regime_window, axis=0)
            active_regimes = np.where(avg_regime_probs > min_regime_probability)[0]
            
            # Detect regime transitions
            if len(regime_transitions) > 0:
                prev_regimes = set(regime_transitions[-1]['active_regimes'])
                curr_regimes = set(active_regimes)
                if prev_regimes != curr_regimes:
                    transition_info = {
                        'window_start': start_idx,
                        'window_end': end_idx,
                        'prev_regimes': list(prev_regimes),
                        'curr_regimes': list(curr_regimes),
                        'transition_magnitude': np.sum(np.abs(avg_regime_probs - 
                                                             regime_transitions[-1]['regime_probs']))
                    }
                    regime_transitions.append(transition_info)
            
            regime_transitions.append({
                'window_start': start_idx,
                'window_end': end_idx,
                'active_regimes': list(active_regimes),
                'regime_probs': avg_regime_probs.copy()
            })
            
            # Fit models for active regimes in this window
            window_models = {}
            for regime in active_regimes:
                regime_weights = regime_window[:, regime]
                effective_samples = np.sum(regime_weights > 0.1)
                
                if effective_samples >= 10:
                    # Create and fit model for this window
                    window_model = self._create_weighted_sbl_model(regime)
                    self._fit_weighted_model(window_model, X_window, y_window, regime_weights)
                    window_models[regime] = window_model
                    
                    # Track selected features
                    if window_model.selected_features is not None:
                        dynamic_features[regime].append({
                            'window_start': start_idx,
                            'window_end': end_idx,
                            'selected_features': list(window_model.selected_features),
                            'feature_importance': window_model.feature_importance.copy() if window_model.feature_importance is not None else None,
                            'n_features': len(window_model.selected_features)
                        })
        
        # Calculate feature stability across windows
        for regime in range(self.n_regimes):
            if dynamic_features[regime]:
                # Calculate feature consistency across windows
                all_selected = [set(window['selected_features']) for window in dynamic_features[regime]]
                
                if len(all_selected) > 1:
                    # Jaccard similarity between consecutive windows
                    similarities = []
                    for i in range(len(all_selected) - 1):
                        intersection = len(all_selected[i].intersection(all_selected[i+1]))
                        union = len(all_selected[i].union(all_selected[i+1]))
                        similarity = intersection / union if union > 0 else 0
                        similarities.append(similarity)
                    
                    feature_stability[regime] = {
                        'avg_similarity': np.mean(similarities),
                        'std_similarity': np.std(similarities),
                        'min_similarity': np.min(similarities),
                        'max_similarity': np.max(similarities),
                        'stability_score': np.mean(similarities) - np.std(similarities)  # Higher is more stable
                    }
                    
                    # Feature importance stability
                    if all(window.get('feature_importance') is not None for window in dynamic_features[regime]):
                        importance_series = []
                        for window in dynamic_features[regime]:
                            importance_series.append(window['feature_importance'])
                        
                        importance_matrix = np.array(importance_series)
                        feature_stability[regime]['importance_correlations'] = np.corrcoef(importance_matrix)
                        feature_stability[regime]['avg_importance_stability'] = np.mean(np.diag(np.corrcoef(importance_matrix.T), k=1))
        
        return {
            'dynamic_features': dynamic_features,
            'feature_stability': feature_stability,
            'regime_transitions': regime_transitions,
            'analysis_summary': {
                'n_windows_analyzed': len(regime_transitions),
                'n_regime_transitions': len([t for t in regime_transitions if 'transition_magnitude' in t]),
                'avg_features_per_regime': {regime: np.mean([w['n_features'] for w in windows]) 
                                          for regime, windows in dynamic_features.items() if windows},
                'feature_stability_by_regime': {regime: stability.get('stability_score', 0) 
                                              for regime, stability in feature_stability.items()}
            }
        }
    
    def regime_switching_analysis(self, 
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 regime_probabilities: np.ndarray,
                                 switching_threshold: float = 0.7) -> Dict:
        """
        Analyze regime-switching patterns and their impact on feature selection.
        
        Args:
            X: Feature matrix
            y: Target vector
            regime_probabilities: Regime probability matrix
            switching_threshold: Threshold for regime switching detection
        
        Returns:
            Dictionary with regime switching analysis
        """
        self.logger.info("Performing regime switching analysis")
        
        X = self._validate_input(X)
        y = self._validate_input(y).flatten()
        regime_probabilities = self._validate_input(regime_probabilities)
        
        n_samples, n_features = X.shape
        
        # Detect regime switches
        dominant_regimes = np.argmax(regime_probabilities, axis=1)
        regime_confidences = np.max(regime_probabilities, axis=1)
        
        # Identify stable and switching periods
        stable_periods = regime_confidences > switching_threshold
        switching_points = np.where(np.diff(dominant_regimes) != 0)[0] + 1
        
        # Analyze switching patterns
        switching_analysis = {
            'n_switches': len(switching_points),
            'switch_frequency': len(switching_points) / n_samples,
            'avg_regime_duration': np.mean(np.diff(np.concatenate([[0], switching_points, [n_samples]]))),
            'stable_period_ratio': np.sum(stable_periods) / n_samples
        }
        
        # Analyze feature selection around switches
        switch_window = 20  # Days around switch to analyze
        feature_switch_analysis = []
        
        for switch_point in switching_points:
            if switch_point > switch_window and switch_point < n_samples - switch_window:
                # Before and after switch periods
                before_start = max(0, switch_point - switch_window)
                before_end = switch_point
                after_start = switch_point
                after_end = min(n_samples, switch_point + switch_window)
                
                # Regime probabilities before and after
                regime_before = np.mean(regime_probabilities[before_start:before_end], axis=0)
                regime_after = np.mean(regime_probabilities[after_start:after_end], axis=0)
                
                # Dominant regimes
                dominant_before = np.argmax(regime_before)
                dominant_after = np.argmax(regime_after)
                
                # Fit models for both periods
                models_before = {}
                models_after = {}
                
                for regime in [dominant_before, dominant_after]:
                    # Before switch
                    if regime_before[regime] > 0.3:
                        X_before = X[before_start:before_end]
                        y_before = y[before_start:before_end]
                        weights_before = regime_probabilities[before_start:before_end, regime]
                        
                        model_before = self._create_weighted_sbl_model(regime)
                        self._fit_weighted_model(model_before, X_before, y_before, weights_before)
                        models_before[regime] = model_before
                    
                    # After switch
                    if regime_after[regime] > 0.3:
                        X_after = X[after_start:after_end]
                        y_after = y[after_start:after_end]
                        weights_after = regime_probabilities[after_start:after_end, regime]
                        
                        model_after = self._create_weighted_sbl_model(regime)
                        self._fit_weighted_model(model_after, X_after, y_after, weights_after)
                        models_after[regime] = model_after
                
                # Compare feature selections
                feature_changes = {}
                for regime in set(list(models_before.keys()) + list(models_after.keys())):
                    before_features = set(models_before[regime].selected_features) if regime in models_before and models_before[regime].selected_features is not None else set()
                    after_features = set(models_after[regime].selected_features) if regime in models_after and models_after[regime].selected_features is not None else set()
                    
                    feature_changes[regime] = {
                        'features_before': list(before_features),
                        'features_after': list(after_features),
                        'features_added': list(after_features - before_features),
                        'features_removed': list(before_features - after_features),
                        'feature_turnover': len(before_features.symmetric_difference(after_features)) / max(len(before_features.union(after_features)), 1)
                    }
                
                switch_analysis = {
                    'switch_point': switch_point,
                    'regime_before': dominant_before,
                    'regime_after': dominant_after,
                    'regime_confidence_before': regime_before[dominant_before],
                    'regime_confidence_after': regime_after[dominant_after],
                    'feature_changes': feature_changes,
                    'avg_feature_turnover': np.mean([changes['feature_turnover'] for changes in feature_changes.values()])
                }
                
                feature_switch_analysis.append(switch_analysis)
        
        # Aggregate switching statistics
        if feature_switch_analysis:
            switching_analysis['avg_feature_turnover'] = np.mean([s['avg_feature_turnover'] for s in feature_switch_analysis])
            switching_analysis['feature_switch_correlation'] = np.corrcoef([s['avg_feature_turnover'] for s in feature_switch_analysis], 
                                                                          [abs(s['regime_confidence_after'] - s['regime_confidence_before']) for s in feature_switch_analysis])[0, 1]
        
        return {
            'switching_patterns': switching_analysis,
            'individual_switches': feature_switch_analysis,
            'regime_persistence': {
                'avg_regime_duration': switching_analysis['avg_regime_duration'],
                'regime_transition_matrix': self._calculate_transition_matrix(dominant_regimes),
                'regime_stability_scores': self._calculate_regime_stability(regime_probabilities)
            }
        }
    
    def _calculate_transition_matrix(self, regime_sequence: np.ndarray) -> np.ndarray:
        """Calculate regime transition probability matrix."""
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regime_sequence) - 1):
            current_regime = regime_sequence[i]
            next_regime = regime_sequence[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def _calculate_regime_stability(self, regime_probabilities: np.ndarray) -> Dict:
        """Calculate stability scores for each regime."""
        stability_scores = {}
        
        for regime in range(self.n_regimes):
            regime_probs = regime_probabilities[:, regime]
            
            # Calculate various stability metrics
            mean_prob = np.mean(regime_probs)
            std_prob = np.std(regime_probs)
            persistence = np.mean(regime_probs > 0.5)  # Fraction of time regime is dominant
            
            # Autocorrelation as stability measure
            if len(regime_probs) > 1:
                autocorr = np.corrcoef(regime_probs[:-1], regime_probs[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0.0
            else:
                autocorr = 0.0
            
            stability_scores[regime] = {
                'mean_probability': mean_prob,
                'probability_volatility': std_prob,
                'persistence': persistence,
                'autocorrelation': autocorr,
                'stability_score': mean_prob * autocorr / (std_prob + 1e-8)  # Combined stability measure
            }
        
        return stability_scores
    
    def adaptive_regularization(self, 
                              X: np.ndarray,
                              y: np.ndarray,
                              regime_probabilities: np.ndarray,
                              base_penalty: float = 0.1) -> 'RegimeConditionalSBL':
        """
        Apply adaptive regularization based on regime uncertainty and feature stability.
        
        Args:
            X: Feature matrix
            y: Target vector  
            regime_probabilities: Regime probability matrix
            base_penalty: Base regularization penalty
        
        Returns:
            Self for method chaining
        """
        self.logger.info("Applying adaptive regularization")
        
        X = self._validate_input(X)
        y = self._validate_input(y).flatten()
        regime_probabilities = self._validate_input(regime_probabilities)
        
        # Calculate regime uncertainty
        regime_uncertainty = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-8), axis=1)
        avg_uncertainty = np.mean(regime_uncertainty)
        
        # Adjust regularization based on uncertainty
        uncertainty_factor = 1.0 + avg_uncertainty  # Higher uncertainty = more regularization
        
        # Calculate feature instability penalty
        if hasattr(self, 'regime_features') and len(self.regime_features) > 1:
            feature_consistency = self._calculate_feature_consistency()
            instability_factor = 1.0 + (1.0 - feature_consistency)  # Lower consistency = more regularization
        else:
            instability_factor = 1.0
        
        # Adaptive penalties
        adaptive_regime_penalty = base_penalty * uncertainty_factor
        adaptive_cross_regime_penalty = self.cross_regime_regularization * instability_factor
        
        # Update regularization parameters
        self.regime_switching_penalty = adaptive_regime_penalty
        self.cross_regime_regularization = adaptive_cross_regime_penalty
        
        self.logger.info(f"Adaptive regularization: regime_penalty={adaptive_regime_penalty:.4f}, "
                        f"cross_regime_penalty={adaptive_cross_regime_penalty:.4f}")
        
        # Refit with adaptive regularization
        return self.fit(X, y, regime_probabilities)
    
    def _calculate_feature_consistency(self) -> float:
        """Calculate average feature consistency across regimes."""
        if len(self.regime_features) < 2:
            return 1.0
        
        regime_list = list(self.regime_features.keys())
        similarities = []
        
        for i in range(len(regime_list)):
            for j in range(i + 1, len(regime_list)):
                features_i = set(self.regime_features[regime_list[i]])
                features_j = set(self.regime_features[regime_list[j]])
                
                intersection = len(features_i.intersection(features_j))
                union = len(features_i.union(features_j))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0