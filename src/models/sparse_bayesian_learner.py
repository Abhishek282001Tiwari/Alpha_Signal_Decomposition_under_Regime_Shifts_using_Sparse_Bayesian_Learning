import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats, special
from scipy.optimize import minimize
import logging
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

class SparseBayesianLearner:
    """
    Comprehensive Sparse Bayesian Learning implementation with multiple prior types
    and variational inference for automatic feature selection and uncertainty quantification.
    """
    
    def __init__(self, 
                 prior_type: str = 'laplace',
                 alpha_prior: float = 1.0,
                 beta_prior: float = 1.0,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6,
                 automatic_relevance_determination: bool = True):
        """
        Initialize Sparse Bayesian Learner.
        
        Args:
            prior_type: Type of sparsity prior ('laplace', 'horseshoe', 'spike_slab')
            alpha_prior: Hyperparameter for precision prior
            beta_prior: Hyperparameter for precision prior
            max_iterations: Maximum iterations for optimization
            convergence_threshold: Convergence threshold for ELBO
            automatic_relevance_determination: Enable ARD for feature selection
        """
        self.prior_type = prior_type
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.ard_enabled = automatic_relevance_determination
        
        # Model parameters
        self.weights_mean = None
        self.weights_cov = None
        self.alpha = None  # Precision parameters
        self.beta = None   # Noise precision
        self.gamma = None  # Relevance parameters (ARD)
        
        # Variational parameters
        self.variational_params = {}
        self.elbo_history = []
        
        # Feature importance and selection
        self.feature_importance = None
        self.selected_features = None
        
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'SparseBayesianLearner':
        """
        Fit Sparse Bayesian model to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Optional feature names
        
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting Sparse Bayesian model with {self.prior_type} prior")
        
        X = self._validate_input(X)
        y = self._validate_input(y)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        self.feature_names = feature_names or [f'feature_{i}' for i in range(n_features)]
        
        # Initialize parameters
        self._initialize_parameters(n_features)
        
        # Coordinate Ascent Variational Inference (CAVI)
        self._run_cavi(X, y)
        
        # Feature selection based on relevance
        self._select_features()
        
        self.logger.info(f"Model fitted. Selected {len(self.selected_features)} out of {n_features} features")
        return self
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.array(X)
    
    def _initialize_parameters(self, n_features: int):
        """Initialize model parameters."""
        # Initialize weights
        self.weights_mean = np.zeros(n_features)
        self.weights_cov = np.eye(n_features)
        
        # Initialize precision parameters
        if self.ard_enabled:
            self.alpha = np.ones(n_features) * self.alpha_prior
        else:
            self.alpha = self.alpha_prior
        
        self.beta = self.beta_prior
        
        # Initialize relevance parameters for ARD
        if self.ard_enabled:
            self.gamma = np.ones(n_features) * 0.5
        
        # Initialize variational parameters based on prior type
        if self.prior_type == 'spike_slab':
            self.variational_params['spike_probs'] = np.ones(n_features) * 0.5
            self.variational_params['slab_variance'] = np.ones(n_features)
        elif self.prior_type == 'horseshoe':
            self.variational_params['tau'] = np.ones(n_features)
            self.variational_params['lambda'] = np.ones(n_features)
    
    def _run_cavi(self, X: np.ndarray, y: np.ndarray):
        """Run Coordinate Ascent Variational Inference."""
        prev_elbo = -np.inf
        
        for iteration in range(self.max_iterations):
            # Update variational parameters based on prior type
            if self.prior_type == 'laplace':
                self._update_laplace_params(X, y)
            elif self.prior_type == 'horseshoe':
                self._update_horseshoe_params(X, y)
            elif self.prior_type == 'spike_slab':
                self._update_spike_slab_params(X, y)
            
            # Update weights
            self._update_weights(X, y)
            
            # Update precision parameters
            if self.ard_enabled:
                self._update_ard_parameters(X, y)
            
            # Calculate ELBO
            elbo = self._calculate_elbo(X, y)
            self.elbo_history.append(elbo)
            
            # Check convergence
            if abs(elbo - prev_elbo) < self.convergence_threshold:
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            prev_elbo = elbo
            
            if iteration % 100 == 0:
                self.logger.debug(f"Iteration {iteration}, ELBO: {elbo:.6f}")
    
    def _update_laplace_params(self, X: np.ndarray, y: np.ndarray):
        """Update parameters for Laplace prior."""
        if self.ard_enabled:
            # ARD for Laplace prior
            for i in range(self.n_features):
                # Update alpha_i based on weight posterior
                weight_var = self.weights_cov[i, i]
                weight_mean_sq = self.weights_mean[i]**2
                
                # Update using empirical Bayes
                self.alpha[i] = np.sqrt(2.0 / (weight_mean_sq + weight_var + 1e-8))
        else:
            # Global Laplace prior
            weight_l1_norm = np.sum(np.abs(self.weights_mean))
            self.alpha = self.n_features / (weight_l1_norm + 1e-8)
    
    def _update_horseshoe_params(self, X: np.ndarray, y: np.ndarray):
        """Update parameters for Horseshoe prior."""
        # Update local shrinkage parameters (lambda)
        for i in range(self.n_features):
            weight_var = self.weights_cov[i, i]
            weight_mean_sq = self.weights_mean[i]**2
            
            # Update lambda_i
            scale_param = np.sqrt(weight_mean_sq + weight_var + 1e-8)
            self.variational_params['lambda'][i] = 1.0 / (scale_param + 1e-8)
        
        # Update global shrinkage parameter (tau)
        total_shrinkage = np.sum(1.0 / (self.variational_params['lambda'] + 1e-8))
        self.variational_params['tau'] = np.ones(self.n_features) / (total_shrinkage + 1e-8)
        
        # Update alpha based on horseshoe structure
        if self.ard_enabled:
            self.alpha = 1.0 / (self.variational_params['tau'] * 
                               self.variational_params['lambda'] + 1e-8)
    
    def _update_spike_slab_params(self, X: np.ndarray, y: np.ndarray):
        """Update parameters for Spike-and-Slab prior."""
        # Update spike probabilities
        for i in range(self.n_features):
            weight_var = self.weights_cov[i, i]
            weight_mean_sq = self.weights_mean[i]**2
            
            # Evidence for slab vs spike
            slab_evidence = -0.5 * np.log(2 * np.pi * self.variational_params['slab_variance'][i]) - \
                           0.5 * (weight_mean_sq + weight_var) / self.variational_params['slab_variance'][i]
            
            spike_evidence = -0.5 * np.log(2 * np.pi * 1e-6) - \
                            0.5 * (weight_mean_sq + weight_var) / 1e-6
            
            # Update spike probability using softmax
            log_odds = slab_evidence - spike_evidence
            self.variational_params['spike_probs'][i] = 1.0 / (1.0 + np.exp(-log_odds))
        
        # Update slab variances
        for i in range(self.n_features):
            weight_var = self.weights_cov[i, i]
            weight_mean_sq = self.weights_mean[i]**2
            
            self.variational_params['slab_variance'][i] = weight_mean_sq + weight_var + 1e-6
        
        # Update alpha based on spike-slab mixture
        if self.ard_enabled:
            spike_precision = 1e6  # Very high precision for spike
            for i in range(self.n_features):
                spike_prob = 1 - self.variational_params['spike_probs'][i]
                slab_precision = 1.0 / self.variational_params['slab_variance'][i]
                
                self.alpha[i] = spike_prob * spike_precision + \
                               (1 - spike_prob) * slab_precision
    
    def _update_weights(self, X: np.ndarray, y: np.ndarray):
        """Update weight posterior distribution."""
        n_samples, n_features = X.shape
        
        # Precision matrix for weights
        if self.ard_enabled:
            precision_matrix = self.beta * (X.T @ X) + np.diag(self.alpha)
        else:
            precision_matrix = self.beta * (X.T @ X) + self.alpha * np.eye(n_features)
        
        # Covariance matrix (inverse of precision)
        try:
            self.weights_cov = np.linalg.inv(precision_matrix)
        except np.linalg.LinAlgError:
            # Add regularization if matrix is singular
            precision_matrix += 1e-6 * np.eye(n_features)
            self.weights_cov = np.linalg.inv(precision_matrix)
        
        # Mean of weight posterior
        self.weights_mean = self.beta * self.weights_cov @ (X.T @ y)
    
    def _update_ard_parameters(self, X: np.ndarray, y: np.ndarray):
        """Update Automatic Relevance Determination parameters."""
        if not self.ard_enabled:
            return
        
        # Update gamma (relevance parameters)
        for i in range(self.n_features):
            # Empirical Bayes update for ARD
            weight_var = self.weights_cov[i, i]
            weight_mean_sq = self.weights_mean[i]**2
            
            # Update gamma using the evidence
            self.gamma[i] = 1.0 - self.alpha[i] * weight_var
            self.gamma[i] = max(0.0, min(1.0, self.gamma[i]))  # Clamp to [0, 1]
        
        # Update alpha based on gamma
        for i in range(self.n_features):
            if self.gamma[i] > 1e-6:  # Feature is relevant
                weight_var = self.weights_cov[i, i]
                weight_mean_sq = self.weights_mean[i]**2
                self.alpha[i] = self.gamma[i] / (weight_mean_sq + weight_var + 1e-8)
            else:  # Feature is irrelevant
                self.alpha[i] = 1e6  # Very high precision (shrinks to zero)
    
    def _calculate_elbo(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate Evidence Lower Bound (ELBO)."""
        n_samples, n_features = X.shape
        
        # Likelihood term
        y_pred = X @ self.weights_mean
        residuals = y - y_pred
        
        # Account for uncertainty in weights
        uncertainty_term = 0.0
        for i in range(n_samples):
            x_i = X[i, :]
            uncertainty_term += x_i.T @ self.weights_cov @ x_i
        
        likelihood_term = -0.5 * n_samples * np.log(2 * np.pi / self.beta) - \
                         0.5 * self.beta * (np.sum(residuals**2) + uncertainty_term)
        
        # Prior term (depends on prior type)
        if self.prior_type == 'laplace':
            prior_term = self._laplace_prior_term()
        elif self.prior_type == 'horseshoe':
            prior_term = self._horseshoe_prior_term()
        elif self.prior_type == 'spike_slab':
            prior_term = self._spike_slab_prior_term()
        else:
            prior_term = 0.0
        
        # Entropy term (variational distribution)
        entropy_term = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * self.weights_cov))
        
        elbo = likelihood_term + prior_term + entropy_term
        return elbo
    
    def _laplace_prior_term(self) -> float:
        """Calculate prior term for Laplace prior."""
        if self.ard_enabled:
            # ARD Laplace prior
            prior_term = 0.0
            for i in range(self.n_features):
                # Log probability under Laplace prior
                alpha_i = self.alpha[i]
                weight_mean = self.weights_mean[i]
                weight_var = self.weights_cov[i, i]
                
                # Approximation for log|w_i| expectation under normal distribution
                log_abs_expectation = 0.5 * np.log(2 / np.pi) + 0.5 * np.log(weight_var) + \
                                     np.log(special.erfc(-np.abs(weight_mean) / np.sqrt(2 * weight_var)))
                
                prior_term += np.log(alpha_i / 2) - alpha_i * (np.abs(weight_mean) + log_abs_expectation)
        else:
            # Global Laplace prior
            l1_norm_expectation = np.sum(np.sqrt(2 / np.pi * np.diag(self.weights_cov)) + 
                                       np.abs(self.weights_mean))
            prior_term = self.n_features * np.log(self.alpha / 2) - self.alpha * l1_norm_expectation
        
        return prior_term
    
    def _horseshoe_prior_term(self) -> float:
        """Calculate prior term for Horseshoe prior."""
        prior_term = 0.0
        
        for i in range(self.n_features):
            # Local shrinkage parameter
            lambda_i = self.variational_params['lambda'][i]
            tau_i = self.variational_params['tau'][i]
            
            # Log probability under horseshoe prior
            weight_mean_sq = self.weights_mean[i]**2
            weight_var = self.weights_cov[i, i]
            
            # Approximation for horseshoe prior term
            shrinkage = lambda_i * tau_i
            prior_term += -0.5 * np.log(2 * np.pi * shrinkage) - \
                         0.5 * (weight_mean_sq + weight_var) / shrinkage
        
        return prior_term
    
    def _spike_slab_prior_term(self) -> float:
        """Calculate prior term for Spike-and-Slab prior."""
        prior_term = 0.0
        
        for i in range(self.n_features):
            spike_prob = 1 - self.variational_params['spike_probs'][i]
            slab_prob = self.variational_params['spike_probs'][i]
            slab_var = self.variational_params['slab_variance'][i]
            
            weight_mean_sq = self.weights_mean[i]**2
            weight_var = self.weights_cov[i, i]
            
            # Spike component (delta function at zero)
            spike_term = spike_prob * (-1e6 * (weight_mean_sq + weight_var))
            
            # Slab component (normal distribution)
            slab_term = slab_prob * (-0.5 * np.log(2 * np.pi * slab_var) - 
                                   0.5 * (weight_mean_sq + weight_var) / slab_var)
            
            prior_term += np.log(spike_prob + np.exp(slab_term - spike_term))
        
        return prior_term
    
    def _select_features(self):
        """Select relevant features based on posterior analysis."""
        if self.ard_enabled:
            # Use gamma values for feature selection
            relevance_threshold = 0.1  # Features with gamma > threshold are selected
            relevant_indices = np.where(self.gamma > relevance_threshold)[0]
        else:
            # Use weight magnitude for feature selection
            weight_magnitudes = np.abs(self.weights_mean)
            weight_threshold = np.percentile(weight_magnitudes, 75)  # Top 25% features
            relevant_indices = np.where(weight_magnitudes > weight_threshold)[0]
        
        self.selected_features = relevant_indices
        
        # Calculate feature importance
        if self.ard_enabled:
            self.feature_importance = self.gamma.copy()
        else:
            weight_magnitudes = np.abs(self.weights_mean)
            self.feature_importance = weight_magnitudes / np.sum(weight_magnitudes)
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the fitted model.
        
        Args:
            X: Feature matrix for prediction
            return_uncertainty: Whether to return prediction uncertainty
        
        Returns:
            Predictions and optionally prediction uncertainty
        """
        X = self._validate_input(X)
        
        # Mean prediction
        y_pred = X @ self.weights_mean
        
        if return_uncertainty:
            # Prediction uncertainty
            pred_var = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                x_i = X[i, :]
                pred_var[i] = x_i.T @ self.weights_cov @ x_i + 1.0 / self.beta
            
            pred_std = np.sqrt(pred_var)
            return y_pred, pred_std
        
        return y_pred
    
    def predict_with_selected_features(self, X: np.ndarray, 
                                     return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using only selected features.
        
        Args:
            X: Feature matrix for prediction
            return_uncertainty: Whether to return prediction uncertainty
        
        Returns:
            Predictions using selected features only
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Fit the model first.")
        
        X = self._validate_input(X)
        X_selected = X[:, self.selected_features]
        
        # Use only selected feature weights
        weights_selected = self.weights_mean[self.selected_features]
        
        y_pred = X_selected @ weights_selected
        
        if return_uncertainty:
            # Uncertainty with selected features
            cov_selected = self.weights_cov[np.ix_(self.selected_features, self.selected_features)]
            pred_var = np.zeros(X_selected.shape[0])
            
            for i in range(X_selected.shape[0]):
                x_i = X_selected[i, :]
                pred_var[i] = x_i.T @ cov_selected @ x_i + 1.0 / self.beta
            
            pred_std = np.sqrt(pred_var)
            return y_pred, pred_std
        
        return y_pred
    
    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_k: Number of top features to return
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model not fitted yet.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance,
            'weight_mean': self.weights_mean,
            'weight_std': np.sqrt(np.diag(self.weights_cov)),
            'selected': np.isin(np.arange(len(self.feature_names)), self.selected_features)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        if top_k is not None:
            importance_df = importance_df.head(top_k)
        
        return importance_df
    
    def calculate_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate marginal likelihood for model comparison.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            Log marginal likelihood
        """
        # Use the final ELBO as approximation to marginal likelihood
        return self.elbo_history[-1] if self.elbo_history else -np.inf
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model summary statistics
        """
        if self.weights_mean is None:
            raise ValueError("Model not fitted yet.")
        
        summary = {
            'prior_type': self.prior_type,
            'n_features_total': self.n_features,
            'n_features_selected': len(self.selected_features) if self.selected_features is not None else 0,
            'selection_ratio': len(self.selected_features) / self.n_features if self.selected_features is not None else 0,
            'final_elbo': self.elbo_history[-1] if self.elbo_history else None,
            'convergence_iterations': len(self.elbo_history),
            'ard_enabled': self.ard_enabled,
            'weight_sparsity': np.sum(np.abs(self.weights_mean) < 1e-6) / self.n_features,
        }
        
        # Add prior-specific information
        if self.prior_type == 'spike_slab':
            summary['avg_spike_probability'] = np.mean(1 - self.variational_params['spike_probs'])
        elif self.prior_type == 'horseshoe':
            summary['avg_shrinkage'] = np.mean(self.variational_params['lambda'] * 
                                             self.variational_params['tau'])
        
        if self.ard_enabled:
            summary['avg_relevance'] = np.mean(self.gamma)
            summary['highly_relevant_features'] = np.sum(self.gamma > 0.5)
        
        return summary