"""
Enhanced nonparametric regression methods for Shapley value analysis.

This module provides sophisticated local linear regression implementations
that closely match the R 'npreg' functionality.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings


class LocalLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Local linear regression estimator with automatic bandwidth selection.
    
    This class provides a sophisticated implementation of local linear
    regression that closely matches R's npreg functionality.
    """
    
    def __init__(self, 
                 bandwidth: Optional[Union[float, str, np.ndarray]] = 'cv.aic',
                 kernel: str = 'gaussian',
                 degree: int = 1,
                 cv_folds: int = 5,
                 bandwidth_grid_size: int = 20):
        """
        Initialize the local linear regressor.
        
        Parameters
        ----------
        bandwidth : float, str, or np.ndarray, default='cv.aic'
            Bandwidth selection method
        kernel : str, default='gaussian'
            Kernel function ('gaussian', 'epanechnikov', 'uniform')
        degree : int, default=1
            Degree of local polynomial (1 for local linear)
        cv_folds : int, default=5
            Number of cross-validation folds
        bandwidth_grid_size : int, default=20
            Size of bandwidth grid for cross-validation
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.degree = degree
        self.cv_folds = cv_folds
        self.bandwidth_grid_size = bandwidth_grid_size
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'LocalLinearRegressor':
        """
        Fit the local linear regression model.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        y : np.ndarray or pd.Series
            Target vector
            
        Returns
        -------
        LocalLinearRegressor
            Fitted model
        """
        # Convert to numpy arrays
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y).flatten()
        self.n_samples_, self.n_features_ = self.X_train_.shape
        
        # Store feature names for R compatibility
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
            self.xnames = self.feature_names_in_
        else:
            self.feature_names_in_ = [f'X{i+1}' for i in range(self.n_features_)]
            self.xnames = self.feature_names_in_
        
        # Select bandwidth
        if isinstance(self.bandwidth, str):
            if self.bandwidth == 'cv.aic':
                self.bw_ = self._cross_validate_bandwidth(criterion='aic')
            elif self.bandwidth == 'cv.ls':
                self.bw_ = self._cross_validate_bandwidth(criterion='mse')
            elif self.bandwidth == 'scott':
                self.bw_ = self._scott_bandwidth()
            elif self.bandwidth == 'silverman':
                self.bw_ = self._silverman_bandwidth()
            else:
                raise ValueError(f"Unknown bandwidth method: {self.bandwidth}")
        elif isinstance(self.bandwidth, (int, float)):
            if self.n_features_ == 1:
                self.bw_ = float(self.bandwidth)
            else:
                self.bw_ = np.full(self.n_features_, float(self.bandwidth))
        elif isinstance(self.bandwidth, np.ndarray):
            self.bw_ = np.array(self.bandwidth)
        else:
            raise ValueError("Invalid bandwidth specification")
        
        # Store bandwidth in R-compatible format
        self.bw = self.bw_ if np.isscalar(self.bw_) else self.bw_[0]
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using local linear regression.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Points to predict
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        # Convert to numpy
        X_pred = np.array(X)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1)
        
        predictions = np.zeros(X_pred.shape[0])
        
        for i, x_point in enumerate(X_pred):
            predictions[i] = self._predict_point(x_point)
        
        return predictions
    
    def _predict_point(self, x_point: np.ndarray) -> float:
        """
        Predict a single point using local linear regression.
        """
        # Compute distances and weights
        weights = self._compute_weights(x_point)
        
        # Filter out zero weights for efficiency
        nonzero_idx = weights > 1e-10
        if not np.any(nonzero_idx):
            # If no points have positive weight, return global mean
            return np.mean(self.y_train_)
        
        X_local = self.X_train_[nonzero_idx]
        y_local = self.y_train_[nonzero_idx]
        w_local = weights[nonzero_idx]
        
        # Center data around prediction point
        X_centered = X_local - x_point
        
        # Set up design matrix for local polynomial
        if self.degree == 0:
            # Local constant (Nadaraya-Watson)
            return np.average(y_local, weights=w_local)
        elif self.degree == 1:
            # Local linear
            if self.n_features_ == 1:
                # Univariate case
                X_design = np.column_stack([np.ones(len(X_centered)), X_centered])
            else:
                # Multivariate case
                X_design = np.column_stack([np.ones(len(X_centered)), X_centered])
        else:
            raise NotImplementedError("Higher degree polynomials not implemented")
        
        # Solve weighted least squares
        try:
            # W^(1/2) X
            W_sqrt = np.sqrt(w_local)
            X_weighted = X_design * W_sqrt[:, np.newaxis]
            y_weighted = y_local * W_sqrt
            
            # Solve normal equations
            beta = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
            
            # Prediction is just the intercept (since we centered at x_point)
            return beta[0]
        
        except np.linalg.LinAlgError:
            # Fallback to weighted average if matrix is singular
            return np.average(y_local, weights=w_local)
    
    def _compute_weights(self, x_point: np.ndarray) -> np.ndarray:
        """
        Compute kernel weights for local regression.
        """
        if self.n_features_ == 1:
            # Univariate case
            distances = np.abs(self.X_train_.flatten() - x_point[0])
            scaled_distances = distances / self.bw_
        else:
            # Multivariate case
            if np.isscalar(self.bw_):
                # Same bandwidth for all dimensions
                distances = np.sqrt(np.sum((self.X_train_ - x_point)**2, axis=1))
                scaled_distances = distances / self.bw_
            else:
                # Different bandwidth for each dimension
                scaled_diffs = (self.X_train_ - x_point) / self.bw_
                scaled_distances = np.sqrt(np.sum(scaled_diffs**2, axis=1))
        
        # Apply kernel function
        if self.kernel == 'gaussian':
            weights = np.exp(-0.5 * scaled_distances**2)
        elif self.kernel == 'epanechnikov':
            weights = np.maximum(0, 0.75 * (1 - scaled_distances**2))
        elif self.kernel == 'uniform':
            weights = (scaled_distances <= 1).astype(float)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        return weights
    
    def _cross_validate_bandwidth(self, criterion: str = 'mse') -> Union[float, np.ndarray]:
        """
        Select bandwidth using cross-validation.
        """
        # Get rule-of-thumb bandwidth as starting point
        rot_bw = self._scott_bandwidth()
        
        # Create bandwidth grid
        min_factor, max_factor = 0.1, 5.0
        if np.isscalar(rot_bw):
            min_bw = rot_bw * min_factor
            max_bw = rot_bw * max_factor
            bw_grid = np.logspace(np.log10(min_bw), np.log10(max_bw), self.bandwidth_grid_size)
        else:
            # Multivariate case - use same factor for all dimensions (simplified)
            factors = np.logspace(np.log10(min_factor), np.log10(max_factor), self.bandwidth_grid_size)
            bw_grid = [rot_bw * factor for factor in factors]
        
        best_score = float('inf')
        best_bandwidth = rot_bw
        
        # Cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for bw in bw_grid:
            scores = []
            
            for train_idx, val_idx in kf.split(self.X_train_):
                X_train_cv = self.X_train_[train_idx]
                y_train_cv = self.y_train_[train_idx]
                X_val_cv = self.X_train_[val_idx]
                y_val_cv = self.y_train_[val_idx]
                
                # Fit model with current bandwidth
                temp_model = LocalLinearRegressor(bandwidth=bw, kernel=self.kernel)
                temp_model.X_train_ = X_train_cv
                temp_model.y_train_ = y_train_cv
                temp_model.n_samples_, temp_model.n_features_ = X_train_cv.shape
                temp_model.bw_ = bw
                
                # Predict validation set
                y_pred = np.array([temp_model._predict_point(x) for x in X_val_cv])
                
                # Compute score
                if criterion == 'mse':
                    score = mean_squared_error(y_val_cv, y_pred)
                elif criterion == 'aic':
                    mse = mean_squared_error(y_val_cv, y_pred)
                    # Approximate AIC (simplified)
                    score = len(y_val_cv) * np.log(mse + 1e-10) + 2 * self._effective_parameters(bw)
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")
                
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_bandwidth = bw
        
        return best_bandwidth
    
    def _effective_parameters(self, bandwidth: Union[float, np.ndarray]) -> float:
        """
        Estimate effective number of parameters for AIC computation.
        """
        if np.isscalar(bandwidth):
            return min(self.n_samples_ * 0.1 / max(bandwidth, 0.01), self.n_samples_ / 2)
        else:
            return min(self.n_samples_ * 0.1 / max(np.prod(bandwidth), 0.01), self.n_samples_ / 2)
    
    def _scott_bandwidth(self) -> Union[float, np.ndarray]:
        """
        Compute Scott's rule-of-thumb bandwidth.
        """
        n = self.n_samples_
        d = self.n_features_
        
        if d == 1:
            sigma = np.std(self.X_train_)
            return sigma * (n ** (-1.0 / (d + 4)))
        else:
            sigmas = np.std(self.X_train_, axis=0)
            return sigmas * (n ** (-1.0 / (d + 4)))
    
    def _silverman_bandwidth(self) -> Union[float, np.ndarray]:
        """
        Compute Silverman's rule-of-thumb bandwidth.
        """
        n = self.n_samples_
        d = self.n_features_
        
        if d == 1:
            sigma = np.std(self.X_train_)
            return 1.06 * sigma * (n ** (-1.0 / 5))
        else:
            sigmas = np.std(self.X_train_, axis=0)
            return 1.06 * sigmas * (n ** (-1.0 / (d + 4)))


def test_nonparametric_regression():
    """Test the enhanced nonparametric regression."""
    print("Testing enhanced nonparametric regression...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    
    # 1D test
    print("Testing 1D regression...")
    X_1d = np.random.uniform(-2, 2, (n_samples, 1))
    y_1d = np.sin(X_1d.flatten()) + 0.1 * np.random.randn(n_samples)
    
    model_1d = LocalLinearRegressor(bandwidth='cv.ls')
    model_1d.fit(X_1d, y_1d)
    
    # Test prediction
    X_test = np.array([[0.0], [1.0], [-1.0]])
    pred_1d = model_1d.predict(X_test)
    print(f"1D predictions at [0, 1, -1]: {pred_1d}")
    print(f"1D bandwidth: {model_1d.bw:.4f}")
    
    # 2D test
    print("\nTesting 2D regression...")
    X_2d = np.random.uniform(-2, 2, (n_samples, 2))
    y_2d = X_2d[:, 0]**2 + X_2d[:, 1]**2 + 0.1 * np.random.randn(n_samples)
    
    model_2d = LocalLinearRegressor(bandwidth='scott')
    model_2d.fit(X_2d, y_2d)
    
    # Test prediction
    X_test_2d = np.array([[0.0, 0.0], [1.0, 1.0]])
    pred_2d = model_2d.predict(X_test_2d)
    print(f"2D predictions at [0,0] and [1,1]: {pred_2d}")
    print(f"2D bandwidth: {model_2d.bw}")
    
    print("✓ All nonparametric regression tests completed!")


if __name__ == "__main__":
    test_nonparametric_regression() 