"""
Model management utilities for Shapley value analysis.

This module provides functions for creating, fitting, and managing
nonparametric regression models for different variable subsets.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings


class NonparametricRegressor(BaseEstimator):
    """
    Nonparametric regression using local linear methods.
    
    This class provides a scikit-learn compatible interface for
    nonparametric regression, approximating the R 'npreg' functionality.
    """
    
    def __init__(self, bandwidth: Optional[Union[float, str]] = 'scott',
                 kernel: str = 'gaussian', 
                 degree: int = 1):
        """
        Initialize the nonparametric regressor.
        
        Parameters
        ----------
        bandwidth : float or str, default='scott'
            Bandwidth for kernel regression. If string, uses automatic selection.
        kernel : str, default='gaussian'
            Kernel type for regression
        degree : int, default=1
            Degree of local polynomial (1 for local linear)
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.degree = degree
        self.fitted_ = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NonparametricRegressor':
        """
        Fit the nonparametric regression model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
            
        Returns
        -------
        NonparametricRegressor
            Fitted model
        """
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        self.n_features_in_ = X.shape[1]
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
            self.xnames = self.feature_names_in_  # R compatibility
        else:
            self.feature_names_in_ = [f'X{i+1}' for i in range(self.n_features_in_)]
            self.xnames = self.feature_names_in_
        
        # Estimate bandwidth if needed
        if isinstance(self.bandwidth, str):
            self.bw_ = self._estimate_bandwidth(X, y)
        else:
            self.bw_ = self.bandwidth
            
        # Store bandwidth in R-compatible format
        self.bw = self.bw_
        
        self.fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted nonparametric regression.
        
        Parameters
        ----------
        X : np.ndarray
            Points to predict
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X_pred = np.array(X)
        
        # Simple kernel regression implementation
        predictions = []
        for x_point in X_pred:
            pred = self._predict_point(x_point)
            predictions.append(pred)
            
        return np.array(predictions)
    
    def _predict_point(self, x_point: np.ndarray) -> float:
        """Predict a single point using kernel regression."""
        # Compute distances
        if self.X_train_.ndim == 1:
            distances = np.abs(self.X_train_ - x_point)
        else:
            distances = np.sqrt(np.sum((self.X_train_ - x_point)**2, axis=1))
        
        # Compute weights using Gaussian kernel
        weights = np.exp(-0.5 * (distances / self.bw_)**2)
        weights /= np.sum(weights + 1e-10)  # Normalize
        
        # Weighted average prediction
        return np.sum(weights * self.y_train_)
    
    def _estimate_bandwidth(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate bandwidth using simple heuristic."""
        n = len(X)
        if X.ndim == 1:
            std_x = np.std(X)
        else:
            std_x = np.mean(np.std(X, axis=0))
        
        # Scott's rule adaptation
        bandwidth = std_x * (n ** (-1/(4 + X.shape[1])))
        return max(bandwidth, 1e-6)  # Avoid zero bandwidth


def create_model_subset(X: pd.DataFrame, y: pd.Series, 
                       subset_names: List[str],
                       bandwidth: Optional[float] = None) -> NonparametricRegressor:
    """
    Create and fit a nonparametric regression model for a subset of variables.
    
    This function mirrors the R model_subset functionality.
    
    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix
    y : pd.Series
        Target vector
    subset_names : List[str]
        Names of variables to include in the subset
    bandwidth : float, optional
        Bandwidth for regression. If None, uses automatic selection.
        
    Returns
    -------
    NonparametricRegressor
        Fitted model for the variable subset
    """
    # Select subset of variables
    X_subset = X[subset_names]
    
    # Create and fit model
    if bandwidth is None:
        model = NonparametricRegressor(bandwidth='scott')
    else:
        model = NonparametricRegressor(bandwidth=bandwidth)
    
    model.fit(X_subset, y)
    
    return model


def create_model_list(X: pd.DataFrame, y: pd.Series, 
                     subsets: List[np.ndarray],
                     bandwidths: Optional[Dict] = None) -> List[NonparametricRegressor]:
    """
    Create a list of fitted models for all variable subsets.
    
    This function mirrors the R model_list_fct functionality.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with column names X1, X2, etc.
    y : pd.Series
        Target vector
    subsets : List[np.ndarray]
        List of subset combinations from generate_subsets()
    bandwidths : Dict, optional
        Dictionary mapping subset sizes to bandwidth values
        
    Returns
    -------
    List[NonparametricRegressor]
        List of fitted models for all subsets
    """
    model_list = []
    
    for subset_size_idx, subset_size in enumerate(subsets):
        for subset_idx in range(subset_size.shape[0]):
            subset_indices = subset_size[subset_idx]
            
            # Convert indices to variable names (1-indexed to 0-indexed)
            subset_names = [X.columns[i-1] for i in subset_indices]
            
            # Get bandwidth for this subset size
            if bandwidths and (subset_size_idx + 1) in bandwidths:
                bw = bandwidths[subset_size_idx + 1]
            else:
                bw = None
            
            # Create and fit model
            model = create_model_subset(X, y, subset_names, bandwidth=bw)
            model_list.append(model)
    
    return model_list


def cross_validate_bandwidth(X: pd.DataFrame, y: pd.Series,
                           subset_names: List[str],
                           bandwidth_grid: Optional[List[float]] = None,
                           cv_folds: int = 5) -> float:
    """
    Select optimal bandwidth using cross-validation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    subset_names : List[str]
        Variables to include in the model
    bandwidth_grid : List[float], optional
        Grid of bandwidth values to test
    cv_folds : int, default=5
        Number of cross-validation folds
        
    Returns
    -------
    float
        Optimal bandwidth
    """
    X_subset = X[subset_names]
    
    if bandwidth_grid is None:
        # Create default bandwidth grid
        n = len(X_subset)
        base_bw = np.std(X_subset.values) * (n ** (-1/5))
        bandwidth_grid = [base_bw * factor for factor in [0.1, 0.5, 1.0, 2.0, 5.0]]
    
    best_score = float('inf')
    best_bandwidth = bandwidth_grid[0]
    
    # Simple cross-validation (could be improved)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for bandwidth in bandwidth_grid:
        scores = []
        
        for train_idx, val_idx in kf.split(X_subset):
            X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit model
            model = NonparametricRegressor(bandwidth=bandwidth)
            model.fit(X_train, y_train)
            
            # Predict and compute MSE
            y_pred = model.predict(X_val)
            mse = np.mean((y_val - y_pred)**2)
            scores.append(mse)
        
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_bandwidth = bandwidth
    
    return best_bandwidth


def model_predictions_matrix(model_list: List[NonparametricRegressor],
                           X_eval: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions from all models for evaluation points.
    
    Parameters
    ----------
    model_list : List[NonparametricRegressor]
        List of fitted models
    X_eval : pd.DataFrame
        Evaluation points
        
    Returns
    -------
    np.ndarray
        Matrix where each column contains predictions from one model
    """
    n_points = len(X_eval)
    n_models = len(model_list)
    predictions = np.zeros((n_points, n_models))
    
    for i, model in enumerate(model_list):
        # Get subset of variables for this model
        X_subset = X_eval[model.xnames]
        pred = model.predict(X_subset)
        predictions[:, i] = pred
    
    return predictions


def test_model_management():
    """Test the model management functions."""
    print("Testing model management functions...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame({
        'X1': np.random.randn(n_samples),
        'X2': np.random.randn(n_samples),
        'X3': np.random.randn(n_samples)
    })
    y = pd.Series(X['X1'] + 0.5 * X['X2'] + np.random.randn(n_samples) * 0.1)
    
    # Test single model creation
    print("Testing single model creation...")
    model = create_model_subset(X, y, ['X1', 'X2'])
    print(f"Model xnames: {model.xnames}")
    print(f"Model bandwidth: {model.bw:.6f}")
    
    # Test prediction
    X_test = pd.DataFrame({'X1': [0.0], 'X2': [0.0]})
    pred = model.predict(X_test)
    print(f"Test prediction: {pred[0]:.6f}")
    
    # Test model list creation
    print("\nTesting model list creation...")
    from subset_generation import generate_subsets
    subsets = generate_subsets(X)
    model_list = create_model_list(X, y, subsets)
    print(f"Created {len(model_list)} models")
    
    # Test predictions matrix
    X_eval = pd.DataFrame({
        'X1': [0.0, 1.0],
        'X2': [0.0, 0.0], 
        'X3': [0.0, 0.0]
    })
    pred_matrix = model_predictions_matrix(model_list, X_eval)
    print(f"Prediction matrix shape: {pred_matrix.shape}")
    
    print("✓ All model management tests completed!")


if __name__ == "__main__":
    test_model_management() 