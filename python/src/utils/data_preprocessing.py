"""
Data preprocessing utilities for Shapley value analysis.

This module provides functions for loading, cleaning, and preparing data
for nonparametric Shapley value estimation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional, Dict, Any
from pathlib import Path


def load_data(filepath: Union[str, Path], 
              target_col: Optional[str] = None,
              feature_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from CSV file and separate features and target.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file
    target_col : str, optional
        Name of the target column. If None, uses the last column.
    feature_cols : list, optional
        List of feature column names. If None, uses all columns except target.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (X) and target (y)
    """
    # Load data
    data = pd.read_csv(filepath)
    
    if target_col is None:
        # Use last column as target
        target_col = data.columns[-1]
    
    if feature_cols is None:
        # Use all columns except target as features
        feature_cols = [col for col in data.columns if col != target_col]
    
    X = data[feature_cols]
    y = data[target_col]
    
    return X, y


def prepare_r_style_data(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Prepare data in R-style format with standardized column names.
    
    This function creates a combined DataFrame with features named X1, X2, ..., Xd
    and target named Y, matching the R implementation convention.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
        
    Returns
    -------
    pd.DataFrame
        Combined data with R-style column names
    """
    # Create copy to avoid modifying original data
    X_renamed = X.copy()
    
    # Rename columns to X1, X2, ..., Xd
    n_features = X.shape[1]
    new_names = [f'X{i+1}' for i in range(n_features)]
    X_renamed.columns = new_names
    
    # Combine with target
    data = X_renamed.copy()
    data['Y'] = y
    
    return data


def create_prediction_points(X: pd.DataFrame, 
                           variable_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                           n_points: int = 48) -> Dict[str, np.ndarray]:
    """
    Create prediction points for Shapley curve estimation.
    
    This function creates evaluation points for each variable while holding
    others at their mean values, similar to the R implementation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with column names X1, X2, etc.
    variable_ranges : Dict[str, Tuple[float, float]], optional
        Dictionary mapping variable names to (min, max) ranges.
        If None, uses data-driven ranges.
    n_points : int, default=48
        Number of evaluation points for each variable
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping variable names to evaluation points
    """
    prediction_points = {}
    
    for col in X.columns:
        if variable_ranges and col in variable_ranges:
            min_val, max_val = variable_ranges[col]
        else:
            # Use data range with some padding
            min_val = X[col].min()
            max_val = X[col].max()
            
        # Create evaluation points
        points = np.linspace(min_val, max_val, n_points)
        prediction_points[col] = points
    
    return prediction_points


def create_evaluation_dataframes(X: pd.DataFrame,
                                prediction_points: Dict[str, np.ndarray],
                                variable: str) -> pd.DataFrame:
    """
    Create evaluation DataFrame for a specific variable's Shapley curve.
    
    Parameters
    ----------
    X : pd.DataFrame
        Original feature matrix
    prediction_points : Dict[str, np.ndarray]
        Dictionary of evaluation points for each variable
    variable : str
        Variable name (e.g., 'X1') to vary
        
    Returns
    -------
    pd.DataFrame
        Evaluation points with specified variable varying and others at mean
    """
    n_points = len(prediction_points[variable])
    
    # Create DataFrame with all variables at their mean values
    eval_df = pd.DataFrame()
    for col in X.columns:
        if col == variable:
            eval_df[col] = prediction_points[variable]
        else:
            eval_df[col] = X[col].mean()
    
    return eval_df


def generate_bootstrap_samples(X: pd.DataFrame, y: pd.Series, 
                             n_bootstrap: int = 800,
                             random_state: Optional[int] = None) -> list:
    """
    Generate bootstrap samples from the data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    n_bootstrap : int, default=800
        Number of bootstrap samples
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list
        List of (X_boot, y_boot) tuples
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    bootstrap_samples = []
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        X_boot = X.iloc[indices].reset_index(drop=True)
        y_boot = y.iloc[indices].reset_index(drop=True)
        
        bootstrap_samples.append((X_boot, y_boot))
    
    return bootstrap_samples


def create_mammen_noise(n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate Mammen bootstrap noise (two-point mass distribution).
    
    This implements the wild bootstrap procedure used in the R code.
    
    Parameters
    ----------
    n_samples : int
        Number of noise samples to generate
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Mammen noise vector
        
    Notes
    -----
    Mammen distribution: P(V = (1-√5)/2) = (√5+1)/(2√5), P(V = (1+√5)/2) = (√5-1)/(2√5)
    This has mean 0 and variance 1.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Mammen distribution parameters
    sqrt5 = np.sqrt(5)
    v1 = (1 - sqrt5) / 2  # ≈ -0.618
    v2 = (1 + sqrt5) / 2  # ≈ 1.618
    p1 = (sqrt5 + 1) / (2 * sqrt5)  # ≈ 0.724
    
    # Generate random values
    u = np.random.random(n_samples)
    noise = np.where(u < p1, v1, v2)
    
    return noise


def validate_data_format(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Validate that data is in the expected format for Shapley analysis.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
        
    Raises
    ------
    ValueError
        If data format is invalid
    """
    # Check that X and y have same number of samples
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")
    
    # Check for missing values
    if X.isnull().any().any():
        raise ValueError("Feature matrix X contains missing values")
    
    if y.isnull().any():
        raise ValueError("Target vector y contains missing values")
    
    # Check for infinite values
    if np.isinf(X.values).any():
        raise ValueError("Feature matrix X contains infinite values")
    
    if np.isinf(y.values).any():
        raise ValueError("Target vector y contains infinite values")
    
    # Check minimum number of samples
    if len(X) < 10:
        raise ValueError(f"Insufficient data: need at least 10 samples, got {len(X)}")


def summary_statistics(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Compute summary statistics for the dataset.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary statistics
    """
    stats = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_stats': X.describe(),
        'target_stats': y.describe(),
        'correlations': X.corrwith(y),
        'feature_correlations': X.corr()
    }
    
    return stats


def test_data_preprocessing():
    """Test the data preprocessing functions."""
    print("Testing data preprocessing functions...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    y = pd.Series(X.sum(axis=1) + np.random.randn(n_samples) * 0.1)
    
    # Test R-style data preparation
    r_data = prepare_r_style_data(X, y)
    print(f"R-style data shape: {r_data.shape}")
    print(f"R-style columns: {list(r_data.columns)}")
    
    # Test prediction points creation
    pred_points = create_prediction_points(r_data[['X1', 'X2', 'X3']])
    print(f"Prediction points created for: {list(pred_points.keys())}")
    
    # Test Mammen noise
    noise = create_mammen_noise(100)
    print(f"Mammen noise - mean: {noise.mean():.6f}, var: {noise.var():.6f}")
    
    # Test validation
    try:
        validate_data_format(X, y)
        print("✓ Data validation passed")
    except ValueError as e:
        print(f"✗ Data validation failed: {e}")
    
    print("✓ All data preprocessing tests completed!")


if __name__ == "__main__":
    test_data_preprocessing() 