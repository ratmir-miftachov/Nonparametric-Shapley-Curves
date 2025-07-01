"""
Subset generation utilities for Shapley value computation.

This module provides functions to generate all possible subsets of variables
for Shapley value calculations, mirroring the R implementation.
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple, Union
import pandas as pd


def generate_subsets(X: Union[np.ndarray, pd.DataFrame]) -> List[np.ndarray]:
    """
    Generate all possible subsets of variables for Shapley value computation.
    
    This function creates all possible combinations of variables for each subset size
    from 1 to d (number of dimensions), matching the R implementation.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input data matrix with shape (n_samples, n_features)
        
    Returns
    -------
    List[np.ndarray]
        List of length d, where each element contains all combinations 
        of variables for that subset size. Each combination is represented
        as a numpy array of variable indices (1-indexed to match R).
        
    Examples
    --------
    >>> X = np.random.randn(100, 3)
    >>> subsets = generate_subsets(X)
    >>> len(subsets)  # Should be 3 (number of dimensions)
    3
    >>> subsets[0].shape  # Combinations of size 1
    (3, 1)
    >>> subsets[1].shape  # Combinations of size 2  
    (3, 2)
    >>> subsets[2].shape  # Combinations of size 3
    (1, 3)
    """
    # Get number of dimensions
    if isinstance(X, pd.DataFrame):
        d = X.shape[1]
    else:
        d = X.shape[1]
    
    # Total number of possible subsets is 2^d
    N_subs = 2**d
    
    # Create sequence of variable indices (1-indexed to match R)
    seq = np.arange(1, d + 1)
    
    # Initialize list to store combinations for each subset size
    subset_list = []
    
    # Generate combinations for each subset size k from 1 to d
    for k in range(1, d + 1):
        # Get all combinations of size k
        combs = list(combinations(seq, k))
        # Convert to numpy array with shape (n_combinations, k)
        comb_array = np.array(combs)
        subset_list.append(comb_array)
    
    return subset_list


def get_subset_names(X: pd.DataFrame, subset_indices: np.ndarray) -> List[str]:
    """
    Convert subset indices to variable names.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input data with column names
    subset_indices : np.ndarray
        Array of variable indices (1-indexed)
        
    Returns
    -------
    List[str]
        List of variable names corresponding to the indices
    """
    # Convert from 1-indexed to 0-indexed for Python
    zero_indexed = subset_indices - 1
    return [X.columns[i] for i in zero_indexed]


def subset_to_column_mask(subset_indices: np.ndarray, n_features: int) -> np.ndarray:
    """
    Convert subset indices to boolean mask for column selection.
    
    Parameters
    ----------
    subset_indices : np.ndarray
        Array of variable indices (1-indexed)
    n_features : int
        Total number of features
        
    Returns
    -------
    np.ndarray
        Boolean mask of length n_features
    """
    mask = np.zeros(n_features, dtype=bool)
    # Convert from 1-indexed to 0-indexed
    zero_indexed = subset_indices - 1
    mask[zero_indexed] = True
    return mask


def print_subset_structure(subsets: List[np.ndarray]) -> None:
    """
    Print the structure of generated subsets for debugging.
    
    Parameters
    ----------
    subsets : List[np.ndarray]
        List of subset combinations
    """
    print(f"Generated {len(subsets)} subset sizes:")
    for i, subset_size in enumerate(subsets, 1):
        print(f"  Size {i}: {subset_size.shape[0]} combinations")
        print(f"    Shape: {subset_size.shape}")
        print(f"    First few: {subset_size[:min(3, len(subset_size))]}")
        print()


# Test function to verify implementation matches R output
def test_subset_generation():
    """Test the subset generation function with a simple example."""
    # Create test data (3 variables)
    X = pd.DataFrame({
        'X1': np.random.randn(10),
        'X2': np.random.randn(10), 
        'X3': np.random.randn(10)
    })
    
    subsets = generate_subsets(X)
    
    print("Testing subset generation:")
    print_subset_structure(subsets)
    
    # Expected output for 3 variables:
    # Size 1: [[1], [2], [3]]
    # Size 2: [[1,2], [1,3], [2,3]]  
    # Size 3: [[1,2,3]]
    
    assert len(subsets) == 3, "Should have 3 subset sizes"
    assert subsets[0].shape == (3, 1), "Size 1 should have 3 combinations"
    assert subsets[1].shape == (3, 2), "Size 2 should have 3 combinations"
    assert subsets[2].shape == (1, 3), "Size 3 should have 1 combination"
    
    print("✓ All tests passed!")
    return subsets


if __name__ == "__main__":
    test_subset_generation() 