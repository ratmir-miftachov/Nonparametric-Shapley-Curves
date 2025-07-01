"""
Weight functions for Shapley value computation.

This module provides functions to compute the weights used in the Shapley value
weighted sum formula, mirroring the R implementation.
"""

import numpy as np
from scipy.special import comb
from typing import List, Union
import pandas as pd


def shapley_weight(j: int, subset_vars: Union[List[str], List[int], np.ndarray], 
                   d: int, use_names: bool = True) -> float:
    """
    Compute the Shapley weight for variable j given a subset of variables.
    
    This function implements the weight calculation used in the Shapley value
    weighted sum formula, matching the R implementation.
    
    Parameters
    ----------
    j : int
        Variable index (1-indexed to match R convention)
    subset_vars : List[str], List[int], or np.ndarray
        List of variable names or indices in the subset
    d : int
        Total number of variables/dimensions
    use_names : bool, default=True
        Whether subset_vars contains variable names (True) or indices (False)
        
    Returns
    -------
    float
        Shapley weight for the given variable and subset
        
    Notes
    -----
    The weight formula is: sign * (1/d) * (choose(d-1, |S| - indicator))^(-1)
    where:
    - sign = +1 if variable j is in subset S, -1 otherwise
    - |S| is the cardinality (size) of subset S
    - indicator = 1 if j ∈ S, 0 otherwise
    - choose(n,k) is the binomial coefficient
    
    Examples
    --------
    >>> # Variable 1 is in subset {X1, X2}
    >>> weight = shapley_weight(1, ['X1', 'X2'], d=3)
    >>> weight > 0  # Should be positive since variable is in subset
    True
    
    >>> # Variable 3 is not in subset {X1, X2}  
    >>> weight = shapley_weight(3, ['X1', 'X2'], d=3)
    >>> weight < 0  # Should be negative since variable not in subset
    True
    """
    if use_names:
        # Convert variable index to variable name (X1, X2, etc.)
        var_name = f"X{j}"
        # Check if variable j is in the subset
        indicator = int(var_name in subset_vars)
    else:
        # subset_vars contains indices
        indicator = int(j in subset_vars)
    
    # Sign function: +1 if variable j is in subset, -1 otherwise
    sign = 1 if indicator > 0 else -1
    
    # Cardinality (size) of the subset
    card_s = len(subset_vars)
    
    # Compute weight using Shapley formula
    # Note: comb(n, k) computes binomial coefficient "n choose k"
    binomial_coeff = comb(d - 1, card_s - indicator)
    
    # Handle edge case where binomial coefficient is 0
    if binomial_coeff == 0:
        return 0.0
    
    weight = sign * (1.0 / d) * (1.0 / binomial_coeff)
    
    return weight


def compute_all_weights(j: int, subsets: List[np.ndarray], d: int) -> np.ndarray:
    """
    Compute Shapley weights for variable j across all possible subsets.
    
    Parameters
    ----------
    j : int
        Variable index (1-indexed)
    subsets : List[np.ndarray]
        List of all subset combinations from generate_subsets()
    d : int
        Total number of variables
        
    Returns
    -------
    np.ndarray
        Array of weights for each subset, flattened across all subset sizes
    """
    weights = []
    
    for subset_size_idx, subset_size in enumerate(subsets):
        for subset_idx in range(subset_size.shape[0]):
            subset_indices = subset_size[subset_idx]
            weight = shapley_weight(j, subset_indices, d, use_names=False)
            weights.append(weight)
    
    return np.array(weights)


def weight_by_model_index(j: int, model_index: int, model_list: List, d: int) -> float:
    """
    Compute Shapley weight for variable j and model k using model list.
    
    This function matches the exact R implementation of weight(j,k).
    
    Parameters
    ----------
    j : int
        Variable index (1-indexed)
    model_index : int
        Model index in model_list
    model_list : List
        List of fitted models
    d : int
        Total number of variables
        
    Returns
    -------
    float
        Shapley weight
    """
    # Get the subset of variables for this model
    model = model_list[model_index]
    subset_vars = model.xnames
    
    # Convert to variable indices (1-indexed)
    if hasattr(subset_vars, '__iter__') and not isinstance(subset_vars, str):
        # List of variable names like ['X1', 'X2']
        subset_indices = [int(var[1:]) for var in subset_vars if var.startswith('X')]
    else:
        # Single variable name like 'X1'
        subset_indices = [int(subset_vars[1:])]
    
    return shapley_weight(j, subset_indices, d)


def empty_set_weight(d: int) -> float:
    """
    Compute the weight for the empty set in Shapley value calculation.
    
    This corresponds to the constant term: -(1/d) * (choose(d-1, 0))^(-1)
    
    Parameters
    ----------
    d : int
        Total number of variables
        
    Returns
    -------
    float
        Weight for the empty set
    """
    return -(1.0 / d) * (1.0 / comb(d - 1, 0))


def verify_weight_properties(d: int) -> bool:
    """
    Verify that the weight function implementation is consistent.
    
    This checks that the weight calculations are mathematically sound,
    but doesn't require them to sum to 1 since the R implementation
    uses a different formulation.
    
    Parameters
    ----------
    d : int
        Number of variables
        
    Returns
    -------
    bool
        True if weights are calculated consistently
    """
    try:
        from .subset_generation import generate_subsets
    except ImportError:
        from subset_generation import generate_subsets
    
    # Create dummy data to generate subsets
    X_dummy = np.random.randn(10, d)
    subsets = generate_subsets(X_dummy)
    
    # Test basic properties: weights should be symmetric for variables
    # and opposite sign for inclusion vs exclusion
    
    # Test symmetry: weight for X1 in {X1,X2} should equal weight for X2 in {X1,X2}
    w1_in_12 = shapley_weight(1, [1, 2], d, use_names=False)
    w2_in_12 = shapley_weight(2, [1, 2], d, use_names=False)
    
    if abs(w1_in_12 - w2_in_12) > 1e-10:
        return False
    
    # Test sign: variable in subset should have positive weight,
    # variable not in subset should have negative weight
    w1_in_1 = shapley_weight(1, [1], d, use_names=False)
    w1_not_in_2 = shapley_weight(1, [2], d, use_names=False)
    
    if w1_in_1 <= 0 or w1_not_in_2 >= 0:
        return False
    
    # Test that implementation matches the R formula exactly
    # For d=3, j=1, subset {2}: should be -1/3 * 1/choose(2,1) = -1/6
    if d == 3:
        expected_weight = -1/3 * (1/2)  # -1/6
        actual_weight = shapley_weight(1, [2], 3, use_names=False)
        if abs(actual_weight - expected_weight) > 1e-10:
            return False
    
    return True


def test_weight_functions():
    """Test the weight function implementation."""
    print("Testing weight functions...")
    
    # Test with 3 variables
    d = 3
    
    # Test individual weights
    # Variable 1 in subset {X1}
    w1 = shapley_weight(1, ['X1'], d)
    print(f"Weight for X1 in {{X1}}: {w1:.6f}")
    
    # Variable 1 not in subset {X2, X3}
    w2 = shapley_weight(1, ['X2', 'X3'], d)
    print(f"Weight for X1 in {{X2, X3}}: {w2:.6f}")
    
    # Test empty set weight
    w_empty = empty_set_weight(d)
    print(f"Empty set weight: {w_empty:.6f}")
    
    # Test weight properties
    properties_ok = verify_weight_properties(d)
    print(f"Weight properties verified: {properties_ok}")
    
    if properties_ok:
        print("✓ All weight function tests passed!")
    else:
        print("✗ Weight function tests failed!")
    
    return properties_ok


if __name__ == "__main__":
    test_weight_functions() 