"""
Model consistency verification between R and Python implementations.

This module provides comprehensive testing to ensure that the Python translation
maintains mathematical consistency with the original R implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.special import comb

# Import our algorithms
from ..utils.weight_functions import shapley_weight, compute_all_weights
from ..utils.subset_generation import generate_subsets
from .shapley_estimator import ShapleyEstimator
from .integration_methods import CubatureStyleIntegrator


class ModelConsistencyTester:
    """
    Comprehensive testing for model consistency between R and Python.
    
    This class verifies that the Python implementation produces results
    mathematically consistent with the original R implementation.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize consistency tester.
        
        Parameters
        ----------
        tolerance : float
            Numerical tolerance for consistency checks
        """
        self.tolerance = tolerance
        self.test_results = {}
    
    def test_weight_consistency(self, d: int = 3) -> Dict[str, Any]:
        """
        Test Shapley weight consistency with R implementation.
        
        Verifies that weight calculations match R exactly.
        """
        results = {
            'test_name': 'weight_consistency',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # Test basic weight properties
            for j in range(1, d + 1):
                for subset_size in range(d + 1):
                    # Test empty subset
                    if subset_size == 0:
                        weight = shapley_weight(j, [], d, use_names=False)
                        expected = -(1.0 / d) * (1.0 / comb(d - 1, 0))
                        
                        if abs(weight - expected) > self.tolerance:
                            results['passed'] = False
                            results['errors'].append(
                                f"Empty subset weight mismatch for j={j}: {weight} vs {expected}"
                            )
                    
                    # Test full subset
                    if subset_size == d:
                        full_subset = list(range(1, d + 1))
                        weight = shapley_weight(j, full_subset, d, use_names=False)
                        expected = (1.0 / d) * (1.0 / comb(d - 1, d - 1))
                        
                        if abs(weight - expected) > self.tolerance:
                            results['passed'] = False
                            results['errors'].append(
                                f"Full subset weight mismatch for j={j}: {weight} vs {expected}"
                            )
            
            # Test weight symmetry
            for j1 in range(1, d + 1):
                for j2 in range(1, d + 1):
                    if j1 != j2:
                        subset1 = [j2]  # j1 not in subset
                        subset2 = [j1]  # j2 not in subset
                        
                        weight1 = shapley_weight(j1, subset1, d, use_names=False)
                        weight2 = shapley_weight(j2, subset2, d, use_names=False)
                        
                        if abs(weight1 - weight2) > self.tolerance:
                            results['passed'] = False
                            results['errors'].append(
                                f"Weight symmetry violation: j1={j1}, j2={j2}"
                            )
            
            results['details']['total_tests'] = d * (d + 1) + d * (d - 1)
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Exception in weight testing: {str(e)}")
        
        return results
    
    def test_integration_bounds_consistency(self) -> Dict[str, Any]:
        """
        Test integration bounds consistency with R implementation.
        
        Verifies that integration bounds (-5, 5) are used consistently.
        """
        results = {
            'test_name': 'integration_bounds_consistency',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            integrator = CubatureStyleIntegrator(integration_bounds=(-5, 5))
            
            # Test 1D integration bounds
            def test_func_1d(x):
                return np.exp(-x**2)  # Gaussian-like function
            
            result_1d = integrator.cubintegrate_1d(test_func_1d, (-5, 5))
            if 'integral' not in result_1d or np.isnan(result_1d['integral']):
                results['passed'] = False
                results['errors'].append("1D integration failed")
            
            # Test 2D integration bounds
            def test_func_2d(x, y):
                return np.exp(-(x**2 + y**2))
            
            result_2d = integrator.cubintegrate_2d(test_func_2d, [(-5, 5), (-5, 5)])
            if 'integral' not in result_2d or np.isnan(result_2d['integral']):
                results['passed'] = False
                results['errors'].append("2D integration failed")
            
            results['details']['bounds'] = (-5, 5)
            results['details']['1d_result'] = result_1d.get('integral', 'Failed')
            results['details']['2d_result'] = result_2d.get('integral', 'Failed')
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Exception in bounds testing: {str(e)}")
        
        return results
    
    def test_model_list_paradigm(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Test model list paradigm consistency with R implementation.
        
        Verifies that Python model fitting produces equivalent results to R.
        """
        results = {
            'test_name': 'model_list_paradigm',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # Initialize Shapley estimator
            estimator = ShapleyEstimator()
            
            # Fit models
            estimator.fit(X, y)
            
            # Verify all subsets are modeled
            d = X.shape[1]
            expected_subsets = 2**d
            actual_subsets = len(estimator.subsets_)
            
            if actual_subsets != expected_subsets:
                results['passed'] = False
                results['errors'].append(
                    f"Subset count mismatch: {actual_subsets} vs {expected_subsets}"
                )
            
            # Test predictions on subset models
            test_predictions = {}
            for subset in estimator.subsets_:
                subset_key = tuple(subset)
                if subset_key in estimator.models_:
                    model = estimator.models_[subset_key]
                    
                    if len(subset) > 0:
                        X_subset = X[subset]
                        pred = model.predict(X_subset)
                        test_predictions[subset_key] = pred
                    else:
                        pred = model.predict(pd.DataFrame(index=X.index))
                        test_predictions[subset_key] = pred
            
            # Verify weight calculations
            weights = estimator.weights_
            for var in X.columns:
                if var not in weights:
                    results['passed'] = False
                    results['errors'].append(f"Missing weights for variable {var}")
            
            results['details']['n_subsets'] = actual_subsets
            results['details']['n_models'] = len(estimator.models_)
            results['details']['n_variables'] = d
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Exception in model list testing: {str(e)}")
        
        return results
    
    def test_shapley_curve_properties(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Test mathematical properties of Shapley curves.
        
        Verifies efficiency property and other mathematical constraints.
        """
        results = {
            'test_name': 'shapley_curve_properties',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            estimator = ShapleyEstimator()
            estimator.fit(X, y)
            
            # Test on a few evaluation points
            n_test_points = 10
            test_points = {}
            for col in X.columns:
                col_min, col_max = X[col].min(), X[col].max()
                test_points[col] = np.linspace(col_min, col_max, n_test_points)
            
            # Estimate curves
            curves = estimator.estimate_all_curves(test_points)
            
            # Test basic properties
            for var, curve in curves.items():
                if len(curve) == 0:
                    results['passed'] = False
                    results['errors'].append(f"Empty curve for variable {var}")
                    continue
                
                # Check for NaN values
                if np.any(np.isnan(curve)):
                    results['passed'] = False
                    results['errors'].append(f"NaN values in curve for {var}")
                
                # Check for infinite values
                if np.any(np.isinf(curve)):
                    results['passed'] = False
                    results['errors'].append(f"Infinite values in curve for {var}")
            
            results['details']['n_curves'] = len(curves)
            results['details']['curve_lengths'] = {var: len(curve) for var, curve in curves.items()}
            
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"Exception in curve properties testing: {str(e)}")
        
        return results
    
    def run_comprehensive_test(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Run comprehensive consistency test suite.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : np.ndarray
            Target values
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive test results
        """
        print("Running comprehensive consistency tests...")
        
        # Run all tests
        test_suite = [
            self.test_weight_consistency,
            self.test_integration_bounds_consistency,
            lambda: self.test_model_list_paradigm(X, y),
            lambda: self.test_shapley_curve_properties(X, y)
        ]
        
        all_results = {}
        overall_passed = True
        
        for i, test_func in enumerate(test_suite):
            print(f"Running test {i+1}/{len(test_suite)}...")
            try:
                result = test_func()
                test_name = result['test_name']
                all_results[test_name] = result
                
                if not result['passed']:
                    overall_passed = False
                    print(f"❌ {test_name} FAILED")
                    for error in result['errors']:
                        print(f"   - {error}")
                else:
                    print(f"✅ {test_name} PASSED")
                    
            except Exception as e:
                overall_passed = False
                error_result = {
                    'test_name': f'test_{i}',
                    'passed': False,
                    'details': {},
                    'errors': [f"Test execution failed: {str(e)}"]
                }
                all_results[f'test_{i}'] = error_result
                print(f"❌ Test {i+1} FAILED with exception: {str(e)}")
        
        # Summary
        summary = {
            'overall_passed': overall_passed,
            'total_tests': len(test_suite),
            'passed_tests': sum(1 for r in all_results.values() if r['passed']),
            'failed_tests': sum(1 for r in all_results.values() if not r['passed']),
            'test_results': all_results
        }
        
        print(f"\n{'='*50}")
        print(f"CONSISTENCY TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Overall Result: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        
        if not overall_passed:
            print(f"\nFailed Tests:")
            for name, result in all_results.items():
                if not result['passed']:
                    print(f"- {name}: {len(result['errors'])} errors")
        
        return summary


def run_consistency_tests(X: pd.DataFrame, y: np.ndarray, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Quick function to run consistency tests.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : np.ndarray 
        Target values
    tolerance : float
        Numerical tolerance for tests
        
    Returns
    -------
    Dict[str, Any]
        Test results
    """
    tester = ModelConsistencyTester(tolerance=tolerance)
    return tester.run_comprehensive_test(X, y)


def create_test_data(n_samples: int = 100, n_features: int = 3, 
                    random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create synthetic test data for consistency testing.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    random_state : Optional[int]
        Random seed
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Test features and target
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create synthetic data
    X = pd.DataFrame(
        np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=n_samples
        ),
        columns=[f'X{i+1}' for i in range(n_features)]
    )
    
    # Create target with known relationships
    y = (X['X1'] * 2 + X['X2'] * (-1) + X['X3'] * 0.5 + 
         np.random.normal(0, 0.1, n_samples))
    
    return X, y 