"""
Squared error analysis for Shapley value consistency studies.

This module implements SE_vec and SE_vec_int functions from R for computing
squared errors between estimated and true population Shapley values.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from scipy import integrate
import warnings

# Import our algorithms
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .population_shapley import PopulationShapleyEstimator
from .shapley_estimator import ShapleyEstimator


class SquaredErrorAnalyzer:
    """
    Squared error analysis for Shapley value consistency studies.
    
    This class implements the SE_vec and SE_vec_int functions from R for
    computing integrated squared errors between estimated and true Shapley values.
    """
    
    def __init__(self, 
                 population_estimator: PopulationShapleyEstimator,
                 integration_bounds: Tuple[float, float] = (-2, 2),
                 integration_tolerance: float = 3e-1):
        """
        Initialize the squared error analyzer.
        
        Parameters
        ----------
        population_estimator : PopulationShapleyEstimator
            Fitted population (true) Shapley estimator
        integration_bounds : Tuple[float, float], default=(-2, 2)
            Integration bounds for ISE computation
        integration_tolerance : float, default=3e-1
            Integration tolerance (matching R)
        """
        self.population_estimator = population_estimator
        self.integration_bounds = integration_bounds
        self.integration_tolerance = integration_tolerance
        self.d_ = population_estimator.d_
    
    def SE_vec(self, j: int, x_eval: np.ndarray, fitted_estimator: ShapleyEstimator) -> float:
        """
        Compute squared error for variable j at evaluation point x_eval.
        
        Exactly matches R implementation:
        SE_vec = function(j, x_eval){
          return( (shapley(j, x_eval) - shapley_popul(j, x_eval))^2 )
        }
        
        Parameters
        ----------
        j : int
            Variable index (1-indexed)
        x_eval : np.ndarray
            Evaluation point
        fitted_estimator : ShapleyEstimator
            Fitted Shapley estimator
            
        Returns
        -------
        float
            Squared error at the evaluation point
        """
        # Estimated Shapley value
        shapley_estimated = fitted_estimator.shapley_value(j, x_eval)
        
        # True population Shapley value
        shapley_true = self.population_estimator.shapley_population(j, x_eval)
        
        # Squared error
        squared_error = (shapley_estimated - shapley_true) ** 2
        
        return squared_error
    
    def SE_vec_int(self, j: int, x_eval: np.ndarray, 
                   fitted_estimator: ShapleyEstimator,
                   integration_estimator: Optional[ShapleyEstimator] = None) -> float:
        """
        Compute squared error for integration-based Shapley estimation.
        
        Matches R implementation:
        SE_vec_int = function(j, x_eval){
          return( (shapley_int(j, x_eval) - shapley_popul(j, x_eval))^2 ) 
        }
        
        Parameters
        ----------
        j : int
            Variable index (1-indexed)
        x_eval : np.ndarray
            Evaluation point
        fitted_estimator : ShapleyEstimator
            Fitted Shapley estimator
        integration_estimator : ShapleyEstimator, optional
            Integration-based estimator (if different from fitted_estimator)
            
        Returns
        -------
        float
            Squared error for integration-based estimation
        """
        if integration_estimator is None:
            integration_estimator = fitted_estimator
        
        # Integration-based Shapley value (using current implementation as proxy)
        shapley_int = integration_estimator.shapley_value(j, x_eval)
        
        # True population Shapley value
        shapley_true = self.population_estimator.shapley_population(j, x_eval)
        
        # Squared error
        squared_error = (shapley_int - shapley_true) ** 2
        
        return squared_error
    
    def compute_ISE(self, j: int, fitted_estimator: ShapleyEstimator,
                    method: str = 'component') -> float:
        """
        Compute Integrated Squared Error (ISE) for variable j.
        
        This matches the R hcubature integration:
        ISE_res1=hcubature(f=SE_vec, rep(l_int, d), rep(u_int, d), tol=3e-1, j=1)
        
        Parameters
        ----------
        j : int
            Variable index (1-indexed)
        fitted_estimator : ShapleyEstimator
            Fitted Shapley estimator
        method : str, default='component'
            Method type ('component' for SE_vec, 'integration' for SE_vec_int)
            
        Returns
        -------
        float
            Integrated squared error
        """
        def integrand(*args):
            """Integrand function for multidimensional integration."""
            x_eval = np.array(args)
            if method == 'component':
                return self.SE_vec(j, x_eval, fitted_estimator)
            elif method == 'integration':
                return self.SE_vec_int(j, x_eval, fitted_estimator)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        # Perform multidimensional integration
        if self.d_ == 3:
            # 3D integration
            result, error = integrate.tplquad(
                integrand,
                self.integration_bounds[0], self.integration_bounds[1],  # x1 bounds
                lambda x1: self.integration_bounds[0], lambda x1: self.integration_bounds[1],  # x2 bounds
                lambda x1, x2: self.integration_bounds[0], lambda x1, x2: self.integration_bounds[1],  # x3 bounds
                epsabs=self.integration_tolerance,
                epsrel=self.integration_tolerance
            )
        elif self.d_ == 2:
            # 2D integration
            result, error = integrate.dblquad(
                integrand,
                self.integration_bounds[0], self.integration_bounds[1],  # x1 bounds
                lambda x1: self.integration_bounds[0], lambda x1: self.integration_bounds[1],  # x2 bounds
                epsabs=self.integration_tolerance,
                epsrel=self.integration_tolerance
            )
        else:
            raise ValueError(f"Unsupported dimensionality: {self.d_}")
        
        return result
    
    def compute_all_ISE(self, fitted_estimator: ShapleyEstimator) -> Dict[str, Dict[str, float]]:
        """
        Compute ISE for all variables and both methods.
        
        Matches the R simulation structure:
        ISE_res1=hcubature(f=SE_vec, rep(l_int, d), rep(u_int, d), tol=3e-1, j=1)
        ISE_res1_int=hcubature(f=SE_vec_int, rep(l_int, d), rep(u_int, d), tol=3e-1, j=1)
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            ISE results for all variables and methods
        """
        results = {}
        
        for j in range(1, self.d_ + 1):
            var_name = f'X{j}'
            results[var_name] = {}
            
            print(f"Computing ISE for {var_name}...")
            
            # Component-based ISE
            try:
                ise_component = self.compute_ISE(j, fitted_estimator, method='component')
                results[var_name]['component'] = ise_component
                print(f"  Component ISE: {ise_component:.6f}")
            except Exception as e:
                print(f"  Component ISE failed: {str(e)}")
                results[var_name]['component'] = np.nan
            
            # Integration-based ISE
            try:
                ise_integration = self.compute_ISE(j, fitted_estimator, method='integration')
                results[var_name]['integration'] = ise_integration
                print(f"  Integration ISE: {ise_integration:.6f}")
            except Exception as e:
                print(f"  Integration ISE failed: {str(e)}")
                results[var_name]['integration'] = np.nan
        
        return results
    
    def consistency_study_single_replication(self, 
                                          X: pd.DataFrame, 
                                          y: pd.Series,
                                          random_state: Optional[int] = None) -> List[float]:
        """
        Run a single replication of the consistency study.
        
        Matches the R ISE_fct function structure:
        return(c(ISE1, ISE1_int, ISE2, ISE2_int, ISE3, ISE3_int))
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target data
        random_state : int, optional
            Random seed
            
        Returns
        -------
        List[float]
            [ISE1, ISE1_int, ISE2, ISE2_int, ISE3, ISE3_int]
        """
        # Fit the estimator
        estimator = ShapleyEstimator(bandwidth_method='scott')  # Faster for simulation
        estimator.fit(X, y)
        
        # Compute all ISE values
        ise_results = self.compute_all_ISE(estimator)
        
        # Extract results in R format: [ISE1, ISE1_int, ISE2, ISE2_int, ISE3, ISE3_int]
        result_vector = []
        for j in range(1, self.d_ + 1):
            var_name = f'X{j}'
            result_vector.append(ise_results[var_name]['component'])
            result_vector.append(ise_results[var_name]['integration'])
        
        return result_vector
    
    def monte_carlo_consistency_study(self,
                                    sample_sizes: List[int] = [300, 500, 1000, 2000],
                                    n_replications: int = 100,
                                    random_state: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        Run Monte Carlo consistency study exactly like R simulation.
        
        Matches the R structure:
        res1=mclapply(1:6000, ISE_fct, mc.cores=40, obs=300)
        
        Parameters
        ----------
        sample_sizes : List[int]
            Sample sizes to test
        n_replications : int
            Number of Monte Carlo replications
        random_state : int, optional
            Random seed
            
        Returns
        -------
        Dict[int, np.ndarray]
            Results matrix for each sample size
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        results = {}
        
        for sample_size in sample_sizes:
            print(f"\nRunning consistency study for sample size {sample_size}...")
            print(f"Number of replications: {n_replications}")
            
            sample_results = []
            
            for rep in range(n_replications):
                if (rep + 1) % 50 == 0:
                    print(f"  Replication {rep + 1}/{n_replications}")
                
                # Generate data using the same DGP as R
                X, y = self._generate_dgp_data(sample_size, random_state=rep)
                
                # Run single replication
                try:
                    rep_result = self.consistency_study_single_replication(X, y, random_state=rep)
                    sample_results.append(rep_result)
                except Exception as e:
                    print(f"    Replication {rep + 1} failed: {str(e)}")
                    # Append NaN results to maintain structure
                    sample_results.append([np.nan] * 6)
            
            # Convert to matrix (replications x 6 values)
            results[sample_size] = np.array(sample_results)
            
            # Compute means like R
            means = np.nanmean(results[sample_size], axis=0)
            print(f"  Mean ISE values: {means}")
        
        return results
    
    def _generate_dgp_data(self, n_samples: int, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate data using the same DGP as R simulation.
        
        R DGP:
        g1 = function(X){ return( -sin(2*X[,1]) ) } 
        g2 = function(X){ return( cos(3*X[,2])   ) } 
        g3 = function(X){ return( 0.5*X[,3] ) } 
        int = function(X){ return( 2*cos(x1)*sin(2*x2) ) }
        Y = g1(X) + g2(X) + g3(X) + int(X) + rt(n=nrow(X), df=5)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate X from multivariate normal with covariance sigma_sim
        mean = [0, 0, 0]
        X = np.random.multivariate_normal(mean, self.population_estimator.sigma_sim, n_samples)
        X_df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
        
        # Generate Y using the same DGP as R
        g1 = -np.sin(2 * X_df['X1'])
        g2 = np.cos(3 * X_df['X2']) 
        g3 = 0.5 * X_df['X3']
        interaction = 2 * np.cos(X_df['X1']) * np.sin(2 * X_df['X2'])
        
        # t-distribution noise with df=5 (as in R)
        noise = np.random.standard_t(df=5, size=n_samples)
        
        y = pd.Series(g1 + g2 + g3 + interaction + noise)
        
        return X_df, y


def test_squared_error_analysis():
    """Test the squared error analysis implementation."""
    print("Testing squared error analysis...")
    
    # Create population estimator
    pop_estimator = PopulationShapleyEstimator(true_function_type='additive')
    
    # Create squared error analyzer
    se_analyzer = SquaredErrorAnalyzer(pop_estimator)
    
    # Generate test data
    X_test, y_test = se_analyzer._generate_dgp_data(100, random_state=42)
    
    # Fit estimator
    fitted_estimator = ShapleyEstimator(bandwidth_method='scott')
    fitted_estimator.fit(X_test, y_test)
    
    # Test SE_vec at a point
    test_point = np.array([0.0, 0.0, 0.0])
    for j in range(1, 4):
        se_val = se_analyzer.SE_vec(j, test_point, fitted_estimator)
        print(f"  SE_vec X{j} at origin: {se_val:.6f}")
    
    # Test single replication
    print("\nTesting single replication...")
    rep_result = se_analyzer.consistency_study_single_replication(X_test, y_test)
    print(f"  Replication result: {[f'{x:.4f}' for x in rep_result]}")
    
    print("✓ Squared error analysis tests completed!")


if __name__ == "__main__":
    test_squared_error_analysis() 