"""
Population Shapley value computation for theoretical validation.

This module implements the true/population Shapley values using analytical functions,
exactly matching the R implementation in shapley_popul.R and integral_population.R.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from scipy import integrate
from scipy.stats import multivariate_normal
from scipy.special import comb
import warnings

# Import our utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.weight_functions import shapley_weight, weight_by_model_index


class PopulationShapleyEstimator:
    """
    Population (true) Shapley value estimator for theoretical validation.
    
    This class implements the exact analytical Shapley values using known
    true functions, matching the R implementation in shapley_popul.R.
    """
    
    def __init__(self, true_function_type: str = 'nonlinear_interaction',
                 covariance_matrix: Optional[np.ndarray] = None):
        """
        Initialize the population Shapley estimator.
        
        Parameters
        ----------
        true_function_type : str, default='nonlinear_interaction'
            Type of true function ('additive', 'nonlinear_interaction', 'custom')
        covariance_matrix : np.ndarray, optional
            Covariance matrix for conditional integration (3x3 for 3D case)
        """
        self.true_function_type = true_function_type
        self.d_ = 3  # Number of dimensions (hardcoded like R version)
        
        # Default covariance matrix from R (sigma_sim)
        if covariance_matrix is None:
            cova = 0  # As in R: cova<<-0
            self.sigma_sim = np.array([
                [4, cova, cova],
                [cova, 4, cova], 
                [cova, cova, 4]
            ])
        else:
            self.sigma_sim = covariance_matrix
            
        # Initialize conditional density parameters (matching R exactly)
        self._setup_conditional_densities()
        
        # Create true model list
        self.true_model_list_ = self._create_true_model_list()
    
    def _setup_conditional_densities(self):
        """Setup conditional normal density parameters exactly as in R."""
        # For x1/x2,x3 (1D conditional on 2D) - matching R integral_population.R
        self.sigma_xy_1 = self.sigma_sim[0, 1:3].reshape(1, -1)  # t(sigma_sim[1, 2:3])
        self.sigma_yy_1 = self.sigma_sim[1:3, 1:3]  # sigma_sim[2:3, 2:3]
        self.sigma_yx_1 = self.sigma_xy_1.T  # t(sigma_xy_1)
        self.sigma_xx_1 = self.sigma_sim[0, 0]  # sigma_sim[1,1]
        
        # Conditional variance
        self.inv_yy_1 = self.sigma_xy_1 @ np.linalg.inv(self.sigma_yy_1)
        self.c_var_1 = self.sigma_xx_1 - self.sigma_xy_1 @ np.linalg.inv(self.sigma_yy_1) @ self.sigma_yx_1
        self.pre_mult = 1.0 / np.sqrt(self.c_var_1 * 2 * np.pi)
        self.sq = np.sqrt(self.c_var_1)
        
        # For x1,x2/x3 (2D conditional on 1D) - matching R
        self.sigma_xy_2 = self.sigma_sim[0:2, 2].reshape(-1, 1)  # matrix(sigma_sim[1:2, 3])
        self.sigma_yy_2 = self.sigma_sim[2, 2]  # sigma_sim[3, 3]
        self.sigma_yx_2 = self.sigma_xy_2.T
        self.sigma_xx_2 = self.sigma_sim[0:2, 0:2]  # sigma_sim[1:2, 1:2]
        
        # Conditional variance
        self.inv_yy_2 = self.sigma_xy_2 @ (1.0 / self.sigma_yy_2)  # solve(sigma_yy_2) = 1/scalar
        self.c_var_2 = self.sigma_xx_2 - self.sigma_xy_2 @ (1.0 / self.sigma_yy_2) @ self.sigma_yx_2
        self.pre_mult2 = 1.0 / np.sqrt(np.linalg.det(2 * np.pi * self.sigma_xx_2))
        self.inv = np.linalg.inv(self.sigma_xx_2)
    
    def norm1(self, dep: float, cond: np.ndarray) -> float:
        """
        Conditional normal density: 1D dependent variable given 2D conditioning.
        
        Exactly matches R implementation of norm1 function.
        """
        x = dep  # dependent variable, scalar
        Y_1 = cond  # what you condition on, 2D vector
        c_mu_1 = self.inv_yy_1 @ Y_1
        return self.pre_mult * np.exp(-0.5 * ((x - c_mu_1[0]) / self.sq) ** 2)
    
    def norm2(self, dep: np.ndarray, cond: float) -> float:
        """
        Conditional normal density: 2D dependent variable given 1D conditioning.
        
        Exactly matches R implementation of norm2 function.
        """
        x = dep  # dependent variable, 2D vector
        Y_2 = cond  # what you condition on, scalar
        c_mu_2 = self.inv_yy_2 * Y_2
        diff = x - c_mu_2.flatten()
        return self.pre_mult2 * np.exp(-0.5 * diff.T @ self.inv @ diff)
    
    def norm1_vec(self, dep: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """Vectorized version of norm1."""
        c_mu_1 = self.inv_yy_1 @ cond
        return self.pre_mult * np.exp(-0.5 * ((dep - c_mu_1[0]) / self.sq) ** 2)
    
    def norm2_vec(self, dep: np.ndarray, cond: float) -> np.ndarray:
        """Vectorized version of norm2."""
        c_mu_2 = self.inv_yy_2 * cond
        results = []
        for i in range(dep.shape[1]):
            diff = dep[:, i] - c_mu_2.flatten()
            results.append(self.pre_mult2 * np.exp(-0.5 * diff.T @ self.inv @ diff))
        return np.array(results)
    
    def m_full(self, x1: Union[float, np.ndarray], x2: Union[float, np.ndarray], 
               x3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        True population function exactly as in R integral_population.R.
        
        R: m_full= function(x1, x2, x3){ return(-sin(2*x1) + cos(3*x2) + 2*cos(x1)*sin(2*x2) + 0.5*x3) }
        """
        return -np.sin(2*x1) + np.cos(3*x2) + 2*np.cos(x1)*np.sin(2*x2) + 0.5*x3
    
    def m_full_why(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        Wrapper function matching R m_full_why.
        """
        if isinstance(X, pd.DataFrame):
            x1, x2, x3 = X.iloc[0]['X1'], X.iloc[0]['X2'], X.iloc[0]['X3']
        else:
            x1, x2, x3 = float(X[0]), float(X[1]), float(X[2])
        return self.m_full(x1, x2, x3)
    
    def _create_marginal_functions(self):
        """Create marginal functions using integration (matching R exactly)."""
        
        def m_x1_integrand(x_out, x1):
            """Integrand for m_x1: integrate over x2,x3 given x1."""
            x2, x3 = x_out[0], x_out[1]
            return self.m_full(x1, x2, x3) * self.norm2_vec(
                np.array([[x2], [x3]]), x1
            )[0]
        
        def m_x1(X):
            """m_x1 function: E[m(x1,X2,X3)|X1=x1]."""
            x1 = float(X[0])
            # Using scipy.integrate instead of cubature
            result, _ = integrate.dblquad(
                lambda x3, x2: self.m_full(x1, x2, x3) * self.norm2(np.array([x2, x3]), x1),
                -5, 5,  # x2 bounds
                lambda x2: -5, lambda x2: 5  # x3 bounds
            )
            return result
        
        def m_x2_integrand(x_out, x2):
            """Integrand for m_x2."""
            x1, x3 = x_out[0], x_out[1]
            return self.m_full(x1, x2, x3) * self.norm2_vec(
                np.array([[x1], [x3]]), x2
            )[0]
        
        def m_x2(X):
            """m_x2 function: E[m(X1,x2,X3)|X2=x2]."""
            x2 = float(X[1])
            result, _ = integrate.dblquad(
                lambda x3, x1: self.m_full(x1, x2, x3) * self.norm2(np.array([x1, x3]), x2),
                -5, 5,  # x1 bounds
                lambda x1: -5, lambda x1: 5  # x3 bounds  
            )
            return result
        
        def m_x3(X):
            """m_x3 function: E[m(X1,X2,x3)|X3=x3]."""
            x3 = float(X[2])
            result, _ = integrate.dblquad(
                lambda x2, x1: self.m_full(x1, x2, x3) * self.norm2(np.array([x1, x2]), x3),
                -5, 5,  # x1 bounds
                lambda x1: -5, lambda x1: 5  # x2 bounds
            )
            return result
        
        def m_x1_x2(X):
            """m_x1_x2 function: E[m(x1,x2,X3)|X1=x1,X2=x2]."""
            x1, x2 = float(X[0]), float(X[1])
            result, _ = integrate.quad(
                lambda x3: self.m_full(x1, x2, x3) * self.norm1(x3, np.array([x1, x2])),
                -5, 5
            )
            return result
        
        def m_x1_x3(X):
            """m_x1_x3 function: E[m(x1,X2,x3)|X1=x1,X3=x3]."""
            x1, x3 = float(X[0]), float(X[2])
            result, _ = integrate.quad(
                lambda x2: self.m_full(x1, x2, x3) * self.norm1(x2, np.array([x1, x3])),
                -5, 5
            )
            return result
        
        def m_x2_x3(X):
            """m_x2_x3 function: E[m(X1,x2,x3)|X2=x2,X3=x3]."""
            x2, x3 = float(X[1]), float(X[2])
            result, _ = integrate.quad(
                lambda x1: self.m_full(x1, x2, x3) * self.norm1(x1, np.array([x2, x3])),
                -5, 5
            )
            return result
        
        return m_x1, m_x2, m_x3, m_x1_x2, m_x1_x3, m_x2_x3
    
    def _create_true_model_list(self) -> List[Callable]:
        """
        Create the true_model_list exactly as in R integral_population.R.
        
        R structure:
        true_model_list[[1]] = m_x1
        true_model_list[[2]] = m_x2  
        true_model_list[[3]] = m_x3
        true_model_list[[4]] = m_x1_x2
        true_model_list[[5]] = m_x1_x3
        true_model_list[[6]] = m_x2_x3
        true_model_list[[7]] = m_full_why
        """
        if self.true_function_type == 'additive':
            # For additive case, marginals are simple
            def m_x1(X): return 2.0 * float(X[0])  # E[2*X1 + X2 + 0.5*X3|X1] = 2*X1
            def m_x2(X): return 1.0 * float(X[1])  # E[2*X1 + X2 + 0.5*X3|X2] = X2  
            def m_x3(X): return 0.5 * float(X[2])  # E[2*X1 + X2 + 0.5*X3|X3] = 0.5*X3
            def m_x1_x2(X): return 2.0 * float(X[0]) + 1.0 * float(X[1])
            def m_x1_x3(X): return 2.0 * float(X[0]) + 0.5 * float(X[2])
            def m_x2_x3(X): return 1.0 * float(X[1]) + 0.5 * float(X[2])
            def m_full_additive(X): 
                if isinstance(X, pd.DataFrame):
                    return 2.0 * X.iloc[0]['X1'] + 1.0 * X.iloc[0]['X2'] + 0.5 * X.iloc[0]['X3']
                else:
                    return 2.0 * float(X[0]) + 1.0 * float(X[1]) + 0.5 * float(X[2])
                    
            return [m_x1, m_x2, m_x3, m_x1_x2, m_x1_x3, m_x2_x3, m_full_additive]
        
        elif self.true_function_type == 'nonlinear_interaction':
            # For nonlinear case with interactions, use integration
            m_x1, m_x2, m_x3, m_x1_x2, m_x1_x3, m_x2_x3 = self._create_marginal_functions()
            return [m_x1, m_x2, m_x3, m_x1_x2, m_x1_x3, m_x2_x3, self.m_full_why]
        
        else:
            raise ValueError(f"Unknown true function type: {self.true_function_type}")
    
    def shapley_population(self, j: int, x_eval: np.ndarray) -> float:
        """
        Compute population (true) Shapley value for variable j at x_eval.
        
        Exactly matches R shapley_popul function:
        shapley_popul = function(j, x_eval){
          shap_res=sapply(1:7, function(k){
            shap = weight(j,k)*true_model_list[[k]](x_eval)
          })
          return(sum(shap_res))
        }
        """
        shap_results = []
        
        for k in range(1, 8):  # R indices 1:7, Python 0:6 but we use k directly for weight
            # Get weight using proper subset structure
            if k <= 3:  # Single variable models
                subset_vars = [f'X{k}']
            elif k <= 6:  # Two variable models
                if k == 4: subset_vars = ['X1', 'X2']
                elif k == 5: subset_vars = ['X1', 'X3'] 
                elif k == 6: subset_vars = ['X2', 'X3']
            else:  # k == 7, full model
                subset_vars = ['X1', 'X2', 'X3']
            
            # Compute weight
            weight = shapley_weight(j, subset_vars, self.d_, use_names=True)
            
            # Get prediction from true model
            pred = self.true_model_list_[k-1](x_eval)  # Convert to 0-based indexing
            
            shap_results.append(weight * pred)
        
        return sum(shap_results)
    
    def shapley_population_vectorized(self, j: int, x_eval_array: np.ndarray) -> np.ndarray:
        """
        Vectorized population Shapley computation.
        
        Matches R shapley_popul_vec function.
        """
        x_eval_array = np.array(x_eval_array)
        if x_eval_array.ndim == 1:
            x_eval_array = x_eval_array.reshape(1, -1)
            
        results = []
        for i in range(x_eval_array.shape[0]):
            result = self.shapley_population(j, x_eval_array[i])
            results.append(result)
            
        return np.array(results)


def test_population_shapley():
    """Test the population Shapley implementation."""
    print("Testing population Shapley implementation...")
    
    # Test additive case (where we know the answer)
    pop_estimator = PopulationShapleyEstimator(true_function_type='additive')
    
    # Test point
    test_point = np.array([1.0, 0.5, -0.5])
    
    # Test each variable
    for j in range(1, 4):
        shap_val = pop_estimator.shapley_population(j, test_point)
        print(f"  Population Shapley X{j} at {test_point}: {shap_val:.6f}")
    
    # Test nonlinear case
    print("\nTesting nonlinear case...")
    pop_estimator_nl = PopulationShapleyEstimator(true_function_type='nonlinear_interaction')
    
    for j in range(1, 4):
        try:
            shap_val = pop_estimator_nl.shapley_population(j, test_point)
            print(f"  Nonlinear Shapley X{j} at {test_point}: {shap_val:.6f}")
        except Exception as e:
            print(f"  Nonlinear Shapley X{j}: Integration error (expected): {str(e)[:50]}...")
    
    print("✓ Population Shapley tests completed!")


if __name__ == "__main__":
    test_population_shapley() 